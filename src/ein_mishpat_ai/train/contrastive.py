"""M4 contrastive + distillation training of the students.

torchrun --nproc_per_node 8 scripts/train.py fit configs/train/<run>.yaml

- Each step is one task (in-batch negatives are meaningful); the task's group (d1 / d2 / d3a) is
  drawn with the configured mixture, deterministically from the seed.
- Loss: InfoNCE of each query against all positives + mined negatives of the global batch
  (gathered over GPUs; other copies of the query's own positives are masked), symmetric
  (positive → query) for translation tasks; plus λ · KL(teacher ‖ student) over the query's own
  candidates (positive + its negatives) with the teacher's stored scores. Tasks without any mined
  negative (D3a he→en; D3a en→he before round-trip mining) train on in-batch negatives only.
- Large batches by gradient caching: embeddings without grad, loss on the embeddings, then each
  chunk is re-encoded with grad and receives its embedding gradient (Gao et al. 2021, GradCache).
- Checkpoint every `save_every` steps (model + optimizer + step); a restart resumes from the last.
- Dev: R@1 on the held-out D1 slice (data/train/mix/dev_d1.parquet) every `eval_every` steps.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ein_mishpat_ai.utils.code_hash import code_hash
from ein_mishpat_ai.utils.io import load_yaml, now_iso, project_root, resolve
from ein_mishpat_ai.utils.log import get_logger
from ein_mishpat_ai.utils.seed import set_seed

log = get_logger("train", "train.log")
SYMMETRIC = {"bitext_he2en", "bitext_en2he", "d3a_he2en", "d3a_en2he"}


def schedule(ex: pl.DataFrame, cfg: dict[str, Any]) -> list[tuple[str, np.ndarray]]:
    """Global batches (task, row indices) for the whole run; identical on every rank."""
    rng = np.random.default_rng(cfg["seed"])
    B = cfg["batch_size"]
    mixed = cfg["mixture"]
    groups = {g: w for g, w in mixed.items() if w > 0 and (ex["group"] == g).any()}
    wsum = sum(groups.values())
    n_steps = int(cfg["data_fraction"] * cfg["epochs"] * ex.height / B)
    tasks: dict[str, list[tuple[str, np.ndarray]]] = {}
    for g in groups:
        rows = ex.with_row_index("i").filter(pl.col("group") == g)
        tasks[g] = [
            (t, d["i"].to_numpy()) for (t,), d in rows.group_by("task", maintain_order=True)
        ]

    def batches(g: str):
        while True:  # one pass over the group's tasks, reshuffled every pass
            out = []
            for t, idx in tasks[g]:
                p = rng.permutation(idx)
                out += [(t, p[i : i + B]) for i in range(0, len(p) - B + 1, B)]
            for k in rng.permutation(len(out)):
                yield out[k]

    its = {g: batches(g) for g in groups}
    names = list(groups)
    draw = rng.choice(len(names), size=n_steps, p=[groups[g] / wsum for g in names])
    return [next(its[names[k]]) for k in draw]


def _lr(step: int, total: int, warmup: float) -> float:
    w = max(1, int(warmup * total))
    if step < w:
        return (step + 1) / w
    return 0.5 * (1 + math.cos(math.pi * (step - w) / max(1, total - w)))


def fit(config: str) -> None:
    import torch
    import torch.distributed as dist
    from sentence_transformers import SentenceTransformer

    cfg = load_yaml(config)
    tc = load_yaml("configs/teacher.yaml")
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    set_seed(cfg["seed"])
    out = resolve(cfg["out_dir"])
    B = cfg["batch_size"]
    assert B % world == 0, "batch_size must be divisible by the number of GPUs"
    b = B // world
    k = cfg["negatives"]

    mix = resolve(cfg["mix_dir"])
    ex = pl.read_parquet(mix / "examples.parquet")
    tx = pl.read_parquet(mix / "texts.parquet")
    text = dict(zip((tx["lang"] + "\t" + tx["ref"]).to_list(), tx["text"].to_list(), strict=True))
    del tx
    sched = schedule(ex, cfg)
    total = len(sched)
    instr = {t: v["instruction"] for t, v in tc["tasks"].items()}
    instr["d3a_he2en"] = instr["d3a_en2he"] = instr["bitext_he2en"]
    instr["d3b_he2he"] = instr["d3b_en2he"] = load_yaml("configs/gen.yaml")["d3b"]["instruction"]
    cols = {c: ex[c].to_list() for c in ex.columns}
    mined = {
        t for t, n in zip(cols["task"], cols["negatives"], strict=True) if n
    }  # tasks with hard negatives

    # model: resume from the last checkpoint if there is one
    ckpts = sorted((out / "ckpt").glob("step*"), key=lambda p: int(p.name[4:]))
    start = 0
    src = load_yaml("configs/eval/models.yaml")["models"][cfg["model"]]["hf_id"]
    lora = cfg.get("lora")  # flagship: frozen base model + trainable LoRA adapter
    if ckpts:
        start = int(ckpts[-1].name[4:])
        if not lora:
            src = str(ckpts[-1])
    dtype = (
        torch.bfloat16 if lora else torch.float32
    )  # frozen base: half the memory; adapter stays float32
    model = SentenceTransformer(src, device=str(dev), model_kwargs={"torch_dtype": dtype})
    model[0].auto_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    peft_model = None
    if lora:  # peft adds the LoRA layers in place; only peft_model can save just the adapter
        init = cfg.get("init_adapter")  # continuing an earlier LoRA run (new learning-rate cycle)
        adapter = ckpts[-1] / "adapter" if ckpts else resolve(init) if init else None
        peft_model = _add_lora(model[0].auto_model, lora, adapter)
        if rank == 0:
            n = sum(p.numel() for p in model.parameters() if p.requires_grad)
            log.info(
                "lora: %d trainable of %d parameters", n, sum(p.numel() for p in model.parameters())
            )
    model.train()
    net = (
        torch.nn.parallel.DistributedDataParallel(model, device_ids=[local]) if world > 1 else model
    )
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _lr(s, total, cfg["warmup"]))
    if ckpts:
        st = torch.load(ckpts[-1] / "trainer_state.pt", map_location=dev)
        opt.load_state_dict(st["optimizer"])
        sch.load_state_dict(st["scheduler"])

    def tok(texts: list[str], max_len: int) -> dict[str, Any]:
        model.max_seq_length = max_len
        f = model.tokenize(texts)
        return {n: v.to(dev) if hasattr(v, "to") else v for n, v in f.items()}

    def embed(texts: list[str], max_len: int, chunk: int) -> torch.Tensor:
        outs = []
        for i in range(0, len(texts), chunk):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outs.append(net(tok(texts[i : i + chunk], max_len))["sentence_embedding"].float())
        return torch.cat(outs)

    def gather(x: torch.Tensor) -> torch.Tensor:
        """All ranks' rows, in rank order; this rank's slice keeps its autograd graph."""
        if world == 1:
            return x
        parts = [torch.empty_like(x) for _ in range(world)]
        dist.all_gather(parts, x.detach().contiguous())
        parts[rank] = x
        return torch.cat(parts)

    def gather_obj(x: Any) -> list[Any]:
        if world == 1:
            return [x]
        parts: list[Any] = [None] * world
        dist.all_gather_object(parts, x)
        return parts

    run_id = None
    if rank == 0:
        import mlflow

        exp = cfg["experiment"]
        if mlflow.get_experiment_by_name(exp) is None:
            mlflow.create_experiment(
                exp, artifact_location=(project_root() / "mlruns" / "artifacts").as_uri()
            )
        mlflow.set_experiment(exp)
        rid = out / "mlflow_run_id"
        out.mkdir(parents=True, exist_ok=True)
        run = mlflow.start_run(
            run_id=rid.read_text().strip() if rid.exists() else None, run_name=cfg["run_name"]
        )
        run_id = run.info.run_id
        rid.write_text(run_id)
        if not start:
            mlflow.set_tag("code_hash", code_hash())
            mlflow.log_params({k_: v for k_, v in _flat(cfg).items()})
            (out / "config.yaml").write_text((project_root() / config).read_text())
            (out / "schedule_stats.json").write_text(
                json.dumps(
                    {
                        "steps": total,
                        "by_task": {
                            t: sum(1 for x, _ in sched if x == t)
                            for t in sorted({x for x, _ in sched})
                        },
                        "code_hash": code_hash(),
                        "started": now_iso(),
                    },
                    indent=2,
                )
            )
        log.info(
            "fit %s: %d steps, start %d, world %d, examples %d",
            cfg["run_name"],
            total,
            start,
            world,
            ex.height,
        )

    tau, lam = cfg["temperature"], cfg["distill"]["weight"]
    tq, ts = cfg["distill"]["teacher_temperature"], cfg["distill"]["student_temperature"]
    ml = cfg["max_length"]
    if start == 0:  # the base model's dev numbers, at step 0
        if rank == 0:
            import mlflow

            r = dev_d1(model, text, mix, instr["bitext_he2en"], ml)
            mlflow.log_metrics(r, step=0)
            log.info("dev after 0 steps: %s", r)
            (out / "dev_start.json").write_text(json.dumps(r))
            model.train()
        if world > 1:
            dist.barrier()
    for step in range(start, total):
        t0 = time.time()
        task, idx = sched[step]
        mine = idx[rank * b : (rank + 1) * b]
        rng = np.random.default_rng([cfg["seed"], step, rank])
        prefix = f"Instruct: {instr[task]}\nQuery: "
        q_txt, pos_txt, neg_txt, pos_ref, neg_ref, pos_sets, t_rows = [], [], [], [], [], [], []
        kk = k if task in mined else 0
        for i in mine:
            ql, tl = cols["qlang"][i], cols["tlang"][i]
            ps = cols["positives"][i]
            j = int(rng.integers(len(ps)))
            q_txt.append(prefix + text[f"{ql}\t{cols['query_ref'][i]}"])
            pos_txt.append(text[f"{tl}\t{ps[j]}"])
            pos_ref.append(ps[j])
            pos_sets.append(set(ps))
            if kk:
                negs = cols["negatives"][i]
                if not negs:
                    # ~45% of commentary/untyped queries keep no mined negative (every candidate
                    # scored > 0.95 × the positive): borrow other rows' mined negatives of this
                    # batch (masked if they are this row's positives); no teacher scores → no KL
                    pool = [n for x in mine for n in cols["negatives"][x]] or [
                        cols["positives"][x][0] for x in mine if x != i
                    ]
                    sel = rng.choice(len(pool), size=kk, replace=len(pool) < kk)
                    neg_txt += [text[f"{tl}\t{pool[s]}"] for s in sel]
                    neg_ref += [pool[s] for s in sel]
                    t_rows.append(None)
                    continue
                sel = rng.choice(len(negs), size=kk, replace=len(negs) < kk)
                neg_txt += [text[f"{tl}\t{negs[s]}"] for s in sel]
                neg_ref += [negs[s] for s in sel]
                tp, tn = cols["t_pos"][i], cols["t_neg"][i]
                row = [tp[j], *[tn[s] for s in sel]] if tp is not None else None
                t_rows.append(None if row is None or None in row else row)  # no score → no KL
        # local doc layout: [positive of row 0..b-1, kk negatives of row 0, of row 1, ...]
        local_docs = pos_txt + neg_txt

        # 1) embeddings without grad
        with torch.no_grad():
            rq = embed(q_txt, ml["query"], cfg["chunk_nograd"])
            rd = embed(local_docs, ml["doc"], cfg["chunk_nograd"])
        rq.requires_grad_(True)
        rd.requires_grad_(True)

        # 2) loss on the embeddings of the whole global batch
        Q, D = gather(rq), gather(rd)
        doc_refs = sum(gather_obj(pos_ref + neg_ref), [])
        all_pos = sum(gather_obj(pos_sets), [])
        teacher = sum(gather_obj(t_rows), []) if kk else []
        loss, parts = _loss(
            Q, D, b, kk, world, doc_refs, all_pos, teacher, tau, lam, tq, ts, task in SYMMETRIC
        )
        (loss * world).backward()
        gq, gd = rq.grad, rd.grad

        # 3) re-encode each chunk with grad and push its embedding gradient through the model
        chunks = [("q", i) for i in range(0, len(q_txt), cfg["chunk"])]
        chunks += [("d", i) for i in range(0, len(local_docs), cfg["chunk"])]
        for n, (kind, i) in enumerate(chunks):
            last = n == len(chunks) - 1
            ctx = net.no_sync() if world > 1 and not last else _null()
            with ctx:
                if kind == "q":
                    e = embed(q_txt[i : i + cfg["chunk"]], ml["query"], cfg["chunk"])
                    (e * gq[i : i + cfg["chunk"]]).sum().backward()
                else:
                    e = embed(local_docs[i : i + cfg["chunk"]], ml["doc"], cfg["chunk"])
                    (e * gd[i : i + cfg["chunk"]]).sum().backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
        opt.step()
        sch.step()
        opt.zero_grad(set_to_none=True)

        if rank == 0 and (step % cfg["log_every"] == 0 or step == total - 1):
            import mlflow

            m = {
                **parts,
                "loss": float(loss),
                "grad_norm": float(gn),
                "lr": sch.get_last_lr()[0],
                "step_s": time.time() - t0,
            }
            if lora:
                m["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
            mlflow.log_metrics(m, step=step)
            log.info(
                "step %d/%d %s %s",
                step,
                total,
                task,
                {k_: float(f"{v:.4g}") for k_, v in m.items()},
            )
        if (step + 1) % cfg["eval_every"] == 0 or step == total - 1:
            if rank == 0:
                r = dev_d1(model, text, mix, instr["bitext_he2en"], ml)
                import mlflow

                mlflow.log_metrics(r, step=step + 1)  # dev metrics are indexed by steps done
                log.info("dev after %d steps: %s", step + 1, r)
                model.train()
            if world > 1:
                dist.barrier()
        if (step + 1) % cfg["save_every"] == 0 or step == total - 1:
            if rank == 0:
                _save(model, opt, sch, out, step + 1, cfg["keep_ckpts"], peft_model)
                if step + 1 in cfg.get("keep_steps", []):  # adapters kept for the learning curve
                    shutil.copytree(
                        out / "ckpt" / f"step{step + 1}" / "adapter",
                        out / "curve" / f"step{step + 1}",
                        dirs_exist_ok=True,
                    )
            if world > 1:
                dist.barrier()

    if rank == 0:
        if lora:
            peft_model.save_pretrained(str(out / "final" / "adapter"))
            if cfg.get("save_merged", True):  # the release model: adapter merged into the base
                peft_model.merge_and_unload()  # in place, on the model inside `model`
                if any("lora_" in n for n in model.state_dict()):
                    raise RuntimeError("LoRA layers still present after merging")
                log.info(
                    "dev after merging: %s", dev_d1(model, text, mix, instr["bitext_he2en"], ml)
                )
                model.save(str(out / "final"))
        else:
            model.save(str(out / "final"))
        (out / "final" / "trained_from.json").write_text(
            json.dumps(
                {
                    "config": config,
                    "steps": total,
                    "code_hash": code_hash(),
                    "finished": now_iso(),
                    "mlflow_run_id": run_id,
                },
                indent=2,
            )
        )
        import mlflow

        mlflow.end_run()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


def _loss(
    Q: Any,
    D: Any,
    b: int,
    kk: int,
    world: int,
    doc_refs: list[str],
    pos_sets: list[set[str]],
    teacher: list[list[float] | None],
    tau: float,
    lam: float,
    tq: float,
    ts: float,
    symmetric: bool,
) -> tuple[Any, dict[str, float]]:
    """InfoNCE over the global batch (+ reverse direction for translation tasks) + λ·KL to the
    teacher over each query's own candidates. Ranks' docs are laid out as in `fit`."""
    import torch
    import torch.nn.functional as F

    n, per = Q.shape[0], b * (1 + kk)
    r, j = np.divmod(np.arange(n), b)
    labels = r * per + j
    where: dict[str, list[int]] = {}
    for x, ref in enumerate(doc_refs):
        where.setdefault(ref, []).append(x)
    mask = torch.zeros(n, D.shape[0], dtype=torch.bool)
    for g, ps in enumerate(pos_sets):  # other copies of the query's own positives are not negatives
        for ref in ps:
            for x in where.get(ref, ()):
                if x != labels[g]:
                    mask[g, x] = True
    lab = torch.from_numpy(labels).to(Q.device)
    s = (Q @ D.T / tau).masked_fill(mask.to(Q.device), float("-inf"))
    ce = F.cross_entropy(s, lab)
    parts = {"ce": float(ce)}
    if symmetric:
        ce2 = F.cross_entropy(D[lab] @ Q.T / tau, torch.arange(n, device=Q.device))
        parts["ce_rev"] = float(ce2)
        ce = (ce + ce2) / 2
    loss = ce
    rows = [g for g, t in enumerate(teacher) if t is not None]
    if kk and lam > 0 and rows:
        cand = np.stack([[labels[g], *(r[g] * per + b + j[g] * kk + np.arange(kk))] for g in rows])
        c = torch.from_numpy(cand).to(Q.device)
        st = (Q[rows].unsqueeze(1) * D[c]).sum(-1) / ts
        tt = torch.tensor([teacher[g] for g in rows], device=Q.device) / tq
        kl = F.kl_div(
            F.log_softmax(st, -1), F.log_softmax(tt, -1), log_target=True, reduction="batchmean"
        )
        parts["kl"] = float(kl)
        loss = loss + lam * kl
    return loss, parts


def dev_d1(
    model: Any, text: dict[str, str], mix: Path, instruction: str, ml: dict[str, int]
) -> dict[str, float]:
    """R@1 he→en and en→he on the held-out D1 slice (pool = the slice's other side)."""
    import torch

    refs = pl.read_parquet(mix / "dev_d1.parquet")["ref"].to_list()
    he = [text[f"he\t{r}"] for r in refs]
    en = [text[f"en\t{r}"] for r in refs]
    p = f"Instruct: {instruction}\nQuery: "
    model.eval()
    res = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for name, q, d in (("he2en", he, en), ("en2he", en, he)):
            model.max_seq_length = ml["query"]
            qe = model.encode(
                [p + x for x in q], batch_size=64, normalize_embeddings=True, convert_to_tensor=True
            )
            model.max_seq_length = ml["doc"]
            de = model.encode(d, batch_size=64, normalize_embeddings=True, convert_to_tensor=True)
            hit = (qe.float() @ de.float().T).argmax(1).cpu().numpy() == np.arange(len(refs))
            res[f"dev_d1_{name}_r1"] = float(hit.mean())
    return res


def _add_lora(auto_model: Any, lc: dict[str, Any], adapter: Path | None) -> Any:
    """Wrap the frozen base in a LoRA adapter: a new one, or the checkpoint's when resuming."""
    from peft import LoraConfig, PeftModel, get_peft_model

    auto_model.enable_input_require_grads()
    if adapter:
        return PeftModel.from_pretrained(auto_model, str(adapter), is_trainable=True)
    return get_peft_model(
        auto_model,
        LoraConfig(
            r=lc["r"],
            lora_alpha=lc["alpha"],
            target_modules="all-linear",
            lora_dropout=lc.get("dropout", 0.0),
        ),
    )


def _save(
    model: Any, opt: Any, sch: Any, out: Path, step: int, keep: int, peft_model: Any = None
) -> None:
    import torch

    d = out / "ckpt" / f"step{step}"
    tmp = d.with_name(d.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    if peft_model is not None:
        peft_model.save_pretrained(str(tmp / "adapter"))
    else:
        model.save(str(tmp))
    torch.save(
        {"optimizer": opt.state_dict(), "scheduler": sch.state_dict(), "step": step},
        tmp / "trainer_state.pt",
    )
    tmp.rename(d)  # a checkpoint exists only once complete
    old = sorted(
        (out / "ckpt").glob("step*"), key=lambda p: int(p.name[4:]) if p.name[4:].isdigit() else -1
    )
    for o in old[:-keep]:
        if o.name[4:].isdigit():
            shutil.rmtree(o)
    log.info("saved %s", d)


def _flat(cfg: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k_, v in cfg.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k_}."))
        else:
            out[f"{prefix}{k_}"] = v
    return out


class _null:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *a: Any) -> None:
        return None
