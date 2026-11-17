"""The rule for keeping the 500-step flagship, fixed before the run (2026-11-07)."""

from ein_mishpat_ai.eval.kesher import SCORE_SETTINGS, flagship_choice, kesher_score, topic_hit10


def metrics(score: float, he: float, en: float) -> dict[str, float]:
    m = {f"{s}/ndcg_at_10": score for s in SCORE_SETTINGS}
    m["K5_topic_sources/he2he/hit_at_10"] = he
    m["K5_topic_sources/en2he/hit_at_10"] = en
    return m


def test_score_and_topic_mean() -> None:
    m = metrics(0.5, 0.4, 0.2)
    assert abs(kesher_score(m) - 0.5) < 1e-12
    assert abs(topic_hit10(m) - 0.3) < 1e-12


def test_rule() -> None:
    short = metrics(0.5, 0.4, 0.4)
    assert flagship_choice(short, metrics(0.51, 0.41, 0.40))
    assert flagship_choice(short, metrics(0.51, 0.40, 0.40))  # topic search not lower: enough
    assert flagship_choice(short, metrics(0.51, 0.43, 0.38))  # the mean decides, not each language
    assert not flagship_choice(short, metrics(0.50, 0.50, 0.50))  # the score must be higher
    assert not flagship_choice(short, metrics(0.60, 0.39, 0.40))  # topic search lower

