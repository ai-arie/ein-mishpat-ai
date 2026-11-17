"""D3b synthetic search queries: answer parsing (gen/d3b.py)."""

from ein_mishpat_ai.gen.d3b import messages, parse


def test_parse_accepts_one_he_en_pair_per_segment():
    s = '```json\n{"1": {"he": "מהו דין מוקצה", "en": "laws of muktzeh"}, "2": {"he": "למה", "en": "why"}}\n```'
    assert parse(s, 2) == [("מהו דין מוקצה", "laws of muktzeh"), ("למה", "why")]


def test_parse_rejects_wrong_count_missing_or_non_hebrew():
    assert parse('{"1": {"he": "מהו", "en": "what"}}', 2) is None
    assert parse('{"1": {"he": "", "en": "what"}}', 1) is None
    assert parse('{"1": {"he": "what is it", "en": "what"}}', 1) is None
    assert parse("not json", 1) is None


def test_messages_number_segments_without_refs():
    m = messages("Some Work", ["Some Work 1:1", "Some Work 1:2"], ["א", "ב"])
    assert m[1]["content"].endswith("[1] א\n\n[2] ב") and "1:1" not in m[1]["content"]


def test_budget_check_stops_before_passing_the_budget():
    from ein_mishpat_ai.gen.d3b import budget_check

    p = [{"requests": 1000, "tokens_all": 1_000_000}] * 3
    assert not budget_check(p, 50_000, 80_000_000)["stop"]
    assert budget_check(p, 50_000, 80_000_000)["projected"] == 50_000_000
    assert budget_check(p * 26, 50_000, 80_000_000)["spent"] == 78_000_000
    assert not budget_check(p * 26, 50_000, 80_000_000)["stop"]  # 78M + 1.2M: still under
    assert budget_check(p * 26 + p[:1], 50_000, 80_000_000)["stop"]  # 79M + 1.2M: would pass


def test_budget_check_flags_the_projection_only_after_three_blocks():
    from ein_mishpat_ai.gen.d3b import budget_check

    p = [{"requests": 1000, "tokens_all": 2_000_000}]
    assert not budget_check(p * 2, 50_000, 80_000_000)["flag"]
    b = budget_check(p * 3, 50_000, 80_000_000)
    assert b["flag"] and b["projected"] == 100_000_000 and not b["stop"]
    assert not budget_check([], 50_000, 80_000_000)["stop"]
