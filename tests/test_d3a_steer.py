"""D3a concurrency steering (gen/d3a.py `_steer`): the rules of 2026-11-01/02 (16 → 32 also in working hours)."""

from datetime import datetime, timedelta, timezone

from ein_mishpat_ai.gen.d3a import _steer
from ein_mishpat_ai.utils.io import load_yaml

DC = load_yaml("configs/gen.yaml")["d3a"]
BASE = 0.024
MON_13 = datetime(2026, 11, 2, 13, 0, tzinfo=timezone.utc)  # working hours
MON_18 = datetime(2026, 11, 2, 18, 0, tzinfo=timezone.utc)  # evening
SAT_13 = datetime(2026, 11, 7, 13, 0, tzinfo=timezone.utc)


def chunk(spt=0.025, n=800, errors=0, cached=0):
    return {
        "requests_batched": n,
        "requests_single": 0,
        "cached": cached,
        "request_errors": errors,
        "s_per_out_token_median": spt,
    }


def test_working_hours_up_to_32_only():
    assert _steer(16, chunk(), {}, 1000, {}, DC, BASE, MON_13)[:2] == (32, "up: endpoint fine")
    assert _steer(32, chunk(), {}, 1000, {}, DC, BASE, MON_13)[:2] == (32, "keep")


def test_working_hours_cap():
    assert _steer(48, chunk(), {}, 1000, {}, DC, BASE, MON_13)[:2] == (32, "down: working hours")


def test_evening_steps_up_one_level_at_a_time():
    st: dict = {}
    assert _steer(16, chunk(), {}, 1000, st, DC, BASE, MON_18)[0] == 32
    assert _steer(32, chunk(), {}, 1500, st, DC, BASE, MON_18 + timedelta(minutes=15))[0] == 48
    assert _steer(48, chunk(), {}, 1800, st, DC, BASE, MON_18 + timedelta(minutes=30))[0] == 48


def test_slow_endpoint_steps_down_and_waits_an_hour():
    st: dict = {}
    new, why, problem = _steer(32, chunk(spt=1.9 * BASE), {}, 1000, st, DC, BASE, MON_18)
    assert (new, problem) == (16, True) and why.startswith("down")
    assert _steer(16, chunk(), {}, 1000, st, DC, BASE, MON_18 + timedelta(minutes=30))[0] == 16
    assert _steer(16, chunk(), {}, 1000, st, DC, BASE, MON_18 + timedelta(minutes=61))[0] == 32


def test_step_up_without_throughput_gain_goes_back():
    st: dict = {}
    assert _steer(32, chunk(), {}, 1000, st, DC, BASE, MON_18)[0] == 48
    new, why, _ = _steer(48, chunk(), {}, 1050, st, DC, BASE, MON_18 + timedelta(minutes=15))
    assert new == 32 and "no throughput gain" in why


def test_errors_and_retries_are_problems():
    assert _steer(32, chunk(errors=20), {}, 1000, {}, DC, BASE, MON_18)[0] == 16
    assert _steer(32, chunk(), {"IncompleteRead": 60}, 1000, {}, DC, BASE, MON_18)[0] == 16


def test_cached_chunk_changes_nothing():
    assert _steer(16, chunk(cached=790), {}, 99999, {}, DC, BASE, MON_18)[:2] == (16, "keep")


def test_weekend_counts_as_evening():
    assert _steer(32, chunk(), {}, 1000, {}, DC, BASE, SAT_13)[0] == 48


def test_night_tolerates_more_slowdown_than_working_hours():
    assert _steer(32, chunk(spt=1.5 * BASE), {}, 1000, {}, DC, BASE, MON_13)[0] == 16
    assert _steer(32, chunk(spt=1.5 * BASE), {}, 1000, {}, DC, BASE, MON_18)[0] == 48
    assert _steer(32, chunk(spt=1.9 * BASE), {}, 1000, {}, DC, BASE, MON_18)[0] == 16
