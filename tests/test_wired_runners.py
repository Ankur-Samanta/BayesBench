"""
Wired-runner tests: drive the ACTUAL runner entry points
(run_experiment / run_multi_turn_active_batch) end-to-end with a deterministic
fake extraction backend. This exercises the full production path —
run_experiment -> Environment.from_model -> orchestration engine -> make_poll ->
TrajectoryResult -> real metrics — without a GPU (only the token-level
extraction is stubbed).

Run from the repo root:
    PYTHONPATH=. python tests/test_wired_runners.py
"""

import glob
import json
import sys
import types


# ── stub the torch-bound extraction modules BEFORE importing the runners ──────

def _social_judgment_extraction_stub():
    m = types.ModuleType("bayesbench.social_judgment.extraction")
    probe = {"p_yta": 0.6, "p_yta_v1": 0.62, "p_yta_v2": 0.58,
             "position_bias": 0.04, "ab_mass": 0.97}
    m.setup_model = lambda *a, **k: (object(), object())
    m.generate = lambda model, tok, conv, **k: "RESP"
    m.generate_batch = lambda model, tok, convs, **k: ["RESP"] * len(convs)
    m.extract_p_yta = lambda model, tok, conv, **k: dict(probe)
    m.extract_p_yta_batch = lambda model, tok, convs, **k: [dict(probe) for _ in convs]
    return m


def _triage_extraction_stub():
    """Deterministic stub whose MCQ result depends ONLY on the option strings, so
    the sequential (extract_urgency/profile/4way_counterbalanced) and batched
    (extract_4way_many) paths return identical distributions for the same probe —
    letting the tests assert true batched-vs-sequential parity."""
    from bayesbench.medical_triage.config import URGENCY_ORDER, PROFILE_ORDER
    m = types.ModuleType("bayesbench.medical_triage.extraction")

    def mcq(options):
        vals = [len(o) % 7 + 1 for o in options]
        s = float(sum(vals))
        dist = [v / s for v in vals]
        idx = max(range(len(dist)), key=lambda i: dist[i])
        return dist, idx, 0.0, 1.0

    u_opts = [u.value for u in URGENCY_ORDER]
    p_opts = [p.value for p in PROFILE_ORDER]

    m.setup_model = lambda *a, **k: (object(), object())
    m.generate = lambda model, tok, conv, **k: "RESP"
    m.generate_batch = lambda model, tok, convs, **k: ["RESP"] * len(convs)
    # Sequential primitives: options are internal (urgency vs profile order).
    m.extract_urgency = lambda *a, **k: mcq(u_opts)
    m.extract_profile = lambda *a, **k: mcq(p_opts)
    # 4way_counterbalanced(model, tok, base, options_canonical, build, ...) — key on options.
    m.extract_4way_counterbalanced = lambda model, tok, base, options, build, **k: mcq(options)
    # Batched: one result per (base, options, build) request, keyed on its options.
    m.extract_4way_many = lambda model, tok, requests, **k: [mcq(opts) for (_b, opts, _f) in requests]
    m.index_to_urgency = lambda i: u_opts[i]
    m.index_to_profile = lambda i: p_opts[i]
    return m


sys.modules["bayesbench.social_judgment.extraction"] = _social_judgment_extraction_stub()
sys.modules["bayesbench.medical_triage.extraction"] = _triage_extraction_stub()

import bayesbench.social_judgment.runner as ar          # noqa: E402
import bayesbench.medical_triage.runner as tr                  # noqa: E402
from bayesbench.social_judgment.config import (          # noqa: E402
    ExperimentConfig as ACfg, Condition as ACond, AccountStyle)
from bayesbench.medical_triage.config import (                  # noqa: E402
    ExperimentConfig as TCfg, Condition as TCond, PopInfo, PatientProfile, UrgencyTier)

DUMMY = object()


def test_social_judgment_runner():
    pid = sorted(p.rsplit("/", 1)[-1][:-5]
                 for p in glob.glob("bayesbench/social_judgment/storyboards/*.json"))[0]
    _, sb = ar.load_storyboard(pid); N = len(sb)

    cfg = ACfg(model_name="dummy", condition=ACond.MULTI_TURN_ACTIVE,
               post_id=pid, post_index=0, style=AccountStyle.DEFENDING)
    r = ar.run_experiment(cfg, DUMMY, DUMMY)
    assert r.metrics is not None and [p.t for p in r.polls] == list(range(N + 1))

    cfgp = ACfg(model_name="dummy", condition=ACond.MULTI_TURN_PASSIVE,
                post_id=pid, post_index=0)
    rp = ar.run_experiment(cfgp, DUMMY, DUMMY)
    assert rp.metrics is not None and [p.t for p in rp.polls] == list(range(N + 1))

    # batched active path (production uses --batch-size > 1)
    post, sb3 = ar.load_storyboard(pid)
    rb = ar.run_multi_turn_active_batch(DUMMY, DUMMY, [(cfg, post, sb3)])
    assert len(rb) == 1 and rb[0].metrics is not None
    assert [p.t for p in rb[0].polls] == list(range(N + 1))

    # batched PASSIVE path — must match the sequential passive result exactly.
    rpb = ar.run_multi_turn_passive_batch(DUMMY, DUMMY, [(cfgp, post, sb3)])
    assert len(rpb) == 1 and rpb[0].metrics is not None
    assert rpb[0].to_dict()["polls"] == rp.to_dict()["polls"], \
        "social_judgment batched passive != sequential passive"
    print(f"  ok: social_judgment active+passive, batched active + batched passive parity (N={N})")


def test_medical_triage_runner():
    cid = sorted(int(p.rsplit("/", 1)[-1][:-5])
                 for p in glob.glob("bayesbench/medical_triage/storyboards/*.json"))[0]
    case = tr.load_storyboard(cid); N = len(case["storyboard"])

    cfg = TCfg(model_name="dummy", pop_info=PopInfo.ZERO_SHOT, case_id=cid, case_index=0,
               profile=PatientProfile.CYBERCHONDRIAC,
               true_urgency=UrgencyTier(case["true_urgency"]),
               condition=TCond.MULTI_TURN_ACTIVE)
    r = tr.run_experiment(cfg, DUMMY, DUMMY)
    assert r.metrics is not None and [p.t for p in r.polls] == list(range(N))
    assert r.polls[0].cot_reasoning is not None and r.polls[0].profile_prediction is not None

    cfgp = TCfg(model_name="dummy", pop_info=PopInfo.ZERO_SHOT, case_id=cid, case_index=0,
                profile=PatientProfile.ACCURATE,
                true_urgency=UrgencyTier(case["true_urgency"]),
                condition=TCond.MULTI_TURN_PASSIVE)
    rp = tr.run_experiment(cfgp, DUMMY, DUMMY)
    assert rp.metrics is not None and [p.t for p in rp.polls] == list(range(N))
    assert rp.polls[0].user_message == case["storyboard"][0]["content"]
    assert rp.polls[0].judge_response == "Noted." and rp.polls[0].cot_reasoning is None

    # batched ACTIVE path — full 4-poll chain batched (probe_batch). Must match
    # the sequential active result exactly (deterministic option-keyed stub).
    rab = tr.run_triage_active_batch(DUMMY, DUMMY, [(cfg, case)])
    assert len(rab) == 1 and rab[0].metrics is not None
    assert [p.t for p in rab[0].polls] == list(range(N))
    assert rab[0].to_dict()["polls"] == r.to_dict()["polls"], \
        "triage batched active != sequential active"

    # batched PASSIVE over a chunk — each item matches its standalone result.
    rpb = tr.run_triage_passive_batch(DUMMY, DUMMY, [(cfgp, case)])
    assert len(rpb) == 1
    assert rpb[0].to_dict()["polls"] == rp.to_dict()["polls"], \
        "triage batched passive != single passive"
    print(f"  ok: triage active+passive, batched active parity + batched passive (N={N})")


if __name__ == "__main__":
    test_social_judgment_runner()
    test_medical_triage_runner()
    print("ALL WIRED-RUNNER TESTS PASSED")
