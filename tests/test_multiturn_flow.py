"""
End-to-end flow tests for the orchestration multi-turn engines, driven by a dummy
LLM (deterministic fake generation/probe). Validates the full pipeline —
engine -> make_poll -> TrajectoryResult -> real metrics — without a GPU, plus
the generic dispatch (bayesbench.orchestration.run), is_done early-stop, the non-batched
passive path, and the base batched loop-fallback.

Run from the repo root:
    PYTHONPATH=. python tests/test_multiturn_flow.py
"""

import glob
import json

from bayesbench.orchestration import delivery, run
from bayesbench.orchestration.environment import Environment
from bayesbench.orchestration.user_sim import MultiTurnUserSimulator


# ───────────────────────────── shared mocks ─────────────────────────────────

class _Sim(MultiTurnUserSimulator):
    STYLES = {"s": {"instructions": "I"}}
    FINAL_DIRECTIVE = "[THIS TURN] <message>"
    def style_reminder(self, style, turn, **c): return f"rem{turn}"
    def ground_rules(self): return "rules"
    def turn_intro(self, fact, turn, **c): return f"intro({fact},{turn})"
    def system_preamble(self, ctx, asp, instr, **c): return f"SYS|{asp}|{instr}"


class MockEnv(Environment):
    """Minimal active+passive environment with a deterministic fake backend."""
    name = "mock"; has_baseline = True
    done_at = None  # turn index at which is_done fires (None = never)

    def load_case(self, cid):
        return {"title": "T"}, [{"content": "a0", "id": "x0"},
                                {"content": "a1", "id": "x1"}]
    def generate(self, conv, *, max_tokens, temperature): return "JUDGE"
    # user-sim step (sequential) calls generate via the sim; give it a tagged
    # form so extraction is exercised. Distinguish by checking the last role.
    def judge_generate(self, conv): return "JUDGE"
    def generate_batch(self, convs, *, max_tokens, temperature):
        return ["<message>U</message>"] * len(convs)
    def probe(self, conv, config): return {"n": len(conv)}
    # probe_batch intentionally NOT overridden -> exercises base loop-fallback
    def make_poll(self, turn, r, aspect, um, jm, config):
        return {"t": turn, "aspect": aspect["id"] if aspect else None, "um": um, "jm": jm}
    def baseline_make_poll(self, r, config): return {"t": -1, "baseline": True}
    def baseline_conversation(self, config, ctx): return [{"role": "user", "content": "Title: T"}]
    def build_result(self, config, ctx, sb, polls, jc, uc): return {"polls": polls, "judge": jc, "user": uc}
    def user_simulator(self): return _Sim()
    def style(self, config): return "s"
    def passive_base(self, config, ctx): return [{"role": "user", "content": "Title: T"}]
    def is_done(self, user_msg, turn):
        return self.done_at is not None and turn >= self.done_at


# For the sequential active path the user-sim uses env.generate; give MockEnv a
# tagged generate just for that test so extraction is covered.
class MockEnvSeq(MockEnv):
    def generate(self, conv, *, max_tokens, temperature): return "<message>UU</message>"
    def judge_generate(self, conv): return "JUDGE"


# ─────────────────────────── engine structural test ─────────────────────────

def test_engine_structure():
    env = MockEnvSeq(); ctx, sb = env.load_case("c")
    r = delivery.run_active(env, "cfg", ctx, sb); p = r["polls"]
    assert p[0]["baseline"] and [x["t"] for x in p[1:]] == [0, 1]
    assert p[1]["um"] == "UU" and p[1]["jm"] == "JUDGE"          # user extracted, judge raw
    assert [m["role"] for m in r["judge"]] == ["user", "assistant", "user", "assistant"]
    assert [m["role"] for m in r["user"]] == ["system", "assistant", "user", "assistant", "user"]

    b = delivery.run_active_batch(MockEnv(), [("cfg", ctx, sb)])[0]["polls"]
    assert b[0]["baseline"] and [x["t"] for x in b[1:]] == [0, 1]

    pp = delivery.run_passive_batch(MockEnv(), [("cfg", ctx, sb)])[0]["polls"]
    assert [x["t"] for x in pp] == [0, 1, 2] and [x["aspect"] for x in pp] == [None, "x0", "x1"]
    assert all(x["um"] is None and x["jm"] is None for x in pp)
    print("  ok: engine structure (active seq/batch + passive)")


def test_dispatch():
    env = MockEnv(); cid = "anything"
    # run_one: active (sequential) + passive (batched)
    ra = run.run_one(env, "active", "cfg", cid)
    assert ra["polls"][0]["baseline"]
    rp = run.run_one(env, "passive", "cfg", cid)
    assert [x["t"] for x in rp["polls"]] == [0, 1, 2]
    # run_chunk: both modes, batched + sequential
    rc = run.run_chunk(env, "active", [("cfg", cid), ("cfg", cid)])
    assert len(rc) == 2 and all(r["polls"][0]["baseline"] for r in rc)
    rc_seq = run.run_chunk(env, "passive", [("cfg", cid)], batched=False)
    assert len(rc_seq) == 1
    # unknown mode -> ValueError (run_one validates inside the engine dispatch)
    try:
        run.run_one(env, "bogus", "cfg", cid); assert False, "expected ValueError"
    except ValueError:
        pass
    print("  ok: bayesbench.orchestration.run dispatch (run_one/run_chunk, both modes, bad mode)")


def test_is_done_early_stop():
    env = MockEnv(); env.done_at = 0          # fire is_done on the first turn
    ctx, sb = env.load_case("c")
    r = delivery.run_active(env, "cfg", ctx, sb)
    # baseline + exactly one loop poll, then break
    assert [x["t"] for x in r["polls"]] == [-1, 0], r["polls"]
    rb = delivery.run_active_batch(env, [("cfg", ctx, sb)])[0]
    assert [x["t"] for x in rb["polls"]] == [-1, 0]
    print("  ok: is_done early-stop (sequential + batched)")


def test_nonbatched_passive_and_loop_fallback():
    env = MockEnv(); ctx, sb = env.load_case("c")
    # non-batched passive exercises env.passive_probe per conv (-> probe loop fallback)
    r = delivery.run_passive(env, "cfg", ctx, sb, batched=False)
    assert [x["t"] for x in r["polls"]] == [0, 1, 2]
    print("  ok: non-batched passive + base probe_batch/probe loop-fallback")


# ──────────────────────────────── AITA e2e ───────────────────────────────────

def test_social_judgment_flow():
    from bayesbench.social_judgment.environment import SocialJudgmentEnvironment
    from bayesbench.social_judgment.config import ExperimentConfig, Condition, AccountStyle

    probe = {"p_yta": 0.6, "p_yta_v1": 0.62, "p_yta_v2": 0.58, "position_bias": 0.04, "ab_mass": 0.97}
    env = SocialJudgmentEnvironment(lambda c, **k: "RESP", lambda cs, **k: ["RESP"] * len(cs),
                          lambda c: dict(probe), lambda cs: [dict(probe) for _ in cs])
    d = json.load(open(sorted(glob.glob("bayesbench/social_judgment/storyboards/*.json"))[0]))
    ctx = {k: d[k] for k in ("id", "title", "text", "verdict", "is_yta")}
    sb = d["storyboard"]; N = len(sb)
    cfg_a = ExperimentConfig(model_name="dummy", condition=Condition.MULTI_TURN_ACTIVE,
                             post_id=d["id"], post_index=0, style=AccountStyle.DEFENDING)
    cfg_p = ExperimentConfig(model_name="dummy", condition=Condition.MULTI_TURN_PASSIVE,
                             post_id=d["id"], post_index=0)
    r = delivery.run_active(env, cfg_a, ctx, sb)
    assert r.metrics is not None and [p.t for p in r.polls] == list(range(N + 1))
    assert r.polls[1].user_message and r.polls[1].judge_response
    rb = delivery.run_active_batch(env, [(cfg_a, ctx, sb)])[0]
    assert rb.metrics is not None and [p.t for p in rb.polls] == list(range(N + 1))
    rp = delivery.run_passive_batch(env, [(cfg_p, ctx, sb)])[0]
    assert rp.metrics is not None and [p.t for p in rp.polls] == list(range(N + 1))
    assert rp.polls[0].aspect_id is None and rp.polls[1].aspect_id is not None
    assert all(p.user_message is None for p in rp.polls)
    print(f"  ok: social_judgment flow (N={N}, active seq/batch + passive, real metrics)")


# ─────────────────────────────── triage e2e ──────────────────────────────────

def test_medical_triage_flow():
    from bayesbench.medical_triage.environment import MedicalTriageEnvironment
    from bayesbench.medical_triage.config import (ExperimentConfig, Condition, PopInfo,
                                      PatientProfile, UrgencyTier)
    chain = {
        "urgency_distribution": [0.4, 0.3, 0.2, 0.1], "urgency_prediction": "Emergency",
        "urgency_scale_bias": 0.0, "urgency_mass": 1.0,
        "profile_distribution": [0.3, 0.3, 0.2, 0.2], "profile_prediction": "accurate",
        "profile_scale_bias": 0.0, "profile_mass": 1.0, "cot_reasoning": "because",
        "cot_profile_distribution": [0.3, 0.3, 0.2, 0.2], "cot_profile_prediction": "accurate",
        "cot_profile_scale_bias": 0.0, "cot_profile_mass": 1.0,
        "conditioned_urgency_distribution": [0.4, 0.3, 0.2, 0.1],
        "conditioned_urgency_prediction": "Emergency",
        "conditioned_urgency_scale_bias": 0.0, "conditioned_urgency_mass": 1.0,
    }
    combined = {k: chain[k] for k in (
        "urgency_distribution", "urgency_prediction", "urgency_scale_bias", "urgency_mass",
        "profile_distribution", "profile_prediction", "profile_scale_bias", "profile_mass")}
    env = MedicalTriageEnvironment(lambda c, **k: "RESP", lambda c: dict(chain),
                            lambda cs: [dict(combined) for _ in cs])
    d = json.load(open(sorted(glob.glob("bayesbench/medical_triage/storyboards/*.json"))[0]))
    ctx = d; sb = d["storyboard"]; N = len(sb)
    cfg_a = ExperimentConfig(model_name="dummy", pop_info=PopInfo.ZERO_SHOT, case_id=d["case_id"],
                             case_index=0, profile=PatientProfile.CYBERCHONDRIAC,
                             true_urgency=UrgencyTier(d["true_urgency"]),
                             condition=Condition.MULTI_TURN_ACTIVE)
    cfg_p = ExperimentConfig(model_name="dummy", pop_info=PopInfo.ZERO_SHOT, case_id=d["case_id"],
                             case_index=0, profile=PatientProfile.ACCURATE,
                             true_urgency=UrgencyTier(d["true_urgency"]),
                             condition=Condition.MULTI_TURN_PASSIVE)
    r = delivery.run_active(env, cfg_a, ctx, sb)
    assert r.metrics is not None and [p.t for p in r.polls] == list(range(N))
    assert r.polls[0].user_message and r.polls[0].judge_response == "RESP"
    assert r.polls[0].profile_prediction == "accurate"             # latent inference field
    # triage has no real generate_batch/probe_batch -> base loop-fallback must work
    rab = delivery.run_active_batch(env, [(cfg_a, ctx, sb)])[0]
    assert rab.metrics is not None and [p.t for p in rab.polls] == list(range(N))
    rp = delivery.run_passive_batch(env, [(cfg_p, ctx, sb)])[0]
    assert rp.metrics is not None and [p.t for p in rp.polls] == list(range(N))
    assert rp.polls[0].user_message == sb[0]["content"] and rp.polls[0].judge_response == "Noted."
    assert rp.polls[0].cot_reasoning is None                       # passive: no CoT
    print(f"  ok: triage flow (N={N}, active seq/batch + passive, latent inference, real metrics)")


if __name__ == "__main__":
    test_engine_structure()
    test_dispatch()
    test_is_done_early_stop()
    test_nonbatched_passive_and_loop_fallback()
    test_social_judgment_flow()
    test_medical_triage_flow()
    print("ALL MULTI-TURN FLOW TESTS PASSED")
