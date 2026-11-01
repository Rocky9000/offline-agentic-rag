import json
from pathlib import Path

from fastapi.testclient import TestClient

from agentrag import AgenticRAG, ScriptedLLM
from agentrag.agents import ABSTAIN, ANSWER_SYSTEM, CRITIC_SYSTEM, PLANNER_SYSTEM
from agentrag.api import create_app
from agentrag.evaluate import end_to_end_metrics, load_dataset, retrieval_metrics

ROOT = Path(__file__).resolve().parents[1]


def scripted(answers, verdicts, plan=None):
    """Return an LLM that plays planner, answerer and critic from scripts."""
    answers, verdicts = list(answers), list(verdicts)

    def respond(prompt, system):
        if system == PLANNER_SYSTEM:
            return json.dumps({"queries": plan or []})
        if system == ANSWER_SYSTEM:
            return answers.pop(0)
        if system == CRITIC_SYSTEM:
            return json.dumps(verdicts.pop(0))
        raise AssertionError("unexpected system prompt")

    return ScriptedLLM(respond)


def test_happy_path_single_iteration(handbook_index):
    llm = scripted(["Store at 3.8 volts per cell [1]."], [{"grounded": True, "reason": "ok"}])
    r = AgenticRAG(handbook_index, llm).ask("What is the storage voltage?")
    assert r.grounded and r.iterations == 1 and not r.abstained
    assert len(r.sources) == 1
    assert [s.name for s in r.trace] == ["plan", "retrieve", "answer", "critique"]


def test_critic_triggers_follow_up_retrieval(handbook_index):
    llm = scripted(
        ["Packs are stored at 3.8 V [1].", "3.8 V [1]; propellers every 25 hours [2]."],
        [
            {"grounded": False, "reason": "propeller part missing", "follow_up_query": "propeller replacement hours"},
            {"grounded": True, "reason": "ok"},
        ],
    )
    r = AgenticRAG(handbook_index, llm, max_iterations=3).ask("Storage voltage and propeller schedule?")
    assert r.grounded and r.iterations == 2
    names = [s.name for s in r.trace]
    assert names.count("retrieve") == 2
    assert "propeller part missing" in llm.calls[-2][0]  # feedback reached the answerer


def test_loop_is_bounded(handbook_index):
    bad = {"grounded": False, "reason": "no", "follow_up_query": ""}
    llm = scripted(["x [1]."] * 5, [bad] * 5)
    r = AgenticRAG(handbook_index, llm, max_iterations=2).ask("anything")
    assert r.iterations == 2 and not r.grounded


def test_planner_queries_are_searched(handbook_index):
    llm = scripted(["ok [1]."], [{"grounded": True}], plan=["hotel limit metro"])
    r = AgenticRAG(handbook_index, llm).ask("travel?")
    assert r.trace[0].detail["queries"] == ["travel?", "hotel limit metro"]


def test_ablation_flags(handbook_index):
    llm = scripted(["Answer [1]."], [])
    r = AgenticRAG(handbook_index, llm, use_planner=False, use_critic=False).ask("q")
    assert r.grounded and [s.name for s in r.trace] == ["plan", "retrieve", "answer"]
    assert all(system == ANSWER_SYSTEM for _, system in llm.calls)


def test_retrieval_metrics_on_handbook(handbook_index):
    items = load_dataset(ROOT / "eval" / "handbook_qa.jsonl")
    m = retrieval_metrics(handbook_index, items, k=3, mode="hybrid")
    assert m["n"] == 22
    assert m["hit@3"] >= 0.8  # regression guard for the offline baseline


def test_end_to_end_metrics_with_oracle_llm(handbook_index):
    items = load_dataset(ROOT / "eval" / "handbook_qa.jsonl")[-2:]  # two unanswerable
    llm = scripted([ABSTAIN] * 4, [{"grounded": False, "follow_up_query": ""}] * 4)
    m = end_to_end_metrics(AgenticRAG(handbook_index, llm), items)
    assert m["abstention_accuracy"] == 1.0


def test_api(handbook_index):
    llm = scripted(["Channel 9 [1]."], [{"grounded": True}])
    client = TestClient(create_app(AgenticRAG(handbook_index, llm)))
    assert client.get("/health").json()["status"] == "ok"
    r = client.post("/query", json={"question": "emergency radio channel?"})
    assert r.status_code == 200 and r.json()["answer"] == "Channel 9 [1]."
    assert client.post("/query", json={"question": ""}).status_code == 422
    assert "Agentic RAG" in client.get("/").text
