import json

from agentrag import Chunk, ScriptedLLM, SearchHit
from agentrag.agents import ABSTAIN, Answerer, Critic, Planner, extract_citations, parse_json_object

HITS = [
    SearchHit(Chunk("a", "battery-care", "Store packs at 3.8 volts per cell.", 0, 1), 1.0),
    SearchHit(Chunk("b", "maintenance", "Replace propellers every 25 hours.", 0, 1), 0.5),
]


def test_parse_json_handles_fences_and_chatter():
    raw = 'Sure! ```json\n{"queries": ["a", "b"]}\n``` hope that helps'
    assert parse_json_object(raw) == {"queries": ["a", "b"]}
    assert parse_json_object("no json here") is None
    assert parse_json_object("{broken") is None
    assert parse_json_object("[1, 2]") is None


def test_extract_citations():
    assert extract_citations("A [1]. B [2][3]. C [12].") == [1, 2, 3, 12]


def test_planner_keeps_original_and_dedupes():
    llm = ScriptedLLM(lambda p, s: json.dumps({"queries": ["storage voltage", "Q?", "storage voltage", " "]}))
    assert Planner(llm).plan("Q?") == ["Q?", "storage voltage"]


def test_planner_falls_back_on_garbage():
    assert Planner(ScriptedLLM(lambda p, s: "I cannot do that")).plan("Q?") == ["Q?"]


def test_planner_caps_queries():
    llm = ScriptedLLM(lambda p, s: json.dumps({"queries": [f"q{i}" for i in range(10)]}))
    assert len(Planner(llm, max_queries=2).plan("Q?")) == 3


def test_answerer_numbers_context_and_includes_feedback():
    llm = ScriptedLLM(lambda p, s: "3.8 volts [1]")
    Answerer(llm).answer("voltage?", HITS, feedback="cite properly")
    prompt = llm.calls[0][0]
    assert "[1] (source: battery-care)" in prompt and "[2] (source: maintenance)" in prompt
    assert "cite properly" in prompt


def test_answerer_abstains_without_calling_llm_when_no_context():
    llm = ScriptedLLM(lambda p, s: "should not be called")
    assert Answerer(llm).answer("q", []) == ABSTAIN
    assert llm.calls == []


def test_critic_rejects_out_of_range_citation_without_llm():
    llm = ScriptedLLM(lambda p, s: "unused")
    v = Critic(llm).review("q", "Answer [7].", HITS)
    assert not v.grounded and v.invalid_citations == [7]
    assert llm.calls == []


def test_critic_rejects_uncited_answer():
    v = Critic(ScriptedLLM(lambda p, s: "unused")).review("q", "It is 3.8 volts.", HITS)
    assert not v.grounded


def test_critic_uses_llm_judgement():
    llm = ScriptedLLM(lambda p, s: '{"grounded": false, "reason": "x", "follow_up_query": "propeller schedule"}')
    v = Critic(llm).review("q", "Answer [1].", HITS)
    assert not v.grounded and v.follow_up_query == "propeller schedule"


def test_critic_requires_literal_true():
    llm = ScriptedLLM(lambda p, s: '{"grounded": "yes"}')
    assert not Critic(llm).review("q", "Answer [1].", HITS).grounded


def test_critic_treats_unparseable_judge_as_pass():
    v = Critic(ScriptedLLM(lambda p, s: "looks fine to me")).review("q", "Answer [1].", HITS)
    assert v.grounded
