"""The Claude answer path (app/answer/claude.py), with no network: the real SDK sends its request into a mocked
transport, so a parameter the installed SDK or the documented API does not take fails here, not in production (the
reranker-bug class: a call that never worked, hidden behind a fallback)."""

import json
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from app.answer import claude

EXCERPTS = [claude.Excerpt(chunk_id="n1", path="configs/run_047.yaml", text="learning_rate: 3e-4\nbatch: 8", start_byte=10),
            claude.Excerpt(chunk_id="n2", path="logs/run_047.out", text="run 47 finished", start_byte=None)]
QUESTION = "What learning rate did run 47 use?"


def _reply(content, *, stop="end_turn", model="claude-opus-5-5", usage=None):
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": model, "content": content,
            "stop_reason": stop, "stop_sequence": None,
            "usage": usage or {"input_tokens": 900, "output_tokens": 150, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}


CITED = [{"type": "text", "text": "Run 47 used "},
         {"type": "text", "text": "3e-4", "citations": [{"type": "char_location", "cited_text": "learning_rate: 3e-4",
                                                          "document_index": 0, "document_title": "configs/run_047.yaml",
                                                          "start_char_index": 0, "end_char_index": 19}]}]


def _sdk(reply, seen):
    def handler(request):
        seen.append({"url": str(request.url), "headers": dict(request.headers), "body": json.loads(request.content)})
        status, body = reply if isinstance(reply, tuple) else (200, reply)
        return httpx2.Response(status, json=body)

    return anthropic.Anthropic(api_key="test-key", max_retries=0,
                               http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)))


def test_the_installed_sdk_sends_the_documented_request_and_the_reply_is_parsed():
    seen = []
    result = claude.answer(_sdk(_reply(CITED), seen), QUESTION, EXCERPTS, claude.AnswerSettings())
    (call,) = seen
    assert call["url"] == "https://api.anthropic.com/v1/messages"
    assert call["headers"]["anthropic-version"] == "2026-10-21"
    assert call["headers"]["anthropic-beta"] == claude.FALLBACK_BETA
    body = call["body"]
    assert body["model"] == "claude-opus-5-5" and body["max_tokens"] == claude.DEFAULT_MAX_TOKENS
    assert body["fallbacks"] == "default" and body["output_config"] == {"effort": "low"}
    assert body["system"] == claude.SYSTEM_PROMPT and "cache_control" not in json.dumps(body)
    assert not {"temperature", "top_p", "top_k"} & set(body)  # removed on API version 2026-10-21
    (message,) = body["messages"]
    assert message["role"] == "user"
    *documents, question = message["content"]
    assert question == {"type": "text", "text": QUESTION}
    assert [d["source"] for d in documents] == [{"type": "text", "media_type": "text/plain", "data": e.text} for e in EXCERPTS]
    assert all(d["type"] == "document" and d["citations"] == {"enabled": True} for d in documents)
    assert [d["title"] for d in documents] == ["configs/run_047.yaml", "logs/run_047.out"]
    assert result.delivered and result.reason is None and result.text == "Run 47 used 3e-4"
    (cite,) = result.citations
    assert (cite.chunk_id, cite.path, cite.start_char, cite.end_char) == ("n1", "configs/run_047.yaml", 0, 19)
    assert (cite.start_byte, cite.end_byte) == (10, 29)  # the hit's start_byte plus the UTF-8 length before each char
    assert result.cost_usd == pytest.approx((900 * 4.00 + 150 * 20.00) / 1e6)


def test_haiku_has_no_server_side_fallback_so_none_is_sent():
    seen = []
    claude.answer(_sdk(_reply(CITED, model="claude-haiku-5-5"), seen), QUESTION, EXCERPTS,
                  claude.AnswerSettings(model="claude-haiku-5-5"))
    (call,) = seen
    assert "anthropic-beta" not in call["headers"] and "fallbacks" not in call["body"]


def test_a_rejected_request_is_raised_never_turned_into_a_retrieval_only_answer():
    seen = []
    error = {"type": "error", "error": {"type": "invalid_request_error", "message": "bad field"}}
    with pytest.raises(anthropic.BadRequestError):
        claude.answer(_sdk((400, error), seen), QUESTION, EXCERPTS, claude.AnswerSettings())


# The 400s that say the account cannot pay (claude.UNPAID): a workspace's and an organization's own spend limit, in the
# words of Anthropic's Rate limits page, and prepaid credit that ran out, as the API words it.
UNPAID_MESSAGES = [
    "You have reached your specified workspace API usage limits. You will regain access on 2026-12-01 at 00:00 UTC.",
    "You have reached your specified API usage limits. You will regain access on 2026-12-01 at 00:00 UTC.",
    "Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase "
    "credits.",
]


@pytest.mark.parametrize("message", UNPAID_MESSAGES)
def test_a_400_saying_the_account_cannot_pay_is_a_reason_that_costs_nothing(message):
    seen = []
    error = {"type": "error", "error": {"type": "invalid_request_error", "message": message}, "request_id": "req_1"}
    result = claude.answer(_sdk((400, error), seen), QUESTION, EXCERPTS, claude.AnswerSettings(model="claude-haiku-5-5"))
    assert len(seen) == 1  # one call: a 400 is not retried
    assert not result.delivered and result.reason == claude.UNPAID_REASON == "answer spend limit reached"
    assert result.text is None and result.citations == [] and result.model == "claude-haiku-5-5"
    assert result.cost_usd == 0.0  # nothing was billed: the route gives the call's whole reservation back


@pytest.mark.parametrize("status, kind, message", [
    (400, "invalid_request_error", "messages.0.content: Input should be a valid list"),  # a malformed request: a bug
    (400, "invalid_request_error", "messages.0: You have reached your specified API usage limits"),  # not at the start
    (400, "permission_error", UNPAID_MESSAGES[2]),  # the words, under another error type
    (402, "billing_error", "There is an issue with your billing or payment information."),  # a payment problem
])
def test_every_other_client_error_is_still_raised(status, kind, message):
    seen = []
    error = {"type": "error", "error": {"type": kind, "message": message}}
    with pytest.raises(anthropic.APIStatusError) as raised:
        claude.answer(_sdk((status, error), seen), QUESTION, EXCERPTS, claude.AnswerSettings(model="claude-haiku-5-5"))
    assert raised.value.status_code == status and not claude.unpaid(raised.value)


def test_a_transient_failure_is_not_delivered_and_says_why():
    seen = []
    error = {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}
    result = claude.answer(_sdk((529, error), seen), QUESTION, EXCERPTS, claude.AnswerSettings())
    assert not result.delivered and result.reason.startswith("upstream") and result.cost_usd is None


def test_a_wrong_parameter_raises_instead_of_degrading():
    class Strict:
        class beta:  # noqa: N801 - mirrors client.beta.messages
            class messages:  # noqa: N801
                @staticmethod
                def create(*, model, max_tokens, messages, system, output_config, betas):  # no `fallbacks`
                    raise AssertionError("not reached")

    with pytest.raises(TypeError):
        claude.answer(Strict(), QUESTION, EXCERPTS, claude.AnswerSettings())


def _parse(content, **kw):
    response = SimpleNamespace(content=[SimpleNamespace(**block) for block in content], model=kw.get("model", "claude-opus-5-5"),
                               stop_reason=kw.get("stop", "end_turn"), stop_details=kw.get("details"),
                               usage=SimpleNamespace(input_tokens=100, output_tokens=10, cache_creation_input_tokens=0,
                                                     cache_read_input_tokens=0))
    return claude.parse(response, EXCERPTS)


def _cited(index=0):
    return {"type": "text", "text": "3e-4", "citations": [SimpleNamespace(type="char_location", cited_text="learning_rate: 3e-4",
                                                                           document_index=index, start_char_index=0, end_char_index=19)]}


def test_only_a_normal_cited_reply_is_delivered():
    assert _parse([_cited()]).delivered
    refused = _parse([{"type": "text", "text": "", "citations": None}], stop="refusal", details=SimpleNamespace(category="cyber"))
    assert not refused.delivered and refused.reason == "refused (cyber)" and refused.text is None
    cut = _parse([_cited()], stop="max_tokens")
    assert not cut.delivered and cut.reason == "cut at max_tokens" and cut.text is None
    uncited = _parse([{"type": "text", "text": "The excerpts do not say.", "citations": None}])
    assert not uncited.delivered and uncited.reason == "uncited" and uncited.text == "The excerpts do not say."
    outside = _parse([_cited(index=7)])  # a document index that was never sent
    assert not outside.delivered and outside.reason == "uncited" and outside.citations == []


def test_a_fallback_reply_is_marked_and_priced_by_the_model_that_served_it():
    result = _parse([{"type": "fallback"}, _cited()], model="claude-opus-4-8")
    assert result.fallback_used and result.model == "claude-opus-4-8"
    assert result.cost_usd == pytest.approx((100 * 5.00 + 10 * 25.00) / 1e6)
    unknown = _parse([_cited()], model="claude-future-9")
    assert unknown.cost_usd == pytest.approx((100 * claude.DEAREST["input"] + 10 * claude.DEAREST["output"]) / 1e6)


def test_the_reservation_covers_the_reply_it_reserves_for():
    settings = claude.AnswerSettings(max_tokens=2048)
    worst = claude.worst_case_usd(settings, QUESTION, EXCERPTS)
    chars = len(claude.SYSTEM_PROMPT) + len(QUESTION) + sum(len(e.text) for e in EXCERPTS)
    # a reply that used every output token, with input at one token per character, still fits the reservation
    assert claude.cost_usd("claude-opus-5-5", {"input_tokens": chars // 3, "output_tokens": 2048}) <= worst
    assert claude.cost_usd("claude-opus-4-8", {"input_tokens": chars // 3, "output_tokens": 2048}) <= worst  # a fallback


def test_settings_come_from_the_environment_with_safe_defaults(monkeypatch):
    for name in ("ANSWER_MODEL", "ANSWER_EFFORT", "ANSWER_MAX_TOKENS", "ANSWER_TIMEOUT_SECONDS", "ANSWER_MAX_RETRIES"):
        monkeypatch.delenv(name, raising=False)
    assert claude.AnswerSettings.from_env() == claude.AnswerSettings()
    monkeypatch.setenv("ANSWER_MODEL", "claude-haiku-5-5")
    monkeypatch.setenv("ANSWER_MAX_TOKENS", "not a number")
    s = claude.AnswerSettings.from_env()
    assert s.model == "claude-haiku-5-5" and s.max_tokens == claude.DEFAULT_MAX_TOKENS
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert claude.make_client(s) is None  # no key: answers off, and no CLI profile is ever used on a server
