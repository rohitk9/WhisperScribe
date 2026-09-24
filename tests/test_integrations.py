"""Tests for integrations.py against a fake local HTTP server (no Ollama or AnythingLLM needed)."""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import integrations  # noqa: E402
from engine import Result, Segment  # noqa: E402


class FakeServer:
    """Records requests; `routes[(method, path)]` is a dict/list to return as JSON, a callable, or SSE lines."""

    def __init__(self):
        self.requests, self.routes = [], {}
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _handle(self, method):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"null")
                outer.requests.append({"method": method, "path": self.path, "body": body,
                                       "auth": self.headers.get("Authorization")})
                route = outer.routes.get((method, self.path))
                if callable(route):
                    route = route(body)
                if route is None:
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(b'{"error": "not found"}')
                    return
                status, payload = route if isinstance(route, tuple) else (200, route)
                self.send_response(status)
                if isinstance(payload, str):  # server-sent events
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    self.wfile.write(payload.encode())
                else:
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(payload).encode())

            def do_GET(self):
                self._handle("GET")

            def do_POST(self):
                self._handle("POST")

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def server():
    s = FakeServer()
    yield s
    s.close()


def result():
    return Result(segments=[Segment(0, 2, " Ship it Friday.", "Speaker 1"), Segment(2, 4, " Agreed.", "Speaker 2")],
                  language="en", duration=240, confidence=90, summary="- Ship Friday", num_speakers=2)


# ---------- AnythingLLM ----------
def test_check_sends_bearer_key(server):
    server.routes[("GET", "/api/v1/auth")] = {"authenticated": True}
    assert integrations.AnythingLLMClient("secret", server.url).check()
    assert server.requests[0]["auth"] == "Bearer secret"


def test_check_without_key_explains_where_to_get_one():
    with pytest.raises(integrations.IntegrationError, match="Developer API"):
        integrations.AnythingLLMClient("", "http://127.0.0.1:9").check()


def test_bad_key_surfaces_http_status(server):
    server.routes[("GET", "/api/v1/auth")] = (403, {"error": "Invalid API Key"})
    with pytest.raises(integrations.IntegrationError, match="403"):
        integrations.AnythingLLMClient("wrong", server.url).check()


def test_unreachable_server_is_a_friendly_error():
    with pytest.raises(integrations.IntegrationError, match="Can't reach"):
        integrations.AnythingLLMClient("k", "http://127.0.0.1:9").workspaces()


def test_ensure_workspace_reuses_existing(server):
    server.routes[("GET", "/api/v1/workspaces")] = {"workspaces": [{"name": "Meeting Transcripts", "slug": "mt"}]}
    assert integrations.AnythingLLMClient("k", server.url).ensure_workspace("Meeting Transcripts") == "mt"
    assert [r["path"] for r in server.requests] == ["/api/v1/workspaces"]


def test_ensure_workspace_creates_and_configures(server):
    server.routes[("GET", "/api/v1/workspaces")] = {"workspaces": []}
    server.routes[("POST", "/api/v1/workspace/new")] = {"workspace": {"slug": "meeting-transcripts"}}
    server.routes[("POST", "/api/v1/workspace/meeting-transcripts/update")] = {"workspace": {}}
    slug = integrations.AnythingLLMClient("k", server.url).ensure_workspace(
        "Meeting Transcripts", integrations.workspace_settings("qwen3.5:9b"))
    assert slug == "meeting-transcripts"
    update = server.requests[-1]["body"]
    assert update["chatProvider"] == "ollama" and update["chatModel"] == "qwen3.5:9b"
    assert "Speaker 1" in update["openAiPrompt"]


def test_workspace_settings_without_model_keeps_anythingllm_default():
    assert "chatModel" not in integrations.workspace_settings("")


def test_sync_transcript_uploads_and_opens_thread(server):
    server.routes[("GET", "/api/v1/workspaces")] = {"workspaces": [{"name": "Meeting Transcripts", "slug": "mt"}]}
    server.routes[("POST", "/api/v1/document/raw-text")] = {"documents": [{"location": "custom-documents/x.json"}]}
    server.routes[("POST", "/api/v1/workspace/mt/thread/new")] = {"thread": {"slug": "t-1"}}
    entry = integrations.sync_transcript(integrations.AnythingLLMClient("k", server.url), result(), "Standup",
                                         "Summarize", "24 Sep 2026")
    assert entry == {"title": "Standup", "workspace": "mt", "thread": "t-1", "document": "custom-documents/x.json",
                     "recorded": "24 Sep 2026"}
    upload = next(r["body"] for r in server.requests if r["path"] == "/api/v1/document/raw-text")
    assert upload["addToWorkspaces"] == "mt"
    assert upload["metadata"]["title"] == "Standup"
    assert "Speaker 1: Ship it Friday." in upload["textContent"]
    assert "- Ship Friday" in upload["textContent"]


def test_stream_chat_parses_events_until_close(server):
    events = [{"type": "textResponseChunk", "textResponse": "Hel", "close": False},
              {"type": "textResponseChunk", "textResponse": "lo", "close": False},
              {"type": "finalizeResponseStream", "sources": [{"title": "Standup"}], "close": True},
              {"type": "textResponseChunk", "textResponse": "IGNORED", "close": False}]
    server.routes[("POST", "/api/v1/workspace/mt/thread/t-1/stream-chat")] = "".join(
        f"data: {json.dumps(e)}\n\n" for e in events)
    got = list(integrations.AnythingLLMClient("k", server.url).stream_chat("mt", "hi", "t-1"))
    assert "".join(e.get("textResponse") or "" for e in got) == "Hello"
    assert got[-1]["sources"] == [{"title": "Standup"}]
    assert server.requests[0]["body"] == {"message": "hi", "mode": "chat"}


def test_parse_sse_raises_on_error_event():
    with pytest.raises(integrations.IntegrationError, match="model not found"):
        list(integrations.parse_sse([b'data: {"error": "model not found", "close": true}']))


def test_parse_sse_skips_noise():
    lines = [b": keep-alive", b"", b"data: not json", b'data: {"textResponse": "ok", "close": true}']
    assert [e["textResponse"] for e in integrations.parse_sse(lines)] == ["ok"]


def test_scope_message_names_the_recording():
    assert integrations.scope_message("Who owns it?", "Standup") == 'About the recording "Standup": Who owns it?'
    assert integrations.scope_message("Who owns it?") == "Who owns it?"


def test_transcript_document_layout():
    doc = integrations.transcript_document(result(), "Standup", "Summarize", "24 Sep 2026")
    assert doc.splitlines()[:3] == ["Recording: Standup", "Recorded: 24 Sep 2026",
                                    "Duration: 4 min · Language: en · Speakers: 2"]
    assert doc.index("AI summary (Summarize):") < doc.index("Transcript:")


@pytest.mark.skipif(os.name != "nt", reason="Windows Credential Manager")
def test_api_key_roundtrip_in_credential_manager():
    target = "WhisperScribe/unit-test-key"
    try:
        integrations.save_api_key("ABC-123 é", target)
        assert integrations.load_api_key(target) == "ABC-123 é"
        integrations.save_api_key("", target)  # empty key deletes it
        assert integrations._win_cred_read(target) == ""
    finally:
        integrations._win_cred_write(target, "")


# ---------- Ollama ----------
def test_ollama_models_skips_embedders(server):
    server.routes[("GET", "/api/tags")] = {"models": [{"name": "qwen3.5:9b"}, {"name": "nomic-embed-text:latest"},
                                                      {"name": "phi4:latest"}]}
    assert integrations.OllamaClient(server.url).models() == ["phi4:latest", "qwen3.5:9b"]


def test_ollama_chat_retries_without_think_flag(server):
    def chat(body):
        if "think" in body:
            return 400, {"error": "\"phi4\" does not support thinking"}
        return {"message": {"content": "hi"}, "eval_count": 1}

    server.routes[("POST", "/api/chat")] = chat
    r = integrations.OllamaClient(server.url).chat("phi4", [{"role": "user", "content": "hey"}], num_ctx=4096)
    assert r["message"]["content"] == "hi"
    assert len(server.requests) == 2 and server.requests[1]["body"]["options"]["num_ctx"] == 4096


def test_ollama_context_length(server):
    server.routes[("POST", "/api/show")] = {"model_info": {"general.architecture": "qwen35",
                                                           "qwen35.context_length": 262144}}
    assert integrations.OllamaClient(server.url).context_length("qwen3.5:9b") == 262144


def test_nothink_variant_reuses_base_weights(server):
    digest = "a" * 64
    server.routes[("GET", "/api/tags")] = {"models": [{"name": "qwen3.5:9b"}]}
    server.routes[("POST", "/api/show")] = {"modelfile": f"# comment\nFROM C:\\models\\blobs\\sha256-{digest}\n"}
    server.routes[("POST", "/api/create")] = {"status": "success"}
    assert integrations.OllamaClient(server.url).ensure_nothink_variant("qwen3.5:9b", "qwen3.5-nothink:9b")
    create = server.requests[-1]["body"]
    assert create["model"] == "qwen3.5-nothink:9b"
    assert create["files"] == {"model.gguf": f"sha256:{digest}"}
    assert create["template"].endswith("<think>\n\n</think>\n\n")


def test_nothink_variant_skips_when_present_or_base_missing(server):
    server.routes[("GET", "/api/tags")] = {"models": [{"name": "qwen3.5-nothink:9b"}]}
    client = integrations.OllamaClient(server.url)
    assert client.ensure_nothink_variant("qwen3.5:9b", "qwen3.5-nothink:9b")
    server.routes[("GET", "/api/tags")] = {"models": []}
    assert not client.ensure_nothink_variant("qwen3.5:9b", "qwen3.5-nothink:9b")
    assert all(r["path"] == "/api/tags" for r in server.requests)


def test_existing_workspace_only_gets_its_model_updated(server):
    server.routes[("GET", "/api/v1/workspaces")] = {"workspaces": [
        {"name": "Meeting Transcripts", "slug": "mt", "chatModel": "qwen3.5:9b", "openAiPrompt": "my own prompt"}]}
    server.routes[("POST", "/api/v1/workspace/mt/update")] = {"workspace": {}}
    integrations.AnythingLLMClient("k", server.url).ensure_workspace(
        "Meeting Transcripts", integrations.workspace_settings("qwen3.5-nothink:9b"))
    assert server.requests[-1]["body"] == {"chatProvider": "ollama", "chatModel": "qwen3.5-nothink:9b"}


def test_last_sources_comes_from_saved_history(server):
    server.routes[("GET", "/api/v1/workspace/mt/thread/t/chats")] = {"history": [
        {"role": "user", "content": "q"}, {"role": "assistant", "content": "a", "sources": [{"title": "x.txt"}]}]}
    sources = integrations.AnythingLLMClient("k", server.url).last_sources("mt", "t")
    assert [integrations.source_title(s) for s in sources] == ["x"]


def test_ollama_available_false_when_down():
    assert not integrations.OllamaClient("http://127.0.0.1:9").available()
