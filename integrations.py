"""Local HTTP integrations: Ollama (LLM runtime) and AnythingLLM (knowledge base + chat).

Standard library only, so it adds no dependencies. Everything talks to localhost by default.
"""
import json
import logging
import re
import urllib.error
import urllib.request
from typing import Iterator, Optional

log = logging.getLogger("whisperscribe.integrations")

OLLAMA_URL = "http://127.0.0.1:11434"
# ChatML with an empty <think></think> already written for the assistant = Qwen's "thinking off" mode.
NOTHINK_TEMPLATE = ("{{- range .Messages }}<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>\n{{ end }}"
                    "<|im_start|>assistant\n<think>\n\n</think>\n\n")
ANYTHINGLLM_URL = "http://localhost:3001"
KEYRING_SERVICE = "WhisperScribe"
KEYRING_USER = "anythingllm-api-key"


class IntegrationError(Exception):
    pass


def _request(url, method="GET", payload=None, headers=None, timeout=30):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Content-Type": "application/json", "Accept": "application/json", **(headers or {})})
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")[:300]
        raise IntegrationError(f"{method} {url} → HTTP {exc.code}: {body}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise IntegrationError(f"Can't reach {url}: {getattr(exc, 'reason', exc)}") from exc


def _json(url, method="GET", payload=None, headers=None, timeout=30):
    with _request(url, method, payload, headers, timeout) as resp:
        body = resp.read().decode()
    try:
        return json.loads(body or "null")
    except json.JSONDecodeError:
        return {"text": body}  # some endpoints answer with plain "OK"


# ---------------------------------------------------------------- Ollama
class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_URL):
        self.base_url = base_url.rstrip("/")

    def available(self) -> bool:
        try:
            _json(f"{self.base_url}/api/version", timeout=2)
            return True
        except IntegrationError:
            return False

    def models(self) -> list:
        """Installed chat models (embedding-only models are skipped)."""
        tags = _json(f"{self.base_url}/api/tags", timeout=5).get("models", [])
        return sorted(m["name"] for m in tags if "embed" not in m["name"] and "bert" not in
                      (m.get("details", {}).get("family") or ""))

    def loaded(self) -> list:
        """Models currently in memory, with their VRAM use (bytes)."""
        return _json(f"{self.base_url}/api/ps", timeout=5).get("models", [])

    def context_length(self, model: str) -> int:
        """The model's maximum context window, from its metadata (0 if unknown)."""
        info = _json(f"{self.base_url}/api/show", "POST", {"model": model}, timeout=10).get("model_info", {})
        return next((int(v) for k, v in info.items() if k.endswith(".context_length")), 0)

    def ensure_nothink_variant(self, base: str, variant: str) -> bool:
        """Create `variant`: `base` with its thinking phase pre-filled as empty (Qwen's official non-thinking mode).

        AnythingLLM can't send Ollama's think=false, so a thinking model spends 5-40 s "reasoning" before every
        answer and the reasoning leaks into replies. The variant reuses the base model's weights (no download,
        no extra disk). Returns False if the base model isn't installed.
        """
        installed = {m["name"] for m in _json(f"{self.base_url}/api/tags", timeout=5).get("models", [])}
        if variant in installed:
            return True
        if base not in installed:
            return False
        modelfile = _json(f"{self.base_url}/api/show", "POST", {"model": base}, timeout=10).get("modelfile", "")
        match = re.search(r"^FROM\s+.*?(sha256-[0-9a-f]{64})", modelfile, re.M)
        if not match:
            raise IntegrationError(f"Couldn't find the weights of {base} to build {variant}.")
        _json(f"{self.base_url}/api/create", "POST", {
            "model": variant, "stream": False,
            "files": {"model.gguf": match.group(1).replace("sha256-", "sha256:")},
            "template": NOTHINK_TEMPLATE,
            "parameters": {"stop": ["<|im_end|>", "<|im_start|>"], "temperature": 0.7, "top_p": 0.8,
                           "top_k": 20, "presence_penalty": 1.5},
        }, timeout=300)
        log.info("Created Ollama model %s from %s", variant, base)
        return True

    def unload(self, model: str):
        """Free a model's VRAM immediately (keep_alive=0)."""
        _json(f"{self.base_url}/api/generate", "POST", {"model": model, "keep_alive": 0}, timeout=30)

    def chat(self, model: str, messages: list, num_ctx: int = 16384, max_tokens: int = 900,
             temperature: float = 0.0, timeout: int = 600) -> dict:
        """Non-streaming chat. Returns Ollama's response dict (message.content, eval_count, durations…)."""
        payload = {"model": model, "messages": messages, "stream": False, "think": False,
                   "options": {"num_ctx": num_ctx, "num_predict": max_tokens, "temperature": temperature}}
        try:
            return _json(f"{self.base_url}/api/chat", "POST", payload, timeout=timeout)
        except IntegrationError as exc:
            if "think" not in str(exc):
                raise
            payload.pop("think")  # model without a thinking mode rejects the flag
            return _json(f"{self.base_url}/api/chat", "POST", payload, timeout=timeout)


# ---------------------------------------------------------- AnythingLLM
def load_api_key() -> str:
    try:
        import keyring
        return keyring.get_password(KEYRING_SERVICE, KEYRING_USER) or ""
    except Exception:
        log.warning("Could not read AnythingLLM key from the credential store", exc_info=True)
        return ""


def save_api_key(key: str):
    import keyring
    if key:
        keyring.set_password(KEYRING_SERVICE, KEYRING_USER, key)
    else:
        try:
            keyring.delete_password(KEYRING_SERVICE, KEYRING_USER)
        except Exception:
            pass


class AnythingLLMClient:
    """Thin wrapper over AnythingLLM's developer API (/api/v1, see http://localhost:3001/api/docs)."""

    def __init__(self, api_key: str, base_url: str = ANYTHINGLLM_URL):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _url(self, path):
        return f"{self.base_url}/api/v1{path}"

    def _call(self, path, method="GET", payload=None, timeout=60):
        return _json(self._url(path), method, payload, {"Authorization": f"Bearer {self.api_key}"}, timeout)

    # --- setup
    def check(self) -> bool:
        """True if the server is up and the API key is accepted."""
        if not self.api_key:
            raise IntegrationError("No AnythingLLM API key set (AnythingLLM → Settings → Developer API).")
        return bool(self._call("/auth", timeout=5).get("authenticated"))

    def ollama_context(self) -> int:
        """The context size AnythingLLM asks Ollama for (0 if unknown)."""
        settings = self._call("/system", timeout=10).get("settings", {})
        try:
            return int(settings.get("OllamaLLMTokenLimit") or 0)
        except (TypeError, ValueError):
            return 0

    def workspaces(self) -> list:
        return self._call("/workspaces").get("workspaces", [])

    def ensure_workspace(self, name: str, settings: Optional[dict] = None) -> str:
        """Return the slug of workspace `name`, creating it (and applying `settings`) if needed.

        For an existing workspace only the chat model is kept in sync, so edits made in AnythingLLM survive."""
        for ws in self.workspaces():
            if ws.get("name") == name:
                model = (settings or {}).get("chatModel")
                if model and ws.get("chatModel") != model:
                    self.update_workspace(ws["slug"], {"chatProvider": settings["chatProvider"], "chatModel": model})
                return ws["slug"]
        slug = self._call("/workspace/new", "POST", {"name": name})["workspace"]["slug"]
        if settings:
            self.update_workspace(slug, settings)
        log.info("Created AnythingLLM workspace %s", slug)
        return slug

    def update_workspace(self, slug: str, settings: dict):
        self._call(f"/workspace/{slug}/update", "POST", settings)

    # --- documents
    def add_text_document(self, text: str, title: str, workspace_slug: str, description: str = "",
                          source: str = "") -> str:
        """Store `text` as a document and embed it into the workspace. Returns the document's location."""
        payload = {"textContent": text, "addToWorkspaces": workspace_slug,
                   "metadata": {"title": title, "description": description, "docSource": source,
                                "docAuthor": "WhisperScribe"}}
        docs = self._call("/document/raw-text", "POST", payload, timeout=600).get("documents") or []
        if not docs:
            raise IntegrationError("AnythingLLM accepted the upload but returned no document.")
        return docs[0].get("location", "")

    # --- threads & chat
    def new_thread(self, slug: str, name: str) -> str:
        return self._call(f"/workspace/{slug}/thread/new", "POST", {"name": name[:100]})["thread"]["slug"]

    def thread_history(self, slug: str, thread_slug: str) -> list:
        """Earlier messages: [{"role": "user"|"assistant", "content": …, "sources": […]}]."""
        return self._call(f"/workspace/{slug}/thread/{thread_slug}/chats").get("history", [])

    def last_sources(self, slug: str, thread_slug: str) -> list:
        """Sources of the latest answer. AnythingLLM saves them with the chat but doesn't always stream them."""
        for msg in reversed(self.thread_history(slug, thread_slug)):
            if msg.get("role") == "assistant":
                return msg.get("sources") or []
        return []

    def stream_chat(self, slug: str, message: str, thread_slug: Optional[str] = None,
                    mode: str = "chat") -> Iterator[dict]:
        """Yield AnythingLLM stream events: {"type": "textResponseChunk", "textResponse": …, "sources": […]}."""
        path = f"/workspace/{slug}/thread/{thread_slug}/stream-chat" if thread_slug else \
            f"/workspace/{slug}/stream-chat"
        resp = _request(self._url(path), "POST", {"message": message, "mode": mode},
                        {"Authorization": f"Bearer {self.api_key}", "Accept": "text/event-stream"}, timeout=600)
        with resp:
            yield from parse_sse(resp)


def parse_sse(lines) -> Iterator[dict]:
    """Parse a text/event-stream of `data: {json}` lines. Stops after an event with close=true."""
    for raw in lines:
        line = raw.decode("utf-8", errors="replace").strip() if isinstance(raw, bytes) else raw.strip()
        if not line.startswith("data:"):
            continue
        body = line[5:].strip()
        if not body:
            continue
        try:
            event = json.loads(body)
        except json.JSONDecodeError:
            continue
        if event.get("error"):
            raise IntegrationError(str(event["error"]))
        yield event
        if event.get("close"):
            return


DEFAULT_WORKSPACE = "Meeting Transcripts"
# Picked with benchmarks/bench_ollama.py (see docs/BENCHMARKS.md). Set only on the transcripts workspace,
# so other AnythingLLM workspaces keep their own model.
RECOMMENDED_BASE_MODEL = "qwen3.5:9b"
RECOMMENDED_CHAT_MODEL = "qwen3.5-nothink:9b"  # built from the base by OllamaClient.ensure_nothink_variant
WORKSPACE_PROMPT = (
    "You answer questions about the user's recorded meetings and voice notes. Each document is one recording: it "
    "starts with the recording's title and date, then an optional AI summary, then a transcript where people are "
    "labelled 'Speaker 1', 'Speaker 2' and so on. Answer only from the provided context. If the context doesn't "
    "contain the answer, say that it wasn't mentioned. When it helps, say which recording the answer comes from "
    "by its title, but never refer to 'Context 1', 'Context 2' or similar. When someone says 'I' or 'me' "
    "(\"I'll write it up\"), the owner is the speaker label on that line, not a person they mentioned earlier."
)


def workspace_settings(chat_model: str = "") -> dict:
    settings = {"openAiPrompt": WORKSPACE_PROMPT, "openAiTemp": 0.2, "openAiHistory": 20, "chatMode": "chat",
                "topN": 8, "similarityThreshold": 0.25,
                "queryRefusalResponse": "I couldn't find that in your recordings."}
    if chat_model:
        settings.update(chatProvider="ollama", chatModel=chat_model)
    return settings


def sync_transcript(client: "AnythingLLMClient", result, title: str, prompt: str = "", recorded: str = "",
                    workspace: str = DEFAULT_WORKSPACE, chat_model: str = "") -> dict:
    """Upload one recording and open a thread for it. Returns the entry the chat panel keeps."""
    slug = client.ensure_workspace(workspace, workspace_settings(chat_model))
    doc = client.add_text_document(transcript_document(result, title, prompt, recorded), title, slug,
                                   description=f"WhisperScribe transcript recorded {recorded}".strip(),
                                   source="WhisperScribe")
    thread = client.new_thread(slug, title)
    return {"title": title, "workspace": slug, "thread": thread, "document": doc, "recorded": recorded}


def source_title(source: dict) -> str:
    """AnythingLLM reports a raw-text document's title as a slug file name ("launch-sync.txt")."""
    title = source.get("title") or source.get("name") or ""
    return re.sub(r"\.txt$", "", title)


def scope_message(question: str, title: str = "") -> str:
    """Name the recording in the message so AnythingLLM's search favours that recording's chunks."""
    return f'About the recording "{title}": {question}' if title else question


def transcript_document(result, title: str, prompt: str = "", recorded: str = "") -> str:
    """Text uploaded to AnythingLLM: a header the retriever can match on, the summary, then the transcript."""
    lines = [f"Recording: {title}"]
    if recorded:
        lines.append(f"Recorded: {recorded}")
    lines.append(f"Duration: {result.duration / 60:.0f} min · Language: {result.language}"
                 + (f" · Speakers: {result.num_speakers}" if result.num_speakers else ""))
    if result.summary:
        lines += ["", f"AI summary ({prompt}):" if prompt else "AI summary:", result.summary]
    lines += ["", "Transcript:", result.speaker_text]
    return "\n".join(lines)
