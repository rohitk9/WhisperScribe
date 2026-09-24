"""Chat tab: talk to your transcripts through AnythingLLM, plus its connection settings dialog."""
import json
import logging
import os
import re
import threading

import customtkinter as ctk

import integrations
from ui_theme import ACCENT, ACCENT_HOVER, BORDER, CARD, DANGER, FIELD, MUTED, TEXT

log = logging.getLogger("whisperscribe.chat")

ALL_MEETINGS = "All meetings"
THREADS_PATH = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "WhisperScribe", "chat_threads.json")
SETUP_TEXT = (
    "Chat with your recordings using AnythingLLM, running locally.\n\n"
    "1.  In AnythingLLM, open Settings → Developer API → Generate New API Key.\n"
    "2.  Click ⚙ above, paste the key, and press Test.\n"
    "3.  Transcribe something with \"Send to AnythingLLM\" switched on, then ask away.\n\n"
    "Transcripts go into the \"Meeting Transcripts\" workspace, one thread per recording, so you can also carry on "
    "the conversation inside AnythingLLM."
)


def tidy_markdown(text: str) -> str:
    """Plain-text rendering of an LLM answer: bullets become •, headings lose their #. **bold** is kept
    for the text box to render."""
    out = []
    for line in text.splitlines():
        line = re.sub(r"^(\s*)[*\-+]\s+", lambda m: m.group(1) + "•  ", line)
        line = re.sub(r"^#{1,6}\s+(.*)", r"**\1**", line)
        out.append(line)
    return "\n".join(out)


def load_threads() -> dict:
    try:
        with open(THREADS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"recordings": [], "all_thread": {}}
    except Exception:
        log.warning("Could not read %s", THREADS_PATH, exc_info=True)
        return {"recordings": [], "all_thread": {}}


def save_threads(data: dict):
    os.makedirs(os.path.dirname(THREADS_PATH), exist_ok=True)
    with open(THREADS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)


class AnythingLLMSettingsDialog(ctk.CTkToplevel):
    def __init__(self, app, on_saved):
        super().__init__(app)
        self.app, self.on_saved = app, on_saved
        self.title("AnythingLLM connection")
        self.geometry("460x330")
        self.resizable(False, False)
        self.configure(fg_color=CARD)
        self.transient(app)
        self.after(100, self.grab_set)

        pad = {"padx": 20, "sticky": "ew"}
        self.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(self, text="AnythingLLM connection", font=app.f_h2, text_color=TEXT).grid(
            row=0, column=0, pady=(18, 2), **pad)
        ctk.CTkLabel(self, text="Everything stays on this computer. The key is kept in Windows Credential Manager.",
                     font=app.f_small, text_color=MUTED, wraplength=420, justify="left").grid(row=1, column=0, **pad)

        self.url = self._field(2, "Server address", app.settings.get("anythingllm_url", integrations.ANYTHINGLLM_URL))
        self.key = self._field(4, "Developer API key", integrations.load_api_key(), show="•")
        self.workspace = self._field(6, "Workspace for transcripts",
                                     app.settings.get("anythingllm_workspace", integrations.DEFAULT_WORKSPACE))

        row = ctk.CTkFrame(self, fg_color="transparent")
        row.grid(row=8, column=0, pady=(16, 0), **pad)
        self.status = ctk.CTkLabel(row, text="", font=app.f_small, text_color=MUTED, anchor="w")
        self.status.pack(side="left", fill="x", expand=True)
        ctk.CTkButton(row, text="Save", width=80, fg_color=ACCENT, hover_color=ACCENT_HOVER,
                      command=self._save).pack(side="right")
        app._ghost_button(row, "Test", self._test, width=70).pack(side="right", padx=8)

    def _field(self, row, label, value, **kw):
        ctk.CTkLabel(self, text=label, font=self.app.f_small, text_color=MUTED, anchor="w").grid(
            row=row, column=0, padx=20, pady=(10, 0), sticky="ew")
        entry = ctk.CTkEntry(self, font=self.app.f_body, fg_color=FIELD, border_color=BORDER, **kw)
        entry.insert(0, value)
        entry.grid(row=row + 1, column=0, padx=20, sticky="ew")
        return entry

    def _client(self):
        return integrations.AnythingLLMClient(self.key.get().strip(), self.url.get().strip())

    def _test(self):
        self.status.configure(text="Testing…", text_color=MUTED)
        client = self._client()

        def run():
            try:
                client.check()
                n = len(client.workspaces())
                msg, color = f"✓ Connected · {n} workspace(s)", ACCENT
            except Exception as exc:
                msg, color = f"✗ {exc}"[:90], DANGER
            self.app.post(self.status.configure, text=msg, text_color=color)

        threading.Thread(target=run, daemon=True).start()

    def _save(self):
        integrations.save_api_key(self.key.get().strip())
        self.app.settings["anythingllm_url"] = self.url.get().strip() or integrations.ANYTHINGLLM_URL
        self.app.settings["anythingllm_workspace"] = self.workspace.get().strip() or integrations.DEFAULT_WORKSPACE
        self.app._save_settings()
        self.on_saved()
        self.destroy()


class ChatPanel:
    """Lives inside the "Chat" tab of the results card."""

    def __init__(self, app, parent):
        self.app = app
        self.data = load_threads()
        self.busy = False
        self.client = None

        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(1, weight=1)

        top = ctk.CTkFrame(parent, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ctk.CTkLabel(top, text="Ask about", font=app.f_small, text_color=MUTED).pack(side="left")
        self.scope_var = ctk.StringVar(value=ALL_MEETINGS)
        self.scope_menu = app._option_menu(top, [ALL_MEETINGS], self.scope_var, lambda _: self._on_scope(),
                                           height=28, width=240, dynamic_resizing=False)
        self.scope_menu.pack(side="left", padx=8)
        app._ghost_button(top, "⚙", self.open_settings, width=32, height=28).pack(side="right")
        app._ghost_button(top, "New chat", self.new_chat, width=10, height=28).pack(side="right", padx=6)

        self.box = ctk.CTkTextbox(parent, font=app.f_body, fg_color=FIELD, border_width=0, corner_radius=10,
                                  wrap="word")
        self.box.grid(row=1, column=0, sticky="nsew")
        self.box.tag_config("you", foreground=ACCENT)
        self.box.tag_config("muted", foreground="#8B93A1")
        self.box.tag_config("error", foreground=DANGER)
        self.box._textbox.tag_config("bold", font=app.f_body_bold)
        self.box._textbox.tag_config("label", font=app.f_body_bold)
        self.box.configure(state="disabled")

        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        row.grid_columnconfigure(0, weight=1)
        self.entry = ctk.CTkEntry(row, font=app.f_body, fg_color=FIELD, border_color=BORDER, height=36,
                                  placeholder_text="Ask anything about your recordings…")
        self.entry.grid(row=0, column=0, sticky="ew")
        self.entry.bind("<Return>", lambda e: self.send())
        self.send_btn = ctk.CTkButton(row, text="Send", width=72, height=36, fg_color=ACCENT,
                                      hover_color=ACCENT_HOVER, command=self.send)
        self.send_btn.grid(row=0, column=1, padx=(8, 0))
        self.status = ctk.CTkLabel(parent, text="", font=app.f_small, text_color=MUTED, anchor="w")
        self.status.grid(row=3, column=0, sticky="ew", pady=(4, 0))

        self._refresh_scopes()
        self.reconnect()

    # ------------------------------------------------------------- setup
    def reconnect(self):
        key = integrations.load_api_key()
        self.client = integrations.AnythingLLMClient(
            key, self.app.settings.get("anythingllm_url", integrations.ANYTHINGLLM_URL)) if key else None
        if not self.client:
            self._show_setup()
            self.status.configure(text="Not connected", text_color=MUTED)
            return
        self.status.configure(text="Connecting to AnythingLLM…", text_color=MUTED)
        client = self.client

        def run():
            try:
                client.check()
                text, color = f"Connected · workspace “{self.workspace_name}”", MUTED
                try:  # match AnythingLLM's context size so Ollama never reloads between chat and summary
                    self.app.engine.ollama_num_ctx = client.ollama_context() or self.app.engine.ollama_num_ctx
                except Exception:
                    pass
            except Exception as exc:
                text, color = f"AnythingLLM: {exc}"[:120], DANGER
            self.app.post(self.status.configure, text=text, text_color=color)
            if color == MUTED:
                self.app.post(self._on_scope)

        threading.Thread(target=run, daemon=True).start()

    @property
    def workspace_name(self):
        return self.app.settings.get("anythingllm_workspace", integrations.DEFAULT_WORKSPACE)

    def open_settings(self):
        AnythingLLMSettingsDialog(self.app, self.reconnect)

    def _show_setup(self):
        self._reset()
        self._write(SETUP_TEXT, "muted")

    # ------------------------------------------------------ recordings
    def add_recording(self, entry: dict):
        """Called after a transcript was synced: remember its thread and make it the chat scope."""
        recs = [r for r in self.data["recordings"] if r.get("thread") != entry["thread"]]
        self.data["recordings"] = ([entry] + recs)[:100]
        save_threads(self.data)
        self._refresh_scopes()
        self.scope_var.set(self._label(entry))
        self._on_scope()

    @staticmethod
    def _label(entry):
        return f"{entry['title']} · {entry.get('recorded', '')}".strip(" ·")

    def _refresh_scopes(self):
        labels = [ALL_MEETINGS] + [self._label(r) for r in self.data["recordings"]]
        self.scope_menu.configure(values=labels)

    def _current(self):
        label = self.scope_var.get()
        return next((r for r in self.data["recordings"] if self._label(r) == label), None)

    def _on_scope(self):
        """Show the selected thread's earlier messages (fetched from AnythingLLM)."""
        if not self.client:
            return self._show_setup()
        rec = self._current()
        thread = rec["thread"] if rec else self.data.get("all_thread", {}).get(self.workspace_name)
        workspace = rec["workspace"] if rec else None
        self._reset()
        intro = (f"Asking about “{rec['title']}”. Answers come from this recording's transcript and summary."
                 if rec else "Asking across every recording in the workspace.")
        self._write(intro + "\n\n", "muted")
        if not thread:
            return
        client = self.client

        def run():
            try:
                slug = workspace or client.ensure_workspace(self.workspace_name)
                history = client.thread_history(slug, thread)
            except Exception as exc:
                log.warning("Could not load chat history: %s", exc)
                return
            self.app.post(self._render_history, history, rec)

        threading.Thread(target=run, daemon=True).start()

    def _render_history(self, history, rec):
        if self._current() is not rec:
            return  # user switched scope meanwhile
        for msg in history:
            content = msg.get("content", "")
            if msg.get("role") == "user":
                if rec:
                    content = re.sub(r'^About the recording ".*?": ', "", content)
                self._turn("You", content)
            else:
                self._turn("Assistant", content, msg.get("sources"))

    def new_chat(self):
        """Start a fresh thread for the current scope (the old one stays in AnythingLLM)."""
        if not self.client or self.busy:
            return
        rec = self._current()
        client = self.client

        def run():
            try:
                slug = rec["workspace"] if rec else client.ensure_workspace(self.workspace_name)
                thread = client.new_thread(slug, (rec["title"] if rec else "All meetings") + " (new)")
            except Exception as exc:
                return self.app.post(self._write, f"\n{exc}\n", "error")
            if rec:
                rec["thread"] = thread
            else:
                self.data.setdefault("all_thread", {})[self.workspace_name] = thread
            save_threads(self.data)
            self.app.post(self._on_scope)

        threading.Thread(target=run, daemon=True).start()

    # ------------------------------------------------------------ chat
    def send(self):
        question = self.entry.get().strip()
        if not question or self.busy:
            return
        if not self.client:
            return self.open_settings()
        self.entry.delete(0, "end")
        rec = self._current()
        self._turn("You", question)
        self._write("Assistant\n", "label")
        self.box.mark_set("answer_start", "end-1c")
        self.box.mark_gravity("answer_start", "left")
        self.busy = True
        self.send_btn.configure(state="disabled")
        self.status.configure(text="Thinking…", text_color=MUTED)
        threading.Thread(target=self._stream, args=(question, rec), daemon=True).start()

    def _stream(self, question, rec):
        app, client = self.app, self.client
        try:
            if not app.running:  # leave the GPU to Ollama while chatting
                app.engine.unload_summarizer()
                app.engine.unload_whisper()
            if rec:
                slug, thread = rec["workspace"], rec["thread"]
            else:
                slug = client.ensure_workspace(self.workspace_name, integrations.workspace_settings(
                    app.chat_model_for_workspace()))
                thread = self.data.setdefault("all_thread", {}).get(self.workspace_name)
                if not thread:
                    thread = client.new_thread(slug, "All meetings")
                    self.data["all_thread"][self.workspace_name] = thread
                    save_threads(self.data)
            sources = []
            message = integrations.scope_message(question, rec["title"] if rec else "")
            for event in client.stream_chat(slug, message, thread):
                if event.get("textResponse"):
                    app.post(self._write, event["textResponse"])
                if event.get("sources"):
                    sources = event["sources"]
            if not sources:
                try:
                    sources = client.last_sources(slug, thread)
                except Exception:
                    pass
            app.post(self._finish_answer, sources, None)
        except Exception as exc:
            log.warning("Chat failed", exc_info=True)
            app.post(self._finish_answer, [], exc)

    def _finish_answer(self, sources, error):
        # Re-draw the streamed answer tidily: chunk boundaries can split markdown markers mid-stream.
        answer = self.box.get("answer_start", "end-1c")
        self.box.configure(state="normal")
        self.box.delete("answer_start", "end-1c")
        self.box.configure(state="disabled")
        if answer.strip():
            self._write(tidy_markdown(answer).strip() + "\n")
        if error:
            self._write(f"{error}\n", "error")
        self._write_sources(sources)
        self._write("\n")
        self.busy = False
        self.send_btn.configure(state="normal")
        self.status.configure(text=f"Connected · workspace “{self.workspace_name}”" if not error else
                              "Something went wrong. See the message above.",
                              text_color=MUTED if not error else DANGER)
        self.entry.focus_set()

    # ---------------------------------------------------------- rendering
    def _turn(self, who, text, sources=None):
        self._write(f"{who}\n", "you" if who == "You" else "label")
        self._write((text if who == "You" else tidy_markdown(text)).strip() + "\n")
        if who != "You":
            self._write_sources(sources)
        self._write("\n")

    def _write_sources(self, sources):
        titles = []
        for s in sources or []:
            t = integrations.source_title(s)
            if t and t not in titles:
                titles.append(t)
        if titles:
            self._write("Sources: " + " · ".join(titles[:5]) + "\n", "muted")

    def _reset(self):
        self.box.configure(state="normal")
        self.box.delete("1.0", "end")
        self.box.configure(state="disabled")

    def _write(self, text, tag=None):
        self.box.configure(state="normal")
        if tag in (None, "label"):  # render **bold** from the model without the asterisks
            for i, part in enumerate(re.split(r"\*\*(.+?)\*\*", text)):
                part = part.replace("**", "")  # a marker split across streamed chunks
                self.box.insert("end", part, "bold" if (i % 2 or tag == "label") else ())
        else:
            self.box.insert("end", text, tag)
        self.box.see("end")
        self.box.configure(state="disabled")

    def text(self):
        return self.box.get("1.0", "end-1c")
