"""
Intelligent CLI Assistant — Project 1-I-A

A terminal-based AI assistant on a chosen topic domain (cooking, history,
or programming), with conversation memory, topic switching, and Rich
formatted output.

Architecture (per spec):
    User CLI -> LangChain ChatModel (Groq) -> Conversation Buffer Memory
    -> Rich Console Output

VERIFIED API NOTES (checked against the actually-installed package
versions before writing this, not assumed from docs/blog posts):
    - langchain-groq==1.1.3: ChatGroq's model-name parameter is
      `model_name`, NOT `model` — several docs/tutorials online show
      `model=`, which does not match this installed version.
    - langchain-core==1.4.8: RunnableWithMessageHistory works correctly
      (memory genuinely accumulates across turns, verified with a mock
      model before writing this file) but emits a real
      LangChainDeprecationWarning pointing to LangGraph's persistence
      layer as the future-recommended replacement. The older
      ConversationBufferMemory/ConversationChain classes are MORE
      deprecated (explicit removal="1.0" decorator) and were avoided
      entirely. RunnableWithMessageHistory is used here deliberately,
      with this warning left visible rather than suppressed — see the
      README for the reasoning.

Extra features beyond the spec, all low-risk additions reusing patterns
already verified elsewhere in this internship's projects:
    - streaming responses (token-by-token, via Rich's Live display)
    - /tokens — turn counter; per-turn token counts are NOT shown, since
      Groq's streaming responses don't expose usage stats in the field
      langchain-groq reads (confirmed during development — see
      ask_streaming()'s docstring for detail)
    - /save — export the conversation transcript to a markdown file
    - /history — view the full transcript without scrolling

Run:
    python cli_assistant.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

MODEL_NAME = "openai/gpt-oss-120b"
SESSION_ID = "cli-session"  # single local user, single session — no need
                             # for multi-user session IDs in this CLI tool

# Topic domains, each with a system prompt that scopes the assistant's
# persona and a short welcome blurb shown when switching into it.
TOPICS = {
    "cooking": {
        "label": "Cooking",
        "system_prompt": (
            "You are a knowledgeable, friendly cooking assistant. Answer "
            "questions about recipes, techniques, ingredient substitutions, "
            "and kitchen science. Keep answers practical and concise unless "
            "asked for detail. If asked something outside cooking/food, "
            "politely say this assistant is focused on cooking and suggest "
            "switching topics with /topic."
        ),
        "welcome": "Ask me about recipes, techniques, substitutions, or kitchen science.",
    },
    "history": {
        "label": "History",
        "system_prompt": (
            "You are a knowledgeable history assistant. Answer questions "
            "about historical events, figures, and eras accurately and "
            "with appropriate nuance — note where historical accounts are "
            "disputed or incomplete. If asked something outside history, "
            "politely say this assistant is focused on history and suggest "
            "switching topics with /topic."
        ),
        "welcome": "Ask me about historical events, figures, or eras.",
    },
    "programming": {
        "label": "Programming",
        "system_prompt": (
            "You are a knowledgeable programming assistant. Answer "
            "questions about code, languages, debugging, and software "
            "design clearly, with code examples in markdown code blocks "
            "where useful. If asked something outside programming, "
            "politely say this assistant is focused on programming and "
            "suggest switching topics with /topic."
        ),
        "welcome": "Ask me about code, languages, debugging, or design.",
    },
}

console = Console()


def get_llm() -> ChatGroq:
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        console.print(
            "[bold red]GROQ_API_KEY not set.[/bold red] Create a .env file "
            "with GROQ_API_KEY=gsk_... or export it as an environment variable."
        )
        sys.exit(1)
    # model_name, not model — verified against the installed langchain-groq
    # version (see module docstring).
    return ChatGroq(model_name=MODEL_NAME, temperature=0.5, streaming=True)


class AssistantSession:
    """Wraps the LangChain runnable + memory + current topic state for one
    CLI session. Kept as a small class (rather than module-level globals)
    so /topic switching can cleanly swap the system prompt without losing
    the underlying chat history object."""

    def __init__(self, llm: ChatGroq, topic_key: str) -> None:
        self.llm = llm
        self.topic_key = topic_key
        self._history_store: dict[str, BaseChatMessageHistory] = {
            SESSION_ID: InMemoryChatMessageHistory()
        }
        self.turn_count = 0

        self._chain = RunnableWithMessageHistory(
            self.llm,
            self._get_session_history,
        )

    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        return self._history_store[session_id]

    @property
    def topic(self) -> dict:
        return TOPICS[self.topic_key]

    def switch_topic(self, new_topic_key: str) -> None:
        """Switches topic WITHOUT clearing conversation history — the
        assistant's persona/scope changes, but earlier turns are still
        visible to it. This is a deliberate choice: a real assistant
        switching domains mid-conversation would still remember what was
        said, it would just start answering in a different scope."""
        self.topic_key = new_topic_key

    def _build_messages(self, user_input: str) -> list:
        # The system prompt is re-sent fresh every turn (not stored in
        # history) so that switching topics immediately changes behavior
        # without needing to touch/clear the stored history.
        return [SystemMessage(content=self.topic["system_prompt"]), HumanMessage(content=user_input)]

    def ask_streaming(self, user_input: str):
        """Streams the response token-by-token. Yields text chunks.

        TWO BUGS FOUND VIA LIVE TESTING, both fixed here:

        1. Token tracking during streaming is unreliable. Verified live:
           ChatGroq streaming chunks have usage_metadata=None. Confirmed
           via Groq's own community/GitHub discussions: Groq's streaming
           response puts usage stats in a nonstandard `x_groq.usage`
           field (not the standard OpenAI-compatible `usage` field),
           which langchain-groq does not currently surface into
           usage_metadata during streaming. Rather than show a /tokens
           counter that always silently reads 0 (misleading — looks like
           a working feature that just reports nothing happened), token
           tracking is OFF while streaming; see /tokens command for the
           user-facing explanation.

        2. SystemMessage was leaking into persisted history. Verified
           live: RunnableWithMessageHistory saves whatever message list
           is passed to it, including the SystemMessage re-sent every
           turn for topic-scoping — meaning every turn was silently
           appending ANOTHER system message into history, which both
           bloats context unnecessarily and undermines the topic-switch
           design (old topic instructions never actually leave history).
           Fixed by stripping any SystemMessage out of history immediately
           after each turn completes.
        """
        config = {"configurable": {"session_id": SESSION_ID}}

        for chunk in self._chain.stream(self._build_messages(user_input), config=config):
            piece = getattr(chunk, "content", "") or ""
            yield piece

        self.turn_count += 1

        # Bug fix #2: strip any SystemMessage(s) that just got persisted
        # into history by the call above.
        history = self._history_store[SESSION_ID]
        history.messages = [m for m in history.messages if not isinstance(m, SystemMessage)]

    def get_history_messages(self) -> list:
        return self._history_store[SESSION_ID].messages


def render_welcome(session: AssistantSession) -> None:
    console.print(
        Panel(
            f"[bold]{session.topic['label']} Assistant[/bold]\n"
            f"{session.topic['welcome']}\n\n"
            f"[dim]/topic [cooking|history|programming] · /history · /tokens "
            f"· /save · /clear · /help · /exit[/dim]",
            border_style="cyan",
        )
    )


def render_help() -> None:
    table = Table(title="Commands", show_header=True, header_style="bold cyan")
    table.add_column("Command")
    table.add_column("Description")
    table.add_row("/topic <name>", "Switch topic domain (cooking, history, programming)")
    table.add_row("/history", "Show the full conversation so far")
    table.add_row("/tokens", "Show total tokens used this session")
    table.add_row("/save", "Save the conversation to a markdown file")
    table.add_row("/clear", "Clear conversation history (keeps current topic)")
    table.add_row("/help", "Show this table")
    table.add_row("/exit", "Quit")
    console.print(table)


def render_history(session: AssistantSession) -> None:
    messages = session.get_history_messages()
    if not messages:
        console.print("[dim]No conversation yet.[/dim]")
        return
    for msg in messages:
        if isinstance(msg, HumanMessage):
            console.print(Panel(msg.content, title="You", border_style="blue", title_align="left"))
        elif isinstance(msg, AIMessage):
            console.print(Panel(Markdown(msg.content), title="Assistant", border_style="green", title_align="left"))


def save_transcript(session: AssistantSession) -> Path:
    messages = session.get_history_messages()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(f"transcript_{timestamp}.md")

    lines = [f"# Conversation transcript — {session.topic['label']}", ""]
    for msg in messages:
        if isinstance(msg, HumanMessage):
            lines.append(f"**You:** {msg.content}\n")
        elif isinstance(msg, AIMessage):
            lines.append(f"**Assistant:** {msg.content}\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def handle_command(command: str, session: AssistantSession) -> bool:
    """Returns False if the session should end, True otherwise."""
    parts = command.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip().lower() if len(parts) > 1 else ""

    if cmd in ("/exit", "/quit"):
        console.print("[dim]Goodbye.[/dim]")
        return False

    if cmd == "/help":
        render_help()

    elif cmd == "/topic":
        if arg in TOPICS:
            session.switch_topic(arg)
            console.print(f"[bold cyan]Switched to {TOPICS[arg]['label']}.[/bold cyan]")
            console.print(f"[dim]{TOPICS[arg]['welcome']}[/dim]")
        else:
            console.print(
                f"[red]Unknown topic '{arg}'.[/red] Available: {', '.join(TOPICS.keys())}"
            )

    elif cmd == "/history":
        render_history(session)

    elif cmd == "/tokens":
        console.print(
            "[dim]Per-turn token counts aren't available in streaming mode — "
            "Groq's streaming responses don't expose usage stats through "
            "the standard field langchain-groq reads (verified during "
            "development; this is a provider-side limitation, not a bug "
            "here). Turns so far this session: "
            f"{session.turn_count}[/dim]"
        )

    elif cmd == "/save":
        path = save_transcript(session)
        console.print(f"[green]Saved transcript to {path}[/green]")

    elif cmd == "/clear":
        session._history_store[SESSION_ID] = InMemoryChatMessageHistory()
        console.print("[dim]Conversation history cleared.[/dim]")

    else:
        console.print(f"[red]Unknown command: {cmd}[/red] (try /help)")

    return True


def choose_starting_topic() -> str:
    console.print(Panel("[bold]Intelligent CLI Assistant[/bold]", border_style="cyan"))
    topic = Prompt.ask(
        "Choose a topic to start",
        choices=list(TOPICS.keys()),
        default="programming",
    )
    return topic


def main() -> None:
    llm = get_llm()
    topic_key = choose_starting_topic()
    session = AssistantSession(llm, topic_key)
    render_welcome(session)

    while True:
        try:
            user_input = Prompt.ask(f"[bold blue]you[/bold blue]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input.strip():
            continue

        if user_input.startswith("/"):
            should_continue = handle_command(user_input, session)
            if not should_continue:
                break
            continue

        # Stream the response with a live-updating panel, rendered as
        # markdown as it grows — this is the extra streaming feature
        # beyond spec; verified .stream() works against the real
        # installed ChatGroq before writing this loop.
        accumulated = ""
        with Live(console=console, refresh_per_second=12) as live:
            for piece in session.ask_streaming(user_input):
                accumulated += piece
                live.update(
                    Panel(Markdown(accumulated or "..."), title="Assistant", border_style="green", title_align="left")
                )


if __name__ == "__main__":
    main()