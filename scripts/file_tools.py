# file_tools.py

import os
import hashlib
from typing import Annotated
from typing_extensions import NotRequired

from langchain.agents import AgentState
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from langgraph.types import Command


# -------------------------
# Shared Agent State
# -------------------------

class DeepAgentState(AgentState):
    """
    Shared state for all agents (orchestrator, researcher, editor).

    Inherits from LangChain's AgentState and adds:
    - user_id:    separate users
    - thread_id:  separate conversations per user

    Files are stored on REAL disk, not in state.
    """
    user_id: NotRequired[str]
    thread_id: NotRequired[str]


BASE_FILE_DIR = os.getenv("AGENT_FILE_BASE_DIR", "agent_files")


def _thread_folder(state: DeepAgentState) -> str:
    """Return the folder for this user/thread, create if missing."""
    user = state.get("user_id") or "default_user"
    thread = state.get("thread_id") or "default_thread"
    folder = os.path.join(BASE_FILE_DIR, user, thread)
    os.makedirs(folder, exist_ok=True)
    return folder


def _researcher_folder(state: DeepAgentState) -> str:
    """Return the researcher subfolder for this user/thread, create if missing."""
    thread_folder = _thread_folder(state)
    researcher_folder = os.path.join(thread_folder, "researcher")
    os.makedirs(researcher_folder, exist_ok=True)
    return researcher_folder


def generate_hash(text: str, length: int = 6) -> str:
    """Generate a short hash from text for unique file naming."""
    return hashlib.md5(text.encode()).hexdigest()[:length]


def _disk_path(state: DeepAgentState, file_path: str) -> str:
    """
    Build real filesystem path as:
        agent_files/<user_id>/<thread_id>/<file_path>
    """
    folder = _thread_folder(state)
    safe_path = file_path.lstrip("/\\")
    full = os.path.join(folder, safe_path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    return full


# -------------------------
# Tools
# -------------------------

@tool(parse_docstring=True)
def ls(
    state: Annotated[DeepAgentState, InjectedState],
    path: str = ""
) -> list[str]:
    """
    List available files for this user/thread on the real filesystem.

    Args:
        state: Injected agent state providing user_id/thread_id.
        path: Optional subdirectory path (e.g., "researcher").
              If empty, lists root thread folder.

    Returns:
        A sorted list of filenames in the specified folder.
    """
    folder = _thread_folder(state)
    if path:
        # Join with the subdirectory path, removing leading slashes
        folder = os.path.join(folder, path.lstrip("/\\"))
    if not os.path.exists(folder):
        return []
    return sorted(os.listdir(folder))


@tool(parse_docstring=True)
def read_file(
    file_path: str,
    state: Annotated[DeepAgentState, InjectedState],
    offset: int = 0,
    limit: int = 2000,
) -> str:
    """
    Read a file from the real filesystem.

    Args:
        file_path: Relative file path under this user/thread folder.
        state: Injected agent state providing user_id/thread_id.
        offset: Line number to start from (0-based).
        limit: Maximum number of lines to return.

    Returns:
        File content with line numbers, or an error message.
    """
    path = _disk_path(state, file_path)

    if not os.path.exists(path):
        return f"Error: File '{file_path}' does not exist."

    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    end = min(offset + limit, len(lines))
    numbered = [
        f"{i + 1:5d}  {line}"
        for i, line in enumerate(lines[offset:end])
    ]
    return "\n".join(numbered)


@tool(parse_docstring=True)
def write_file(
    file_path: str,
    content: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Write content to a file on the real filesystem.

    Args:
        file_path: Relative path (e.g., "plan.md", "notes/sources.txt").
        content: Text content to write (overwrites existing file).
        state: Injected agent state providing user_id/thread_id.
        tool_call_id: Tool call ID used to attach a ToolMessage.

    Side effects:
        - Overwrites the file on disk for this user/thread.

    Returns:
        Command that adds a ToolMessage confirming the write.
    """
    path = _disk_path(state, file_path)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    msg = f"[FILE WRITTEN] {file_path} -> {path}"
    return Command(
        update={
            "messages": [ToolMessage(msg, tool_call_id=tool_call_id)]
        }
    )


@tool(parse_docstring=True)
def cleanup_files(
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Delete ALL files for this user/thread from the real filesystem.

    IMPORTANT:
    - This is a destructive operation.
    - Use ONLY when the human user explicitly asks to wipe, reset,
      or clean the workspace for this conversation.

    Args:
        state: Injected agent state with user_id/thread_id.
        tool_call_id: Tool call ID for attaching a ToolMessage.

    Behavior:
        - Looks in: agent_files/<user>/<thread>/
        - Deletes all regular files in that folder (does not recurse).

    Returns:
        Command with a ToolMessage summarizing what was deleted.
    """
    folder = _thread_folder(state)

    if not os.path.exists(folder):
        msg = "[CLEANUP] No folder found, nothing to delete."
        return Command(update={"messages": [ToolMessage(msg, tool_call_id=tool_call_id)]})

    deleted = []
    for name in os.listdir(folder):
        full = os.path.join(folder, name)
        if os.path.isfile(full):
            try:
                os.remove(full)
                deleted.append(name)
            except Exception as e:
                deleted.append(f"{name} (error: {e})")

    if not deleted:
        msg = "[CLEANUP] No files to delete."
    else:
        msg = f"[CLEANUP] Deleted files: {deleted}"

    return Command(update={"messages": [ToolMessage(msg, tool_call_id=tool_call_id)]})
