#!/usr/bin/env python3
"""List, control, start, and warm notebook kernels.

This script supports two modes:
1. Jupyter Server managed sessions via the server API.
2. Raw kernels started directly by tools such as the VS Code Jupyter extension.

Examples:
  ./restart_jupyter_kernel.py --list
  ./restart_jupyter_kernel.py --start
  ./restart_jupyter_kernel.py --start --warm
  ./restart_jupyter_kernel.py --hot-reload-help
  ./restart_jupyter_kernel.py --interrupt
  ./restart_jupyter_kernel.py --interrupt --pid 3690
  ./restart_jupyter_kernel.py --kernel-id <kernel-id>
  ./restart_jupyter_kernel.py --session-id <session-id>

Hot reload principle:
  Keep the warmed raw kernel alive, preserve the in-memory model objects
  (`asr`, `mt_backend`; plus `gemma_llm` / `gemma_tokenizer` only as legacy convenience handles), reload only the Python modules with
  `importlib.reload(...)`, rebind the preserved model objects into the reloaded
  `qwen3asr_gemma_cascade_core` module, reset only `core.state`, then call
  `qwen3asr_gemma_cascade_core.run_baseline(...)` directly. This updates the
  code path without paying the model reload cost again.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable
from urllib.parse import urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen


KERNEL_COMMAND_PATTERN = re.compile(
    r"ipykernel_launcher|jupyter-kernel|python[^ ]* .*ipykernel"
)
RUNTIME_PATH_PATTERN = re.compile(r"(?:--f|-f)(?:=|\s+)(\S*kernel-[^\s]+\.json)")
DEFAULT_TMUX_SESSION = "cascade-inference-kernel"
DEFAULT_RUNTIME_FILE = "kernel-cascade-simultaneous.json"
LOAD_MODELS_BEGIN_MARKER = "LOAD_MODELS_BEGIN"
LOAD_MODELS_DONE_MARKER = "LOAD_MODELS_DONE"
HOT_RELOAD_GUIDE = """
Hot reload principle
--------------------
1. Keep the persistent raw `.venv-inference` kernel alive and warmed.
2. Save `qwen3asr_gemma_cascade_core.asr` and `mt_backend`.
3. Optionally keep `gemma_llm` and `gemma_tokenizer` in sync as legacy convenience handles.
4. Reload `cascade_translation_variants`, `cascade_mt_backend`, and `qwen3asr_gemma_cascade_core`
   with `importlib.reload(...)`.
5. Rebind the saved model objects into the reloaded core module.
6. Reset only `core.state`.
7. Call `qwen3asr_gemma_cascade_core.run_baseline(...)` directly.

Why call the core module directly?
  The notebook facade imports function references eagerly. After a hot reload,
  running the reloaded `qwen3asr_gemma_cascade_core` module directly avoids
  accidentally using stale function objects that were imported before reload.
""".strip()
LOAD_MODELS_CODE = (
    "from qwen3asr_gemma_cascade_notebook import load_models\n"
    f"print({LOAD_MODELS_BEGIN_MARKER!r})\n"
    "load_models()\n"
    f"print({LOAD_MODELS_DONE_MARKER!r})\n"
)


@dataclass(frozen=True)
class ServerInfo:
    base_url: str
    port: int
    secure: bool
    token: str

    @property
    def local_base(self) -> str:
        scheme = "https" if self.secure else "http"
        base_path = self.base_url if self.base_url.endswith("/") else f"{self.base_url}/"
        return urlunparse((scheme, f"127.0.0.1:{self.port}", base_path, "", "", ""))


@dataclass(frozen=True)
class KernelTarget:
    kind: str
    display_name: str
    session_id: str | None = None
    kernel_id: str | None = None
    kernel_name: str | None = None
    notebook_path: str | None = None
    pid: int | None = None
    runtime_file: str | None = None
    server: ServerInfo | None = None


def _run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=True, capture_output=True, text=True)


def _run_jupyter_server_list() -> list[dict]:
    try:
        result = _run_command(["jupyter", "server", "list", "--jsonlist"])
    except FileNotFoundError:
        return []
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() or exc.stdout.strip()
        raise RuntimeError(f"`jupyter server list` failed: {stderr or 'unknown error'}") from exc

    stdout = result.stdout.strip()
    if not stdout:
        return []
    return json.loads(stdout)


def _get_json(url: str) -> object:
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request) as response:
        return json.load(response)


def _post_json(url: str) -> object:
    request = Request(
        url,
        data=b"",
        headers={"Accept": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:
        body = response.read()
        if not body:
            return None
        return json.loads(body)


def _normalize_path(path: str) -> PurePosixPath:
    return PurePosixPath(path.replace("\\", "/"))


def _matches_path(actual: str, wanted: str) -> bool:
    actual_norm = _normalize_path(actual)
    wanted_norm = _normalize_path(wanted)
    return actual_norm == wanted_norm or actual_norm.as_posix().endswith(wanted_norm.as_posix())


def discover_servers() -> list[ServerInfo]:
    servers: list[ServerInfo] = []
    for raw in _run_jupyter_server_list():
        servers.append(
            ServerInfo(
                base_url=raw.get("base_url", "/"),
                port=int(raw["port"]),
                secure=bool(raw.get("secure", False)),
                token=raw.get("token", ""),
            )
        )
    return servers


def _server_api_url(server: ServerInfo, path: str) -> str:
    query = urlencode({"token": server.token}) if server.token else ""
    parsed = urlparse(server.local_base)
    api_path = parsed.path.rstrip("/") + path
    return urlunparse(parsed._replace(path=api_path, query=query))


def discover_server_targets(servers: Iterable[ServerInfo]) -> list[KernelTarget]:
    targets: list[KernelTarget] = []
    for server in servers:
        raw_sessions = _get_json(_server_api_url(server, "/api/sessions"))
        if not isinstance(raw_sessions, list):
            raise RuntimeError("Unexpected response from Jupyter sessions API.")
        for raw in raw_sessions:
            kernel = raw.get("kernel") or {}
            notebook_path = raw.get("path", "")
            targets.append(
                KernelTarget(
                    kind="server",
                    display_name=notebook_path or kernel.get("id", "(unknown session)"),
                    session_id=raw["id"],
                    kernel_id=kernel["id"],
                    kernel_name=kernel.get("name", ""),
                    notebook_path=notebook_path,
                    server=server,
                )
            )
    return targets


def _runtime_dir() -> Path:
    env_dir = os.environ.get("JUPYTER_RUNTIME_DIR")
    if env_dir:
        return Path(env_dir)

    try:
        result = _run_command(["jupyter", "--runtime-dir"])
        runtime = result.stdout.strip()
        if runtime:
            return Path(runtime)
    except Exception:
        pass

    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "jupyter" / "runtime"


def _list_kernel_processes() -> list[tuple[int, str]]:
    result = _run_command(["ps", "-eo", "pid=,command="])
    processes: list[tuple[int, str]] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, command = stripped.partition(" ")
        if not pid_text.isdigit():
            continue
        if KERNEL_COMMAND_PATTERN.search(command):
            processes.append((int(pid_text), command.strip()))
    return processes


def discover_raw_targets() -> list[KernelTarget]:
    targets: list[KernelTarget] = []
    runtime_dir = _runtime_dir()
    for pid, command in _list_kernel_processes():
        match = RUNTIME_PATH_PATTERN.search(command)
        runtime_path = match.group(1) if match else ""
        runtime_file = Path(runtime_path).name if runtime_path else None
        if runtime_path and not Path(runtime_path).exists() and runtime_file:
            candidate = runtime_dir / runtime_file
            if candidate.exists():
                runtime_path = str(candidate)
        targets.append(
            KernelTarget(
                kind="raw",
                display_name=runtime_file or f"pid-{pid}",
                pid=pid,
                runtime_file=runtime_file or runtime_path or None,
                kernel_name="ipykernel",
            )
        )
    return targets


def discover_targets() -> list[KernelTarget]:
    server_targets = discover_server_targets(discover_servers())
    if server_targets:
        return server_targets
    return discover_raw_targets()


def print_targets(targets: list[KernelTarget]) -> None:
    if not targets:
        print("No active notebook kernels found.")
        print(
            "Tip: start a persistent .venv-inference kernel with "
            "`./restart_jupyter_kernel.py --start --warm`."
        )
        return

    print(f"Active notebook kernels: {len(targets)}")
    for target in targets:
        if target.kind == "server":
            print(f"- kind: server")
            print(f"  notebook: {target.notebook_path or '(unknown path)'}")
            print(f"  session_id: {target.session_id}")
            print(f"  kernel_id: {target.kernel_id}")
            print(f"  kernel_name: {target.kernel_name or '(unknown kernel)'}")
            print(f"  server: {target.server.local_base if target.server else '(unknown server)'}")
        else:
            print(f"- kind: raw")
            print(f"  pid: {target.pid}")
            print(f"  runtime: {target.runtime_file or '(unknown runtime)'}")
            print(f"  command: ipykernel")


def select_targets(
    targets: list[KernelTarget],
    *,
    session_id: str | None,
    kernel_id: str | None,
    notebook_path: str | None,
    pid: int | None,
    all_targets: bool,
) -> list[KernelTarget]:
    if all_targets:
        return targets

    selected = targets
    if session_id:
        selected = [target for target in selected if target.session_id == session_id]
    if kernel_id:
        selected = [target for target in selected if target.kernel_id == kernel_id]
    if notebook_path:
        selected = [
            target
            for target in selected
            if target.notebook_path and _matches_path(target.notebook_path, notebook_path)
        ]
    if pid is not None:
        selected = [target for target in selected if target.pid == pid]

    if session_id or kernel_id or notebook_path or pid is not None:
        if notebook_path and len(selected) > 1:
            raise RuntimeError(
                "More than one target matched that path. Use --list and select one by id."
            )
        return selected

    if not targets:
        return []

    if len(targets) == 1:
        return targets

    raise RuntimeError(
        "Multiple active kernels found. Use --list, then select one with "
        "--path, --session-id, --kernel-id, or --pid."
    )


def control_targets(targets: list[KernelTarget], *, action: str) -> int:
    if not targets:
        print("No matching notebook kernels found.", file=sys.stderr)
        return 1

    for target in targets:
        if target.kind == "server":
            if target.server is None or target.kernel_id is None:
                raise RuntimeError("Incomplete server target information.")
            _post_json(_server_api_url(target.server, f"/api/kernels/{target.kernel_id}/{action}"))
            verb = "Interrupted" if action == "interrupt" else "Restarted"
            print(f"{verb} kernel {target.kernel_id} for {target.notebook_path or '(unknown path)'}")
            continue

        if target.pid is None:
            raise RuntimeError("Incomplete raw kernel information.")
        if action == "restart":
            raise RuntimeError(
                "Restart is not supported for raw kernels from this script. "
                "Use --interrupt to stop the current cell without losing memory."
            )
        os.kill(target.pid, signal.SIGINT)
        print(f"Interrupted raw kernel PID {target.pid} ({target.runtime_file or 'unknown runtime'})")

    return 0


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _runtime_path(runtime_file: str) -> Path:
    path = Path(runtime_file)
    if path.is_absolute():
        return path
    return _runtime_dir() / path.name


def _pid_is_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _find_raw_target_by_runtime_file(runtime_file: str) -> KernelTarget | None:
    runtime_name = Path(runtime_file).name
    for target in discover_raw_targets():
        if (target.runtime_file or "") == runtime_name:
            return target
    return None


def _tmux_has_session(session_name: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def start_persistent_raw_kernel(
    *,
    session_name: str,
    runtime_file: str,
) -> KernelTarget:
    if shutil.which("tmux") is None:
        raise RuntimeError("`tmux` is required to start a persistent raw kernel.")

    repo_root = _repo_root()
    python_path = repo_root / ".venv-inference" / "bin" / "python"
    if not python_path.exists():
        raise RuntimeError(
            f"Expected inference Python at {python_path}, but it does not exist."
        )

    runtime_path = _runtime_path(runtime_file)
    runtime_path.parent.mkdir(parents=True, exist_ok=True)

    existing = _find_raw_target_by_runtime_file(runtime_path.name)
    if existing and _pid_is_alive(existing.pid):
        print(
            f"Persistent raw kernel already running: PID {existing.pid} "
            f"({existing.runtime_file or runtime_path.name})"
        )
        return existing

    if _tmux_has_session(session_name):
        subprocess.run(
            ["tmux", "kill-session", "-t", session_name],
            check=False,
            capture_output=True,
            text=True,
        )

    if runtime_path.exists():
        runtime_path.unlink()

    subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            session_name,
            "-c",
            str(repo_root),
            str(python_path),
            "-m",
            "ipykernel_launcher",
            "-f",
            str(runtime_path),
        ],
        check=True,
    )

    deadline = time.time() + 30.0
    while time.time() < deadline:
        target = _find_raw_target_by_runtime_file(runtime_path.name)
        if target and _pid_is_alive(target.pid):
            print(
                f"Started persistent raw kernel: PID {target.pid} "
                f"({target.runtime_file or runtime_path.name})"
            )
            return target
        time.sleep(0.1)

    raise RuntimeError(
        f"Started tmux session {session_name!r}, but no raw kernel became discoverable."
    )


def warm_raw_kernel(target: KernelTarget) -> int:
    if target.kind != "raw":
        raise RuntimeError("Warm-up is only supported for raw kernels.")
    if not target.runtime_file:
        raise RuntimeError("The raw kernel does not expose a runtime file.")
    if not _pid_is_alive(target.pid):
        raise RuntimeError("The selected raw kernel is no longer running.")

    runtime_path = _runtime_path(target.runtime_file)
    if not runtime_path.exists():
        raise RuntimeError(f"Missing runtime file: {runtime_path}")

    from jupyter_client import BlockingKernelClient

    client = BlockingKernelClient(connection_file=str(runtime_path))
    client.load_connection_file()
    client.start_channels()
    try:
        client.wait_for_ready(timeout=120)
        print(f"Connected to raw kernel PID {target.pid} via {runtime_path.name}")
        msg_id = client.execute(LOAD_MODELS_CODE, store_history=False)
        saw_done = False
        start = time.time()
        last_wait_notice = start

        while True:
            try:
                message = client.get_iopub_msg(timeout=5)
            except queue.Empty:
                now = time.time()
                if now - last_wait_notice >= 30:
                    print(f"Still warming kernel; elapsed={int(now - start)}s")
                    last_wait_notice = now
                continue

            if message.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            msg_type = message["header"]["msg_type"]
            content = message["content"]
            if msg_type == "stream":
                text = content.get("text", "")
                if text:
                    print(text, end="")
                    if LOAD_MODELS_DONE_MARKER in text:
                        saw_done = True
            elif msg_type == "error":
                traceback = "\n".join(content.get("traceback", []))
                raise RuntimeError(f"Kernel warm-up failed:\n{traceback}")
            elif (
                msg_type == "status"
                and content.get("execution_state") == "idle"
                and saw_done
            ):
                print("Kernel warm-up finished.")
                return 0
    finally:
        client.stop_channels()

    raise RuntimeError("Kernel warm-up ended unexpectedly.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List, start, warm, interrupt, or restart active notebook kernels."
    )
    parser.add_argument(
        "--hot-reload-help",
        action="store_true",
        help="Print the recommended code-only hot reload workflow for the warmed raw kernel.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List active kernels instead of controlling them.",
    )
    parser.add_argument(
        "--start",
        action="store_true",
        help=(
            "Start a persistent raw .venv-inference ipykernel inside tmux. "
            "If it already exists, reuse it."
        ),
    )
    parser.add_argument(
        "--warm",
        action="store_true",
        help=(
            "Run `load_models()` inside the selected raw kernel. "
            "Useful with --start to pre-load ASR and Gemma once."
        ),
    )
    parser.add_argument(
        "--interrupt",
        action="store_true",
        help="Interrupt matching kernel(s) instead of restarting them.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Control every discovered kernel.",
    )
    parser.add_argument(
        "--path",
        help="Control the server-backed session whose notebook path matches this value.",
    )
    parser.add_argument("--session-id", help="Control a server-backed session by session id.")
    parser.add_argument("--kernel-id", help="Control a server-backed session by kernel id.")
    parser.add_argument("--pid", type=int, help="Control a raw kernel by process id.")
    parser.add_argument(
        "--tmux-session",
        default=DEFAULT_TMUX_SESSION,
        help=f"tmux session name used by --start (default: {DEFAULT_TMUX_SESSION}).",
    )
    parser.add_argument(
        "--runtime-file",
        default=DEFAULT_RUNTIME_FILE,
        help=(
            "Runtime filename used by --start or when resolving a raw kernel "
            f"(default: {DEFAULT_RUNTIME_FILE})."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.hot_reload_help:
            print(HOT_RELOAD_GUIDE)
            return 0

        if args.start:
            target = start_persistent_raw_kernel(
                session_name=args.tmux_session,
                runtime_file=args.runtime_file,
            )
            if args.warm:
                return warm_raw_kernel(target)
            return 0

        targets = discover_targets()
        if args.list:
            print_targets(targets)
            return 0

        if args.warm:
            selected = select_targets(
                targets,
                session_id=args.session_id,
                kernel_id=args.kernel_id,
                notebook_path=args.path,
                pid=args.pid,
                all_targets=args.all,
            )
            if len(selected) != 1:
                raise RuntimeError(
                    "Warm-up requires exactly one selected kernel. Use --list first, "
                    "then select a raw kernel with --pid, or use --start --warm."
                )
            return warm_raw_kernel(selected[0])

        action = "interrupt" if args.interrupt else "restart"
        selected = select_targets(
            targets,
            session_id=args.session_id,
            kernel_id=args.kernel_id,
            notebook_path=args.path,
            pid=args.pid,
            all_targets=args.all,
        )
        return control_targets(selected, action=action)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
