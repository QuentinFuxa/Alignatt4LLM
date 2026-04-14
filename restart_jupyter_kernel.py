#!/usr/bin/env python3
"""List and control active notebook kernels.

This script supports two modes:
1. Jupyter Server managed sessions via the server API.
2. Raw kernels started directly by tools such as the VS Code Jupyter extension.

Examples:
  ./restart_jupyter_kernel.py --list
  ./restart_jupyter_kernel.py --interrupt
  ./restart_jupyter_kernel.py --interrupt --pid 3690
  ./restart_jupyter_kernel.py --kernel-id <kernel-id>
  ./restart_jupyter_kernel.py --session-id <session-id>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable
from urllib.parse import urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen


KERNEL_COMMAND_PATTERN = re.compile(
    r"ipykernel_launcher|jupyter-kernel|python[^ ]* .*ipykernel"
)
RUNTIME_PATH_PATTERN = re.compile(r"--f(?:=|\s+)(\S*kernel-[^\s]+\.json)")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List, interrupt, or restart active notebook kernels."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List active kernels instead of controlling them.",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        targets = discover_targets()
        if args.list:
            print_targets(targets)
            return 0

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
