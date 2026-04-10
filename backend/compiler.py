"""
Secure CadQuery Script Execution Engine.

Executes AI-generated CadQuery Python scripts in an isolated subprocess
with strict timeout enforcement. This is the runtime behind the QA Inspector
(Critic Loop) — failed executions return tracebacks that get fed back to the
Machinist subagent for correction.
"""

import ast
import asyncio
import logging
import os
import pathlib
import signal
import subprocess
import sys
import tempfile

try:
    import resource

    _HAS_RESOURCE = True
except ImportError:          # Windows — no resource module
    _HAS_RESOURCE = False
    logging.getLogger("mirum.compiler").warning(
        "Running on Windows: subprocess memory limits are NOT enforced. "
        "Production deployments MUST use Linux/Docker."
    )

# Directory containing cad_utils.py and other backend modules.
# Added to PYTHONPATH so subprocess-executed scripts can import them.
_BACKEND_DIR = str(pathlib.Path(__file__).parent.resolve())

# ---------------------------------------------------------------------------
# AST Security Scanner (Allowlist Architecture)
# ---------------------------------------------------------------------------
_ALLOWED_MODULES = {"cadquery", "cq", "math", "cad_utils"}
_BANNED_BUILTINS = {
    "eval", "exec", "open", "compile", "__import__", "getattr", "setattr",
    "delattr", "globals", "locals", "vars", "breakpoint", "input",
    "memoryview", "type",
}
_BANNED_DUNDERS = {
    "__import__", "__subclasses__", "__globals__", "__builtins__",
    "__loader__", "__spec__", "__code__",
}
_MEM_LIMIT_BYTES = int(1.5 * 1024 * 1024 * 1024)  # 1.5 GB


def _check_ast_security(script_string: str) -> str | None:
    """Return an error message if the script imports or calls banned targets.

    Uses an ALLOWLIST for imports: only modules in _ALLOWED_MODULES may be
    imported.  All other modules (os, sys, subprocess, importlib, ctypes,
    pickle, socket, shutil, pathlib, etc.) are rejected automatically.
    """
    try:
        tree = ast.parse(script_string)
    except SyntaxError as exc:
        return f"SyntaxError in script: {exc}"

    for node in ast.walk(tree):
        # --- Import allowlist ---
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in _ALLOWED_MODULES:
                    return (
                        f"Security violation: import of '{alias.name}' is not "
                        f"allowed. Permitted modules: {sorted(_ALLOWED_MODULES)}."
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top not in _ALLOWED_MODULES:
                    return (
                        f"Security violation: import from '{node.module}' is "
                        f"not allowed. Permitted modules: {sorted(_ALLOWED_MODULES)}."
                    )

        # --- Banned builtin calls ---
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _BANNED_BUILTINS:
                return (
                    f"Security violation: call to '{func.id}()' is banned."
                )

        # --- Dangerous dunder attribute access ---
        if isinstance(node, ast.Attribute) and node.attr in _BANNED_DUNDERS:
            return (
                f"Security violation: access to '{node.attr}' is banned."
            )

    return None


def _preexec_sandbox() -> None:
    """preexec_fn — new process group + memory cap (POSIX only)."""
    os.setpgrp()
    resource.setrlimit(resource.RLIMIT_AS, (_MEM_LIMIT_BYTES, _MEM_LIMIT_BYTES))


async def execute_cad_script(script_string: str) -> dict:
    """Execute a CadQuery Python script in a sandboxed subprocess.

    The script is written to a temporary file and run as a separate process
    to isolate the main application from crashes, infinite loops, or unsafe
    operations in AI-generated code.

    Args:
        script_string: Complete, self-contained Python/CadQuery source code.

    Returns:
        On success: {"status": "success", "output": <stdout>}
        On failure: {"status": "error", "traceback": <stderr>}
    """
    # --- AST security gate (always enforced) ---
    violation = _check_ast_security(script_string)
    if violation:
        return {"status": "error", "traceback": violation}

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
        ) as tmp:
            tmp.write(script_string)
            tmp_path = tmp.name

        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": _BACKEND_DIR,
            "HOME": os.environ.get("HOME", os.environ.get("USERPROFILE", "/tmp")),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),  # Required on Windows
        }

        popen_kwargs: dict = dict(
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        if _HAS_RESOURCE:
            popen_kwargs["preexec_fn"] = _preexec_sandbox
        else:
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        def _run() -> tuple[int, str, str]:
            proc = subprocess.Popen([sys.executable, tmp_path], **popen_kwargs)
            try:
                stdout, stderr = proc.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                # Kill the entire process group, not just the parent
                if _HAS_RESOURCE:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except OSError:
                        proc.kill()
                else:
                    proc.kill()
                proc.wait()
                raise
            return proc.returncode, stdout, stderr

        returncode, stdout, stderr = await asyncio.to_thread(_run)

        if returncode != 0:
            return {"status": "error", "traceback": stderr}

        return {"status": "success", "output": stdout}

    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "traceback": "TIMEOUT: Script exceeded 30-second execution limit.",
        }
    except Exception as e:
        return {
            "status": "error",
            "traceback": f"Execution error: {e}",
        }

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
