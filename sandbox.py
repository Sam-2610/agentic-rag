"""Run LLM-written Python in a restricted subprocess.

Protection layers (best effort, NOT a perfect security boundary):
  1. AST check: only allow-listed imports, no open/eval/exec, no file-reading or dunder tricks.
  2. Separate Python process in isolated mode (-I), in a temporary folder.
  3. No API keys or other environment secrets are passed to the process.
  4. Hard timeout and output size limit.
Run the app in Docker for an extra layer, and do not expose it publicly without more hardening.
"""
import ast
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from config import ARTIFACT_DIR, MPL_CACHE_DIR, SANDBOX_TIMEOUT

ALLOWED_IMPORTS = {
    "pandas", "numpy", "matplotlib", "math", "statistics", "json", "re", "datetime",
    "collections", "itertools", "io", "csv", "decimal", "fractions",
}
BANNED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "input", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "breakpoint", "exit", "quit",
}
BANNED_ATTRS = {
    "system", "popen", "spawn", "environ", "remove", "rmdir", "unlink", "rename", "mkdir",
    "makedirs", "listdir", "walk", "scandir", "open", "load", "loadtxt", "genfromtxt",
    "fromfile", "tofile", "memmap",
}
MAX_OUTPUT = 4000


def validate(code: str):
    """Return an error message if the code breaks the rules, else None."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e.msg} (line {e.lineno})"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    return f"Import not allowed: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                return f"Import not allowed: {node.module}"
        elif isinstance(node, ast.Name):
            if node.id in BANNED_NAMES or (node.id.startswith("__") and node.id.endswith("__")):
                return f"Use of '{node.id}' is not allowed"
        elif isinstance(node, ast.Attribute):
            if (
                node.attr in BANNED_ATTRS
                or node.attr.startswith("read_")
                or (node.attr.startswith("__") and node.attr.endswith("__"))
            ):
                return f"Use of '.{node.attr}' is not allowed (parse CONTEXT with plain Python instead)"
    return None


def run(code: str, context: str, timeout: int = SANDBOX_TIMEOUT) -> dict:
    """Execute `code` with the variable CONTEXT defined. Returns ok/stdout/error/artifacts."""
    problem = validate(code)
    if problem:
        return {"ok": False, "stdout": "", "error": problem, "artifacts": []}

    with tempfile.TemporaryDirectory(prefix="sandbox_") as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "context.txt").write_text(context, encoding="utf-8")
        prelude = (
            "import matplotlib\nmatplotlib.use('Agg')\n"
            "CONTEXT = open('context.txt', encoding='utf-8').read()\n"
        )
        (tmp_path / "script.py").write_text(prelude + code, encoding="utf-8")

        env = {
            "PATH": os.environ.get("PATH", ""),
            "MPLCONFIGDIR": str(MPL_CACHE_DIR),
            "TEMP": tmp, "TMP": tmp, "TMPDIR": tmp, "HOME": tmp, "USERPROFILE": tmp,
            "PYTHONIOENCODING": "utf-8",
        }
        if os.name == "nt":
            env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")

        try:
            proc = subprocess.run(
                [sys.executable, "-I", "script.py"],
                cwd=tmp, env=env, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "stdout": "", "error": f"Timed out after {timeout}s", "artifacts": []}

        stdout = proc.stdout[:MAX_OUTPUT]
        if proc.returncode != 0:
            tail = "\n".join(proc.stderr.strip().splitlines()[-6:])
            return {"ok": False, "stdout": stdout, "error": tail[-1500:], "artifacts": []}

        artifacts = []
        for f in tmp_path.iterdir():
            if f.suffix.lower() in {".png", ".csv"}:
                name = f"{uuid.uuid4().hex[:8]}_{f.name}"
                shutil.copy(f, ARTIFACT_DIR / name)
                artifacts.append(name)
        if not stdout.strip() and not artifacts:
            return {"ok": False, "stdout": "", "error": "The script printed nothing. Use print() to show results.", "artifacts": []}
        return {"ok": True, "stdout": stdout, "error": "", "artifacts": artifacts}
