"""Python code checks: static analysis (course constraints, security, best practice) and a test runner.

  constraints(code)   for student submissions and generated reference solutions: imports outside ALLOWED_IMPORTS,
                      dangerous builtins and dunder escapes are BLOCKED (the code is not run)
  review(code)        for code in the tutor's chat answers: never blocks, warns about security and style problems
  run_tests(code, tests)  runs code + assert lines in a separate Python process with a time limit

The runner is a guard, not a sandbox: it only ever runs code that passed constraints(), in an isolated interpreter
(-I) inside a temporary folder, with an audit hook that refuses file writes, processes and network, and a timeout.
That is enough for course exercises on your own PC; it is not safe for untrusted code from the internet.
"""
import ast
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ALLOWED_IMPORTS = {"math", "re", "collections", "itertools", "functools", "string", "heapq", "bisect", "statistics",
                   "random", "typing", "dataclasses", "unicodedata", "operator", "copy", "json", "numpy", "fractions",
                   "decimal", "array", "enum", "abc"}
BLOCKED_CALLS = {"eval", "exec", "compile", "__import__", "open", "input", "breakpoint", "globals", "locals", "vars",
                 "exit", "quit", "help", "memoryview"}
BLOCKED_ATTRS = re.compile(r"^__(class|bases|subclasses|globals|builtins|code|closure|dict|mro|loader|import__)__$")
SHADOWED = {"list", "dict", "str", "set", "tuple", "id", "input", "sum", "max", "min", "len", "type", "map",
            "filter", "iter", "next", "range", "sorted", "object", "format", "hash", "all", "any"}
SECRET = re.compile(r"(?i)(password|passwd|secret|api[_-]?key|token)\s*=\s*['\"][^'\"]{4,}['\"]")


def _calls(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
            full = None
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
                full = f"{f.value.id}.{f.attr}"
            yield node, name, full


def _style(tree, code):
    """Best-practice warnings shared by both checks."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            out.append((node.lineno, "bare `except:` also catches KeyboardInterrupt and hides bugs; catch a specific exception"))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in node.args.defaults + node.args.kw_defaults:
                if isinstance(d, (ast.List, ast.Dict, ast.Set)):
                    out.append((node.lineno, f"`{node.name}` has a mutable default argument; it is shared between calls (use None)"))
            if getattr(node, "end_lineno", node.lineno) - node.lineno > 60:
                out.append((node.lineno, f"`{node.name}` is over 60 lines; consider splitting it"))
        if isinstance(node, ast.Compare) and any(isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops) \
                and any(isinstance(c, ast.Constant) and c.value is None for c in node.comparators):
            out.append((node.lineno, "compare with None using `is` / `is not`"))
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in SHADOWED:
                    out.append((node.lineno, f"the variable `{t.id}` hides the built-in `{t.id}`"))
        if isinstance(node, ast.Global):
            out.append((node.lineno, "`global` makes functions hard to test; pass values as arguments"))
    for i, line in enumerate(code.splitlines(), 1):
        if SECRET.search(line):
            out.append((i, "a password or key is written into the code; load it from an environment variable"))
    return out


def undefined_names(tree):
    """Names the snippet reads but never defines, imports or gets from builtins (e.g. del_cost() copied from
    textbook pseudocode)."""
    import builtins
    defined, used = set(dir(builtins)), {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            defined |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, ast.Name):
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                defined.add(node.id)
            else:
                used.setdefault(node.id, node.lineno)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            defined |= set(node.names)
    return sorted((line, name) for name, line in used.items() if name not in defined)


def parse(code):
    try:
        return ast.parse(code), None
    except SyntaxError as e:
        return None, (e.lineno or 0, f"syntax error: {e.msg}")


def constraints(code, allowed=ALLOWED_IMPORTS):
    """{'syntax_ok', 'blocked': [(line, msg)], 'warnings': [(line, msg)]}; anything blocked must not be run."""
    tree, err = parse(code)
    if err:
        return {"syntax_ok": False, "blocked": [err], "warnings": []}
    blocked = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for m in mods:
                if m.split(".")[0] not in allowed:
                    blocked.append((node.lineno, f"`import {m}` is not allowed in this course exercise "
                                                 f"(allowed: {', '.join(sorted(allowed))})"))
        if isinstance(node, ast.Attribute) and BLOCKED_ATTRS.match(node.attr):
            blocked.append((node.lineno, f"`{node.attr}` is not allowed"))
        if isinstance(node, ast.Name) and node.id == "__builtins__":
            blocked.append((node.lineno, "`__builtins__` is not allowed"))
    for node, name, _ in _calls(tree):
        if name in BLOCKED_CALLS and isinstance(node.func, ast.Name):
            blocked.append((node.lineno, f"`{name}()` is not allowed in submissions"))
    return {"syntax_ok": True, "blocked": sorted(set(blocked)), "warnings": sorted(set(_style(tree, code)))}


def review(code):
    """Checks for code the tutor suggests in chat: syntax, security problems, best practice. Never blocks."""
    tree, err = parse(code)
    if err:
        return {"syntax_ok": False, "warnings": [err]}
    warn = []
    for node, name, full in _calls(tree):
        if name in ("eval", "exec") and isinstance(node.func, ast.Name):
            warn.append((node.lineno, f"`{name}()` runs arbitrary code; never use it on user input"))
        if full in ("os.system", "os.popen"):
            warn.append((node.lineno, f"`{full}` passes a string to the shell; use subprocess.run([...]) with a list"))
        if full and full.startswith("subprocess.") and any(k.arg == "shell" and isinstance(k.value, ast.Constant)
                                                          and k.value.value is True for k in node.keywords):
            warn.append((node.lineno, "`shell=True` allows shell injection; pass the command as a list"))
        if full in ("pickle.load", "pickle.loads", "torch.load") and not any(k.arg == "weights_only" for k in node.keywords):
            warn.append((node.lineno, f"`{full}` can run code hidden in the file; only load files you trust"
                         + (" (use weights_only=True)" if full == "torch.load" else "")))
        if full in ("yaml.load",) and not any(k.arg == "Loader" for k in node.keywords):
            warn.append((node.lineno, "`yaml.load` without a Loader can build arbitrary objects; use yaml.safe_load"))
        if name in ("get", "post", "request") and any(k.arg == "verify" and isinstance(k.value, ast.Constant)
                                                     and k.value.value is False for k in node.keywords):
            warn.append((node.lineno, "`verify=False` turns off HTTPS certificate checks"))
        if name in ("execute", "executemany") and node.args and isinstance(node.args[0], (ast.JoinedStr, ast.BinOp)):
            warn.append((node.lineno, "SQL built from a formatted string risks SQL injection; use query parameters"))
        if full in ("random.random", "random.randint", "random.choice") and re.search(r"(?i)token|password|secret",
                                                                                         ast.unparse(node)):
            warn.append((node.lineno, "the `random` module is not secure for secrets; use `secrets`"))
    for line, name in undefined_names(tree):
        warn.append((line, f"`{name}` is used but never defined in this code; it will fail with NameError unless you define it"))
    return {"syntax_ok": True, "warnings": sorted(set(warn + _style(tree, code)))}


RUNNER = r'''
import io, json, sys, time
real_out = sys.stdout
def hook(event, args):
    if event == "open" and len(args) > 1 and isinstance(args[1], str) and any(c in args[1] for c in "wax+"):
        raise PermissionError("writing files is not allowed")
    if event.startswith(("subprocess.", "socket.", "os.exec", "os.spawn", "os.posix_spawn", "ctypes.", "winreg.",
                         "shutil.", "_winapi.")) or event in ("os.system", "os.remove", "os.rmdir", "os.rename",
                                                              "os.mkdir", "os.chmod", "os.kill", "os.startfile"):
        raise PermissionError(event + " is not allowed")
job = json.load(open(sys.argv[1], encoding="utf-8"))
sys.addaudithook(hook)
ns = {"__name__": "submission"}
sys.stdout = io.StringIO()
try:
    exec(compile(job["code"], "submission.py", "exec"), ns)
    real_out.write(json.dumps({"load": "ok"}) + "\n")
except BaseException as e:
    real_out.write(json.dumps({"load": f"{type(e).__name__}: {e}"}) + "\n")
    raise SystemExit(0)
for i, t in enumerate(job["tests"]):
    t0 = time.time()
    try:
        exec(compile(t, f"test_{i}", "exec"), ns)
        r = {"i": i, "ok": True}
    except AssertionError as e:
        r = {"i": i, "ok": False, "error": "wrong result" + (f": {e}" if str(e) else "")}
    except BaseException as e:
        r = {"i": i, "ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
    r["ms"] = round((time.time() - t0) * 1000)
    real_out.write(json.dumps(r) + "\n"); real_out.flush()
'''


def run_tests(code, tests, timeout=10):
    """[{'test', 'ok', 'error'?}] per assert line, plus a load error if the code itself fails to run."""
    with tempfile.TemporaryDirectory(prefix="shannon-run-") as tmp:
        Path(tmp, "runner.py").write_text(RUNNER, encoding="utf-8")
        Path(tmp, "job.json").write_text(json.dumps({"code": code, "tests": tests}), encoding="utf-8")
        try:
            p = subprocess.run([sys.executable, "-I", "-B", "runner.py", "job.json"], cwd=tmp, capture_output=True,
                               text=True, encoding="utf-8", errors="replace", timeout=timeout)
            lines, timed_out = p.stdout.splitlines(), False
        except subprocess.TimeoutExpired as e:
            out = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            lines, timed_out = out.splitlines(), True
    recs = []
    for l in lines:
        try:
            recs.append(json.loads(l))
        except json.JSONDecodeError:
            pass
    load = next((r["load"] for r in recs if "load" in r), None)
    done = {r["i"]: r for r in recs if "i" in r}
    results = []
    for i, t in enumerate(tests):
        r = done.get(i)
        if r:
            results.append({"test": t, "ok": r["ok"], **({"error": r["error"]} if not r["ok"] else {})})
        else:
            why = (f"code failed to run: {load}" if load and load != "ok"
                   else f"time limit ({timeout}s) reached" if timed_out else "not run")
            results.append({"test": t, "ok": False, "error": why})
    return results
