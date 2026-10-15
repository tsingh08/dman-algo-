"""Why is the editor showing thousands of problems in a file that compiles?

This has happened three times and cost an hour the first time (2026-10-12).
Every occurrence looked identical -- thousands of Pylance WARNINGS of the form
`"object" is not defined`, `"float" is not defined` -- and the cause was never
the code. Pylance had lost its view of the standard library, so every builtin
became unresolved.

There are only four things that can do that, and three of them are checkable
from a shell in under a second. Run this before investigating anything else:

    python tools/pylance_env_check.py

A clean run means the file is fine and the language server is stale -- reload
the window (Ctrl+Shift+P -> Developer: Reload Window), which is what fixed it
on 2026-10-12 and again on 2026-10-15.

The specific trap worth remembering: this workspace root is the HOME
directory, so the toolchain lives INSIDE the workspace. Pylance's own typeshed
sits under .vscode/extensions and the interpreter's Lib/site-packages under
AppData/Local/Python. Excluding either from python.analysis.exclude -- which
looks like a reasonable way to stop Pylance walking 44 GB -- hides builtins and
every installed package. Use files.watcherExclude and search.exclude for that
instead; keep Pylance cheap with indexing=false and diagnosticMode=openFilesOnly.
"""
import glob
import io
import json
import os
import sys

HOME = os.path.expanduser("~")
SETTINGS = os.path.join(HOME, ".vscode", "settings.json")
# Excluding either of these hides the toolchain, because the workspace root is
# the home directory. See the module docstring.
HARMFUL = ("**/AppData/**", "**/.vscode/**")

fails = []
notes = []


def check(label, ok, detail):
    print("  %-44s %s" % (label, "ok" if ok else "FAIL"))
    if detail:
        print("       %s" % detail)
    if not ok:
        fails.append(label)


print("Pylance prerequisites (workspace root = %s)" % HOME)
print()

# 1. the analysis excludes must not hide the toolchain
try:
    cfg = json.load(io.open(SETTINGS, encoding="utf-8-sig"))
except Exception as exc:
    cfg = {}
    notes.append("could not read %s (%s)" % (SETTINGS, exc))
excl = list(cfg.get("python.analysis.exclude", []))
bad = [p for p in excl if p in HARMFUL]
check("python.analysis.exclude hides no toolchain", not bad,
      ("remove %s -- it hides typeshed and/or site-packages" % bad) if bad
      else "%d pattern(s), none harmful" % len(excl))

# 2. the interpreter it points at must exist
interp = cfg.get("python.defaultInterpreterPath", "")
check("defaultInterpreterPath exists", bool(interp) and os.path.isfile(interp),
      interp or "not set -- Pylance falls back to whatever is selected in the UI")

# 3. Pylance's own typeshed must be present and not excluded
ts = sorted(glob.glob(os.path.join(
    HOME, ".vscode", "extensions", "ms-python.vscode-pylance-*",
    "dist", "typeshed-fallback", "stdlib", "builtins.pyi")))
check("Pylance typeshed builtins.pyi found", bool(ts),
      ts[-1] if ts else "no builtins.pyi under .vscode/extensions -- "
                        "reinstall or update the Pylance extension")

# 4. a .code-workspace file's own settings outrank .vscode/settings.json
for wsf in sorted(glob.glob(os.path.join(HOME, "*.code-workspace"))):
    try:
        ws = json.load(io.open(wsf, encoding="utf-8-sig"))
    except Exception:
        continue
    wex = [p for p in (ws.get("settings", {}) or {}).get(
        "python.analysis.exclude", []) if p in HARMFUL]
    check("%s adds no harmful exclude" % os.path.basename(wsf), not wex,
          ("workspace-file settings OUTRANK .vscode/settings.json -- remove %s"
           % wex) if wex else "no python.analysis settings")

print()
for n in notes:
    print("  note: %s" % n)
if fails:
    print("  %d prerequisite(s) FAILED -- fix those before touching the code." % len(fails))
    sys.exit(1)
print("  All prerequisites are fine, so the code is not the problem.")
print("  If the editor still shows 'X is not defined' warnings, the language")
print("  server is stale: Ctrl+Shift+P -> Developer: Reload Window.")
print("  If that does not clear it, re-pick the interpreter:")
print("  Ctrl+Shift+P -> Python: Select Interpreter -> pythoncore-3.14-64.")
