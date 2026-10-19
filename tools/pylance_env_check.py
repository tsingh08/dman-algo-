"""Why is the editor showing thousands of problems in a file that compiles?

This has happened three times and cost an hour the first time (2026-10-12).
Every occurrence looked identical -- thousands of Pylance WARNINGS of the form
`"object" is not defined`, `"float" is not defined` -- and the cause was never
the code. Pylance had lost its view of the standard library, so every builtin
became unresolved.

There are only four things that can do that, and three of them are checkable
from a shell in under a second. Run this before investigating anything else:

    python tools/pylance_env_check.py

A clean run USED to mean "reload the window", and on 2026-10-18 that advice
was wrong. The warnings came back within hours because the cause was the one
thing this script did not look at: WHICH FOLDER IS OPEN.

Pylance analyses the open workspace. If dman_algo.py is edited while the open
folder is something else -- that day it was C:/Users/singh/Downloads/files --
the file sits outside every execution environment root, nothing binds the
builtins scope, and every `float`, `str` and `print` in 31,000 lines is
reported undefined. 6,082 of them, all builtins, not one real name. The
home-directory .vscode/settings.json is not loaded either, because it belongs
to a workspace that is not open. A reload re-binds it for a while, which is
exactly what makes this look like staleness instead of the configuration
mistake it is. Check 5 below reads the language server's own log and says so.

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
import re
import sys


def _mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0

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


# "repo root", not "workspace root". Calling HOME the workspace root was the
# assumption that made this script confidently wrong on 2026-10-18 -- check 5
# is the one that asks the editor instead of assuming.
print("Pylance prerequisites (repo root = %s)" % HOME)
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

# 5. the file must actually be INSIDE the open workspace. Pylance's own log
#    is the only honest source for which folder that is -- the editor can have
#    any folder open regardless of where this script is run from.
_logs = sorted(glob.glob(os.path.join(
    os.environ.get("APPDATA", os.path.join(HOME, "AppData", "Roaming")),
    "Code", "logs", "*", "window*", "exthost", "ms-python.python",
    "Python Language Server.log")), key=_mtime)
if not _logs:
    notes.append("no Pylance log found -- cannot verify which folder is open")
else:
    _root, _found = None, None
    for _line in io.open(_logs[-1], encoding="utf-8", errors="replace"):
        _m = re.search(r'for workspace "([^"]+)"', _line)
        if _m:
            _root = _m.group(1)
        _m = re.search(r"Found (\d+) source files", _line)
        if _m:
            _found = int(_m.group(1))
    _inside = bool(_root) and os.path.normcase(os.path.abspath(HOME)).startswith(
        os.path.normcase(os.path.abspath(_root)))
    check("the repo is inside the open workspace", _inside,
          ("open folder is %s -- the repo at %s is OUTSIDE it, so builtins do "
           "not bind and %s/.vscode/settings.json is not loaded. Open %s "
           "(File -> Open Folder) or the biz.code-workspace file."
           % (_root, HOME, HOME, HOME)) if not _inside else
          "open folder is %s" % _root)
    if _found is not None and _found < 10:
        notes.append("Pylance found only %d source file(s) in the open "
                     "workspace -- another sign the wrong folder is open"
                     % _found)

print()
for n in notes:
    print("  note: %s" % n)
if fails:
    print("  %d prerequisite(s) FAILED -- fix those before touching the code." % len(fails))
    sys.exit(1)
print("  All prerequisites are fine, so the code is not the problem.")
print("  Only now is 'the language server is stale' the likely answer:")
print("  Ctrl+Shift+P -> Developer: Reload Window.")
print("  If that does not clear it, re-pick the interpreter:")
print("  Ctrl+Shift+P -> Python: Select Interpreter -> pythoncore-3.14-64.")
