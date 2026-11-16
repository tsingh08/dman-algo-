"""Did a change touch any module-level constant?

    python tools/risk_pins_check.py OLD_FILE NEW_FILE [OLD2 NEW2 ...]

Every risk limit in this system is a module-level UPPER_CASE constant:
position size, loss caps, the kill-switch bars, which setups may trade, the
tradable watchlist, the macro calendars, the AI veto bar. This compares those
between two versions of a file by parsing them (never importing them), and
exits 1 if any constant was changed or removed. New constants are allowed.

It is the gate that lets the nightly review merge its own bug fixes: a fix
that leaves every constant byte-for-byte alone cannot have loosened a limit
by editing one. The review workflow runs the copy of this file from main, so
a branch cannot weaken the check it is judged by.
"""
import ast
import re
import sys

_NAME = re.compile(r"^_?[A-Z][A-Z0-9_]*$")


def constants(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        elif isinstance(node, ast.AugAssign):
            targets, value = [node.target], node
        else:
            continue
        for t in targets:
            names = [t] if isinstance(t, ast.Name) else (
                [e for e in t.elts if isinstance(e, ast.Name)] if isinstance(t, (ast.Tuple, ast.List)) else [])
            for n in names:
                if _NAME.match(n.id):
                    # a name assigned twice keeps every assignment, in order
                    out.setdefault(n.id, []).append(ast.dump(value))
    return out


def compare(old_path: str, new_path: str) -> list:
    old, new = constants(old_path), constants(new_path)
    problems = []
    for name, val in old.items():
        if name not in new:
            problems.append(f"removed: {name}")
        elif new[name] != val:
            problems.append(f"changed: {name}")
    return problems


def main(argv) -> int:
    if len(argv) < 2 or len(argv) % 2:
        print(__doc__)
        return 2
    problems = []
    for i in range(0, len(argv), 2):
        try:
            problems += [f"{argv[i + 1]}: {p}" for p in compare(argv[i], argv[i + 1])]
        except (OSError, SyntaxError) as e:
            problems.append(f"{argv[i + 1]}: could not be read ({type(e).__name__}: {e})")
    if problems:
        print("CONSTANTS TOUCHED:")
        for p in problems[:40]:
            print("  " + p)
        return 1
    print("No module-level constant changed or removed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
