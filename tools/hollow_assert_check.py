"""Find assertIn(needle, inspect.getsource(...)) that only matches PROSE.

A source-text assertion passes when the needle appears anywhere in the function
text, including the comment that explains the very thing being asserted. When
that happens the test cannot fail and deleting the behaviour leaves it green.

First attempt at this audit reported 15 findings, all false: it rebuilt the
stripped source by joining token strings with newlines, which destroys
adjacency, so a needle like "_bzone_entry_on_record(" could never match its own
call. This version blanks comment and docstring spans IN PLACE, preserving every
other character and offset.
"""
import ast, io, os, sys, tokenize, importlib, inspect

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
sys.path.insert(0, ROOT)
a = importlib.import_module("dman_algo")

TEST = os.path.join(ROOT, "test_dman_algo.py")
src = io.open(TEST, encoding="utf-8").read()
tree = ast.parse(src)


def blank_prose(text: str) -> str:
    """Replace comments and docstrings with spaces, keeping all other offsets."""
    lines = text.splitlines(keepends=True)
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except Exception:
        # Indented source (a nested def) will not tokenize; dedent and retry.
        import textwrap
        try:
            text2 = textwrap.dedent(text)
            lines = text2.splitlines(keepends=True)
            toks = list(tokenize.generate_tokens(io.StringIO(text2).readline))
        except Exception:
            return text

    kill = []
    prev = "NEWLINE"
    for t in toks:
        name = tokenize.tok_name[t.type]
        if name == "COMMENT":
            kill.append((t.start, t.end))
        elif name == "STRING":
            q = t.string.lstrip("rRbBfFuU")
            if q[:3] in ('"""', "'''") and prev in ("NEWLINE", "NL", "INDENT",
                                                    "DEDENT", "ENCODING"):
                kill.append((t.start, t.end))
        if name not in ("NL", "COMMENT"):
            prev = name

    arr = [list(l) for l in lines]
    for (sr, sc), (er, ec) in kill:
        for r in range(sr, er + 1):
            if r - 1 >= len(arr):
                break
            row = arr[r - 1]
            lo = sc if r == sr else 0
            hi = ec if r == er else len(row)
            for i in range(lo, min(hi, len(row))):
                if row[i] != "\n":
                    row[i] = " "
    return "".join("".join(r) for r in arr)


def resolve(expr):
    parts, node = [], expr
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    parts.reverse()
    if parts[0] != "a":
        return None
    obj = a
    for p in parts[1:]:
        obj = getattr(obj, p, None)
        if obj is None:
            return None
    return obj


findings, checked = [], 0
for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
    for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef)]:
        srcvars = {}
        for node in ast.walk(fn):
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == "getsource" and node.value.args):
                tgt = resolve(node.value.args[0])
                if tgt is not None and isinstance(node.targets[0], ast.Name):
                    srcvars[node.targets[0].id] = tgt
        if not srcvars:
            continue
        for node in ast.walk(fn):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "assertIn" and len(node.args) >= 2):
                continue
            needle, hay = node.args[0], node.args[1]
            if not (isinstance(needle, ast.Constant) and isinstance(needle.value, str)):
                continue
            if not (isinstance(hay, ast.Name) and hay.id in srcvars):
                continue
            target = srcvars[hay.id]
            try:
                body = inspect.getsource(target)
            except Exception:
                continue
            checked += 1
            code_only = blank_prose(body)
            if needle.value in body and needle.value not in code_only:
                findings.append((cls.name, fn.name, node.lineno, needle.value,
                                 getattr(target, "__name__", str(target))))

print(f"checked {checked} source-text assertion(s)")
if not findings:
    print("no hollow ones: every needle appears in executable code")
else:
    print(f"\n{len(findings)} HOLLOW -- needle exists only in a comment/docstring:\n")
    for cls, fn, ln, needle, tgt in findings:
        print(f"  line {ln}  {cls}.{fn}")
        print(f"     assertIn({needle!r}) vs {tgt}()")
        print(f"     assertIn is carried by prose, so deleting the behaviour "
              f"leaves this test green. Assert it at RUNTIME instead.")
sys.exit(1 if findings else 0)
