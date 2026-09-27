"""Find f-strings that Python 3.11 rejects but 3.12+ accepts.

CI and every trading job run 3.11; local development runs 3.14. PEP 701 relaxed
two separate restrictions in 3.12, and BOTH are a SyntaxError on 3.11:

  1. a same-quote string nested inside a replacement field
     f"{d["k"]}"          -- caught since this file was written
  2. a backslash anywhere inside a replacement field
     f"{'\u2705' if ok else '\u274c'}"   -- added 2026-09-27

Only (1) was checked, which is why (2) reached a commit: a status line written
as f"{'<tick>' if ok else '<cross>'}" parsed fine on 3.14, passed this guard,
and would have failed the 3.11 job. ast.parse(feature_version=(3, 11)) does not
help -- it does not model either rule, so it reports OK on both.

Literal text outside the braces may contain escapes; that has always been legal.
Only the expression parts are checked, tracked by brace depth.

Needs a 3.12+ interpreter to see FSTRING_START tokens."""
import io, sys, tokenize


def _quote(tok_string: str) -> str:
    s = tok_string.lstrip("rRbBfFuU")
    return s[:3] if s[:3] in ('"""', "'''") else s[:1]


def violations(path: str) -> list:
    out, stack = [], []
    depth = 0          # brace nesting INSIDE the innermost f-string
    src = io.open(path, encoding="utf-8").read()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        name = tokenize.tok_name[tok.type]
        if name == "FSTRING_START":
            q = _quote(tok.string)
            if any(q == outer for outer in stack):
                out.append((tok.start[0], tok.line.strip()[:100]))
            stack.append(q)
        elif name == "FSTRING_END":
            if stack:
                stack.pop()
            depth = 0
        elif stack and name == "OP" and tok.string == "{":
            depth += 1
        elif stack and name == "OP" and tok.string == "}":
            depth = max(0, depth - 1)
        elif name == "STRING" and stack:
            # (1) same-quote nesting, and (2) any backslash inside the
            # replacement field -- both legal only from 3.12.
            if _quote(tok.string) in stack or (depth > 0 and "\\" in tok.string):
                out.append((tok.start[0], tok.line.strip()[:100]))
    return out


if __name__ == "__main__":
    bad = 0
    for f in sys.argv[1:]:
        for ln, line in violations(f):
            bad += 1
            print(f"{f}:{ln}: {line}")
    print(f"{bad} Python-3.11-incompatible f-string(s)")
    sys.exit(1 if bad else 0)
