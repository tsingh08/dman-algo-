"""Find f-strings that nest a same-quote string inside a replacement field.
Legal since Python 3.12 (PEP 701), a SyntaxError on 3.11 -- which CI and every
trading job run. Needs a 3.12+ interpreter to see FSTRING_START tokens."""
import io, sys, tokenize


def _quote(tok_string: str) -> str:
    s = tok_string.lstrip("rRbBfFuU")
    return s[:3] if s[:3] in ('"""', "'''") else s[:1]


def violations(path: str) -> list:
    out, stack = [], []
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
        elif name == "STRING" and stack:
            if _quote(tok.string) in stack:
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
