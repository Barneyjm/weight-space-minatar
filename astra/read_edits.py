"""Read the last ```edits block of a reply into a 600-32-32-6 network
(the authors' 'edits' language). Parses statements; never evals code.
Usage: read_edits.py reply.md out.npz"""
import ast, re, sys, numpy as np
text = open(sys.argv[1]).read()
blocks = re.findall(r"```edits[^\n]*\n(.*?)```", text, re.S)
assert blocks, "no edits block"
T = {"W1": np.zeros((600, 32)), "b1": np.zeros(32), "W2": np.zeros((32, 32)),
     "b2": np.zeros(32), "W3": np.zeros((32, 6)), "b3": np.zeros(6)}
lines = [l.split("#", 1)[0].rstrip() for l in blocks[-1].splitlines()]
stmts, cur = [], ""
for l in lines:                      # join continuation lines until brackets balance
    if not l.strip() and not cur: continue
    cur = (cur + " " + l).strip()
    if cur.count("[") == cur.count("]"): stmts.append(cur); cur = ""
if cur: stmts.append(cur)
STMT = re.compile(r"^(W1|b1|W2|b2|W3|b3)\s*(\[[^\]]*\])?\s*(=|\*=|\+=|-=)\s*(.+)$")
def idx(s, nd):
    parts = [p.strip() for p in s.strip("[]").split(",")] if s else [":"] * nd
    out = []
    for p in parts:
        if ":" in p:
            out.append(slice(*[int(v) if v.strip() else None for v in p.split(":")]))
        else:
            out.append(int(p))
    return tuple(out)
ok = err = 0
for s in stmts:
    m = STMT.match(s)
    try:
        assert m, "syntax"
        name, ix, op, val = m.groups(); A = T[name]
        v = np.array(ast.literal_eval(val.strip()), float)
        assert np.all(np.abs(v) <= 1e6)
        k = idx(ix, A.ndim); tgt = A[k]
        v = np.broadcast_to(v, np.shape(tgt)) if np.ndim(v) == 0 else v.reshape(np.shape(tgt))
        if op == "=": A[k] = v
        elif op == "*=": A[k] = tgt * v
        elif op == "+=": A[k] = tgt + v
        else: A[k] = tgt - v
        ok += 1
    except Exception as e:
        err += 1; print("skipped:", s[:100], "|", type(e).__name__, e)
np.savez(sys.argv[2], **{"actor/dense1/kernel": T["W1"].astype(np.float32), "actor/dense1/bias": T["b1"].astype(np.float32),
    "actor/dense2/kernel": T["W2"].astype(np.float32), "actor/dense2/bias": T["b2"].astype(np.float32),
    "actor/logits/kernel": T["W3"].astype(np.float32), "actor/logits/bias": T["b3"].astype(np.float32)})
print(f"statements read {ok}, skipped {err}, nonzero weights {sum(int((a != 0).sum()) for a in T.values())}")
