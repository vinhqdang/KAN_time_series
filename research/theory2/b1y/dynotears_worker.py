"""DYNOTEARS worker (labelled reference). Runs in a separate Python 3.10 environment (/home/user/venv310) because the only
maintained implementation, causalnex 0.12.1 (QuantumBlack; Pamfil et al., AISTATS 2020, arXiv 2002.00498), does not install on
Python >= 3.11. Protocol: reads pickled requests from stdin, writes pickled replies to stdout (length-prefixed).
Request: dict(Z=(bytes, shape) of the (m, p) standardised current slices, Zlag=same for previous slices, lambda_w, lambda_a, w_threshold, max_iter, h_tol).
Reply: dict(W=(p, p) intra-slice weights, A=(p, p) lag-1 weights), W[i, j]: i -> j in the same slice, A[i, j]: i(t) -> j(t+1).
"""
import sys, pickle, struct
import numpy as np


def run(req):
    from causalnex.structure.dynotears import from_numpy_dynamic
    # arrays arrive as (raw float64 bytes, shape): pickled numpy-2 arrays cannot be loaded by numpy 1.x
    dec = lambda t: np.frombuffer(t[0], dtype=np.float64).reshape(t[1]).copy()
    Z, Zl = dec(req["Z"]), dec(req["Zlag"])
    sm = from_numpy_dynamic(Z, Zl, lambda_w=req["lambda_w"], lambda_a=req["lambda_a"], max_iter=req.get("max_iter", 100),
                            h_tol=req.get("h_tol", 1e-8), w_threshold=req["w_threshold"])
    p = Z.shape[1]
    W = np.zeros((p, p)); A = np.zeros((p, p))
    for u, v, data in sm.edges(data=True):
        iu, lu = u.split("_lag"); iv, lv = v.split("_lag")
        w = data["weight"]
        if lu == "0" and lv == "0":
            W[int(iu), int(iv)] = w
        elif lu == "1" and lv == "0":
            A[int(iu), int(iv)] = w
    return dict(W=W.tolist(), A=A.tolist())


def serve():
    rin, rout = sys.stdin.buffer, sys.stdout.buffer
    while True:
        h = rin.read(8)
        if len(h) < 8:
            return
        n = struct.unpack("<Q", h)[0]
        req = pickle.loads(rin.read(n))
        try:
            rep = run(req)
        except Exception as e:                     # noqa: BLE001  report the failure to the caller, never swallow it
            rep = dict(error=repr(e))
        b = pickle.dumps(rep, protocol=4)
        rout.write(struct.pack("<Q", len(b))); rout.write(b); rout.flush()


if __name__ == "__main__":
    serve()
