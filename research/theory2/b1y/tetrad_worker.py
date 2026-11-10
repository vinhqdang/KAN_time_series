"""Tetrad worker: serves BOSS / GRaSP / FGES calls from a separate process (JVM isolated from the forking learner timeouts).
Protocol as dynotears_worker.py (length-prefixed pickles on stdin/stdout); same Python environment, so arrays pickle normally.
Request: dict(method in {fges, boss, grasp}, X). Reply: dict(Mdir, Mund) or dict(error)."""
import sys, os, pickle, struct
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import methods_tetrad as mt


def serve():
    rin, rout = sys.stdin.buffer, sys.stdout.buffer
    fns = dict(fges=mt.fges_tetrad, boss=mt.boss_tetrad, grasp=mt.grasp_tetrad)
    while True:
        h = rin.read(8)
        if len(h) < 8:
            return
        req = pickle.loads(rin.read(struct.unpack("<Q", h)[0]))
        try:
            D, U = fns[req["method"]](req["X"]); rep = dict(Mdir=D, Mund=U)
        except Exception as e:                      # noqa: BLE001
            rep = dict(error=repr(e)[:300])
        b = pickle.dumps(rep, protocol=4); rout.write(struct.pack("<Q", len(b))); rout.write(b); rout.flush()


if __name__ == "__main__":
    serve()
