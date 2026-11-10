"""Tetrad (Java) BOSS, GRaSP and FGES via JPype1 (Tetrad 7.6.10 jars from Maven Central io.github.cmu-phil:tetrad-lib; the PyPI name
`py-tetrad` does not exist). SemBicScore with penalty discount 2 (BIC with the same extra penalty used for FLOP), data as ContinuousVariable columns.
Returns (Mdir, Mund) bool (d, d) like methods_y.*_cpdag. Classpath: B1Y_TETRAD_JARS (default /home/user/tetrad_env/jars/*.jar)."""
import os, glob
import numpy as np

_started = False


def _start():
    global _started
    if _started:
        return
    import jpype, jpype.imports
    jars = glob.glob(os.environ.get("B1Y_TETRAD_JARS", "/home/user/tetrad_env/jars") + "/*.jar")
    if not jpype.isJVMStarted():
        jpype.startJVM("-Xmx2g", classpath=jars)
    _started = True


def _dataset(X):
    import jpype
    from edu.cmu.tetrad.data import BoxDataSet, DoubleDataBox, ContinuousVariable
    from java.util import ArrayList
    n, d = X.shape
    vars_ = ArrayList()
    for j in range(d):
        vars_.add(ContinuousVariable("X%d" % j))
    box = DoubleDataBox(jpype.JArray(jpype.JDouble, 2)(X.astype(float).tolist()))
    return BoxDataSet(box, vars_), vars_


def _to_matrices(G, d):
    from edu.cmu.tetrad.graph import Endpoint
    Mdir = np.zeros((d, d), bool); Mund = np.zeros((d, d), bool)
    for e in G.getEdges():
        i = int(str(e.getNode1().getName())[1:]); j = int(str(e.getNode2().getName())[1:])
        e1, e2 = e.getEndpoint1(), e.getEndpoint2()
        if e1 == Endpoint.TAIL and e2 == Endpoint.ARROW:
            Mdir[i, j] = True
        elif e1 == Endpoint.ARROW and e2 == Endpoint.TAIL:
            Mdir[j, i] = True
        elif e1 == Endpoint.TAIL and e2 == Endpoint.TAIL:
            Mund[i, j] = Mund[j, i] = True
    return Mdir, Mund


def _score(X, penalty=2.0):
    from edu.cmu.tetrad.search.score import SemBicScore
    ds, vars_ = _dataset(X)
    sc = SemBicScore(ds, True)
    sc.setPenaltyDiscount(penalty)
    return sc


def boss_tetrad(X):
    _start()
    from edu.cmu.tetrad.search import Boss, PermutationSearch
    s = PermutationSearch(Boss(_score(X)))
    return _to_matrices(s.search(), X.shape[1])


def grasp_tetrad(X):
    _start()
    from edu.cmu.tetrad.search import Grasp
    sc = _score(X)
    g = Grasp(sc)
    g.setDepth(3); g.setUncoveredDepth(1); g.setNonSingularDepth(1)
    g.bestOrder(sc.getVariables())
    return _to_matrices(g.getGraph(True), X.shape[1])


def fges_tetrad(X):
    _start()
    from edu.cmu.tetrad.search import Fges
    return _to_matrices(Fges(_score(X)).search(), X.shape[1])
