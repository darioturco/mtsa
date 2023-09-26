from src.composition import CompositionGraph, CompositionAnalyzer
from src.environment import Environment

FSP_PATH = "./fsp"
BENCHMARK_PROBLEMS = ["AT", "BW", "CM", "DP", "TA", "TL"]

if __name__ == "__main__":
    d = CompositionGraph("AT", 3, 3)

    d.start_composition(FSP_PATH)
    da = CompositionAnalyzer(d)
    k = 0
    i = 100
    while i and not d._javaEnv.isFinished():
        print(d.getFrontier())
        d.expand(0)
        frontier_features = []
        for trans in d.getFrontier():
            feature = da.compute_features(trans)
            frontier_features.append(feature)

        # frontier_features = [(da.compute_features(trans)) for trans in d.getFrontier()]

        assert (d._expansion_order[-1] in [e[2]["action_with_features"] for e in d.edges(data=True)])

        # k+=sum([sum(da.isLastExpanded(trans[2]["action_with_features"])) for trans in d.edges(data=True)])
        # i-=1

    # print(k)
