package MTSTools.ac.ic.doc.mtstools.model.operations.DCS;

import MTSTools.ac.ic.doc.mtstools.model.LTS;
import MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.Alphabet;
import MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.Compostate;
import MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.FeatureBasedExplorationHeuristic;
import MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.Statistics;
import MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.abstraction.Recommendation;
import MTSTools.ac.ic.doc.mtstools.model.operations.DCS.nonblocking.ExplorationHeuristic;

import java.util.HashMap;
import java.util.List;
import java.util.Set;

public abstract class DirectedControllerSynthesis<State, Action> {
    public abstract void setHeuristic(ExplorationHeuristic<State, Action> heuristic);

    @SuppressWarnings("unchecked")
    public abstract LTS<Long, Action> synthesize(
        List<LTS<State, Action>> ltss,
        Set<Action> controllable,
        boolean reachability,
        HashMap<Integer, Integer> guarantees,
        HashMap<Integer, Integer> assumptions);

    public abstract Statistics getStatistics();
}
