package MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking;
import java.util.List;

public class ActionForPython<State, Action> {
    public HAction<Action> action;
    public Compostate<State, Action> state;
    public Compostate<State, Action> child;
    public List<State> childStates;
    public boolean childMarked;
    public FeatureBasedExplorationHeuristic<State, Action> heuristic;

    ActionForPython(Compostate<State, Action> state, HAction<Action> action, FeatureBasedExplorationHeuristic<State, Action> heuristic) {
        this.heuristic = heuristic;
        this.state = state;
        this.childStates = state.actionChildStates.get(action);
        this.child = this.heuristic.dcs.compostates.get(childStates);
        this.action = action;
        this.childMarked = true;

        for (int lts = 0; childMarked && lts < this.heuristic.dcs.ltssSize; ++lts){
            childMarked = this.heuristic.dcs.defaultTargets.get(lts).contains(childStates.get(lts));
        }
    }

    public String toString() {
        StringBuilder r = new StringBuilder(state.toString() + " " + action.toString() + " " + child.toString());
        return r.toString();
    }
}