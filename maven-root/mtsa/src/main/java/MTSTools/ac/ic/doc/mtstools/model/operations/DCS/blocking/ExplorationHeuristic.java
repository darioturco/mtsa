package MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking;

import MTSTools.ac.ic.doc.commons.relations.Pair;
import MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.Compostate;
import MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.HAction;

import java.util.ArrayList;
import java.util.Set;

public interface ExplorationHeuristic<State, Action> {

    int currentTargetLTSIndex = 0;

    void setLastExpandedStateAction(ActionWithFeatures<State, Action> stateAction);

    int frontierSize();

    void setInitialState(Compostate<State, Action> initial);

    boolean somethingLeftToExplore();

    void expansionDone(Compostate<State, Action> first, HAction<Action> second, Compostate<State, Action> child);

    Pair<Compostate<State,Action>, HAction<Action>> getNextAction(boolean updateUnexploredTransaction);

    ArrayList<Integer> getOrder();

    int getNextActionIndex();

    ActionWithFeatures<State, Action> removeFromFrontier(int idx);

    void filterFrontier();

    void notifyExpandingState(Compostate<State, Action> state, HAction<Action> action, Compostate<State, Action> child);

    void notifyStateIsNone(Compostate<State, Action> state);

    void notifyClosedPotentiallyWinningLoop(Set<Compostate<State, Action>> loop);

    void notifyStateSetErrorOrGoal(Compostate<State, Action> state);

    void newState(Compostate<State, Action> state, Compostate<State, Action> parent);

    void notifyExpansionDidntFindAnything(Compostate<State, Action> parent, HAction<Action> action, Compostate<State, Action> child);

    void notify_end_synthesis();

    boolean fullyExplored(Compostate<State, Action> state);

    boolean hasUncontrollableUnexplored(Compostate<State, Action> state);

    int getIndexOfStateAction(Pair<Compostate<State,Action>, HAction<Action>> actionState);

    void printFrontier();

    void initialize(Compostate<State, Action> state);

    ArrayList<ActionWithFeatures<State, Action>> getFrontier();

}
