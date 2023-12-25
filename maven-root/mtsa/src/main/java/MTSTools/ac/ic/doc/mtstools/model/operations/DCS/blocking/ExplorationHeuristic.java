package MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking;

import MTSTools.ac.ic.doc.commons.relations.Pair;
import MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.Compostate;
import MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.HAction;

import java.util.ArrayList;

public interface ExplorationHeuristic<State, Action> {

    void setLastExpandedStateAction(ActionWithFeatures<State, Action> stateAction);

    int frontierSize();

    void setInitialState(Compostate<State, Action> initial);

    boolean somethingLeftToExplore();

    void expansionDone(Compostate<State, Action> first, HAction<Action> second, Compostate<State, Action> child);

    Pair<Compostate<State,Action>, HAction<Action>> getNextAction();

    ArrayList<Integer> getOrder();

    int getNextActionIndex();

    ActionWithFeatures<State, Action> removeFromFrontier(int idx);

    void filterFrontier();

    void notifyExpandingState(Compostate<State, Action> state, HAction<Action> action, Compostate<State, Action> child);

    void notifyStateIsNone(Compostate<State, Action> state);

    void notifyStateSetErrorOrGoal(Compostate<State, Action> state);

    void newState(Compostate<State, Action> state, Compostate<State, Action> parent);

    void notifyExpansionDidntFindAnything(Compostate<State, Action> parent, HAction<Action> action, Compostate<State, Action> child);

    boolean fullyExplored(Compostate<State, Action> state);

    boolean hasUncontrollableUnexplored(Compostate<State, Action> state);

    int getIndexOfStateAction(Pair<Compostate<State,Action>, HAction<Action>> actionState);

    void printFrontier();

    void initialize(Compostate<State, Action> state);

    ArrayList<ActionWithFeatures<State, Action>> getFrontier();

}
