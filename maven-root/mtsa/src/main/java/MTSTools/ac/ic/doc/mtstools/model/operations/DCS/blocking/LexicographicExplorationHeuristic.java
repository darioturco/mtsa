package MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking;

import MTSTools.ac.ic.doc.commons.relations.Pair;
import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.*;

public class LexicographicExplorationHeuristic<State, Action> implements ExplorationHeuristic<State, Action> {
    public Queue<Pair<Compostate<State, Action>, HAction<Action>>> explorationFrontier;
    public ArrayList<ActionWithFeatures<State, Action>> actionsToExplore;

    //// Update this values
    public int goals_found = 0;
    public int marked_states_found = 0;
    public int closed_potentially_winning_loops = 0;

    public Compostate<State, Action> lastExpandedTo = null;
    public Compostate<State, Action> lastExpandedFrom = null;
    public ActionWithFeatures<State, Action> lastExpandedStateAction = null;

    public void setLastExpandedStateAction(ActionWithFeatures<State, Action> stateAction){
        this.lastExpandedStateAction = stateAction;
    }

    public static class LexicographicCompostateRanker<State, Action> implements Comparator<Pair<Compostate<State, Action>, HAction<Action>>> {
        @Override
        public int compare(Pair<Compostate<State, Action>, HAction<Action>> o1, Pair<Compostate<State, Action>, HAction<Action>> o2) {
            return o1.getSecond().getAction().toString().compareTo(o2.getSecond().getAction().toString());
        }
    }

    public LexicographicExplorationHeuristic() {
        this.explorationFrontier = new PriorityQueue<Pair<Compostate<State, Action>, HAction<Action>>>(new LexicographicCompostateRanker<>());
        this.actionsToExplore = new ArrayList<>();
    }

    public void setInitialState(Compostate<State, Action> initial) {

    }

    public boolean somethingLeftToExplore() {
        return !explorationFrontier.isEmpty();
    }

    public ArrayList<ActionWithFeatures<State, Action>> getFrontier(){return actionsToExplore;}
    public int frontierSize(){
        return actionsToExplore.size();
    }

    public void expansionDone(Compostate<State, Action> state, HAction<Action> action, Compostate<State, Action> child) {
        lastExpandedTo = child;
        lastExpandedFrom = state;
    }

    public Pair<Compostate<State, Action>, HAction<Action>> getNextAction(boolean updateUnexploredTransaction) {
        Pair<Compostate<State, Action>, HAction<Action>> stateAction = explorationFrontier.remove();
        while (!stateAction.getFirst().isStatus(Status.NONE)) {
            stateAction = explorationFrontier.remove();
        }

        stateAction.getFirst().unexploredTransitions--;
        if(!stateAction.getSecond().isControllable())
            stateAction.getFirst().uncontrollableUnexploredTransitions--;
        return stateAction;
    }

    public int getNextActionIndex() {
        return getIndexOfStateAction(explorationFrontier.peek());
    }

    public ArrayList<Integer> getOrder(){
        ArrayList<Integer> res = new ArrayList<>();
        Queue<Pair<Compostate<State, Action>, HAction<Action>>> explorationCopy = new LinkedList<>(explorationFrontier);
        int s = explorationCopy.size();
        for(int i=0; i<s; i++){
            res.add(getIndexOfStateAction(explorationCopy.remove()));
        }
        return res;
    }

    public ActionWithFeatures<State, Action> removeFromFrontier(int idx) {
        explorationFrontier.remove(actionsToExplore.get(idx).toPair());
        ActionWithFeatures<State, Action> stateAction = efficientRemove(idx);

        stateAction.state.unexploredTransitions--;
        if(!stateAction.action.isControllable())
            stateAction.state.uncontrollableUnexploredTransitions--;
        return stateAction;
    }

    private ActionWithFeatures<State, Action> efficientRemove(int idx) {
        // removing last element is more efficient
        ActionWithFeatures<State, Action> stateAction = actionsToExplore.get(idx);
        actionsToExplore.set(idx, actionsToExplore.get(actionsToExplore.size()-1));
        actionsToExplore.remove(actionsToExplore.size()-1);
        return stateAction;
    }

    public int getIndexOfStateAction(Pair<Compostate<State, Action>, HAction<Action>> pairStateAction){
        Compostate<State, Action> state = pairStateAction.getFirst();
        HAction<Action> action = pairStateAction.getSecond();
        int idx = 0;
        for(ActionWithFeatures<State, Action> actionState : actionsToExplore){
            if(actionState.action.toString().equals(action.toString()) && actionState.state.toString().equals(state.toString())){
                return idx;
            }
            idx++;
        }
        return -1;
    }

    public void updateState(Compostate<State, Action> state){
        state.unexploredTransitions = 0;
        state.uncontrollableUnexploredTransitions = 0;
        //state.actionsWithFeatures = new HashMap<>();
        for(HAction<Action> action : state.getTransitions()){
            List<State> childStates = state.dcs.getChildStates(state, action);
            // assertTrue(!dcs.dcs.canReachMarkedFrom(childStates) == state.getEstimate(action).isConflict());
            if(state.dcs.canReachMarkedFrom(childStates)) {
                state.actionChildStates.put(action, childStates);
                state.unexploredTransitions ++;
                if(!action.isControllable()){
                    state.uncontrollableUnexploredTransitions++;
                }
            } else {
                // action is uncontrollable since we have removed controllable conflicts
                state.heuristicStronglySuggestsIsError = true;
                return;
            }
        }
    }

    public void addTransitionsToFrontier(Compostate<State, Action> state, Compostate<State, Action> parent) {
        updateState(state);
        for (HAction<Action> action : state.transitions) {
            explorationFrontier.add(new Pair<>(state, action));
            actionsToExplore.add(new ActionWithFeatures<>(state, action, parent));
        }
    }

    public void filterFrontier(){
        for(int i = 0; i < actionsToExplore.size();) {
            if (!actionsToExplore.get(i).state.isStatus(Status.NONE)) {
                removeFromFrontier(i);
            } else {
                i++;
            }
        }
    }

    public void notifyExpandingState(Compostate<State, Action> state, HAction<Action> action, Compostate<State, Action> child) {
    }

    public void notifyStateIsNone(Compostate<State, Action> state) {
    }

    public void notifyClosedPotentiallyWinningLoop(Set<Compostate<State, Action>> loop) {
        closed_potentially_winning_loops++;
    }

    public void notifyStateSetErrorOrGoal(Compostate<State, Action> state) {
    }

    public void newState(Compostate<State, Action> state, Compostate<State, Action> parent) {
        if(state.isStatus(Status.NONE))
            addTransitionsToFrontier(state, parent);
    }

    public void notifyExpansionDidntFindAnything(Compostate<State, Action> parent, HAction<Action> action, Compostate<State, Action> child) {
    }

    public boolean fullyExplored(Compostate<State, Action> state) {
        return state.unexploredTransitions == 0;
    }

    public boolean hasUncontrollableUnexplored(Compostate<State, Action> state) {
        return state.uncontrollableUnexploredTransitions > 0;
    }

    public void printFrontier(){
        System.out.println("Frontier: ");
        for(ActionWithFeatures<State, Action> stateAction : actionsToExplore){
            System.out.println(new StringBuilder(stateAction.state.toString() + " | " + stateAction.action.toString()));
        }
    }

    public void initialize(Compostate<State, Action> state) {
        state.unexploredTransitions = state.transitions.size();
        state.actionChildStates = new HashMap<>();

        state.uncontrollableUnexploredTransitions = 0;
        for(HAction<Action> action : state.transitions)
            if(!action.isControllable()) state.uncontrollableUnexploredTransitions ++;
        state.uncontrollableTransitions = state.uncontrollableUnexploredTransitions;
    }
}
