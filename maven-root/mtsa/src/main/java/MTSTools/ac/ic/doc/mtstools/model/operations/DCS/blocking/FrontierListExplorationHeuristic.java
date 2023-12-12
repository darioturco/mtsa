package MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking;

import MTSTools.ac.ic.doc.commons.relations.Pair;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;

/** Exploration heuristic with minimal behavior */
public class FrontierListExplorationHeuristic<State, Action> implements ExplorationHeuristic<State, Action> {
    public LinkedList<Pair<Compostate<State, Action>, HAction<Action>>> explorationFrontier;
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

    public FrontierListExplorationHeuristic() {
        this.explorationFrontier = new LinkedList<>();
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

    public Pair<Compostate<State, Action>, HAction<Action>> getNextAction() {
        Pair<Compostate<State, Action>, HAction<Action>> stateAction = explorationFrontier.pop();

        while (!stateAction.getFirst().isStatus(Status.NONE)) {
            stateAction = explorationFrontier.pop();
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
        for(int i=0 ; i<explorationFrontier.size() ; i++){
            res.add(getIndexOfStateAction(explorationFrontier.get(i)));
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


    public void addTransitionsToFrontier(Compostate<State, Action> state) {
        updateState(state);
        for (HAction<Action> action : state.transitions){
            explorationFrontier.add(new Pair<>(state, action));
            actionsToExplore.add(new ActionWithFeatures<>(state,  action));
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


    public void notifyStateSetErrorOrGoal(Compostate<State, Action> state) {

    }

    public void newState(Compostate<State, Action> state, Compostate<State, Action> parent) {
        if(state.isStatus(Status.NONE))
            addTransitionsToFrontier(state);
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
