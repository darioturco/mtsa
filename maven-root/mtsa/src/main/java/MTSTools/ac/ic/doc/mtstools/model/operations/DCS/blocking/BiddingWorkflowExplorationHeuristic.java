package MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking;

import MTSTools.ac.ic.doc.commons.relations.Pair;

import java.util.*;

// Heuristica incompleta, al parecer no es tan facil como crei el hacer una heuristica a mano para BW
public class BiddingWorkflowExplorationHeuristic<State, Action> implements ExplorationHeuristic<State, Action> {
    public LinkedList<Pair<Compostate<State, Action>, HAction<Action>>> explorationFrontier;
    public ArrayList<ActionWithFeatures<State, Action>> actionsToExplore;

    //// Update this values
    public int goals_found = 0;
    public int marked_states_found = 0;
    public int closed_potentially_winning_loops = 0;

    public Compostate<State, Action> lastExpandedTo = null;
    public Compostate<State, Action> lastExpandedFrom = null;
    public ActionWithFeatures<State, Action> lastExpandedStateAction = null;

    public DirectedControllerSynthesisBlocking<State,Action> dcs;

    public int n;
    public int k;

    public boolean returnTurnOffFluent;
    public int crew;
    public int times_rejected;
    public int[] accepted;
    public boolean[] rejected;
    public boolean[] assigned;
    public int actual_step;
    public String search_for;
    public boolean approved;
    public List<Integer> assign_queue;



    public BiddingWorkflowExplorationHeuristic(DirectedControllerSynthesisBlocking<State,Action> dcs) {
        this.explorationFrontier = new LinkedList<>();
        this.actionsToExplore = new ArrayList<>();
        this.dcs = dcs;
        this.n = dcs.n;
        this.k = dcs.k;

        this.crew = 0;
        this.assigned = new boolean[this.n];
        this.accepted = new int[this.n];
        this.rejected = new boolean[this.n];
        this.approved = false;
        this.returnTurnOffFluent = false;





        // No se si utilizo
        this.actual_step = 1;
        this.search_for = "assign";

        this.assign_queue = new ArrayList<>();

    }

    public void setLastExpandedStateAction(ActionWithFeatures<State, Action> stateAction) {
        this.lastExpandedStateAction = stateAction;
    }

    public void setInitialState(Compostate<State, Action> initial) {
    }

    public boolean somethingLeftToExplore() {
        return !explorationFrontier.isEmpty();
    }

    public ArrayList<ActionWithFeatures<State, Action>> getFrontier() {
        return actionsToExplore;
    }

    public int frontierSize() {
        return actionsToExplore.size();
    }

    public void expansionDone(Compostate<State, Action> state, HAction<Action> action, Compostate<State, Action> child) {
        lastExpandedTo = child;
        lastExpandedFrom = state;
        actual_step += 1;
    }

    public Pair<Compostate<State, Action>, HAction<Action>> getNextAction(boolean updateUnexploredTransaction) {
        Pair<Compostate<State, Action>, HAction<Action>> stateAction = explorationFrontier.pop();

        while (!stateAction.getFirst().isStatus(Status.NONE)) {
            stateAction = explorationFrontier.pop();
        }

        stateAction.getFirst().unexploredTransitions--;
        if (!stateAction.getSecond().isControllable())
            stateAction.getFirst().uncontrollableUnexploredTransitions--;
        return stateAction;
    }

    public boolean[] getAssingRow(Compostate<State, Action> state){
        return state.customFeaturesMatrix[0];
    }

    public void setAssing(Compostate<State, Action> state, int crew, boolean value){
        state.customFeaturesMatrix[0][crew] = value;
    }

    public boolean isAssigned(Compostate<State, Action> state, int crew){
        return state.customFeaturesMatrix[0][crew];
    }

    public boolean[] getAcceptedRow(Compostate<State, Action> state){
        return state.customFeaturesMatrix[1];
    }

    public boolean isAccepted(Compostate<State, Action> state, int crew){
        return state.customFeaturesMatrix[1][crew];
    }

    public void setAccepted(Compostate<State, Action> state, int crew, boolean value){
        state.customFeaturesMatrix[1][crew] = value;
    }

    public boolean all_accepted(Compostate<State, Action> state){
        boolean[] accepted = getAcceptedRow(state);
        for(int i =0 ; i<n ; i++){
            if(!accepted[i])
                return false;
        }
        return true;
    }

    public boolean canExpandAcceptOrReject(){
        for(int i=0;i<actionsToExplore.size();i++){
            String action = actionsToExplore.get(i).action.toString();
            if(action.contains("accept") || action.contains("reject")){
                if(actionsToExplore.get(i).entity == crew) {
                    return true;
                }
            }

        }
        return false;
    }

    public int getNextActionIndex() {
        // La file 0 de la matriz de customFeatures tiene si cada entidad fue asignada en ese compostate.
        // La fila 1 de la matriz de customFeatures tiene si cada entidad fue aceptada en ese compostate.
        int h = 7;
        for(int i=0;i<actionsToExplore.size();i++){
            ActionWithFeatures<State, Action> actionWithFeature = actionsToExplore.get(i);
            Compostate<State, Action> state = actionWithFeature.state;
            String action = actionWithFeature.action.toString();
            int entity = actionWithFeature.entity;
            //int index = actionWithFeature.index;

            if(action.contains("accept") && actionWithFeature.expansionStep == 14){
                int g = 56;
            }


            if(entity == crew && state.actualEntity == entity){
                if(!isAssigned(state, entity) && action.contains("assign")){
                    return i;
                }else if(action.contains("accept")){
                    return i;
                }else if(action.contains("reject")){
                    return i;
                }
            }

            if(!approved && all_accepted(state) && action.contains("approve")){
                approved = true;
                return i;
            }

            // Refuse tengo que expanditlo siempre y cuando no lleve a un deadlock
            if(action.contains("refuse") && !goToInvalid(actionWithFeature)) {
                return i;
            }

            if(!returnTurnOffFluent && action.contains("assign") && activeFluent(state)){
                returnTurnOffFluent = true;
                return i;
            }
        }

        return 0;
    }


    public void notifyExpandingState(Compostate<State, Action> state, HAction<Action> action, Compostate<State, Action> child) {
        String actionStr = action.toString();
        child.customFeaturesMatrix[1] = Arrays.copyOf(state.customFeaturesMatrix[1], dcs.n);
        int entity = ActionWithFeatures.getNumber(actionStr, 1);
        if(actionStr.contains("assign")){
            setAssing(child, entity, true);
            child.actualEntity = state.actualEntity;
        }else if(actionStr.contains("accept")){
            accepted[entity] += 1;
            child.customFeaturesMatrix[1][entity] = true;
            if(accepted[entity] >= k && rejected[entity] && !canExpandAcceptOrReject()){
                crew += 1;
            }

            child.actualEntity = crew + 1;
            setAccepted(child, entity, true);
        }else if(actionStr.contains("reject")){
            int index = ActionWithFeatures.getNumber(actionStr, 2);
            if(index == dcs.k){
                child.actualEntity = crew + 1;
                rejected[entity] = true;
            }else{
                child.actualEntity = crew;
                setAssing(child, entity, false);
            }
            if(accepted[entity] >= k && rejected[entity] && !canExpandAcceptOrReject()){
                crew += 1;
            }
        }

    }

    public boolean activeFluent(Compostate<State, Action> state){
        return state.states.get(state.states.size()-1).equals(1L);
    }

    public boolean goToInvalid(ActionWithFeatures<State, Action> actionWithFeature){
        for(State s : actionWithFeature.childStates){
            if(s.equals(-1L)){
                return true;
            }
        }
        return false;
    }

    public ArrayList<Integer> getOrder(){
        ArrayList<Integer> res = new ArrayList<>();
        for(int i=0 ; i<actionsToExplore.size() ; i++){
            res.add(i);
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

            state.actionChildStates.put(action, childStates);
            state.unexploredTransitions ++;
            if(!action.isControllable()){
                state.uncontrollableUnexploredTransitions++;
            }
        }
    }

    public void addTransitionsToFrontier(Compostate<State, Action> state, Compostate<State, Action> parent) {
        updateState(state);
        for (HAction<Action> action : state.transitions){
            explorationFrontier.add(new Pair<>(state, action));
            actionsToExplore.add(new ActionWithFeatures<>(state,  action, parent));
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

    public void notifyStateIsNone(Compostate<State, Action> state) {

    }

    public void notifyClosedPotentiallyWinningLoop(Set<Compostate<State, Action>> loop) {
        closed_potentially_winning_loops++;
    }

    public void notifyStateSetErrorOrGoal(Compostate<State, Action> state) {

    }

    public void notifyExpansionDidntFindAnything(Compostate<State, Action> parent, HAction<Action> action, Compostate<State, Action> child) {

    }

    public void newState(Compostate<State, Action> state, Compostate<State, Action> parent) {
        if(state.isStatus(Status.NONE))
            addTransitionsToFrontier(state, parent);
    }

    public boolean fullyExplored(Compostate<State, Action> state) {
        return state.unexploredTransitions == 0;
    }

    public boolean hasUncontrollableUnexplored(Compostate<State, Action> state) {
        return state.uncontrollableUnexploredTransitions > 0;
    }

    public void notify_end_synthesis(){}

    public void printFrontier(){
        System.out.println("Frontier: ");
        for(int i = 0 ; i<actionsToExplore.size() ; i++){
            ActionWithFeatures<State, Action> stateAction = actionsToExplore.get(i);
            //System.out.println(i + ": " + stateAction.state.toString() + " | " + stateAction.action.toString());
            System.out.println(i + ": " + stateAction.toString());
        }
    }

    public void initialize(Compostate<State, Action> state) {
        state.unexploredTransitions = state.transitions.size();
        state.actionChildStates = new HashMap<>();

        state.uncontrollableUnexploredTransitions = 0;
        for(HAction<Action> action : state.transitions)
            if(!action.isControllable()) state.uncontrollableUnexploredTransitions ++;
        state.uncontrollableTransitions = state.uncontrollableUnexploredTransitions;

        state.actualEntity = 0;

        state.entityPositions = new int[n];
        state.customFeaturesMatrix = new boolean[2][n];

        if(state.parents.size() == 0){
            for (Pair<HAction<Action>, Compostate<State, Action>> ancestorActionAndState : state.parents) {
                Compostate<State, Action> parent = ancestorActionAndState.getSecond();
                state.customFeaturesMatrix[0] = Arrays.copyOf(parent.customFeaturesMatrix[0], dcs.n);
                state.customFeaturesMatrix[1] = Arrays.copyOf(parent.customFeaturesMatrix[1], dcs.n);

                state.entityPositions = Arrays.copyOf(parent.entityPositions, dcs.n);
            }
        }
    }
}
