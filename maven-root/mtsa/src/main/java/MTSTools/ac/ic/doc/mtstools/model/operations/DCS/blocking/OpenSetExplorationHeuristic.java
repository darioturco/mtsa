package MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking;

import MTSTools.ac.ic.doc.commons.collections.BidirectionalMap;
import MTSTools.ac.ic.doc.commons.collections.InitMap;
import MTSTools.ac.ic.doc.commons.collections.QueueSet;
import MTSTools.ac.ic.doc.commons.relations.Pair;
import MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.abstraction.HeuristicMode;
import MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.abstraction.*;

import java.util.*;

import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

import static java.util.Collections.emptyList;

public class OpenSetExplorationHeuristic<State, Action> implements ExplorationHeuristic<State, Action> {

    /** Queue of open states, the most promising state should be expanded first.
     *  Key:index of LTS with marked states -> Value: Queue for that color */
    public HashMap<Integer, Queue<Compostate<State, Action>>> opens;

    /** Abstraction used to rank the transitions from a state.
    *  Key:index of LTS with marked states -> Value: Queue for that color */
    public HashMap<Integer, Abstraction<State,Action>> abstractions;

    public DirectedControllerSynthesisBlocking<State,Action> dcs;

    /** Contains the marked states per color per LTS. */
    /** FIXME: a diferencia del 'markedStates', este tiene
     * en cuenta los distintos tipos de colores, y considera
     * completamente pintados de un color "c" a todos los LTSs
     * que no representen la guarantee "c".
     * Reconsiderar fuertemente renombrar esto y 'markedStates'.
     * */
    public Map<Integer, List<Set<State>>> defaultTargets;

    /** the target we are currently aiming the exploration to */
    public Integer currentTargetColor = 0;
    public Integer currentTargetLTSIndex;

    public Set<State> allColors = null;

    /** has the LTSIndex of all guarantees and assumptions */
    public List<Integer> objectivesIndex;

    /** Initially dcs.guarantees. Updated to dcs.assumptions when it is clear there is no way to win by guarantees*/
    public HashMap<Integer, Integer> currentObjectives;

    /** save the abstraction mode in case we need to reset for assumptions*/
    public HeuristicMode mode;

    private final Logger logger = Logger.getLogger(OpenSetExplorationHeuristic.class.getName());

    public ArrayList<ActionWithFeatures<State, Action>> actionsToExplore;
    public ArrayList<Compostate<State, Action>> allActionsToExplore;

    //// Update this values
    public int goals_found = 0;
    public int marked_states_found = 0;
    public int closed_potentially_winning_loops = 0;

    public Compostate<State, Action> lastExpandedTo = null;
    public Compostate<State, Action> lastExpandedFrom = null;
    public ActionWithFeatures<State, Action> lastExpandedStateAction = null;
    
    public OpenSetExplorationHeuristic(
            DirectedControllerSynthesisBlocking<State,Action> dcs,
            HeuristicMode mode) {

        //LOGGER
        //set logger formatter for more control over logs
        logger.setUseParentHandlers(false);
        ConsoleHandler handler = new ConsoleHandler();
        LogFormatter formatter = new LogFormatter();
        handler.setFormatter(formatter);
        logger.addHandler(handler);
        //Sets the minimum level required for messages to be logged
        // SEVERE > WARNING > INFO > CONFIG > FINE > FINER > FINEST
        // we use fine/finer/finest to log exploration info
        logger.setLevel(Level.FINEST);
        handler.setLevel(Level.FINEST);
        
        this.dcs = dcs;
        this.mode = mode;
        this.actionsToExplore = new ArrayList<>();
        this.allActionsToExplore = new ArrayList<>();
        opens = new HashMap<>();
        abstractions = new HashMap<>();
        objectivesIndex = new ArrayList<>();
        objectivesIndex.addAll(dcs.guarantees.values());
        objectivesIndex.addAll(dcs.assumptions.values());
        currentTargetLTSIndex = dcs.guarantees.get(currentTargetColor); //get index of guarantee 0
        defaultTargets = buildDefaultTargets();
        currentObjectives = dcs.guarantees;

        //FIXME, this is only done here until it can be chosen from the FSP instead of hardcoded
        switch (mode){
            case Monotonic:
                for (Integer color : objectivesIndex) {
                    //MA still isn't updated to handle multiple objectives
                    abstractions.put(color, new MonotonicAbstraction<State, Action>(/*color,*/ dcs.ltss, defaultTargets.get(color), dcs.base, dcs.alphabet));
                    opens.put(color, new PriorityQueue<>(new DefaultCompostateRanker<>()));
                }
                break;
            case Ready:
                for (Integer color : objectivesIndex) {
                    abstractions.put(color, new ReadyAbstraction<>(color, dcs.ltss, defaultTargets.get(color), dcs.alphabet));
                    opens.put(color, new PriorityQueue<>(new ReadyAbstraction.CompostateRanker<State, Action>(color)));
                }
                break;
            default:
                logger.severe("Mode desconocido: " + mode);
                break;
        }

    }

    /*this assumes updateOpen was called just before, so getNextState returns an action that was not previously explored*/
    public Pair<Compostate<State,Action>, HAction<Action>> getNextAction() {
        assert(!opens.get(currentTargetLTSIndex).isEmpty());
        Compostate<State,Action> state = getNextState(currentTargetLTSIndex);
        Recommendation<Action> recommendation = state.nextRecommendation(currentTargetLTSIndex);
        HAction<Action> action = recommendation.getAction();

        //assert(!state.getExploredActions().contains(action));
        return new Pair<>(state, action);
    }

    public int getNextActionIndex() {
        int res = getIndexOfStateAction(getNextAction());
        while(res == -1){
            res = getIndexOfStateAction(getNextAction());
        }
        return res;
    }

    public ArrayList<Integer> getOrder(){
        ArrayList<Integer> res = new ArrayList<>();

        Queue<Compostate<State, Action>> openCopy = new LinkedList<>(opens.get(currentTargetLTSIndex));
        int openSize = openCopy.size();
        for(int i=0 ; i<openSize ; i++){
            Compostate<State, Action> state = openCopy.remove();
            List<Recommendation<Action>> recommendations = state.recommendations.get(currentTargetLTSIndex);

            if(recommendations != null){
                for(Recommendation<Action> recommendation : recommendations){
                    HAction<Action> action = recommendation.getAction();
                    int index = getIndexOfStateAction(new Pair<>(state, action));
                    if (index != -1 && !res.contains(index)) {
                        res.add(index);
                    }
                }
            }
        }

        // Add all the transition that are not in res
        for(int i=0 ; i<actionsToExplore.size() ; i++){
            if(!res.contains(i)){
                res.add(i);
            }
        }

        return res;
    }

    public ActionWithFeatures<State, Action> removeFromFrontier(int idx) {
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

    public Compostate<State,Action> getNextState(Integer color) {
        Compostate<State,Action> state = opens.get(color).remove();
        state.inOpen.put(color, false);
        return state;
    }

    public boolean somethingLeftToExplore() {
        boolean aColorIsUnreachable = false;

        for (Integer it = 0; it< currentObjectives.size(); ++it) {
            Integer ltsIndex = currentObjectives.get((currentTargetColor + it)%(currentObjectives.size()));
            Queue<Compostate<State, Action>> open = opens.get(ltsIndex);
            updateOpen(ltsIndex);
            if (open.isEmpty()){
                aColorIsUnreachable = true;
                break;
            }
        }
        if (!aColorIsUnreachable){
            return true;
        }

        //there is a color that we can never reach
        if(currentObjectives == dcs.guarantees){
            System.out.println("It is not possible to win by fulfilling guarantees, will attempt to prove the " +
                    "assumptions false");
            currentObjectives = dcs.assumptions;
            currentTargetColor = 0;
            currentTargetLTSIndex = dcs.assumptions.get(currentTargetColor); //get index of assumption 0
            return somethingLeftToExplore();
        }
        return false;
    }

    public void setLastExpandedStateAction(ActionWithFeatures<State, Action> stateAction){
        this.lastExpandedStateAction = stateAction;
    }

    public ArrayList<ActionWithFeatures<State, Action>> getFrontier(){return actionsToExplore;}
    public int frontierSize(){
        return actionsToExplore.size();
    }

    /** removes from open the states and transitions that were already explored until the next state recommendation
     * is one that was never explored */
    private void updateOpen(Integer color){
        Queue<Compostate<State, Action>> open = opens.get(color);
        if(open.isEmpty()) return;
        Compostate<State,Action> state = null;

        while(true){
            while(state == null || fullyExplored(state) || !state.isLive() || state.cantWinColor(color)){
                if(open.isEmpty()) return;
                state = getNextState(color);
            }
            assert(state.isEvaluated(color));

            Recommendation<Action> recommendation = state.peekRecommendation(color);
            assert(recommendation != null);
            if(!state.getExploredActions().contains(recommendation.getAction())){
                addToOpen(color, state);
                return;
            }else{
                while(state.peekRecommendation(color) != null &&
                        state.getExploredActions().contains(state.peekRecommendation(color).getAction())){
                    state.updateRecommendation(color);
                }
                if (state.peekRecommendation(color) != null) {
                    addToOpen(color, state);
                } else {
                    System.out.println("CHECK! State was actually fullyExplored");
                }
            }
            state = null; //we dont want to check the same state again
        }
    }

    /** Adds this state to the open queue (reopening it if was previously closed). */
    public boolean open(Compostate<State,Action> state) {
        // System.err.println("opening" + state);
        boolean result = false;
        state.live = true;
        // TODO: considerar agregarlo solamente al open del color que se está explorando ahora.
        for (Integer color : opens.keySet()) {
            if (!state.inOpen.get(color)) {
                if (!state.hasStatusChild(Status.NONE)) {
                    result = addToOpen(color, state);
                } else { // we are reopening a state, thus we reestablish it's exploredChildren instead
                    for (Pair<HAction<Action>,Compostate<State, Action>> transition : state.getExploredChildren()) {
                        Compostate<State, Action> child = transition.getSecond();
                        if (!child.isLive() && child.isStatus(Status.NONE) && !fullyExplored(child)) // !isGoal(child)
                            result |= open(child);
                    }
                    if (!result || state.isControlled()){
                        result = addToOpen(color, state);
                    }
                }
            }
        }

        return result;
    }

    private Map<Integer, List<Set<State>>> buildDefaultTargets() {
        var result = new HashMap<Integer, List<Set<State>>>();
        for (Integer objectiveLTSIndex : objectivesIndex) {
            List<Set<State>> coloredPerLts = new ArrayList<>();
            for (int ltsIndex = 0; ltsIndex < dcs.ltssSize; ltsIndex++) {
                Set<State> colored;
                if (objectiveLTSIndex == ltsIndex) {
                    colored = dcs.markedStates.get(ltsIndex);
                } else {
                    // All colored if it's not the guarantee's LTS
                    colored = dcs.ltss.get(ltsIndex).getStates();
                }
                coloredPerLts.add(colored);
            }
            result.put(objectiveLTSIndex, coloredPerLts);
        }
        return result;
    }
    
    public boolean addToOpen(Integer color, Compostate<State, Action> state) {
        state.inOpen.put(color, true);
        return opens.get(color).add(state);
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

    public void setInitialState(Compostate<State, Action> state) {
        open(state);
        newState(state, null);
    }

    public void notifyStateIsNone(Compostate<State, Action> state) {
        if(!fullyExplored(state))
            open(state);
    }

    public void newState(Compostate<State, Action> state, Compostate<State, Action> parent) {
        if(parent != null){
            state.setTargets(parent.getTargets());
        }
        state.addTargets(state); //we always call addTargets, if the state is not marked, it will not be added to any list

        for (Abstraction<State,Action> abstraction : abstractions.values()) {
            abstraction.eval(state);
        }
        state.clearRA();
        if(currentObjectives == dcs.guarantees){
            for (Integer color : currentObjectives.values()){
                if (state.peekRecommendation(color) == null){
                    //the state has no way of reaching a marked state for this color of guarantee, thus it can only win by assumptions
                    state.markLoserForGuarantees();
                    break;
                }
            }
        }

        //the new state is explored next, if it is marked with the current color, we look for the next color in the future
        if(currentObjectives.size()>1){
            Set<Integer> marked = new HashSet<>();
            if(currentObjectives == dcs.guarantees){
                for(Integer i : state.markedByGuarantee){
                    marked.add(currentObjectives.get(i));
                }
            }else{
                for(Integer i : state.markedByAssumption){
                    marked.add(currentObjectives.get(i));
                }
            }
            if(marked.size() < currentObjectives.size()){//if this state has all the colors, there is no point in looking for the next one
                while(marked.contains(currentTargetLTSIndex)){
                    currentTargetColor = (currentTargetColor + 1)%(currentObjectives.size());
                    currentTargetLTSIndex = currentObjectives.get(currentTargetColor);
                }
            }
        }

        if (!allActionsToExplore.contains(state) && state.isStatus(Status.NONE)){
            allActionsToExplore.add(state);
            addTransitionsToFrontier(state);
        }
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
        for (HAction<Action> action : state.transitions) {
            actionsToExplore.add(new ActionWithFeatures<>(state, action));
        }
    }

    public void notifyStateSetErrorOrGoal(Compostate<State, Action> state) {
        state.live = false;
        state.clearRecommendations();
    }

    public void notifyExpandingState(Compostate<State, Action> parent, HAction<Action> action, Compostate<State, Action> state) {
        if(state.wasExpanded()){ // todo: understand this, i am copying the behavior of the code pre refactor
            state.setTargets(parent.getTargets());
            state.addTargets(state); //we always call addTargets, if the state is not marked, it will not be added to any list
        }
    }

    public void expansionDone(Compostate<State, Action> state, HAction<Action> action, Compostate<State, Action> child) {
        if (state.isControlled() && state.isStatus(Status.NONE) && !fullyExplored(state)) {
            open(state);
        }
        lastExpandedTo = child;
        lastExpandedFrom = state;
    }

    public void notifyExpansionDidntFindAnything(Compostate<State, Action> parent, HAction<Action> action, Compostate<State, Action> child) {
        if (!child.isLive() && !fullyExplored(child)) {
            open(child);
        }
    }

    public void notifyClosedPotentiallyWinningLoop(Set<Compostate<State, Action>> loop) {
        closed_potentially_winning_loops++;
    }

    public boolean fullyExplored(Compostate<State, Action> state) {
        return (state.getExploredActions().size() + state.getDiscardedActions().size()) >= state.transitions.size();
        //return (state.getExploredActions().size()) >= state.transitions.size();
        //puede ser que este cambio rompa casos si recommendation era null antes de ver todas las transiciones
    }
    
    public boolean hasUncontrollableUnexplored(Compostate<State, Action> state) {
        if (state.isControlled()) return false;
        return state.getExploredActions().size() < state.transitions.size();
        //uses that we remove controllable transitions from mixed states in buildTransitions
        //puede ser que este cambio rompa casos si recommendation era null antes de ver todas las transiciones
        //return state.recommendation != null && !state.recommendation.getAction().isControllable();
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

    public void printFrontier(){
        System.out.println("Frontier: ");
        for(ActionWithFeatures<State, Action> stateAction : actionsToExplore){
            System.out.println(new StringBuilder(stateAction.state.toString() + " | " + stateAction.action.toString()));
        }
    }

    public void initialize(Compostate<State, Action> state) {
        state.live = false;
        state.actionChildStates = new HashMap<>();
        for (Integer color : opens.keySet()) {
            state.inOpen.put(color, false);
        }

        state.targets = emptyList();
        state.vertices = new HashSet<>();
        state.edges = new BidirectionalMap<>();
        state.gapCache = new InitMap<>(HashMap.class);
        state.readyInLTS = new InitMap<>(HashSet.class);
    }
}
