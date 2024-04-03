package MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking;

import MTSTools.ac.ic.doc.commons.collections.BidirectionalMap;
import MTSTools.ac.ic.doc.commons.collections.InitMap;
import MTSTools.ac.ic.doc.commons.relations.Pair;
import MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.abstraction.Abstraction;
import MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.abstraction.ReadyAbstraction;
import MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.abstraction.Recommendation;
import ai.onnxruntime.*;

import java.nio.FloatBuffer;
import java.util.*;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

import static java.util.Collections.emptyList;

public class RLExplorationHeuristic<State, Action> implements ExplorationHeuristic<State, Action> {

    /** Queue of states, the most promising state should be expanded first.
     *  Key:index of LTS with marked states -> Value: Queue for that color */
    public HashMap<Integer, Queue<Compostate<State, Action>>> frontiers;

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

    private final Logger logger = Logger.getLogger(RLExplorationHeuristic.class.getName());

    public ArrayList<ActionWithFeatures<State, Action>> actionsToExplore;
    public ArrayList<Compostate<State, Action>> allActionsToExplore;

    public int goals_found = 0;
    public int marked_states_found = 0;
    public int closed_potentially_winning_loops = 0;

    public Compostate<State, Action> lastExpandedTo = null;
    public Compostate<State, Action> lastExpandedFrom = null;
    public ActionWithFeatures<State, Action> lastExpandedStateAction = null;

    /** Name of the set of features used by the learner */
    public String featureGroup;     // TODO: convertirlo a enumerate

    public OrtEnvironment ortEnv;
    public OrtSession session;
    public OrtSession.SessionOptions opts;

    public int nfeatures;

    public DCSFeatures featureMaker;

    public RLExplorationHeuristic(
            DirectedControllerSynthesisBlocking<State,Action> dcs, String featureGroup) {

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
        this.featureGroup = featureGroup;
        this.actionsToExplore = new ArrayList<>();
        this.allActionsToExplore = new ArrayList<>();
        frontiers = new HashMap<>();
        abstractions = new HashMap<>();
        objectivesIndex = new ArrayList<>();
        objectivesIndex.addAll(dcs.guarantees.values());
        objectivesIndex.addAll(dcs.assumptions.values());
        currentTargetLTSIndex = dcs.guarantees.get(currentTargetColor); //get index of guarantee 0
        defaultTargets = buildDefaultTargets();
        currentObjectives = dcs.guarantees;

        for (Integer color : objectivesIndex) {
            abstractions.put(color, new ReadyAbstraction<>(color, dcs.ltss, defaultTargets.get(color), dcs.alphabet));
            frontiers.put(color, new PriorityQueue<>(new ReadyAbstraction.CompostateRanker<>(color)));
        }

        featureMaker = new DCSFeatures<>(featureGroup, this);
    }

    // this assumes updateOpen was called just before, so getNextState returns an action that was not previously explored
    public Pair<Compostate<State,Action>, HAction<Action>> getNextAction(boolean updateUnexploredTransaction) {
        assert(!frontiers.get(currentTargetLTSIndex).isEmpty());
        Compostate<State,Action> state = getNextState(currentTargetLTSIndex, updateUnexploredTransaction);
        Recommendation<Action> recommendation = state.nextRecommendation(currentTargetLTSIndex);
        HAction<Action> action = recommendation.getAction();
        return new Pair<>(state, action);
    }

    /** Load the features of all the actions to explore */
    public void computeFeatures(){
        actionsToExplore.parallelStream().forEach(ActionWithFeatures::updateFeatures);
    }

    public int getNextActionIndex() {
        computeFeatures();
        if(session == null){
            return 0;
        }

        float[][] availableActions = new float[actionsToExplore.size()][nfeatures];
        for(int i=0 ; i<actionsToExplore.size() ; i++){
            availableActions[i] = actionsToExplore.get(i).featureVector;
        }

        OnnxTensor t = null;
        OnnxTensor tRes = null;
        OrtSession.Result results = null;

        FloatBuffer values = null;
        try {
            t = OnnxTensor.createTensor(this.ortEnv, availableActions);
            results = session.run(Collections.singletonMap("X", t));
            tRes = (OnnxTensor)results.get(0);
            values = tRes.getFloatBuffer();
        } catch (OrtException e) {
            e.printStackTrace();
        }
        assert values != null;

        int best = 0;
        float bestValue = values.get();
        for(int i = 1; i < actionsToExplore.size(); i++){
            float v = values.get();
            if(v > bestValue){
                best = i;
                bestValue = v;
            }
        }

        t.close();
        tRes.close();
        results.close();
        return best;
    }

    public void loadModelFromPath(String modelPath) throws OrtException {
        if(modelPath.equals("")) {
            session = null;
        }else{
            ortEnv = OrtEnvironment.getEnvironment();
            opts = new OrtSession.SessionOptions();
            opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT);
            session = ortEnv.createSession(modelPath);
        }
    }

    public ArrayList<Integer> getOrder(){
        ArrayList<Integer> res = new ArrayList<>();
        removeNotLive(currentTargetLTSIndex);

        Queue<Compostate<State, Action>> frontierCopy = new LinkedList<>(frontiers.get(currentTargetLTSIndex));
        int frontierSize = frontierCopy.size();
        for(int i=0 ; i<frontierSize ; i++){
            Compostate<State, Action> state = frontierCopy.remove();
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

    public Compostate<State,Action> getNextState(Integer color, boolean updateUnexploredTransaction) {
        removeNotLive(color);
        Compostate<State,Action> state = frontiers.get(color).remove();
        state.inOpen.put(color, false);
        if(updateUnexploredTransaction)
            state.unexploredTransitions--;
        return state;
    }

    public boolean somethingLeftToExplore() {
        boolean aColorIsUnreachable = false;

        for (Integer it = 0; it< currentObjectives.size(); ++it) {
            Integer ltsIndex = currentObjectives.get((currentTargetColor + it)%(currentObjectives.size()));
            Queue<Compostate<State, Action>> frontier = frontiers.get(ltsIndex);
            updateFrontier(ltsIndex);
            removeNotLive(ltsIndex);
            if (frontier.isEmpty()){
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

    private void removeNotLive(Integer color) {
        while (!frontiers.get(color).isEmpty() && (
                !frontiers.get(color).peek().isStatus(Status.NONE) ||
                        fullyExplored(frontiers.get(color).peek()) ||
                        !frontiers.get(color).peek().isLive()
        )) {
            frontiers.get(color).remove();
        }
    }

    /** removes from frontier the states and transitions that were already explored until the next state recommendation
     * is one that was never explored */
    private void updateFrontier(Integer color){
        Queue<Compostate<State, Action>> frontier = frontiers.get(color);
        if(frontier.isEmpty()) return;
        Compostate<State,Action> state = null;

        while(true){
            while(state == null || fullyExplored(state) || !state.isLive() || state.cantWinColor(color) || state.peekRecommendation(color) == null){
                if(frontier.isEmpty()) return;
                state = getNextState(color, false);
            }
            assert(state.isEvaluated(color));

            Recommendation<Action> recommendation = state.peekRecommendation(color);
            assert(recommendation != null); // state.toString(.equals("[8, 10, 13, 10, 0]"))
            if(!state.getExploredActions().contains(recommendation.getAction())){
                addToFrontier(color, state);
                return;
            }else{
                while(state.peekRecommendation(color) != null &&
                        state.getExploredActions().contains(state.peekRecommendation(color).getAction())){
                    state.updateRecommendation(color);
                }
                if (state.peekRecommendation(color) != null) {
                    addToFrontier(color, state);
                }
            }
            state = null; //we dont want to check the same state again
        }
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

    public void addToAllFrontier(Compostate<State, Action> state) {
        for (Integer color : frontiers.keySet()) {
            addToFrontier(color, state);
        }
    }

    public boolean addToFrontier(Integer color, Compostate<State, Action> state) {
        if (state.isStatus(Status.NONE) && !fullyExplored(state) && !state.inOpen.get(color)) {
            state.inOpen.put(color, true);
            state.live = true;
            return frontiers.get(color).add(state);
        }
        return false;
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

    public void notifyClosedPotentiallyWinningLoop(Set<Compostate<State, Action>> loop) {
        closed_potentially_winning_loops ++;
    }

    public void setInitialState(Compostate<State, Action> state) {
        newState(state, null);
        addToAllFrontier(state);
    }

    public void notifyStateIsNone(Compostate<State, Action> state) {
        if(!fullyExplored(state))
            addToAllFrontier(state);
    }

    public void newState(Compostate<State, Action> state, Compostate<State, Action> parent) {
        if(parent != null){
            state.setTargets(parent.getTargets());
        }

        state.addTargets(state); //we always call addTargets, if the state is not marked, it will not be added to any list
        if(state.isMarked()) {
            marked_states_found++;
        }

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
            addTransitionsToFrontier(state, parent);
        }
    }

    public void updateState(Compostate<State, Action> state){
        state.unexploredTransitions = 0;
        state.uncontrollableUnexploredTransitions = 0;
        //state.actionsWithFeatures = new HashMap<>();
        for(HAction<Action> action : state.getTransitions()){
            List<State> childStates = state.dcs.getChildStates(state, action);

            state.actionChildStates.put(action, childStates);
            state.unexploredTransitions++;
            if(!action.isControllable()){
                state.uncontrollableUnexploredTransitions++;
            }

        }
    }

    public void addTransitionsToFrontier(Compostate<State, Action> state, Compostate<State, Action> parent) {
        updateState(state);
        for (HAction<Action> action : state.transitions) {
            actionsToExplore.add(new ActionWithFeatures<>(state, action, parent));
        }
    }

    public void notifyStateSetErrorOrGoal(Compostate<State, Action> state) {
        state.close();
        state.clearRecommendations();
        if(state.isStatus(Status.GOAL)){
            goals_found++;
        }
    }

    public void notifyExpandingState(Compostate<State, Action> parent, HAction<Action> action, Compostate<State, Action> state) {
        if(state.wasExpanded()){
            state.setTargets(parent.getTargets());
            state.addTargets(state); //we always call addTargets, if the state is not marked, it will not be added to any list
        }
    }

    public void expansionDone(Compostate<State, Action> state, HAction<Action> action, Compostate<State, Action> child) {
        dcs.instanceDomain.updateMatrixFeature(action, child);
        addToAllFrontier(state);
        addToAllFrontier(child);
        lastExpandedFrom = state;
        lastExpandedTo = child;
    }

    public void notifyExpansionDidntFindAnything(Compostate<State, Action> parent, HAction<Action> action, Compostate<State, Action> child) {
        if (!child.isLive() && !fullyExplored(child)) {
            addToAllFrontier(child);
        }
    }

    public boolean fullyExplored(Compostate<State, Action> state) {
        return (state.getExploredActions().size()) >= state.transitions.size();
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

    public void notify_end_synthesis(){
        try {
            session.close();
            ortEnv.close();
            opts.close();
        }catch (Exception e){System.out.println("Error trying to close the onnx runtime environment");}
    }

    public void printFrontier(){
        System.out.println("Frontier: ");
        for(int i=0 ; i<actionsToExplore.size() ; i++){
            System.out.println(i + ": " + actionsToExplore.get(i).toString());
        }
        //for(ActionWithFeatures<State, Action> stateAction : actionsToExplore){
        //    System.out.println(stateAction.toString());
        //}
    }

    public void initialize(Compostate<State, Action> state) {
        state.unexploredTransitions = state.transitions.size();
        state.close();
        state.actionChildStates = new HashMap<>();
        for (Integer color : frontiers.keySet()) {
            state.inOpen.put(color, false);
        }

        state.targets = emptyList();
        state.vertices = new HashSet<>();
        state.edges = new BidirectionalMap<>();
        state.gapCache = new InitMap<>(HashMap.class);
        state.readyInLTS = new InitMap<>(HashSet.class);
    }
}
