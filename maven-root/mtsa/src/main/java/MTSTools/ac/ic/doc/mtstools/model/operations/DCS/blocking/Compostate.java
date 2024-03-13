package MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking;

import MTSTools.ac.ic.doc.commons.collections.BidirectionalMap;
import MTSTools.ac.ic.doc.commons.relations.BinaryRelation;
import MTSTools.ac.ic.doc.commons.relations.BinaryRelationImpl;
import MTSTools.ac.ic.doc.commons.relations.Pair;
import MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.abstraction.HEstimate;
import MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.abstraction.Ranker;
import MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.abstraction.Recommendation;

import java.util.*;

import static java.util.Collections.emptyList;
import static java.util.Collections.emptySet;
import static org.junit.Assert.assertSame;

/** This class represents a state in the parallel composition of the LTSs
 *  in the environment. These states are used to build the fragment of the
 *  environment required to reach the goal on-the-fly. The controller for
 *  the given problem can then be extracted directly from the partial
 *  construction achieved by using these states. */
public class Compostate<State, Action> implements Comparable<Compostate<State, Action>> {
    public final DirectedControllerSynthesisBlocking<State, Action> dcs;
    /** States by each LTS in the environment that conform this state. */
    public final List<State> states; // Note: should be a set of lists for non-deterministic LTSs

    /** Indicates whether this state is a goal (1) or an error (-1) or not yet known (0). */
    public Status status;

    /** The real distance to the goal state from this state. */
    public int distance;

    /** Depth at which this state has been expanded. */
    public int depth;

    /** Indicates whether the state is controlled or not. */
    public boolean controlled;

    /** Indicates by guarantee number what guarantees the compostate fulfills */
    public final Set<Integer> markedByGuarantee;

    /** Indicates by assumptions number what assumptions the compostate negates */
    public final Set<Integer> markedByAssumption;

    private Integer loopID;
    private Integer bestDistToWinLoop;

    /** Children states expanded following a recommendation of this state. */
    public final BinaryRelation<HAction<Action>, Compostate<State, Action>> exploredChildren;

    /** Actions expanded following a recommendation of this state. */
    public final Set<HAction<Action>> exploredActions;

    /** Children states expanded through uncontrollable transitions. */
    public final Set<Compostate<State, Action>> childrenExploredThroughUncontrollable;

    /** Children states expanded through controllable transitions. */
    public final Set<Compostate<State, Action>> childrenExploredThroughControllable;

    /** Parents that expanded into this state. */
    public final BinaryRelation<HAction<Action>, Compostate<State, Action>> parents;

    /** Set of actions enabled from this state. */
    public final Set<HAction<Action>> transitions;

    public HashSet<LinkedList<Integer>> arrivingPaths;

    public boolean hasGoalChild = false;
    public boolean hasErrorChild = false;
    
    /** if the state has a goal child, this action points to a goal child */
    public HAction<Action> actionToGoal;
    
    /** Indicates whether this state was expanded by DCS */
    public boolean wasExpanded = false;

    // Variables for OpenSetExplorationHeuristic --------------------
    /** all "colors" in these attributes, are the index of the LTS which marks that color */

    /** The estimated distance to the goal from this state for each color. */
    public Map<Integer, HEstimate> estimates;

    public final Integer uncontrollablesCount;

    /** A ranking of the outgoing transitions from this state for each color. */
    public Map<Integer, List<Recommendation<Action>>> recommendations;

    /** An iterator for the ranking of recommendations for each color. */
    public Map<Integer,Iterator<Recommendation<Action>>> recommendit;

    /** Current recommendation (or null) for each color. */
    public Map<Integer, Recommendation<Action>> recommendation;

    /** Indicates whether the state is actively being used. */
    public boolean live;

    /** Indicates whether the state is in the open queue of a specific color, by LTS index. */
    public Map<Integer, Boolean> inOpen;
    
    /** Stores target states (i.e., already visited marked states) to reach from this state.
     * targets[i] are the targets to reach in the i-th LTS*/
    public List<Set<State>> targets = emptyList();

    public final Set<HAction<Action>> discardedActions;
    public final HashMap<Integer,Set<HAction<Action>>> discardedActionsByColor;

    private boolean cantWinByGuarantees = false;

    /** Indicates that the state has a uncontrollable conflincting action */
    public boolean heuristicStronglySuggestsIsError = false;

    /** Set of vertices in the RA graph. */
    public Set<HAction<Action>> vertices;

    /** Edges in the RA graph. */
    public BidirectionalMap<HAction<Action>, HAction<Action>> edges;

    /** Cache of gaps between actions. */
    public Map<HAction<Action>, Map<HAction<Action>, Integer>> gapCache;

    public Map<HAction<Action>, Set<Integer>> readyInLTS;

    // Variables for FrontierListExplorationHeuristic --------------------

    /** Number of uncontrollable transitions */
    public int uncontrollableTransitions;

    /** Set of transitions that were not yet expanded by DCS */
    public int unexploredTransitions;

    /** Number of uncontrollable transitions that were not yet expanded by DCS */
    public int uncontrollableUnexploredTransitions;

    public Map<HAction<Action>, List<State>> actionChildStates;

    public HAction<Action> lastExpandedAction;
    public Map<HAction<Action>, boolean[]> missionsCompletes;
    public Map<HAction<Action>, int[]> entityIndexes;


    /** Constructor for a Composed State. */
    public Compostate(DirectedControllerSynthesisBlocking<State, Action> dcs, List<State> states) {
        this.dcs = dcs;
        this.states = states;
        this.status = Status.NONE;
        this.distance = DirectedControllerSynthesisBlocking.INF;
        this.depth = DirectedControllerSynthesisBlocking.INF;
        this.live = false;
        this.inOpen = new HashMap<>();
        this.estimates = new HashMap<>();
        this.recommendation = new HashMap<>();
        this.recommendit = new HashMap<>();
        this.recommendations = new HashMap<>();

        this.controlled = true; // we assume the state is controlled until an uncontrollable recommendation is obtained
        this.exploredChildren = new BinaryRelationImpl<>();
        this.exploredActions = new HashSet<>();
        this.discardedActions = new HashSet<>();
        this.discardedActionsByColor = new HashMap<>();
        this.childrenExploredThroughUncontrollable = new HashSet<>();
        this.childrenExploredThroughControllable = new HashSet<>();
        this.parents = new BinaryRelationImpl<>();
        this.loopID = -1;
        this.bestDistToWinLoop = -1;
        this.markedByGuarantee = new HashSet<>();
        for(Map.Entry<Integer, Integer> entry : dcs.guarantees.entrySet()) {
            int gNumber = entry.getKey();
            int gIndex = entry.getValue();

            this.inOpen.put(gIndex, false);
            discardedActionsByColor.put(gIndex, new HashSet<>());

            if (dcs.markedStates.get(gIndex).contains(states.get(gIndex))) {
                markedByGuarantee.add(gNumber);
                dcs.composByGuarantee.get(gNumber).add(this);
            }
        }

        this.markedByAssumption = new HashSet<>();
        for(Map.Entry<Integer, Integer> entry : dcs.assumptions.entrySet()) {
            int aNumber = entry.getKey();
            int aIndex = entry.getValue();

            this.inOpen.put(aIndex, false);
            discardedActionsByColor.put(aIndex, new HashSet<>());

            if (dcs.markedStates.get(aIndex).contains(states.get(aIndex))) {
                markedByAssumption.add(aNumber);
                dcs.notComposByAssumption.get(aNumber).add(this);
            }
        }

        this.transitions = buildTransitions();
        dcs.heuristic.initialize(this);
        this.uncontrollablesCount = countUncontrollables();
        this.missionsCompletes = new HashMap<>();
        this.entityIndexes = new HashMap<>();
        this.lastExpandedAction = null;
    }

    /** Returns the states that conform this composed state. */
    public List<State> getStates() {
        return states;
    }

    /** Returns the distance from this state to the goal state (INF if not yet computed). */
    public int getDistance() {
        return distance;
    }

    /** Sets the distance from this state to the goal state. */
    public void setDistance(int distance) {
        this.distance = distance;
    }

    /** Returns the depth of this state in the exploration tree. */
    public int getDepth() {
        return depth;
    }

    /** Sets the depth for this state. */
    public void setDepth(int depth) {
        if (this.depth > depth)
            this.depth = depth;
    }

    public boolean isDeadlock(){
        return getTransitions().isEmpty();
    }

    /** Returns this state's status. */
    public Status getStatus() {
        return status;
    }

    /** Sets this state's status. */
    public void setStatus(Status status) {
//            logger.fine(this.toString() + " status was: " + this.status + " now is: " + status);
        if (this.status != Status.ERROR || status == Status.ERROR)
            this.status = status;
    }

    /** Indicates whether this state's status equals some other status. */
    public boolean isStatus(Status status) {
        return this.status == status;
    }

    /** Indicates whether this state is marked by guarantee. */
    public boolean isMarked(){
        return markedByGuarantee.contains(dcs. heuristic.currentTargetLTSIndex);
    }

    public void setLoopID(Integer loopID){
        this.loopID = loopID;
    }

    public Integer getLoopID(){
        return this.loopID;
    }

    public void setBestDistToWinLoop(Integer bestDistToWinLoop){
        this.bestDistToWinLoop = bestDistToWinLoop;
    }

    public Integer getBestDistToWinLoop(){
        return this.bestDistToWinLoop;
    }

    public boolean hasGoalChild(){
        return hasGoalChild;
    }

    public void setHasGoalChild(HAction<Action> actionToGoal) {
        this.actionToGoal = actionToGoal;
        this.hasGoalChild = true;
    }

    public boolean hasErrorChild(){
        return hasErrorChild;
    }

    public void setHasErrorChild() {
        this.hasErrorChild = true;
    }

    /** Returns whether this state has a child with the given status. */
    public boolean hasStatusChild(Status status) {
        for (Pair<HAction<Action>,Compostate<State, Action>> transition : getExploredChildren())
            if (transition.getSecond().isStatus(status)) return true;
        return false;
    }

    /** Closes this state to avoid further exploration. */
    public void close() {
        live = false;
    }

    /** Returns the set of actions enabled from this composed state. */
    public Set<HAction<Action>> getTransitions() {
        return transitions;
    }

    /** Initializes the set of actions enabled from this composed state. */
    private Set<HAction<Action>> buildTransitions() { // Note: here I can code the wia and ia behavior for non-deterministic ltss
        Set<HAction<Action>> result = new HashSet<>();
        if (dcs.facilitators == null) {
            for (int i = 0; i < states.size(); ++i) {
                for (Pair<Action,State> transition : dcs.ltss.get(i).getTransitions(states.get(i))) {
                    HAction<Action> action = dcs.alphabet.getHAction(transition.getFirst());
                    dcs.allowed.add(i, action);
                }
            }
        } else {
            for (int i = 0; i < states.size(); ++i)
                if (!dcs.facilitators.get(i).equals(states.get(i)))
                    for (Pair<Action,State> transition : dcs.ltss.get(i).getTransitions(dcs.facilitators.get(i))) {
                        HAction<Action> action = dcs.alphabet.getHAction(transition.getFirst());
                        dcs.allowed.remove(i, action); // remove old non-shared facilitators transitions
                    }
            for (int i = 0; i < states.size(); ++i)
                if (!dcs.facilitators.get(i).equals(states.get(i)))
                    for (Pair<Action,State> transition : dcs.ltss.get(i).getTransitions(states.get(i))) {
                        HAction<Action> action = dcs.alphabet.getHAction(transition.getFirst());
                        dcs.allowed.add(i, action); // add new non-shared facilitators transitions
                    }
        }
        result.addAll(dcs.allowed.getEnabled());
        dcs.facilitators = states;

        //this removes mixed compostates, if there are uncontrollable transitions we ignore controllable ones
        boolean hasU = false;
        for(HAction<Action> ha : result){
            if (!ha.isControllable()){
                hasU = true;
                controlled = false;
            }
        }
        if(hasU){
            result.removeIf(HAction::isControllable);
        }
        return result;
    }

    /** Adds an expanded child to this state. */
    public void addChild(HAction<Action> action, Compostate<State, Action> child) {
        if(action.isControllable()){
            childrenExploredThroughControllable.add(child);
        } else {
            childrenExploredThroughUncontrollable.add(child);
        }

        exploredChildren.addPair(action, child);
        exploredActions.add(action);
    }

    /** Returns all transition leading to exploredChildren of this state. */
    public BinaryRelation<HAction<Action>, Compostate<State, Action>> getExploredChildren() {
        return exploredChildren;
    }

    public Set<HAction<Action>> getExploredActions() {
        return exploredActions;
    }

    public Set<HAction<Action>> getDiscardedActions() {
        return discardedActions;
    }

    public Set<Compostate<State, Action>> getChildrenExploredThroughUncontrollable() {
        return childrenExploredThroughUncontrollable;
    }

    public Set<Compostate<State, Action>> getChildrenExploredThroughControllable() {
        return childrenExploredThroughControllable;
    }



    /** Returns all exploredChildren of this state. */
    public List<Compostate<State, Action>> getExploredChildrenCompostates() {
        List<Compostate<State, Action>> childrenCompostates = new ArrayList<>();
        for (Pair<HAction<Action>, Compostate<State, Action>> transition : exploredChildren){
            Compostate<State, Action> child = transition.getSecond();
            childrenCompostates.add(child);
        }
        return childrenCompostates;
    }


    /** Returns the distance to the goal of a child of this compostate following a given action. */
    public int getChildDistance(HAction<Action> action) {
        int result = DirectedControllerSynthesisBlocking.UNDEF; // exploredChildren should never be empty or null
        for (Compostate<State, Action> compostate : exploredChildren.getImage(action)) { // Note: maximum of non-deterministic exploredChildren
            if (result < compostate.getDistance())
                result = compostate.getDistance();
        }
        return result;
    }

    /** Adds an expanded parent to this state. */
    public void addParent(HAction<Action> action, Compostate<State, Action> parent) {
        parents.addPair(action, parent);
        setDepth(parent.getDepth() + 1);
    }

    /** Returns the inverse transition leading to parents of this state. */
    public BinaryRelation<HAction<Action>, Compostate<State, Action>> getParents() {
        return parents;
    }

    public Set<Compostate<State, Action>> getParentsOfStatus(Status st){
        Set<Compostate<State, Action>> result = new HashSet<>();
        for (Pair<HAction<Action>, Compostate<State, Action>> ancestorActionAndState : parents) {
            Compostate<State, Action> ancestor = ancestorActionAndState.getSecond();
            if(ancestor.isStatus(st)) result.add(ancestor);
        }
        return result;
    }

    /** Compares two composed states by their estimated distance to a goal by (<=). */
    @Override
    public int compareTo(Compostate o) {
        // int result = estimate.compareTo(o.estimate);
        // if (result == 0)
        //     result = this.depth - o.depth;
        // return result;
        return 1;
    }


    /** Clears the internal state removing parent and exploredChildren. */
    public void clear() {
        exploredChildren.clear();
        exploredActions.clear();
        /** Indicates whether the procedure consider a non-blocking requirement (by default we consider a stronger goal). */
        boolean nonblocking = false;
        if (!nonblocking) // this is a quick fix to allow reopening weak states marked as errors
            dcs.compostates.remove(states);
    }

    /** Returns the string representation of a composed state. */
    @Override
    public String toString() {
        return states.toString();
    }

    public boolean wasExpanded() {
        return wasExpanded;
    }

    public void setExpanded() {
        wasExpanded = true;
    }

    // Methods used by OpenSetExplorationHeuristic -------------------

    public HEstimate getEstimate(Integer color) {
        return estimates.get(color);
    }

    /** Indicates whether this state has been evaluated, that is, if it has
     *  a valid ranking of recommendations. */
    public boolean isEvaluated(Integer color) {
        return recommendations.get(color) != null;
    }

    /** Returns the target states to be reached from this state as a list of sets,
     *  which at the i-th position holds the set of target states of the i-th LTS. */
    public List<Set<State>> getTargets() {
        return targets;
    }

    /** Returns the target states of a given LTS to be reached from this state. */
    @SuppressWarnings("unchecked")
    public Set<State> getTargets(int lts) {
        return targets.isEmpty() ? (Set<State>)emptySet() : targets.get(lts);
    }

    /** Sets the given set as target states for this state (creates
     *  aliasing with the argument set). */
    public void setTargets(List<Set<State>> targets) {
        this.targets = targets;
    }

    /** Adds a state to this state's targets. */
    public void addTargets(Compostate<State, Action> compostate) {
        List<State> states = compostate.getStates();
        if (targets.isEmpty()) {
            targets = new ArrayList<>(dcs.ltssSize);
            for (int lts = 0; lts < dcs.ltssSize; ++lts)
                targets.add(new HashSet<>());
        }
        //we only add the states that are marked for each relevant LTS, since only for those colors should we aim for this compostate
        for(int lts : compostate.inOpen.keySet())
            targets.get(lts).add(states.get(lts));
        assert(targets.get(0).isEmpty()); //the first LTS is the plant, not a guarantee or assumption so it is useless for the abstractions
    }

    /** Sorts this state's recommendations in order to be iterated properly. */
    public Recommendation<Action> rankRecommendations(Integer color) {
        Recommendation<Action> result = null;
        if (!recommendations.get(color).isEmpty()) {
            recommendations.get(color).sort(new Ranker<>());
            result = recommendations.get(color).get(0);
        }
        return result;
    }

    /** Sets up the recommendation list. */
    public void setupRecommendations(Integer color) {
        recommendations.computeIfAbsent(color, k -> new ArrayList<>());
    }

    /** Adds a new recommendation to this state and returns whether an
     *  error action has been introduced (no other recommendations should
     *  be added after an error is detected).
     *  Recommendations should not be added after they have been sorted and
     *  with an iterator in use. */
    public boolean addRecommendation(Integer color, HAction<Action> action, HEstimate estimate) {
        boolean uncontrollableAction = !action.isControllable();
        if (controlled) { // may not work with lists of recommendations
            if (uncontrollableAction)
                controlled = false;
        }
        if (!estimate.isConflict()) {
            // update recommendation
            recommendations.get(color).add(new Recommendation<Action>(action, estimate));
        } else {
            discardedActionsByColor.get(color).add(action);

            boolean discardedByAnyGuarantee = false;
            boolean discardedByAllAssumptions = true;
            //A state is surely ERROR only if it can't reach some guarantee AND it can't win by avoiding any assumption.
            for(Integer gColor : dcs.guarantees.values()){
                Set<HAction<Action>> discAct = discardedActionsByColor.get(gColor);
                if(discAct.contains(action)){
                    discardedByAnyGuarantee = true;
                    break;
                }
            }
            for(Integer aColor : dcs.assumptions.values()){
                Set<HAction<Action>> discAct = discardedActionsByColor.get(aColor);
                if(!discAct.contains(action)){
                    discardedByAllAssumptions = false;
                    break;
                }
            }
            if(discardedByAnyGuarantee && discardedByAllAssumptions){
                discardedActions.add(action);
                if(uncontrollableAction){
                    // an uncontrollable action with at least one INF guarantee estimate and all INF assumptions is an automatic error
                    this.discardedActions.addAll(this.transitions);
                    this.heuristicStronglySuggestsIsError = true;
                    return true;
                }
            }
        }
        return false;
    }

    public void markLoserForGuarantees(){
        cantWinByGuarantees = true;
    }

    public boolean cantWinColor(Integer color){
        return discardedActionsByColor.get(color).size() == transitions.size() ||
                (cantWinByGuarantees && dcs.guarantees.containsValue(color));
    }

    /** Advances the iterator to the next recommendation. */
    public Recommendation<Action> nextRecommendation(Integer color) {
        Recommendation<Action> result = recommendation.get(color);
        updateRecommendation(color);
        return result;
    }

    /** Peek the next recommendation, without advancing the iterator. */
    public Recommendation<Action> peekRecommendation(Integer color) {
        return recommendation.get(color);
    }

    /** Initializes the recommendation iterator guaranteeing representation invariants. */
    public void initRecommendations(Integer color) {
        recommendit.put(color, recommendations.get(color).iterator());
        updateRecommendation(color);
    }

    /* TODO: Comment */
    public boolean[] getLastMissions(){
        return missionsCompletes.get(lastExpandedAction);
    }

    /* TODO: Comment */
    public int[] getLastentityIndex(){
        return entityIndexes.get(lastExpandedAction);
    }


    /** Initializes the recommendation iterator and current estimate for the state. */
    public void updateRecommendation(Integer color) {
        if(recommendit.get(color) == null){
            return;
        }
        if (recommendit.get(color).hasNext()) {
            recommendation.put(color, recommendit.get(color).next());

            // update this state estimate in case the state is reopened
            estimates.put(color, recommendation.get(color).getEstimate());
        } else {
            recommendation.put(color, null);
        }
    }

    /** Clears all recommendations from this state. **/
    public void clearRecommendations() {
        recommendations.clear();
        recommendit.clear();
        recommendation.clear();
    }

    /** Clears RA structure after being evaluated for all colors **/
    public void clearRA(){
        vertices.clear();
        edges.clear();
        gapCache.clear();
        readyInLTS.clear();
    }

    /** Returns whether this state is being actively used. */
    public boolean isLive() {
        return live;
    }

    /** Returns whether this state is controllable or not. */
    public boolean isControlled() {
        return controlled;
    }

    private Integer countUncontrollables() {
        Integer result = 0;
        for (HAction<Action> a : this.transitions) {
            if (!a.isControllable()) result++;
        }
        return result;
    }

    public boolean hasControllableNone() {
        for (Pair<HAction<Action>, Compostate<State, Action>> transition : getExploredChildren())
            if (transition.getFirst().isControllable() && transition.getSecond().isStatus(Status.NONE))
                return true;
        return false;
    }
}
