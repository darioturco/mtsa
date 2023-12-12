package MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.abstraction;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;

import MTSTools.ac.ic.doc.commons.collections.BidirectionalMap;
import MTSTools.ac.ic.doc.commons.collections.InitMap;
import MTSTools.ac.ic.doc.commons.collections.QueueSet;
import MTSTools.ac.ic.doc.commons.collections.InitMap.Factory;
import MTSTools.ac.ic.doc.commons.relations.Pair;
import MTSTools.ac.ic.doc.mtstools.model.LTS;
import MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.Alphabet;
import MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.Compostate;
import MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.DirectedControllerSynthesisBlocking;
import MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.HAction;

/** This class implements the Ready Abstraction (RA). */
public class ReadyAbstraction<State, Action> implements Abstraction<State, Action> {

    /** Mapping of estimates for actions (the result of evaluating the abstraction). */
    private Map<HAction<Action>, HEstimate> estimates;

    /** The minimum estimate per LTS. */
    private Map<Integer, HDist> shortest;

    /** Fresh states discovered at each iteration. */
    private QueueSet<HAction<Action>> fresh;

    /** Mapping of actions to LTSs containing them in their alphabets. */
    private Map<HAction<Action>, Set<Integer>> actionsToLTS;

    /** Maps for each state which other states can be reached.
     *  It also stores which actions lead from source to destination and in how many local steps. */
    private Map<HState<State, Action>, Map<HState<State, Action>, Map<HAction<Action>, Integer>>> manyStepsReachableStates;

    /** Subset of the manyStepsReachableState map, containing only which
     *  marked states can be reached from a given source. This map keeps
     *  aliasing/sharing with the other and it is used only to speed up
     *  iteration while looking for marked states to reach. */
    private Map<HState<State, Action>, Map<HState<State, Action>, Map<HAction<Action>, Integer>>> markedReachableStates;

    /** Maps for each state which actions can be reached.
     *  It also stores which actions lead from source to desired action and in how many local steps. */
    private Map<HState<State, Action>, Map<HAction<Action>, Map<HAction<Action>, Integer>>> manyStepsReachableActions;

    /** Cache of target distances. */
    private Map<Integer, HDist> m0Cache;

    /** Cache of marked distances. */
    private Map<Integer, HDist> m1Cache;

    /** Cache of heuristic states ordered by creation order (used for perfect hashing). */
    public List<HState<State, Action>> stash;

    /** Cache of heuristic states used to reduce dynamic allocations. */
    private Map<Integer, HashMap<State, Integer>> cache;

    private final List<LTS<State, Action>> ltss;
    private final Alphabet<Action> alphabet;
    private final List<Set<State>> defaultTargets;

    /** Whether the debug log is activated or not. */
    private final static Boolean DEBUG_LOG = false;

    /** the objetctive the RA aims to */
    private final Integer color;

    /** Used to log internal RA calculations for debugging purposes. */
    private static void debugLog(Object log, Object... strFmtArgs) {
        if (DEBUG_LOG) {
            if (String.class.isInstance(log)) {
                System.err.println(String.format((String)log, strFmtArgs));
            } else {
                System.err.println(log);
            }
        }
    }

    static public class CompostateRanker<State, Action> implements Comparator<Compostate<State, Action>> {

        private final Integer color;

        public CompostateRanker(Integer color) {
            this.color = color;
        }

        @Override
        public int compare(Compostate<State, Action> c1, Compostate<State, Action> c2) {
            Recommendation<Action> r1 = c1.peekRecommendation(color);
            Recommendation<Action> r2 = c2.peekRecommendation(color);

            // FIXME: parece que pueden quedar compostates con recom en null. Hay que
            // investigar.
            // T: Creo que es por los estados que quedan en los opens de los otros colores al popear de un color
            if (r1 == null)
                return 1;
            if (r2 == null)
                return -1;

            // Uncontrollables have higher priority
            int controllable1 = r1.getAction().isControllable() ? 1 : 0;
            int controllable2 = r2.getAction().isControllable() ? 1 : 0;
            int result = controllable1 - controllable2;

            // If equal controllability, compare their first recommendation
            if (result == 0)
                result = controllable1 == 1 ? r1.compareTo(r2) : r2.compareTo(r1);

            // Depth is the tiebreaker
            if (result == 0)
                result = c1.getDepth() - c2.getDepth();

            return result;
        }
    }

    /** Constructor for the RA.
     *
     * @param ltss
     * @param defaultTargets
     * @param alphabet
     */
    public ReadyAbstraction(Integer color, List<LTS<State, Action>> ltss, List<Set<State>> defaultTargets, Alphabet<Action> alphabet) {
        this.ltss = ltss;
        this.alphabet = alphabet;
        this.defaultTargets = defaultTargets;
        this.color = color;
        stash = new ArrayList<>(this.ltss.size());
        cache = new HashMap<>();

        estimates = new InitMap<>(new Factory<HEstimate>() {
            @Override
            public HEstimate newInstance() {
                HEstimate result = new HEstimate(ltss.size()+1, HDist.chasm);
                result.values.remove(ltss.size());
                return result;
            }
        });
        shortest = new InitMap<>(HDist.chasmFactory);
        fresh = new QueueSet<>();
        actionsToLTS = new InitMap<>(HashSet.class);
        manyStepsReachableStates = new InitMap<>(HashMap.class);
        markedReachableStates = new InitMap<>(HashMap.class);
        manyStepsReachableActions = new InitMap<>(HashMap.class);
        m0Cache = new HashMap<>();
        m1Cache = new HashMap<>();
        init();
    }


    /** Initializes the RA precomputing tables. */
    private void init() {
        computeActionsToLTS();
        computeReachableStates();
        computeReachableActions();
        computeMarkedReachableStates();
    }


    /** Clears the RA internal state. */
    private void clear() {
        estimates.clear();
        shortest.clear();
        fresh.clear();
    }


    /** Evaluates the abstraction by building and exploring the RA. */
    @Override
    public void eval(Compostate<State, Action> compostate) {
        clear();
        buildRA(compostate);
        evaluateRA(compostate);
        extractRecommendations(compostate);
    }


    /** Builds the RA by connecting ready events through edges indicating their causal relationship. */
    private void buildRA(Compostate<State, Action> compostate) {
        if (!compostate.edges.isEmpty()) {
            // It's already built
            return;
        }

        for (int lts = 0; lts < this.ltss.size(); ++lts) {
            HState<State, Action> s = getHState(lts, compostate.getStates().get(lts));
            for (Pair<Action,State> transition : s.getTransitions()) {
                HAction<Action> action = this.alphabet.getHAction(transition.getFirst());
                if (!s.state.equals(transition.getSecond())) { // !s.isSelfLoop(action)
                    compostate.readyInLTS.get(action).add(lts);
                    compostate.vertices.add(action);
                }
            }
        }
        for (HAction<Action> t : compostate.vertices) {
            for (Integer lts : actionsToLTS.get(t)) {
                HState<State, Action> s = getHState(lts, compostate.getStates().get(lts));
                Map<HAction<Action>, Integer> actionsLeadingToTfromS = manyStepsReachableActions.get(s).get(t);
                if (actionsLeadingToTfromS != null) {
                    for (HAction<Action> l : actionsLeadingToTfromS.keySet()) {
                        if (!l.equals(t) && s.contains(l) && !s.isSelfLoop(l)) { // we need an efficient s.contains(l) that returns false for self-loops
                            compostate.edges.put(l, t);
                        }
                    }
                }
            }
        }
        // debugLog("Built RA: %s", edges);
    }


    /** Evaluates the RA by exploring the graph and populating the estimates table. */
    private void evaluateRA(Compostate<State, Action> compostate) {
        for (int lts = 0; lts < this.ltss.size(); ++lts) {
            HState<State, Action> s = getHState(lts, compostate.getStates().get(lts));
            Set<State> markedStates = this.defaultTargets.get(lts);
            Set<State> targetStates = compostate.getTargets(lts);
            Map<HState<State, Action>, Map<HAction<Action>, Integer>> markedReachableStatesFromSource = markedReachableStates.get(s);
            for (Pair<Action,State> transitions : s.getTransitions()) {
                HAction<Action> l = this.alphabet.getHAction(transitions.getFirst());
                State t = transitions.getSecond();
                if (t.equals(-1L)) // CASE 1: action leads to illegal state
                    continue;
                Integer mt = 2, dt = DirectedControllerSynthesisBlocking.INF;
                if (markedStates.contains(t)) {
                    mt = targetStates.contains(t) ? 0 : 1;
                    dt = 1;
                }
                if (!(mt == 0 || (mt == 1 && targetStates.isEmpty()))) { // already best, skip search
                    if (s.state.equals(t)) // a self-loop
                        continue;
                    for (HState<State, Action> g : markedReachableStatesFromSource.keySet()) { // search for best
                        Integer mg = targetStates.contains(g.state) ? 0 : 1;
                        Integer dg = markedReachableStatesFromSource.get(g).get(l);
                        if (dg == null)
                            continue;
                        if (mg < mt || (mg == mt && dg < dt)) {
                            mt = mg;
                            dt = dg;
                        }
                    }
                }
                HDist newlDist = getHDist(mt, dt);
                HDist currentDist = estimates.get(l).get(lts);
                if (newlDist.compareTo(currentDist) < 0) {
                    // CASES 2/3: there's a path in the RA, update with the shortest one
                    estimates.get(l).set(lts, newlDist);
                    // debugLog("Upated estimate for l=%s lts=%d d=%s", l, lts, dt);
                    fresh.add(l);
                    if (compostate.getTransitions().contains(l)) { // register only the shortest distances for enabled events
                        // debugLog("New shortest for lts=%d at comp=%s is d=%d using l=%s", lts, compostate, dt, l);
                        registerShort(lts, newlDist);
                    }
                }
            }
        }

        while (!fresh.isEmpty()) {
            HAction<Action> t = fresh.poll();
            for (HAction<Action> l : compostate.edges.getK(t)) {
                Integer dtl = gap(compostate, l, t);
                // debugLog("At comp=%s gap(%s, %s)=%d", compostate, l, t, dtl);
                for (int lts = 0; lts < this.ltss.size(); ++lts) {
                    if (compostate.readyInLTS.get(l).contains(lts))
                        continue;
                    HDist tDist = estimates.get(t).get(lts);
                    HDist lDist = estimates.get(l).get(lts);
                    Integer dl = addDist(tDist.getSecond(), dtl);
                    HDist newlDist = dl == DirectedControllerSynthesisBlocking.INF ? HDist.chasm : getHDist(tDist.getFirst(), dl);
                    if (newlDist.compareTo(lDist) < 0) {
                        // CASES 2/3 (again): there's a path in the RA, update with the shortest one
                        estimates.get(l).set(lts, newlDist);
                        // debugLog("CLOSURE: Updated estimate for lts=%d at comp=%s,using l1=%s--l2=%s %d = (%d)+%d", lts, compostate, l, t, dl, dtl, tDist.getSecond());
                        fresh.add(l);
                        if (compostate.getTransitions().contains(l)) {
                            // debugLog("CLOSURE: New shortest for lts=%d at comp=%s,using l1=%s--l2=%s %d = (%d)+%d", lts, compostate, l, t, dl, dtl, tDist.getSecond());
                            registerShort(lts, newlDist);
                        }
                    }
                }
            }

        }

        reconcilateShort(compostate);
    }


    /** Returns a distance from cache. */
    private HDist getHDist(Integer m, Integer d) {
        Map<Integer, HDist> mCache = m == 0 ? m0Cache : m1Cache;
        HDist result = mCache.get(d);
        if (result == null)
            mCache.put(d, result = new HDist(m, d));
        return result;
    }


    /** Returns the maximum distance between two actions from the current state of every LTS. */
    private Integer gap(Compostate<State, Action> compostate, HAction<Action> l, HAction<Action> t) {
        Integer result = compostate.gapCache.get(l).get(t);
        if (result != null)
            return result;
        result = 0;
        for (Integer lts : actionsToLTS.get(l)) {
            if (!actionsToLTS.get(t).contains(lts))
                continue;
            HState<State, Action> s = getHState(lts, compostate.getStates().get(lts));
            if (s.contains(l)) {
                Map<HAction<Action>, Integer> actionFromSourceToTarget = manyStepsReachableActions.get(s).get(t);
                Integer dl = actionFromSourceToTarget == null ? null : actionFromSourceToTarget.get(l);
                dl = dl == null ? DirectedControllerSynthesisBlocking.INF : dl - 1;
                if (dl > result)
                    result = dl;
            }
        }
        compostate.gapCache.get(l).put(t, result);
        return result;
    }


    /** Adds to distances (maxing at overflows). */
    private Integer addDist(Integer d1, Integer d2) {
        return (d1 == DirectedControllerSynthesisBlocking.INF || d2 == DirectedControllerSynthesisBlocking.INF) ? DirectedControllerSynthesisBlocking.INF : d1 + d2;
    }


    /** Registers a distance estimated for a given LTS if minimum. */
    private void registerShort(Integer lts, HDist dist) {
        HDist shortDist = shortest.get(lts);
        if (dist.compareTo(shortDist) < 0)
            shortest.put(lts, dist);
    }


    /** Reconciliates the distances for the LTSs for which an action has not been considered. */
    private void reconcilateShort(Compostate<State, Action> compostate) {
        for (int lts = 0; lts < this.ltss.size(); ++lts) { // this loops sets any missing shortest information
            HDist shortLts = shortest.get(lts);
            if (shortLts == HDist.chasm) {
                HState<State, Action> s = getHState(lts, compostate.getStates().get(lts));
                Map<HState<State, Action>, Map<HAction<Action>, Integer>> markedStatesReachableFroms = markedReachableStates.get(s);
                for (Entry<HState<State, Action>, Map<HAction<Action>, Integer>> entry : markedStatesReachableFroms.entrySet()) {
                    HState<State, Action> t = entry.getKey();
                    Integer m = compostate.getTargets(lts).contains(t.state) ? 0 : 1;
                    if (m < shortLts.getFirst()) {
                        for (HAction<Action> a : entry.getValue().keySet()) {
                            Integer d = entry.getValue().get(a);
                            if (d < shortLts.getSecond()) {
                                shortLts = getHDist(m, d);
                                // debugLog("New shortest for lts=%d at comp=%s is d=%d using l=%s", lts, compostate, d, a.getAction());
                            }
                        }
                    }
                }
                shortest.put(lts, shortLts);
            }
        }
        for (HAction<Action> l : compostate.getTransitions()) { // this loops fills missing goals with shortest paths
            HEstimate el = estimates.get(l);
            for (int lts = 0; lts < this.ltss.size(); ++lts) {
                HState<State, Action> s = getHState(lts, compostate.getStates().get(lts));
                if (compostate.readyInLTS.get(l).contains(lts) && !s.isSelfLoop(l))
                    continue;
                HDist lDist = el.get(lts);
                if (lDist == HDist.chasm) {
                    if (s.marked && (!actionsToLTS.get(l).contains(lts) || s.isSelfLoop(l))) {
                        // CASES 4/5
                        // Current state is marked and 'l' is either not in the component's alphabet or a self-loop,
                        // so taking 'l' means staying at a marked state.
                        Integer m = compostate.getTargets(lts).contains(s.state) ? 0 : 1;
                        el.set(lts, getHDist(m, 1));
                        // debugLog("Current is marked for lts=%s and l=%s is either not in alphabet or a self-loop: (%d, 1)", lts, l, m);
                    } else {
                        // CASES 6/7 if shortest.get(lts) is not 'chasm'. If it is 'chasm', we are covering CASE 8
                        // debugLog("Updating dist for lts=%s l=%s using 1 + shortest=%s", lts, l, shortest.get(lts));
                        el.set(lts, shortest.get(lts).inc());
                    }
                }
            }
        }
    }


    /** Extracts recommendations for a state from the estimates table. */
    @SuppressWarnings("unchecked")
    private void extractRecommendations(Compostate<State, Action> compostate) {
        compostate.setupRecommendations(color);
        for (HAction<Action> action : compostate.getTransitions()) {
            HEstimate estimate = estimates.get(action);
            estimate.sortDescending();
            if (compostate.addRecommendation(color, action, estimate))
                break;
        }
        compostate.rankRecommendations(color);
        compostate.initRecommendations(color);
    }


    /** Populates the actionsToLTS map. */
    private void computeActionsToLTS() {
        for (int i = 0; i < this.ltss.size(); ++i) {
            for (Action l : this.ltss.get(i).getActions()) {
                Set<Integer> set = actionsToLTS.computeIfAbsent(this.alphabet.getHAction(l), k -> new HashSet<>());
                set.add(i);
            }
        }
    }


    /** Computes for each state in each LTS which other states can reach and in how many steps. */
    private void computeReachableStates() {
        Map<HState<State, Action>, Map<HState<State, Action>, Set<HAction<Action>>>> oneStepReachableStates = new InitMap<>(HashMap.class);
        Map<HState<State, Action>, Set<Pair<HAction<Action>,HState<State, Action>>>> lastStates = new InitMap<>(HashSet.class);
        Map<HState<State, Action>, Set<Pair<HAction<Action>,HState<State, Action>>>> nextStates = new InitMap<>(HashSet.class);
        boolean statesPopulated = false;

        Map<HState<State, Action>, Map<HAction<Action>, Integer>> manyStepsReachableFromSource;

        for (int lts = 0; lts < this.ltss.size(); ++lts) { // this loop populates one step reachable states
            for (State state : this.ltss.get(lts).getStates()) {
                HState<State, Action> source = buildHState(lts, state);
                manyStepsReachableFromSource = manyStepsReachableStates.get(source);
                Map<HState<State, Action>, Set<HAction<Action>>> oneReachableFromSoruce = oneStepReachableStates.get(source);
                for (Pair<Action, State> transition : this.ltss.get(lts).getTransitions(state)) {
                    HAction<Action> label = this.alphabet.getHAction(transition.getFirst());
                    HState<State, Action> destination = buildHState(lts, transition.getSecond());
                    Set<HAction<Action>> oneStepToDestination = oneReachableFromSoruce.get(destination);
                    Map<HAction<Action>, Integer> manyStepsToDestination = manyStepsReachableFromSource.get(destination);
                    if (oneStepToDestination == null) {
                        oneReachableFromSoruce.put(destination, oneStepToDestination = new HashSet<>());
                        manyStepsReachableFromSource.put(destination, manyStepsToDestination = new HashMap<>());
                    }
                    manyStepsToDestination.put(label, 1);
                    oneStepToDestination.add(label);
                    statesPopulated |= lastStates.get(source).add(Pair.create(label, destination));
                }
            }
        }

        int i = 2;
        while (statesPopulated) { // this loop extends the reachable states in the transitive closure (each iteration adds the states reachable in i steps).
            statesPopulated = false;
            for (HState<State, Action> source : lastStates.keySet()) {
                manyStepsReachableFromSource = manyStepsReachableStates.get(source);
                for (Pair<HAction<Action>, HState<State, Action>> pair : lastStates.get(source)) {
                    HAction<Action> label = pair.getFirst();
                    HState<State, Action> intermediate = pair.getSecond();
                    for (HState<State, Action> target : oneStepReachableStates.get(intermediate).keySet()) {
                        Map<HAction<Action>, Integer> manyStepsReachableFromSourceToTarget = manyStepsReachableFromSource.get(target);
                        if (manyStepsReachableFromSourceToTarget == null)
                            manyStepsReachableFromSource.put(target, manyStepsReachableFromSourceToTarget = new HashMap<>());
                        Integer current = manyStepsReachableFromSourceToTarget.get(label);
                        if (current == null) {
                            manyStepsReachableFromSourceToTarget.put(label, i);
                            statesPopulated |= nextStates.get(source).add(Pair.create(label, target));
                        }
                    }
                }
            }
            for (Set<Pair<HAction<Action>,HState<State, Action>>> set : lastStates.values())
                set.clear();
            Map<HState<State, Action>, Set<Pair<HAction<Action>,HState<State, Action>>>> swap = lastStates;
            lastStates = nextStates;
            nextStates = swap;
            ++i;
        }

    }


    /** Computes for each state in each LTS which other *marked* states can reach and in how many steps. */
    private void computeMarkedReachableStates() {
        for (HState<State, Action> source : manyStepsReachableStates.keySet()) {
            Map<HState<State, Action>, Map<HAction<Action>, Integer>> markedStatesFromSource = markedReachableStates.get(source);
            Map<HState<State, Action>, Map<HAction<Action>, Integer>> reachableStatesFromSource = manyStepsReachableStates.get(source);
            for (HState<State, Action> destination : reachableStatesFromSource.keySet()) {
                if (destination.marked/*isMarked()*/)
                    markedStatesFromSource.put(destination, reachableStatesFromSource.get(destination));
            }
        }
    }


    /** Computes for each state in each LTS which actions can be reached and in how many steps. */
    private void computeReachableActions() {
        for (int lts = 0; lts < this.ltss.size(); ++lts) {
            for (State state : this.ltss.get(lts).getStates()) { // this loop populates the reachable action with the LTSs' transition relations (one step)
                HState<State, Action> source = getHState(lts, state);
                Map<HAction<Action>, Map<HAction<Action>, Integer>> reachableActionsFromSource = manyStepsReachableActions.get(source);
                for (Pair<Action,State> transition : source.getTransitions()) {
                    HAction<Action> label = this.alphabet.getHAction(transition.getFirst());
                    Map<HAction<Action>, Integer> reachableActionsThroughLabel = reachableActionsFromSource.get(label);
                    if (reachableActionsThroughLabel == null)
                        reachableActionsFromSource.put(label, reachableActionsThroughLabel = new HashMap<>());
                    DirectedControllerSynthesisBlocking.putmin(reachableActionsThroughLabel, label, 1);
                }
            }
            for (State state : this.ltss.get(lts).getStates()) { // this loop extends the reachable actions with the outgoing events from reachable states (many steps)
                HState<State, Action> source = getHState(lts, state);
                Map<HState<State, Action>, Map<HAction<Action>, Integer>> statesReachableFromSource = manyStepsReachableStates.get(source);
                Map<HAction<Action>, Map<HAction<Action>, Integer>> actionsReachableFromSource = manyStepsReachableActions.get(source);
                for (HState<State, Action> destination : statesReachableFromSource.keySet()) {
                    for (Entry<HAction<Action>, Integer> entry : statesReachableFromSource.get(destination).entrySet()) {
                        HAction<Action> actionLeadingToDestination = entry.getKey();
                        Integer distanceFromSourceToDestination = entry.getValue();
                        for (Pair<Action,State> transition : destination.getTransitions()) {
                            HAction<Action> target = this.alphabet.getHAction(transition.getFirst());
                            Map<HAction<Action>, Integer> actionsLeadingToTarget = actionsReachableFromSource.get(target);
                            if (actionsLeadingToTarget == null)
                                actionsReachableFromSource.put(target, actionsLeadingToTarget = new HashMap<>());
                            DirectedControllerSynthesisBlocking.putmin(actionsLeadingToTarget, actionLeadingToDestination, distanceFromSourceToDestination + 1);
                        }
                    }
                }
            }
        }

    }

    private HState<State, Action> getHState(int lts, State state){
        // HashMap<State, Integer> a = new HashMap<>(cache.get(lts));
        HashMap<State, Integer> table = cache.computeIfAbsent(lts, k -> new HashMap<>());
        // HashMap<State, Integer> b = new HashMap<>(cache.get(lts));
        // assertTrue(a.equals(b)); // FIXME: no tengo idea por qué pero descomentar esto modifica las expandedTransitions en TR-2-2
        Integer index = table.get(state);
        return stash.get(index);
    }

    /** Builds (or retrieves from cache) a heuristic state. */
    private HState<State, Action> buildHState(int lts, State state) {
        HashMap<State, Integer> table = cache.computeIfAbsent(lts, k -> new HashMap<>());
        Integer index = table.get(state);
        if(index == null){
            HState<State, Action> hstate = new HState<State, Action>(
                    lts,
                    state,
                    this.stash.size(),
                    this.defaultTargets.get(lts).contains(state),
                    this.ltss);

            stash.add(hstate);
            index = hstate.hashCode();
            table.put(state, index);
        }
        return stash.get(index);
    }
}