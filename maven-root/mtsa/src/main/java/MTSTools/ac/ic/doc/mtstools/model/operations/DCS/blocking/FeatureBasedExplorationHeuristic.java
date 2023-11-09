package MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking;

import MTSTools.ac.ic.doc.commons.relations.Pair;
import MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.abstraction.*;
import ai.onnxruntime.*;
import jargs.gnu.CmdLineParser;
import ltsa.dispatcher.TransitionSystemDispatcher;
import ltsa.lts.*;
import ltsa.ui.StandardOutput;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.*;

import static java.util.Collections.emptyList;
import static org.junit.Assert.assertTrue;

public class FeatureBasedExplorationHeuristic<State, Action> implements ExplorationHeuristic<State, Action> {
    FloatBuffer input_buffer;
    DirectedControllerSynthesisBlocking<State, Action> dcs;

    //Feature variables
    public int goals_found = 0;
    public int marked_states_found = 0;
    public int known_transitions = 0;
    public int closed_potentially_winning_loops = 0;
    public Compostate<State, Action> lastExpandedTo = null;
    public Compostate<State, Action> lastExpandedFrom = null;

    public ActionWithFeatures<Long, String> lastExpandedStateAction = null;
    public List<HashMap<State, Integer>> visitCounts;
    public float explorationPercentage = 0;
    public int problem_n, problem_k;

    public static Pair<CompositeState, LTSOutput> compileFSP(String filename){
        String currentDirectory = null;
        try {
            currentDirectory = (new File(".")).getCanonicalPath();
        } catch (IOException e) {
            e.printStackTrace();
        }

        LTSInput input = new LTSInputString(readFile(filename));
        LTSOutput output = new StandardOutput();

        LTSCompiler compiler = new LTSCompiler(input, output, currentDirectory);
        compiler.compile();
        CompositeState c = compiler.continueCompilation("DirectedController");

        TransitionSystemDispatcher.parallelComposition(c, output);

        return new Pair<>(c, output);
    }

    public static String readFile(String filename) {
        String result = null;
        try {
            BufferedReader file = new BufferedReader(new FileReader(filename));
            String thisLine;
            StringBuffer buff = new StringBuffer();
            while ((thisLine = file.readLine()) != null)
                buff.append(thisLine+"\n");
            file.close();
            result = buff.toString();
        } catch (Exception e) {
            System.err.print("Error reading FSP file " + filename + ": " + e);
            System.exit(1);
        }
        return result;
    }


    public void setFeaturesBuffer(FloatBuffer featuresBuffer){
        this.input_buffer = featuresBuffer;
    }

    public void clearBuffer(){
        this.input_buffer.clear();
    }

    public boolean isFinished(){
        return this.dcs.isFinished();
    }

    public void computeFeatures(){
        /*assertTrue("Frontier is empty. The synthesis algorithm has failed.",
                explorationFrontier.size() > 0);
        assertTrue("Frontier did not fit in the buffer. Size was "+explorationFrontier.size(),
                explorationFrontier.size() <= max_frontier);

        explorationFrontier.parallelStream().forEach(ActionWithFeatures::updateFeatures);
        clearBuffer();

        for(ActionWithFeatures<State, Action> action : explorationFrontier){
            this.input_buffer.put(action.feature_vector);
        }


        if(debugging) printFeatures();*/

    }
    /*
    public FloatBuffer getActionsAt(int start, int n){
        this.input_buffer.position(start);
        this.input_buffer.limit(n*featureMaker.n_features);
        return this.input_buffer.slice();
    }

    private void parentPathAdded(LinkedList<Integer> path, HAction<State, Action> action, Compostate<State, Action> state){
        String label = action.toString().replaceAll("[^a-zA-Z]", "");
        int a_idx = featureMaker.labels_idx.get(label);
        LinkedList<Integer> childp = new LinkedList<>(path);
        childp.addLast(a_idx);
        if(childp.size() > featureMaker.state_labels_n) childp.removeFirst();
        boolean new_path = state.arrivingPaths.add(childp);
        if(new_path){
            for(Pair<HAction<State, Action>, Compostate<State, Action>> child : state.getExploredChildren()){
                parentPathAdded(childp, child.getFirst(), child.getSecond());
            }
        }
    }

    private void updatePaths(Compostate<State, Action> state, HAction<State, Action> action, Compostate<State, Action> child){
        String label = action.toString().replaceAll("[^a-zA-Z]", "");
        int a_idx = featureMaker.labels_idx.get(label);
        boolean fixed = false;
        while(!fixed) {
            List<LinkedList<Integer>> new_paths = new LinkedList<>();
            for (List<Integer> p : state.arrivingPaths) {
                LinkedList<Integer> childp = new LinkedList<>(p);
                childp.addLast(a_idx);
                if (childp.size() > featureMaker.state_labels_n) childp.removeFirst();
                if (!child.arrivingPaths.contains(childp)) {
                    new_paths.add(childp);
                }
            }
            fixed = new_paths.isEmpty();
            for (LinkedList<Integer> path : new_paths) {
                child.arrivingPaths.add(path);
                for (Pair<HAction<State, Action>, Compostate<State, Action>> child2 : child.getExploredChildren()) {
                    parentPathAdded(path, child2.getFirst(), child2.getSecond());
                }
            }
        }
    }

    public void addActionLabelToState(Compostate<State, Action> state, HAction<State, Action> action, Compostate<State, Action> parent) {
        if(featureMaker.state_labels_n > 0){
            String label = action.toString().replaceAll("[^a-zA-Z]", "");
            int a_idx = featureMaker.labels_idx.get(label);
            if(parent.arrivingPaths.isEmpty()) {
                LinkedList<Integer> new_path = new LinkedList<>();
                new_path.add(a_idx);
                state.arrivingPaths.add(new_path);
            } else {
                updatePaths(parent, action, state);
            }
        }
    }

    public void addVisit(Compostate<State, Action> s) {
        List<State> states = s.getStates();
        this.explorationPercentage = 0;
        int totalStates = 0;
        for(int i = 0; i < dcs.ltssSize; i++){
            int c = this.visitCounts.get(i).getOrDefault(states.get(i), 0);
            this.visitCounts.get(i).put(states.get(i), c+1);
            this.explorationPercentage += this.visitCounts.get(i).size();
            totalStates += dcs.ltss.get(i).getStates().size();
        }
        this.explorationPercentage /= totalStates;
    }


     */
    // List of actions available for expanding
    public ArrayList<ActionWithFeatures<State, Action>> explorationFrontier;
    public ArrayList<ActionForPython<State, Action>> allActionsWFNoFrontier;

    ReadyAbstraction<State, Action> ra;

    // Environment for running the neural network of the agent
    OrtSession session;
    OrtEnvironment ortEnv;

    // Path to the saved neural network
    public String model_path;

    // If true, prints features observed and values of the model at each step
    public boolean debugging = false;

    public int max_frontier = 1000000;

    // Only used with random agent


    Random rand = new Random();

    public DCSFeatures<State, Action> featureMaker;

    FeatureBasedExplorationHeuristic(String model_path, DCSFeatures<State, Action> featureMaker, boolean debugging){
        this.featureMaker = featureMaker;
        this.model_path = model_path;
        this.debugging = debugging;
    }


    public void set_nk(int n, int k){
        problem_n = n;
        problem_k = k;
    }


    public void startSynthesis(DirectedControllerSynthesisBlocking<State, Action> dcs) {
        //dcs.analyzeReachability(); //Descomentar
        this.dcs = dcs;

        //if(featureMaker.using_ra_feature)
        //    this.ra = new ReadyAbstraction<>(dcs.ltss, dcs.defaultTargets, dcs.alphabet);

        if(model_path.equals("python")){ // the buffer is created from python
            this.session = null;
            this.ortEnv = null;
        } else {
            ByteBuffer bb = ByteBuffer.allocateDirect(featureMaker.n_features*4*max_frontier);
            bb.order(ByteOrder.nativeOrder());
            this.setFeaturesBuffer(bb.asFloatBuffer());

            if(!model_path.equals("random")) {
                this.ortEnv = OrtEnvironment.getEnvironment();
                try {
                    OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
                    opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT);
                    this.session = ortEnv.createSession(model_path, opts);
                } catch (OrtException e) {
                    e.printStackTrace();
                }
            }
        }
        this.explorationFrontier = new ArrayList<>();
        this.allActionsWFNoFrontier = new ArrayList<>();


        this.visitCounts = new ArrayList<>();
        for(int i = 0; i < dcs.ltssSize; i++) this.visitCounts.add(new HashMap<>());
    }

    public int frontierSize(){
        return explorationFrontier.size();
    }


    void filterFrontier(){
        for(int i = 0; i < explorationFrontier.size();) {
            if (!explorationFrontier.get(i).state.isStatus(Status.NONE)) {
                explorationFrontier.get(i).state.actionsWithFeatures.clear();
                removeFromFrontier(i);
            } else {
                i++;
            }
        }
    }
    /*
    public Pair<Compostate<State, Action>, HAction<State, Action>> getNextAction() {
        filterFrontier();
        ActionWithFeatures<State, Action> stateAction;
        if(Objects.equals(model_path, "random")){
            if(debugging) computeFeatures();
            int selected = rand.nextInt(explorationFrontier.size());
            if(debugging) {
                for (int i = 0; i < explorationFrontier.size(); i++) System.out.print("0 ");
                System.out.println();
                System.out.println(selected);
            }
            stateAction = removeFromFrontier(selected);

        } else {
            computeFeatures();
            FloatBuffer availableActions = getActionsAt(0, explorationFrontier.size());
            FloatBuffer values = null;
            try {
                OnnxTensor t = OnnxTensor.createTensor(this.ortEnv, availableActions, new long[]{explorationFrontier.size(), featureMaker.n_features});
                OrtSession.Result results = session.run(Collections.singletonMap("X", t));
                OnnxTensor v = (OnnxTensor) results.get(0);
                values = v.getFloatBuffer();
            } catch (OrtException e) {
                e.printStackTrace();
            }
            assert values != null;
            int best = 0;
            float bestValue = values.get();
            for(int i = 1; i < explorationFrontier.size(); i++){
                float v = values.get();
                if(v > bestValue){
                    best = i;
                    bestValue = v;
                }
            }
            if(debugging) {
                printValues(values);
                System.out.println(best);
            }
            stateAction = removeFromFrontier(best);
        }
        return new Pair<>(stateAction.state, stateAction.action);
    }
    */
    public void newState(Compostate<State, Action> state, List<State> childStates) {
        //if (parent != null)
        //    state.setTargets(parent.getTargets());

        //if (state.marked) {
        //    state.addTargets(state);
        //    if(featureMaker.using_context_features){
        //        marked_states_found ++;
        //    }
        //}

        //if(featureMaker.using_ra_feature)
        //    ra.evalForHeuristic(state, this);

        if(state.isStatus(Status.NONE))
            addTransitionsToFrontier(state);

        for(ActionWithFeatures<State, Action> action : explorationFrontier){
            if(action.child == null && action.childStates.equals(childStates)){
                action.child = state;
            }
        }
    }


    private void addTransitionsToFrontier(Compostate<State, Action> state){
        state.unexploredTransitions = 0;
        state.uncontrollableUnexploredTransitions = 0;
        state.actionsWithFeatures = new HashMap<>();
        for(HAction<Action> action : state.getTransitions()){
            List<State> childStates = dcs.getChildStates(state, action);
            // assertTrue(!dcs.dcs.canReachMarkedFrom(childStates) == state.getEstimate(action).isConflict());
            if(dcs.canReachMarkedFrom(childStates)) {
                state.actionChildStates.put(action, childStates);
                state.unexploredTransitions ++;
                if(!action.isControllable()){
                    state.uncontrollableUnexploredTransitions ++;
                }
            } else {
                // action is uncontrollable since we have removed controllable conflicts
                state.heuristicStronglySuggestsIsError = true;
                return;
            }
        }

        if(featureMaker.using_context_features){
            known_transitions += state.unexploredTransitions;
        }
        ////if(featureMaker.using_ra_feature && state.uncontrollableUnexploredTransitions > 0){
        ////    state.controlled = false;
        ////}
        state.uncontrollableTransitions = state.uncontrollableUnexploredTransitions;

        for(HAction<Action> action : state.getTransitions()){
            ActionWithFeatures<State, Action> actionWF = new ActionWithFeatures<>(state, action, this);
            explorationFrontier.add(actionWF);
            state.actionsWithFeatures.put(action, actionWF);
        }
    }

    /*
    public void notifyExpandingState(Compostate<State, Action> parent, HAction<State, Action> action, Compostate<State, Action> state) {
        if(state.wasExpanded()){ // todo: understand this, i am copying the behavior of the code pre refactor
            state.setTargets(parent.getTargets());
            if (state.marked)
                state.addTargets(state);
        }
        addActionLabelToState(state, action, parent);
        if(featureMaker.using_labelsThatReach_feature) featureMaker.update_child_labelsThatReach(parent,action,state);

    }*/

    public void addToListActions(Compostate<State, Action> parent, HAction<Action> action){
        ActionForPython<State, Action> actionNoFeatures = new ActionForPython<State, Action>(parent, action, this);
        allActionsWFNoFrontier.add(actionNoFeatures);
    }

    public boolean somethingLeftToExplore() {
        return !explorationFrontier.isEmpty();
    }

    ////public void updateUnexploredTransitions(Compostate<State, Action> state, HAction<State, Action> action) {
    public void updateUnexploredTransitions(Compostate<State, Action> state, HAction<Action> action) {////
        state.unexploredTransitions--;
        if(!action.isControllable())
            state.uncontrollableUnexploredTransitions--;
    }



    public void notifyStateSetErrorOrGoal(Compostate<State, Action> state) {
        //if(featureMaker.using_ra_feature)
        //    state.live = false;
        state.uncontrollableUnexploredTransitions = 0;
        state.unexploredTransitions = 0;
        if(featureMaker.using_context_features && state.isStatus(Status.GOAL)){
            goals_found++;
        }
    }

    public void setInitialState(Compostate<State, Action> state) {
        if(featureMaker.using_ra_feature)
            open(state);
    }
    /*
    public void notifyStateIsNone(Compostate<State, Action> state) {
        if(featureMaker.using_ra_feature)
            if(!fullyExplored(state))
                open(state);
    }

    public void notifyPropagatingGoal(Set<Compostate<State, Action>> ancestors){
        if(featureMaker.using_prop_feature) {
            final Iterator<Compostate<State, Action>> each = ancestors.iterator();
            while (each.hasNext()) {
                Compostate<State, Action> s = each.next();
                s.inPropagateGoalAncestors = (int) dcs.statistics.getPropagateGoalsCalls();
                if (dcs.forcedByEnvironmentToLeave(s, ancestors)) {
                    s.forcedToLeavePropagatingLoop = (int) dcs.statistics.getPropagateGoalsCalls();
                    each.remove();
                }
            }
        }

    }

    */
    public void expansionDone(Compostate<State, Action> state, HAction<Action> action, Compostate<State, Action> child) {
        if(featureMaker.using_ra_feature) {
            if (state.isControlled() && state.isStatus(Status.NONE) && !fullyExplored(state)) {
                open(state);
            }
        }
        lastExpandedTo = child;
        lastExpandedFrom = state;
        state.actionsWithFeatures.remove(action);
        //if(featureMaker.using_visits_feature){
        //    addVisit(child);
        //}

    }

    /*
    public void notifyExpansionDidntFindAnything(Compostate<State, Action> parent, HAction<State, Action> action, Compostate<State, Action> child) {
        if(featureMaker.using_ra_feature)
            if (!child.isLive() && !fullyExplored(child)) {
                open(child);
            }

    }
    */
    // Adds this state to the open queue (reopening it if was previously closed).
    public void open(Compostate<State,Action> state) {
        // System.err.println("opening" + state);
        state.live = true;
        if (!state.inOpen) {
            if (!state.hasStatusChild(Status.NONE)) {
                state.inOpen = true;
            } else { // we are reopening a state, thus we reestablish it's exploredChildren instead
                boolean openedChild = false;
                for (Pair<HAction<Action>, Compostate<State, Action>> transition : state.getExploredChildren()) {
                    Compostate<State, Action> child = transition.getSecond();
                    if (!child.isLive() && child.isStatus(Status.NONE) && !fullyExplored(child)) { // !isGoal(child)
                        open(child);
                        openedChild = true;
                    }
                }
                if (!openedChild || state.isControlled()){
                    state.inOpen = true;
                }
            }
        }
    }

    public boolean fullyExplored(Compostate<State, Action> state) {
        return state.unexploredTransitions == 0;
    }

    public boolean hasUncontrollableUnexplored(Compostate<State, Action> state) {
        return state.uncontrollableUnexploredTransitions > 0;
    }



    public void initialize(Compostate<State, Action> state) {
        if(featureMaker.using_ra_feature){
            state.live = false;
            state.inOpen = false;
            state.controlled = true;
        }

        state.actionChildStates = new HashMap<>();
        //state.estimates = new HashMap<>();
        state.targets = emptyList();
        state.arrivingPaths = new HashSet<>();
    }
    /*
    public void notifyClosedPotentiallyWinningLoop(Set<Compostate<State, Action>> loop) {
        closed_potentially_winning_loops ++;

        if(featureMaker.using_prop_feature) {
            int fng_calls = (int) dcs.statistics.getFindNewGoalsCalls();
            Queue<Compostate<State, Action>> q = new LinkedList<>();
            for(Compostate<State, Action> s : loop){
                s.inLoop = fng_calls;
                if(dcs.forcedByEnvironmentToLeave(s, loop)){
                    s.forcedToLeaveInFindNewGoals = fng_calls;
                    if(!hasUncontrollableUnexplored(s)){
                        s.inExploredUncontrollableClausure = fng_calls;
                        q.add(s);
                    }
                }
            }
            while(!q.isEmpty()) {
                Compostate<State, Action> s = q.poll();
                Set<Compostate<State, Action>> uncontrollable_children = s.getChildrenExploredThroughUncontrollable();
                for (Compostate<State, Action> t : uncontrollable_children) {
                    if(t.inExploredUncontrollableClausure != fng_calls && t.isStatus(Status.NONE)){
                        q.add(t);
                        t.inExploredUncontrollableClausure = fng_calls;
                    }
                }
            }
        }
    }
    */
    public ActionWithFeatures<State, Action> removeFromFrontier(int idx) {
        ActionWithFeatures<State, Action> stateAction = efficientRemove(idx);

        if(stateAction.state.isStatus(Status.NONE))
            this.updateUnexploredTransitions(stateAction.state, stateAction.action);

        assert stateAction.state.unexploredTransitions >= 0;

        return stateAction;
    }

    private ActionWithFeatures<State, Action> efficientRemove(int idx) {
        // removing last element is more efficient
        ActionWithFeatures<State, Action> stateAction = explorationFrontier.get(idx);
        explorationFrontier.set(idx, explorationFrontier.get(explorationFrontier.size()-1));
        explorationFrontier.remove(explorationFrontier.size()-1);
        return stateAction;
    }





    public void printFeatures(){
        System.out.println("--------------------------");
        System.out.println(explorationFrontier.size()+" "+featureMaker.n_features);
        for (ActionWithFeatures<State, Action> stateAction : explorationFrontier) {
            System.out.println(stateAction.toString());
        }
    }

    void printValues(FloatBuffer values){
        values.position(0);
        for(int i = 0; i < values.limit(); i++){
            System.out.println(values.get());
        }
    }

    void printFrontier(){
        System.out.println("Frontier: ");
        for(ActionWithFeatures<State, Action> action : explorationFrontier){
            System.out.println(action);
        }
    }

    private static boolean optToBool(CmdLineParser cmdParser, CmdLineParser.Option opt){
        Boolean b = (Boolean)cmdParser.getOptionValue(opt);
        return b != null && b;
    }
    /*

    public static void main(String[] args) {
        CmdLineParser cmdParser= new CmdLineParser();
        CmdLineParser.Option fsp_path_opt = cmdParser.addStringOption('i', "file");
        CmdLineParser.Option model_path_opt = cmdParser.addStringOption('m', "model");
        CmdLineParser.Option features_path_opt = cmdParser.addStringOption('c', "features");
        CmdLineParser.Option labels_path_opt = cmdParser.addStringOption('l', "labels");

        CmdLineParser.Option max_frontier_opt = cmdParser.addIntegerOption('f', "max_frontier");
        CmdLineParser.Option expansion_budget = cmdParser.addIntegerOption('e',"expbud");
        CmdLineParser.Option debug_opt = cmdParser.addBooleanOption('d', "debug");

        try {
            cmdParser.parse(args);
        } catch (CmdLineParser.OptionException e) {
            System.out.println("Invalid option: " + e.getMessage() + "\n");
            System.exit(0);
        }

        String fsp_path = (String)cmdParser.getOptionValue(fsp_path_opt);
        String model_path = (String)cmdParser.getOptionValue(model_path_opt);
        String features_path = (String)cmdParser.getOptionValue(features_path_opt);
        String labels_path = (String)cmdParser.getOptionValue(labels_path_opt);

        Integer max_frontier = (Integer)cmdParser.getOptionValue(max_frontier_opt);
        Integer expansionBudgetValue = (Integer)cmdParser.getOptionValue(expansion_budget);
        boolean debugging = optToBool(cmdParser, debug_opt);

        try {
            Pair<CompositeState, LTSOutput> c = compileFSP(fsp_path);
            if(expansionBudgetValue!=null) DirectedControllerSynthesisNonBlocking.expansion_budget = expansionBudgetValue;

            DirectedControllerSynthesisNonBlocking.mode = DirectedControllerSynthesisNonBlocking.HeuristicMode.TrainedAgent;
            DCSFeatures<Long, String> featureMaker = new DCSFeatures<>(features_path, labels_path, max_frontier, c.getFirst());
            FeatureBasedExplorationHeuristic<Long, String> heuristic = new FeatureBasedExplorationHeuristic<>(model_path, featureMaker, debugging);

            TransitionSystemDispatcher.hcs(heuristic, c.getFirst(), new LTSOutput() {
                public void out(String str) {System.out.print(str);}
                public void outln(String str) { System.out.println(str);}
                public void clearOutput() {}
            }, false);
        } catch (OutOfMemoryError e){
            System.out.println("OutOfMem error during exploration");
        }
    }
    */
}

