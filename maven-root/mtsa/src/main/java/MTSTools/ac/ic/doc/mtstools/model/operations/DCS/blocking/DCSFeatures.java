package MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking;

import MTSTools.ac.ic.doc.commons.relations.Pair;
import java.util.*;

public class DCSFeatures<State, Action> {
    public int nfeatures;
    public int nactions;

    public int randomAmount = 100;

    public LinkedList<ComputeFeature<State, Action>> methodFeatures;
    public HashMap<String, Integer> labels_idx = new HashMap<>();

    public RLExplorationHeuristic heuristic;
    public DirectedControllerSynthesisBlocking<State, Action> dcs;



    DCSFeatures(String featureGroup, RLExplorationHeuristic heuristic){
        this.heuristic = heuristic;
        this.dcs = heuristic.dcs;
        this.methodFeatures = new LinkedList<>();

        SortedSet<String> labels = new TreeSet<>();
        for(int i=0 ; i<dcs.alphabet.actions.size() ; i++){
            labels.add(((String) dcs.alphabet.actions.get(i)).replaceAll("\\d", ""));
        }
        if(dcs.instance.equals("AT")){ // AT 1-1 don't have the label air.crash. then we need to add it manualy or the feature size change only in 1-1.
            labels.add("air.crash.");
        }

        int i = 0;
        for(String l : labels){
            labels_idx.put(l, i);
            i++;
        }
        nactions = labels_idx.size();

        if(featureGroup.equals("2-2")){
            methodFeatures.add(this.action_labels_feature);
            methodFeatures.add(this.state_labels_feature);
            methodFeatures.add(this.controllable_feature);
            methodFeatures.add(this.marked_feature);
            methodFeatures.add(this.context_feature);
            methodFeatures.add(this.child_status_feature);
            methodFeatures.add(this.uncontrollable_neighborhood_feature);
            methodFeatures.add(this.state_child_explored_feature);
            methodFeatures.add(this.just_explored_feature);

        }else if(featureGroup.equals("LRL")){
            methodFeatures.add(this.action_labels_feature);
            methodFeatures.add(this.state_labels_feature);
            methodFeatures.add(this.controllable_feature);
            methodFeatures.add(this.marked_action_feature);
            methodFeatures.add(this.context_feature);
            methodFeatures.add(this.child_status_feature);
            methodFeatures.add(this.uncontrollable_neighborhood_feature);
            methodFeatures.add(this.state_child_explored_feature);
            methodFeatures.add(this.just_explored_feature);
            methodFeatures.add(this.child_deadlock_feature);
            methodFeatures.add(this.mission_feature);
            methodFeatures.add(this.has_index_feature);

        }else if(featureGroup.equals("CRL")){
            methodFeatures.add(this.action_labels_feature);
            methodFeatures.add(this.state_labels_feature);
            methodFeatures.add(this.controllable_feature);
            methodFeatures.add(this.marked_action_feature);
            methodFeatures.add(this.context_feature);
            methodFeatures.add(this.child_status_feature);
            methodFeatures.add(this.uncontrollable_neighborhood_feature);
            methodFeatures.add(this.state_child_explored_feature);
            methodFeatures.add(this.just_explored_feature);
            methodFeatures.add(this.child_deadlock_feature);
            methodFeatures.add(this.mission_feature);
            methodFeatures.add(this.has_index_feature);
            methodFeatures.add(this.custom_feature);

        }else if(featureGroup.equals("RRL")){
            methodFeatures.add(this.action_labels_feature);
            methodFeatures.add(this.state_labels_feature);
            methodFeatures.add(this.controllable_feature);
            methodFeatures.add(this.marked_action_feature);
            methodFeatures.add(this.context_feature);
            methodFeatures.add(this.child_status_feature);
            methodFeatures.add(this.uncontrollable_neighborhood_feature);
            methodFeatures.add(this.state_child_explored_feature);
            methodFeatures.add(this.just_explored_feature);
            methodFeatures.add(this.child_deadlock_feature);
            methodFeatures.add(this.mission_feature);
            methodFeatures.add(this.has_index_feature);
            methodFeatures.add(this.random_feature);
        }

        setAmountOfFeatures();
    }

    public void setAmountOfFeatures(){
        int nfeatures = 0;
        for (ComputeFeature<State, Action> f : methodFeatures) {
            nfeatures += f.size();
        }

        this.nfeatures = nfeatures;
        this.heuristic.nfeatures = nfeatures;
        this.heuristic.dcs.nfeatures = nfeatures;
    }



    interface ComputeFeature<State, Action> {
        void compute(RLExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int start_idx);
        int size();
        boolean requiresUpdate();
        String toString();
    }

    public static float toFloat(boolean b) {
        return b ? 1.0f : 0.0f;
    }

    private final ComputeFeature<State, Action> context_feature = new ComputeFeature<>() {
        public void compute(RLExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            a.featureVector[i] = toFloat(h.goals_found > 0);
            a.featureVector[i+1] = toFloat(h.marked_states_found > 0);
            a.featureVector[i+2] = toFloat(h.closed_potentially_winning_loops > 0);
        }
        public int size() {return 3;}
        public boolean requiresUpdate() { return true; }
        public String toString(){return "context_feature";}
    };

    private final ComputeFeature<State, Action> state_labels_feature = new ComputeFeature<>() {
        public void compute(RLExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            a.resetFeatureVectorSlice(i, i+nactions);
            int idx;
            for(Pair<HAction<Action>, Compostate<State, Action>> parent : a.state.getParents()){
                String actionWithoutIndex = parent.getFirst().toString().replaceAll("\\d", "");
                idx = labels_idx.get(actionWithoutIndex);
                a.featureVector[i+idx] = 1.0f;
            }
        }
        public int size() {return nactions;}
        public boolean requiresUpdate() { return true; }
        public String toString(){return "state_labels_feature";}
    };

    private final ComputeFeature<State, Action> action_labels_feature = new ComputeFeature<>() {
        public void compute(RLExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            a.resetFeatureVectorSlice(i, i+nactions);
            String actionWithoutIndex = a.action.toString().replaceAll("\\d", "");
            int idx = labels_idx.get(actionWithoutIndex);
            a.featureVector[i+idx] = 1.0f;
        }
        public int size() {return nactions;}
        public boolean requiresUpdate() { return true; }
        public String toString(){return "action_labels_feature";}
    };

    private final ComputeFeature<State, Action> controllable_feature = new ComputeFeature<>() {
        public void compute(RLExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            a.featureVector[i] = toFloat(a.action.isControllable());
        }
        public int size() {return 1;}
        public boolean requiresUpdate() { return true; }
        public String toString(){return "controllable_feature";}
    };

    private final ComputeFeature<State, Action> state_child_explored_feature = new ComputeFeature<>() {
        public void compute(RLExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            a.featureVector[i] = toFloat(a.state != null && a.state.unexploredTransitions != a.state.getTransitions().size());
            a.featureVector[i+1] = toFloat(a.child != null && a.child.unexploredTransitions != a.child.getTransitions().size());
        }
        public int size() {return 2;}
        public boolean requiresUpdate() { return true; }
        public String toString(){return "state_child_explored_feature";}
    };

    private final ComputeFeature<State, Action> marked_feature = new ComputeFeature<>() {
        public void compute(RLExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            a.featureVector[i] = toFloat(a.state.isMarked());
        }
        public int size() {return 1;}
        public boolean requiresUpdate() { return true; }
        public String toString(){return "marked_feature";}
    };

    private final ComputeFeature<State, Action> marked_action_feature = new ComputeFeature<>() {
        public void compute(RLExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            a.featureVector[i] = toFloat(a.state.isMarked());
            a.featureVector[i+1] = toFloat(a.childMarked);
        }
        public int size() {return 2;}
        public boolean requiresUpdate() { return true; }
        public String toString(){return "marked_action_feature";}
    };

    private final ComputeFeature<State, Action> child_status_feature = new ComputeFeature<>() {
        public void compute(RLExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            a.featureVector[i] = toFloat(a.child != null && a.child.isStatus(Status.GOAL));
            a.featureVector[i+1] = toFloat(a.child != null && a.child.isStatus(Status.ERROR));
            a.featureVector[i+2] = toFloat(a.child != null && a.child.isStatus(Status.NONE));
        }
        public int size() {return 3;}
        public boolean requiresUpdate() { return true; }
        public String toString(){return "child_status_feature";}
    };

    private final ComputeFeature<State, Action> child_deadlock_feature = new ComputeFeature<>() {
        public void compute(RLExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            a.featureVector[i] = toFloat(a.child != null && (a.child.getTransitions().size() == 0));
        }
        public int size() {return 1;}
        public boolean requiresUpdate() { return true; }
        public String toString(){return "child_deadlock_feature";}
    };

    private final ComputeFeature<State, Action> uncontrollable_neighborhood_feature = new ComputeFeature<>() {
        public void compute(RLExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            a.featureVector[i] = toFloat(a.state.uncontrollableUnexploredTransitions > 0);
            a.featureVector[i+1] = toFloat(a.state.uncontrollableTransitions > 0);
            a.featureVector[i+2] = toFloat(a.child == null || a.child.uncontrollableUnexploredTransitions > 0);
            a.featureVector[i+3] = toFloat(a.child == null || a.child.uncontrollableTransitions > 0);
        }
        public int size() {return 4;}
        public boolean requiresUpdate() { return true; }
        public String toString(){return "uncontrollable_neighborhood_feature";}
    };

    private final ComputeFeature<State, Action> just_explored_feature = new ComputeFeature<>() {
        public void compute(RLExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            a.featureVector[i] = toFloat(a.state == h.lastExpandedTo);
            a.featureVector[i + 1] = toFloat(a.state == h.lastExpandedFrom);
        }
        public int size() {return 2;}
        public boolean requiresUpdate() { return true; }
        public String toString(){return "just_explored_feature";}
    };

    private final ComputeFeature<State, Action> has_index_feature = new ComputeFeature<>() {
        public void compute(RLExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            a.featureVector[i] = toFloat(a.has_entity());
        }
        public int size() {return 1;}
        public boolean requiresUpdate() { return true; }
        public String toString(){return "has_index_feature";}
    };

    private final ComputeFeature<State, Action> random_feature = new ComputeFeature<>() {
        public void compute(RLExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            for(int j=0 ; j<randomAmount ; j++){
                a.featureVector[i+j] = toFloat(Math.random() < 0.5);
            }
        }
        public int size() {return randomAmount;}
        public boolean requiresUpdate() { return true; }
        public String toString(){return "random_feature";}
    };





    private final ComputeFeature<State, Action> mission_feature = new ComputeFeature<>() {
        public void compute(RLExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            boolean mission = dcs.instanceDomain.missionFeature(a);
            a.featureVector[i] = toFloat(mission);
        }
        public int size() {return 1;}
        public boolean requiresUpdate() { return true; }
        public String toString(){return "mission_feature";}
    };

    private final ComputeFeature<State, Action> custom_feature = new ComputeFeature<>() {
        public void compute(RLExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            dcs.instanceDomain.computeCustomFeature(a, i);
        }
        public int size() {return dcs.instanceDomain.size();}
        public boolean requiresUpdate() { return true; }
        public String toString(){return "custom_feature";}
    };
}
