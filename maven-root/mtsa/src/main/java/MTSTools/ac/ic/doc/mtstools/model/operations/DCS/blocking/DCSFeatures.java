package MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking;

import MTSTools.ac.ic.doc.commons.relations.Pair;
import java.util.*;

public class DCSFeatures<State, Action> {
    /** Amount of features is updated when all the features are added in methodFeatures depending on the method group */
    public int nfeatures;

    /** Amount of action (removing the index) on the alphabet */
    public int nactions;

    /** The amount of random features added of the method group is RRL */
    public int randomAmount = 100;

    public LinkedList<ComputeFeature<State, Action>> methodFeatures;
    public HashMap<String, Integer> labels_idx = new HashMap<>();
    public RLExplorationHeuristic heuristic;
    public DirectedControllerSynthesisBlocking<State, Action> dcs;



    DCSFeatures(String featureGroup, RLExplorationHeuristic heuristic){
        this.heuristic = heuristic;
        this.dcs = heuristic.dcs;
        this.dcs.instanceDomain = InstanceDomain.createInstanceDomain(this.dcs, this);
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

        if(featureGroup.equals("RL")){
            methodFeatures.add(this.action_labels_feature);
            methodFeatures.add(this.state_labels_feature);
            methodFeatures.add(this.controllable_feature);
            methodFeatures.add(this.marked_feature);
            methodFeatures.add(this.context_feature);
            methodFeatures.add(this.child_status_feature);
            methodFeatures.add(this.uncontrollable_neighborhood_feature);
            methodFeatures.add(this.state_child_explored_feature);
            methodFeatures.add(this.just_explored_feature);

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
            methodFeatures.add(this.has_entity_feature);
            methodFeatures.add(this.has_index_feature);
            methodFeatures.add(this.child_deadlock_feature);
            methodFeatures.add(this.mission_feature);
            methodFeatures.add(this.custom_feature);

        }else if(featureGroup.equals("RRL")){   // RL with random features
            randomAmount = 100; // Amount of random features to be added

            methodFeatures.add(this.action_labels_feature);
            methodFeatures.add(this.state_labels_feature);
            methodFeatures.add(this.controllable_feature);
            methodFeatures.add(this.marked_action_feature);
            methodFeatures.add(this.context_feature);
            methodFeatures.add(this.child_status_feature);
            methodFeatures.add(this.uncontrollable_neighborhood_feature);
            methodFeatures.add(this.state_child_explored_feature);
            methodFeatures.add(this.just_explored_feature);
            methodFeatures.add(this.random_feature);
        }

        setAmountOfFeatures();
    }



    public void runFeaturesOfListWith(LinkedList<ComputeFeature<State, Action>> list, ActionWithFeatures<State, Action> a, int initial){
        int i = initial;
        for (ComputeFeature<State, Action> f : list){
            f.compute((RLExplorationHeuristic<State, Action>) dcs.heuristic, a, i);
            i += f.size();
        }
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

    public int getCustomFeatureSize(LinkedList<ComputeFeature<State, Action>> list){
        int res = 0;
        for(ComputeFeature<State, Action> f : list){
            res += f.size();
        }
        return res;
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

    private final ComputeFeature<State, Action> has_entity_feature = new ComputeFeature<>() {
        public void compute(RLExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            a.featureVector[i] = toFloat(a.has_entity());
        }
        public int size() {return 1;}
        public boolean requiresUpdate() { return true; }
        public String toString(){return "has_entity_feature";}
    };

    private final ComputeFeature<State, Action> has_index_feature = new ComputeFeature<>() {
        public void compute(RLExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            a.featureVector[i] = toFloat(a.index != -1);
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

    private final ComputeFeature<State, Action> to_error_feature = new ComputeFeature<>() {
        public void compute(RLExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            boolean go_to_error = false;
            for(State s : a.childStates){
                if(s.equals(-1L)){
                    go_to_error = true;
                }
            }
            a.featureVector[i] = toFloat(go_to_error);
        }
        public int size() {return 1;}
        public boolean requiresUpdate() { return true; }
        public String toString(){return "to_error_feature";}
    };

    private final ComputeFeature<State, Action> return_to_start_feature = new ComputeFeature<>() {
        public void compute(RLExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            boolean go_to_zero = false;
            for(int j=0 ; j<a.childStates.size() ; j++){
                if(a.state.states.get(j) != a.childStates.get(j) &&  a.childStates.get(j).equals(0L)){
                    go_to_zero = true;
                }
            }
            a.featureVector[i] = toFloat(go_to_zero);
        }
        public int size() {return 1;}
        public boolean requiresUpdate() { return true; }
        public String toString(){return "return_to_start_feature";}
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

    // Here are all the custom features, those are added  in the class InstanceDomiain deppending the instance to solve.

    // BWFeatures
    public final ComputeFeature<State, Action> entity_was_assigned_BW_feature = new ComputeFeature<>() {
        public void compute(RLExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            if(a.has_entity() && a.action.toString().contains("assign")){
                (a.state.customFeaturesMatrix.get(1))[a.entity] = true;
            }

            boolean entity_was_assigned = a.has_entity() && (a.state.customFeaturesMatrix.get(1))[a.entity];
            a.featureVector[i] = toFloat(entity_was_assigned);
        }
        public int size() {return 1;}
        public boolean requiresUpdate() { return true; }
        public String toString(){return "entity_was_assigned_BW_feature";}
    };

    public final ComputeFeature<State, Action> entity_was_rejected_BW_feature = new ComputeFeature<>() {
        public void compute(RLExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            if(a.has_entity() && a.action.toString().contains("reject")){
                (a.state.customFeaturesMatrix.get(2))[a.entity] = true;
            }

            boolean entity_was_rejected = a.has_entity() && (a.state.customFeaturesMatrix.get(2))[a.entity];
            a.featureVector[i] = toFloat(entity_was_rejected);
        }
        public int size() {return 1;}
        public boolean requiresUpdate() { return true; }
        public String toString(){return "entity_was_rejected_BW_feature";}
    };

    public final ComputeFeature<State, Action> entity_was_accepted_BW_feature = new ComputeFeature<>() {
        public void compute(RLExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            if(a.has_entity() && a.action.toString().contains("accept")){
                (a.state.customFeaturesMatrix.get(3))[a.entity] = true;
            }

            boolean entity_was_accepted = a.has_entity() && (a.state.customFeaturesMatrix.get(3))[a.entity];
            a.featureVector[i] = toFloat(entity_was_accepted);
        }
        public int size() {return 1;}
        public boolean requiresUpdate() { return true; }
        public String toString(){return "entity_was_accepted_BW_feature";}
    };

    public final ComputeFeature<State, Action> almost_rejected_BW_feature = new ComputeFeature<>() {
        public void compute(RLExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            a.featureVector[i] = toFloat(a.has_index() && a.index == (dcs.k-1));
        }
        public int size() {return 1;}
        public boolean requiresUpdate() { return true; }
        public String toString(){return "almost_rejected_BW_feature";}
    };

    // AT Features
    public final ComputeFeature<State, Action> actions_AT_feature = new ComputeFeature<>() {
        public void compute(RLExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            if(a.has_entity()){
                String label = a.action.toString();
                if(label.contains("descend")) {
                    (a.state.customFeaturesMatrix.get(1))[a.entity] = true;
                }else if(label.contains("approach")) {
                    (a.state.customFeaturesMatrix.get(2))[a.entity] = true;
                }
            }

            a.featureVector[i] = toFloat(a.has_entity() && (a.state.customFeaturesMatrix.get(1))[a.entity]);
            a.featureVector[i+1] = toFloat(a.has_entity() && (a.state.customFeaturesMatrix.get(2))[a.entity]);
        }
        public int size() {return 2;}
        public boolean requiresUpdate() { return true; }
        public String toString(){return "actions_AT_feature";}
    };

    public final ComputeFeature<State, Action> first_height_AT_feature = new ComputeFeature<>() {
        public void compute(RLExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            a.featureVector[i] = toFloat(a.index == 0);
        }
        public int size() {return 1;}
        public boolean requiresUpdate() { return true; }
        public String toString(){return "first_height_AT_feature";}
    };

    // TA Features
    public final ComputeFeature<State, Action> actions_TA_feature = new ComputeFeature<>() {
        public void compute(RLExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            if(a.has_entity()){
                String label = a.action.toString();
                if(label.contains("commited") && !label.contains("un")){
                    (a.state.customFeaturesMatrix.get(1))[a.entity] = true;
                }else if(label.contains("uncommitted")){
                    (a.state.customFeaturesMatrix.get(2))[a.entity] = true;
                }
            }

            a.featureVector[i] = toFloat(a.has_entity() && (a.state.customFeaturesMatrix.get(1))[a.entity]);
            a.featureVector[i+1] = toFloat(a.has_entity() && (a.state.customFeaturesMatrix.get(2))[a.entity]);
        }
        public int size() {return 2;}
        public boolean requiresUpdate() { return true; }
        public String toString(){return "actions_TA_feature";}
    };

    public final ComputeFeature<State, Action> next_entity_TA_feature = new ComputeFeature<>() {
        public void compute(RLExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            a.featureVector[i] = 0.0f;
            a.featureVector[i+1] = 0.0f;

            if(a.has_entity() && a.action.toString().contains("query") && a.entity-1 >= 0){
                if(a.state.customFeaturesMatrix.get(4)[a.entity-1])
                    a.featureVector[i] = 1.0f;
                if(a.state.customFeaturesMatrix.get(5)[a.entity-1])
                    a.featureVector[i+1] = 1.0f;
            }
        }
        public int size() {return 2;}
        public boolean requiresUpdate() { return true; }
        public String toString(){return "next_entity_TA_feature";}
    };

    public final ComputeFeature<State, Action> current_service_TA_feature = new ComputeFeature<>() {
        public void compute(RLExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            int service = ((InstanceDomainTA)dcs.instanceDomain).service;
            boolean entity_is_current_service = a.has_entity() && service == a.entity;
            a.featureVector[i] = toFloat(entity_is_current_service);

            boolean entity_is_next_service = a.has_entity() && (service == a.entity+1);
            a.featureVector[i+1] = toFloat(entity_is_next_service);

            boolean entity_is_greater_service = a.has_entity() && (service > a.entity+1);
            a.featureVector[i+2] = toFloat(entity_is_greater_service);
        }
        public int size() {return 3;}
        public boolean requiresUpdate() { return true; }
        public String toString(){return "current_service_TA_feature";}
    };

    // DP Features
    public final ComputeFeature<State, Action> philosopher_took_DP_feature = new ComputeFeature<>() {
        public void compute(RLExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            boolean philosopher_took = a.has_entity() && (a.state.customFeaturesMatrix.get(1))[a.entity];
            a.featureVector[i] = toFloat(philosopher_took);
        }
        public int size() {return 1;}
        public boolean requiresUpdate() { return true; }
        public String toString(){return "philosopher_took_DP_feature";}
    };

    // TL Features
    public final ComputeFeature<State, Action> load_machine_TL_feature = new ComputeFeature<>() {
        public void compute(RLExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            String label = a.action.toString();
            int entity = a.entity;
            if(label.contains("return") || label.contains("put")){
                entity -= 1;
            }

            boolean load_buffer = a.has_entity() && entity < dcs.n && (a.state.customFeaturesMatrix.get(1))[entity];
            a.featureVector[i] = toFloat(load_buffer);
        }
        public int size() {return 1;}
        public boolean requiresUpdate() { return true; }
        public String toString(){return "load_machine_TL_feature";}
    };

    public final ComputeFeature<State, Action> buffer_returned_TL_feature = new ComputeFeature<>() {
        public void compute(RLExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            String label = a.action.toString();
            int entity = a.entity;
            if(label.contains("return") || label.contains("put")){
                entity -= 1;
            }

            boolean buffer_returned = a.entity >= 0 && entity < dcs.n && (a.state.customFeaturesMatrix.get(2))[entity];
            a.featureVector[i] = toFloat(buffer_returned);
        }
        public int size() {return 1;}
        public boolean requiresUpdate() { return true; }
        public String toString(){return "buffer_returned_TL_feature";}
    };

    public final ComputeFeature<State, Action> last_get_TL_feature = new ComputeFeature<>() {
        public void compute(RLExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            boolean last_get = a.action.toString().contains("get") && a.entity == dcs.n;
            a.featureVector[i] = toFloat(last_get);
        }
        public int size() {return 1;}
        public boolean requiresUpdate() { return true; }
        public String toString(){return "last_get_TL_feature";}
    };

    // CM Features
    public final ComputeFeature<State, Action> mouse_closer_CM_feature = new ComputeFeature<>() {
        public void compute(RLExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            String label = a.action.toString();

            boolean mouse_closer = label.contains("mouse") && label.contains("move") && a.state.mousePositions[a.entity] < (2*dcs.n - a.index);
            a.featureVector[i] = toFloat(mouse_closer);
        }
        public int size() {return 1;}
        public boolean requiresUpdate() { return true; }
        public String toString(){return "mouse_closer_CM_feature";}
    };
}