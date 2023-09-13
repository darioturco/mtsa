package MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking;

import MTSTools.ac.ic.doc.commons.relations.BinaryRelation;
import MTSTools.ac.ic.doc.commons.relations.Pair;
import MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.HAction;
////import MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.abstraction.HDist;
import ltsa.lts.CompactState;
import ltsa.lts.CompositeState;

import java.io.File;
import java.io.FileNotFoundException;
import java.nio.FloatBuffer;
import java.util.*;

import static org.junit.Assert.assertFalse;

//public class DCSFeatures {
public class DCSFeatures<State, Action> {
    public int n_features;
    public boolean using_labelsThatReach_feature;

    boolean using_ra_feature;
    boolean using_context_features;
    int state_labels_n;
    boolean using_je_feature;
    boolean using_nk_feature;
    boolean using_prop_feature;
    boolean using_visits_feature;
    boolean only_boolean;

    public int max_frontier;
    public LinkedList<ComputeFeature<State, Action>> methodFeatures;

    public HashMap<String, Integer> labels_idx = new HashMap<>();

    public HashMap<Compostate<State, Action>,HashSet<String>> labelsThatReach;
    /*
    public ComponentWiseFeatures component_wise_features;

    public int philosopher_num_states = 0;


    public class ComponentWiseFeatures{
        HashMap<String, Integer> machine_name_to_dcs_state_index;
        HashMap<Integer, String> dcs_state_index_to_machine_name;
        HashMap<String, Integer> machine_name_to_node_size;

        HashMap<String, CompactState> machine_name_to_machine;

        HashMap<String, Vector<Float>> components_by_state_feature;
        HashMap<String,Integer> component_indices;

        HashMap<String, Integer> component_types_size;

        HashSet<String> n_variable_components;
        ComponentWiseFeatures(CompositeState compiled){
            component_types_size = new HashMap<String, Integer>();
            machine_name_to_dcs_state_index = new HashMap<String, Integer>();
            dcs_state_index_to_machine_name = new HashMap<Integer, String>();
            machine_name_to_node_size = new HashMap<String, Integer>();
            machine_name_to_machine = new HashMap<String, CompactState>();
            components_by_state_feature = new HashMap<String, Vector<Float>>();
            component_indices = new HashMap<String,Integer>();
            n_variable_components = new HashSet<String>(Arrays.asList("Plant.ResponseMonitor", "Plant.RampMonitor", "Plant.HeightMonitor", "Plant.Document", "Plant.Service"));

            for(int i = 0 ; i < compiled.machines.size() ; ++i){
                CompactState machine = compiled.machines.get(i);
                dcs_state_index_to_machine_name.put(i+1,machine.name);
                machine_name_to_dcs_state_index.put(machine.name, i+1);
                if(n_variable_components.contains(this.getComponentType(machine.name))) continue;
                machine_name_to_node_size.put(machine.name, machine.states.length);
                machine_name_to_machine.put(machine.name, machine);
                Vector<Float> components_by_state = new Vector<Float>(machine.states.length);
                for(int j = 0 ; j < machine.states.length ; ++j) components_by_state.add(0.0f);
                components_by_state_feature.put(machine.name, components_by_state);

            }

            for(String name : machine_name_to_node_size.keySet()){
                String type = getComponentType(name);
                component_types_size.put(type,machine_name_to_node_size.get(name));
            }
            int t = 0;
        }

        private String getComponentType(String name) {
            int k = 0;
            while(k < name.length() && name.charAt(k)!='(') k++;
            String type = name.substring(0,k);
            return type;
        }

        ComponentWiseFeatures(){}
    };
    */
    DCSFeatures(String features_path, String labels_path, int max_frontier, CompositeState componentwise_info){
        /*HashMap<String, Integer> feature_values = null;
        if(features_path != null){
            feature_values = readFeatures(features_path);
            if(labels_path != null){
                readLabels(labels_path);
            }
        }
        this.using_ra_feature = feature_values != null && feature_values.get("ra") == 1;
        this.using_context_features = feature_values != null && feature_values.get("context") == 1;
        this.state_labels_n = feature_values != null ? feature_values.get("state_labels") : 0;
        this.using_je_feature = feature_values != null && feature_values.get("je") == 1;
        this.using_nk_feature = feature_values != null && feature_values.get("nk") == 1;
        this.using_prop_feature = feature_values != null && feature_values.get("prop") == 1;
        this.using_visits_feature = feature_values != null && feature_values.get("visits") == 1;
        this.only_boolean = feature_values != null && feature_values.get("boolean") == 1;
        this.using_labelsThatReach_feature = feature_values != null && feature_values.get("ltr") == 1;

        if(feature_values == null || feature_values.get("cbs") != 1){
            componentwise_info = null;
        }

        this.max_frontier = max_frontier;
        this.methodFeatures = new LinkedList<>();

        // this order is maintained for backward compatibility
        if(using_ra_feature) methodFeatures.add(this.ra_feature);
        if(using_context_features) methodFeatures.add(this.context_feature);
        if(state_labels_n > 0) methodFeatures.add(this.state_labels_feature);
        if(labels_idx.size() > 0) methodFeatures.add(this.action_labels_feature);

        methodFeatures.add(this.controllable_feature);
        if(!only_boolean) methodFeatures.add(this.depth_feature); // depth is added even if only_boolean == true
        methodFeatures.add(this.state_explored_feature);
        methodFeatures.add(this.state_uncontrollable_feature);
        methodFeatures.add(this.marked_feature);
        methodFeatures.add(this.child_status_feature);
        methodFeatures.add(this.child_deadlock_feature);
        methodFeatures.add(this.child_uncontrollable_feature);
        methodFeatures.add(this.child_explored_feature);
        if(using_je_feature) methodFeatures.add(this.just_explored_feature);
        if(using_nk_feature) methodFeatures.add(this.nk_feature);
        if(using_prop_feature){
            assertFalse("Not only boolean", only_boolean); // todo: make it boolean
            methodFeatures.add(this.propagation_feature);
        }
        if(using_visits_feature) {
            assertFalse("Not only boolean", only_boolean); // todo: make it boolean
            methodFeatures.add(this.visits_feature);
        }
        if(using_labelsThatReach_feature) {
            labelsThatReach = new HashMap<>();
            methodFeatures.add(labelsThatReach_feature);
        }
        if(componentwise_info != null){
            component_wise_features = new ComponentWiseFeatures(componentwise_info);
            HashSet<String> already_added_types = new HashSet<>();
            for(String name : component_wise_features.component_types_size.keySet()){

                if(!already_added_types.contains(name)) {
                    component_wise_features.component_indices.put(name, philosopher_num_states);
                    philosopher_num_states += component_wise_features.component_types_size.get(name);
                    already_added_types.add(name);
                    component_wise_features.component_indices.put(name + "_finish", philosopher_num_states);

                }
            }
            methodFeatures.add(philosopher_components_by_state);
        }
        int s = 0;
        for (ComputeFeature<State, Action> f : methodFeatures) {
            System.out.println(f);
            s += f.size();
        }
        this.n_features = s;
    */
    }
    /*


    public void update_child_labelsThatReach(Compostate<State, Action> state, HAction<State, Action> action, Compostate<State, Action> child) {
        inherit(state, child);
        labelsThatReach.get(child).add(action.getAction().toString());
        HashSet<Compostate> alreadyPropagated = new HashSet<Compostate>();
        propagateLabelsThatReachFrom(child, alreadyPropagated);
        alreadyPropagated = null;
    }
    private void inherit(Compostate<State, Action> parent, Compostate<State, Action> state) {

        HashSet<String> labels =this.labelsThatReach.get(state) ;
        if(labels==null) {
            labels = new HashSet<String>();
            this.labelsThatReach.put(state, labels);
        }
        HashSet<String> inheritance = getLabelsThatReach(parent);
        if(parent !=null && inheritance!=null)labels.addAll(inheritance);
    }
    public HashSet<String> getLabelsThatReach(Compostate<State, Action> parent) {
        return this.labelsThatReach.get(parent);
    }

    private void propagateLabelsThatReachFrom(Compostate<State,Action> state, HashSet<Compostate> alreadyPropagated) {
        BinaryRelation<HAction<State, Action>, Compostate<State, Action>> children = state.getExploredChildren();
        if(children.size()==0 || alreadyPropagated.contains(state)) return;
        alreadyPropagated.add(state);
        for(Pair<HAction<State, Action>, Compostate<State, Action>> childRelation : children){
            HashSet<String> labelsThatReachPrevious = labelsThatReach.get(state);
            HashSet<String> labelsThatReachNext = labelsThatReach.get(childRelation.getSecond());
            labelsThatReachNext.addAll(labelsThatReachPrevious);
        }
        for(Pair<HAction<State, Action>, Compostate<State, Action>> childRelation : children){
            propagateLabelsThatReachFrom(childRelation.getSecond(), alreadyPropagated);
        }

    }
    */
    interface ComputeFeature<State, Action> {
        void compute(FeatureBasedExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int start_idx);
        int size();
        boolean requiresUpdate();
    }

    private static float toFloat(boolean b) {
        return b ? 1.0f : 0.0f;
    }

    private static float toCategory(int n){
        if(n == 0) return 0.0f;
        else if(n > 100) return 1.0f;
        else if(n > 10) return 0.66f;
        else return 0.33f;
    }
    /*
    // todo: test whether ra feature still works after refactor, and possibly adapt it to only boolean version
    private final ComputeFeature<State, Action> ra_feature = new ComputeFeature<>() {
        public void compute(FeatureBasedExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            HDist firstEstimate = a.state.getEstimate(a.action).get(0);
            a.feature_vector[i] = ((float) firstEstimate.getFirst());
            a.feature_vector[i+1] = (1.0f / ((float) firstEstimate.getSecond()));
            a.feature_vector[i+2] = toFloat(a.state.inOpen);
        }
        public int size() {return 3;}
        public boolean requiresUpdate() { return true; }
    };

    private final ComputeFeature<State, Action> context_feature = new ComputeFeature<>() {
        public void compute(FeatureBasedExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            if(only_boolean){
                a.feature_vector[i] = toFloat(h.goals_found > 0);
                a.feature_vector[i+1] = toFloat(h.marked_states_found > 0);
                a.feature_vector[i+2] = toFloat(h.closed_potentially_winning_loops > 0);
            } else {
                a.feature_vector[i] = toCategory(h.goals_found);
                a.feature_vector[i+1] = toCategory(h.marked_states_found);
                a.feature_vector[i+2] = toCategory(h.closed_potentially_winning_loops);
                a.feature_vector[i+3] = (float) h.frontierSize() / (float) h.known_transitions;
            }
        }
        public int size() {return only_boolean ? 3 : 4;}
        public boolean requiresUpdate() { return true; }
    };

    private final ComputeFeature<State, Action> state_labels_feature = new ComputeFeature<>() {
        private int path2idx(LinkedList<Integer> p){
            int b = labels_idx.size();
            int s = 0;
            int pow = 1;
            for (Integer a : p) {
                s += a * pow;
                pow *= b;
            }
            return s;
        }

        public void compute(FeatureBasedExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            for (LinkedList<Integer> p : a.state.arrivingPaths) {
                if (p.size() == state_labels_n) {
                    a.feature_vector[i + path2idx(p)] = 1.0f;
                }
            }
        }
        public int size() {return (int) Math.pow(labels_idx.size(), state_labels_n);}
        public boolean requiresUpdate() { return true; }
    };

    private final ComputeFeature<State, Action> action_labels_feature = new ComputeFeature<>() {
        public void compute(FeatureBasedExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            String label = a.action.toString().replaceAll("[^a-zA-Z]", "");
            int idx;
            try {
                idx = labels_idx.get(label);
            } catch(Exception e) {
                System.out.println("Unknown action label found! "+a.action);
                throw e;
            }
            for(int j = 0; j < labels_idx.size(); j++) {
                a.feature_vector[i+j] = (toFloat(j == idx));
            }
        }
        public int size() {return labels_idx.size();}
        public boolean requiresUpdate() { return false; }
    };

    private final ComputeFeature<State, Action> controllable_feature = new ComputeFeature<>() {
        public void compute(FeatureBasedExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            a.feature_vector[i] = toFloat(a.action.isControllable());
        }
        public int size() {return 1;}
        public boolean requiresUpdate() { return false; }
    };

    private final ComputeFeature<State, Action> depth_feature = new ComputeFeature<>() {
        public void compute(FeatureBasedExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            a.feature_vector[i] = 1.0f - 1.0f / ((float) a.state.getDepth() + 1);
        }
        public int size() {return 1;}
        public boolean requiresUpdate() { return true; }
    };

    private final ComputeFeature<State, Action> state_explored_feature = new ComputeFeature<>() {
        public void compute(FeatureBasedExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            setExploredFeature(a, a.state, i);
        }
        public int size() {return 1;}
        public boolean requiresUpdate() { return true; }
    };

    private final ComputeFeature<State, Action> state_uncontrollable_feature = new ComputeFeature<>() {
        public void compute(FeatureBasedExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            setUncontrollableFeature(a, a.state, i);
        }
        public int size() {return only_boolean ? 2 : 1;}
        public boolean requiresUpdate() { return false; }
    };

    private final ComputeFeature<State, Action> marked_feature = new ComputeFeature<>() {
        public void compute(FeatureBasedExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            a.feature_vector[i] = toFloat(a.state.marked);
            a.feature_vector[i+1] = toFloat(a.childMarked);
        }
        public int size() {return 2;}
        public boolean requiresUpdate() { return true; }
    };

    private final ComputeFeature<State, Action> child_status_feature = new ComputeFeature<>() {
        public void compute(FeatureBasedExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            a.feature_vector[i] = toFloat(a.child != null && a.child.isStatus(Status.GOAL));
            a.feature_vector[i+1] = toFloat(a.child != null && a.child.isStatus(Status.ERROR));
            a.feature_vector[i+2] = toFloat(a.child != null && a.child.isStatus(Status.NONE));
        }
        public int size() {return 3;}
        public boolean requiresUpdate() { return true; }
    };

    private final ComputeFeature<State, Action> child_deadlock_feature = new ComputeFeature<>() {
        public void compute(FeatureBasedExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            a.feature_vector[i] = toFloat(a.child != null && (a.child.getTransitions().size() == 0));
        }
        public int size() {return 1;}
        public boolean requiresUpdate() { return true; }
    };

    private final ComputeFeature<State, Action> child_explored_feature = new ComputeFeature<>() {
        public void compute(FeatureBasedExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            setExploredFeature(a, a.child, i);
        }
        public int size() {return 1;}
        public boolean requiresUpdate() { return true; }
    };

    private final ComputeFeature<State, Action> child_uncontrollable_feature = new ComputeFeature<>() {
        public void compute(FeatureBasedExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            setUncontrollableFeature(a, a.child, i);
        }
        public int size() {return only_boolean ? 2 : 1;}
        public boolean requiresUpdate() { return true; }
    };

    private final ComputeFeature<State, Action> just_explored_feature = new ComputeFeature<>() {
        public void compute(FeatureBasedExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            a.feature_vector[i] = toFloat(a.state == h.lastExpandedTo);
            a.feature_vector[i + 1] = toFloat(a.state == h.lastExpandedFrom);
        }
        public int size() {return 2;}
        public boolean requiresUpdate() { return true; }
    };

    private final ComputeFeature<State, Action> nk_feature = new ComputeFeature<>() {
        public void compute(FeatureBasedExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            a.feature_vector[i] = h.problem_n;
            a.feature_vector[i + 1] = h.problem_k;
        }
        public int size() {return 2;}
        public boolean requiresUpdate() { return false; }
    };
    private final ComputeFeature<State,Action> labelsThatReach_feature = new ComputeFeature<State, Action>() {
        @Override
        public void compute(FeatureBasedExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int start_idx) {

            HashSet labels = getLabelsThatReach(a.state);

            // if(h.debugging)System.out.println(labels);

            if(labels!=null)          a.feature_vector[start_idx] = toFloat(labels.contains(a.action.toString()));
            else a.feature_vector[start_idx] = 0;
            // if(h.debugging)System.out.println("result: " + a.feature_vector[start_idx]);
        }

        @Override
        public int size() {
            return 1;
        }

        @Override
        public boolean requiresUpdate() {
            return true;
        }
    };
    private final ComputeFeature<State, Action> propagation_feature = new ComputeFeature<>() {
        private float normalizeStep(int s, long timestamp){
            if(s == -1) return 0;
            else return 1.0f / (timestamp - s + 1);
        }

        public void compute(FeatureBasedExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            int fng_calls = (int) h.dcs.getStatistics().getFindNewGoalsCalls();
            int pg_calls = (int) h.dcs.getStatistics().getPropagateGoalsCalls();
            a.feature_vector[i] = normalizeStep(a.state.inLoop, fng_calls);
            a.feature_vector[i+1] = normalizeStep(a.state.forcedToLeaveInFindNewGoals, fng_calls);
            if(a.child != null){
                a.feature_vector[i+2] = normalizeStep(a.child.inLoop, fng_calls);
                a.feature_vector[i+3] = normalizeStep(a.child.forcedToLeaveInFindNewGoals, fng_calls);
            } else {
                a.feature_vector[i+2] = 0;
                a.feature_vector[i+3] = 0;
            }
            a.feature_vector[i+4] = normalizeStep(a.state.inPropagateGoalAncestors, pg_calls);
            a.feature_vector[i+5] = normalizeStep(a.state.forcedToLeavePropagatingLoop, pg_calls);
            a.feature_vector[i+6] = normalizeStep(a.state.inExploredUncontrollableClausure, fng_calls);
        }
        public int size() {return 7;}
        public boolean requiresUpdate() { return true; }
    };

    private final ComputeFeature<State, Action> visits_feature = new ComputeFeature<>() {
        private float getStateVisitedFeature(FeatureBasedExplorationHeuristic<State, Action> h, Compostate<State,Action> state) {
            if(state == null) return 0;
            int exp_trans = h.dcs.getStatistics().getExpandedTransitions();
            float s = exp_trans;
            if(exp_trans != 0){
                for(int i = 0; i < h.dcs.ltssSize; i++){
                    s = Float.min(h.visitCounts.get(i).getOrDefault(state.getStates().get(i), 0), s);
                }
                s /= exp_trans;
            }
            return s;
        }

        public void compute(FeatureBasedExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int i) {
            a.feature_vector[i] = h.explorationPercentage;
            a.feature_vector[i+1] = getStateVisitedFeature(h, a.state);
            a.feature_vector[i+2] = getStateVisitedFeature(h, a.child);
        }
        public int size() {return 3;}
        public boolean requiresUpdate() { return true; }
    };

    private final ComputeFeature<State, Action> philosopher_components_by_state = new ComputeFeature<>(){
        public void compute(FeatureBasedExplorationHeuristic<State, Action> h, ActionWithFeatures<State, Action> a, int start_idx){
            // if(h.debugging)System.out.println(a.state.toString());
            List<State> childStates = h.dcs.getChildStates(a.state,a.action);
            // if(h.debugging)System.out.println("TO");
            // if(h.debugging)System.out.println(childStates.toString());
            // for(int i = 1 ; i < childStates.size(); ++i){
            //     if(h.debugging)System.out.print(component_wise_features.dcs_state_index_to_machine_name.get(i)+ " : " + childStates.get(i) + ", ");
            // }
            // if(h.debugging)System.out.print("\n");
            for (int i = 1 ; i < childStates.size() ; ++i){

                String machine_name = component_wise_features.dcs_state_index_to_machine_name.get(i);
                String machine_type = component_wise_features.getComponentType(machine_name);
                if(component_wise_features.n_variable_components.contains(machine_type)) continue;
                int u = component_wise_features.component_types_size.get(component_wise_features.getComponentType(machine_name));
                Integer state_number = Math.toIntExact((Long) childStates.get(i));
                a.feature_vector[start_idx + state_number + component_wise_features.component_indices.get(machine_type)] = 1.0f;
            }

            for(String type : component_wise_features.component_indices.keySet()){
                if(!type.contains("_finish") && h.debugging) System.out.print("Componentwise " + type + ": ");
                for(int i = start_idx + component_wise_features.component_indices.get(type) ; !type.contains("_finish") && h.debugging && i < start_idx + component_wise_features.component_indices.get(type+"_finish") ; ++i){

                    System.out.print(((Float)a.feature_vector[i]).toString() + " ");
                }
                if(!type.contains("_finish") && h.debugging) System.out.print("\n");
            }

            // if(h.debugging)System.out.println("Next transition");
        }
        public int size(){
            return philosopher_num_states;
        }
        public boolean requiresUpdate(){ return false;}
    };


    private void setUncontrollableFeature(ActionWithFeatures<State, Action> a, Compostate<State, Action> state, int idx){
        if(only_boolean){
            a.feature_vector[idx] = toFloat(state == null || state.uncontrollableUnexploredTransitions > 0);
            a.feature_vector[idx+1] = toFloat(state == null || state.uncontrollableTransitions > 0);
        } else {
            float v = 0.05f;
            if (state != null) {
                if (state.getTransitions().size() == 0) v = 1.0f;
                else v = 1.0f - (float) state.uncontrollableTransitions / state.getTransitions().size();
            }
            a.feature_vector[idx] = v;
        }
    }

    private void setExploredFeature(ActionWithFeatures<State, Action> a, Compostate<State, Action> state, int idx){
        float v = 0.0f;
        if(state != null){
            if(only_boolean){
                v = toFloat(state.unexploredTransitions != state.getTransitions().size());
            } else {
                if (state.getTransitions().size() == 0) v = 1.0f;
                else v = 1.0f - (float) state.unexploredTransitions / state.getTransitions().size();
            }
        }
        a.feature_vector[idx] = v;
    }

    public HashMap<String, Integer> readFeatures(String features_path){
        HashMap<String, Integer> feature_values = new HashMap<>();
        try {
            Scanner reader = new Scanner(new File(features_path));
            while (reader.hasNextLine()) {
                String data = reader.nextLine();
                String[] values = data.split(" ");
                feature_values.put(values[0], Integer.parseInt(values[1]));
            }
            reader.close();
        } catch (FileNotFoundException e) {
            System.out.println("Could not open features file.");
            e.printStackTrace();
        }
        return feature_values;
    }

    public void readLabels(String labels_path){
        LinkedList<String> action_labels = new LinkedList<>();
        if(!Objects.equals(labels_path, "mock")){
            try {
                Scanner reader = new Scanner(new File(labels_path));
                while (reader.hasNextLine()) {
                    String data = reader.nextLine();
                    action_labels.add(data);
                }
                reader.close();
            } catch (FileNotFoundException e) {
                System.out.println("Could not open labels file.");
                e.printStackTrace();
            }
        }

        labels_idx = new HashMap<>();

        int idx = 0;
        for(String action : action_labels){
            labels_idx.put(action, idx);
            idx += 1;
        }
    }*/
}
