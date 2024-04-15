package MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking;

import MTSTools.ac.ic.doc.commons.relations.Pair;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

/** The expancion frontier is a list of ActionWithFeatures (represents a transition to expand).
 */
public class ActionWithFeatures<State, Action> {
    public HAction<Action> action;
    public Compostate<State, Action> state;
    public Compostate<State, Action> parent;
    public Compostate<State, Action> child;
    public List<State> childStates;
    public DirectedControllerSynthesisBlocking<State, Action> dcs;
    public boolean childMarked;
    public int entity;
    public int index;
    public int expansionStep;
    public float[] featureVector;

    ActionWithFeatures(Compostate<State, Action> state, HAction<Action> action, Compostate<State, Action> parent) {
        this.state = state;
        this.action = action;
        this.parent = parent;
        this.dcs = state.dcs;
        this.childStates = state.actionChildStates.get(action);
        this.child = this.dcs.compostates.get(childStates);
        this.entity = getNumber(action.toString(), 1);
        this.index = getNumber(action.toString(), 2);
        this.featureVector = new float[dcs.nfeatures];
        this.expansionStep = dcs.expansionStep;

        if(parent == null){
            state.customFeaturesMatrix = dcs.instanceDomain.newFeatureMatrix(dcs.instanceDomain.compostateFeatureSize);
            state.entityIndexes.put(action, new int[dcs.n]);
        }else{
            state.customFeaturesMatrix = dcs.instanceDomain.copyMatrix(parent);
            state.entityIndexes.put(action, Arrays.copyOf(parent.getLastentityIndex(), dcs.n));
        }

        this.childMarked = true;
        for (int lts = 0; childMarked && lts < this.dcs.ltssSize; ++lts)
            childMarked = this.dcs.defaultTargets.get(lts).contains(childStates.get(lts));
    }

    public void resetFeatureVectorSlice(int init, int end){
        for(int i=init ; i<end ; i++){
            featureVector[i] = 0.0f;
        }
    }

    public Pair<Compostate<State, Action>, HAction<Action>> toPair(){
        return new Pair<>(state, action);
    }

    public String arrayBoolToString(float[] arr){
        String res = "[";
        for(float b : arr){
            res += b + ", ";
        }

        int l = res.length();
        return res.substring(0, l-2) + "]";
    }

    public String toString(){
        return state.toString() + " | " + action.toString();
        //return state.toString() + " | " + action.toString() + " | " + arrayBoolToString(featureVector);
    }

    public static int getNumber(String label, int n){
        String[] values = label.split("\\.");
        for(String s : values){
            if(s.matches("\\d*")){
                n--;
                if(n == 0){
                    return Integer.parseInt(s);
                }
            }
        }
        return -1;
    }

    public boolean has_entity(){
        return entity != -1;
    }

    public boolean has_index(){
        return index != -1;
    }

    public void updateFeatures(){
        int i = 0;
        RLExplorationHeuristic heuristic = ((RLExplorationHeuristic) dcs.heuristic);
        LinkedList<DCSFeatures.ComputeFeature<State, Action>> methods = heuristic.featureMaker.methodFeatures;
        for(DCSFeatures.ComputeFeature<State, Action> f : methods){
            if(f.requiresUpdate()){
                f.compute(heuristic, this, i);
            }
            i += f.size();
        }
    }
}