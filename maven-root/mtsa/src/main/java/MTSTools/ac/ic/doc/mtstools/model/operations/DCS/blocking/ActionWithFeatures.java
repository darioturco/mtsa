package MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking;

import MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.HAction;

import java.util.LinkedList;
import java.util.List;

public class ActionWithFeatures<State, Action> {
    public HAction<Action> action;
    public Compostate<State, Action> state;
    public Compostate<State, Action> child;
    public List<State> childStates;
    public FeatureBasedExplorationHeuristic<State, Action> heuristic;
    public boolean childMarked;
    public float[] feature_vector;

    ActionWithFeatures(Compostate<State, Action> state, HAction<Action> action, FeatureBasedExplorationHeuristic<State, Action> heuristic) {

        feature_vector = new float[heuristic.featureMaker.n_features];
        this.heuristic = heuristic;
        this.state = state;
        this.childStates = state.actionChildStates.get(action);
        this.child = this.heuristic.dcs.compostates.get(childStates);
        this.action = action;

        this.childMarked = true;
        for (int lts = 0; childMarked && lts < this.heuristic.dcs.ltssSize; ++lts)
            childMarked = this.heuristic.dcs.defaultTargets.get(lts).contains(childStates.get(lts));

        // No computo los features en java, eso lo hago luego en python
        /*int i = 0;
        for(DCSFeatures.ComputeFeature<State, Action> f : heuristic.featureMaker.methodFeatures){
            f.compute(heuristic, this, i);
            i += f.size();
        }*/
    }

    public void updateFeatures(){
        /*int i = 0;
        for(DCSFeatures.ComputeFeature<State, Action> f : heuristic.featureMaker.methodFeatures){
            if(f.requiresUpdate()){
                f.compute(heuristic, this, i);
            }
            i += f.size();
        }*/
    }


    public String toString(){
        StringBuilder r = new StringBuilder(state.toString() + " | " + action.toString() + " | ");
        for(float f : feature_vector){
            r.append(" ").append(f);
        }
        return r.toString();
    }
}
