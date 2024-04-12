package MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

public class InstanceDomain<State, Action> {
    /** dcs blocking that keep the curren partial exploration */
    public DirectedControllerSynthesisBlocking dcs;

    /** Boolean that indicate if the dcs didn't expand anything yet */
    public boolean firstExpansion;

    /** Amount of features of custom of each specific instance domain */
    public int globalCustomFeatureSize;

    /** Amount of local features of each compostote */
    public int compostateFeatureSize;

    /** Matrix where each row is the memory used for a diferent feature
     * For example the first row is used to feature 'missionComplete' in that row is saved whether a entity completed his mission or don't */
    public List<boolean[]> featureMatrix;   // TODO: cambiar por un boolean[][]

    public LinkedList<DCSFeatures.ComputeFeature<State, Action>> customFeatureList;

    public DCSFeatures featureMaker;

    InstanceDomain(DirectedControllerSynthesisBlocking dcs){
        this.dcs = dcs;
        this.firstExpansion = true;
        this.customFeatureList = new LinkedList<>();
    }

    public List<boolean[]> copyMatrix(Compostate parent){
        List<boolean[]> res = new ArrayList();
        for(int i=0 ; i<compostateFeatureSize ; i++){
            res.add(Arrays.copyOf((boolean[]) parent.customFeaturesMatrix.get(i), dcs.n));
        }
        return res;
    }

    public List<boolean[]> newFeatureMatrix(int size) {
        List<boolean[]> res = new ArrayList();
        for(int i=0 ; i<size ; i++){
            res.add(new boolean[dcs.n]);
        }
        return res;
    }

    public static InstanceDomain createInstanceDomain(DirectedControllerSynthesisBlocking dcs, DCSFeatures dcsFeatures){
        String domain = dcs.instance;
        if(domain.equals("BW")){
            return new InstanceDomainBW(dcs, dcsFeatures);
        } else if(domain.equals("TA")) {
            return new InstanceDomainTA(dcs, dcsFeatures);
        } else if(domain.equals("AT")) {
            return new InstanceDomainAT(dcs, dcsFeatures);
        } else if(domain.equals("DP")) {
            return new InstanceDomainDP(dcs, dcsFeatures);
        } else if(domain.equals("CM")) {
            return new InstanceDomainCM<>(dcs, dcsFeatures);
        } else if(domain.equals("TL")) {
            return new InstanceDomainTL<>(dcs, dcsFeatures);
        }
        return new InstanceDomain<>(dcs);
    }

    public static float toFloat(boolean b) {
        return b ? 1.0f : 0.0f;
    }
    public int size(){return featureMaker.getCustomFeatureSize(customFeatureList); }

    // TODO: todos deberian tener la misma funcion (no habria hacer que se sobreescriba)
    public void computeCustomFeature(ActionWithFeatures transition, int i){
        featureMaker.runFeaturesOfListWith(customFeatureList, transition, i);
    }

    // Each new InstanceDomain should overwrite this functions
    public boolean missionFeature(ActionWithFeatures transition){return false;}
    public void updateMatrixFeature(HAction<Action> action, Compostate<State, Action> newState){}
}

class InstanceDomainBW extends InstanceDomain{
    InstanceDomainBW(DirectedControllerSynthesisBlocking dcs, DCSFeatures dcsFeatures){
        super(dcs);
        this.globalCustomFeatureSize = 0;
        this.compostateFeatureSize = 3;
        this.featureMatrix = newFeatureMatrix(globalCustomFeatureSize);
        this.featureMaker = dcsFeatures;

        customFeatureList.add(featureMaker.entity_was_assigned_BW_feature);
        customFeatureList.add(featureMaker.entity_was_rejected_BW_feature);
        customFeatureList.add(featureMaker.entity_was_accepted_BW_feature);
        customFeatureList.add(featureMaker.almost_rejected_BW_feature);
    }

    @Override
    public boolean missionFeature(ActionWithFeatures transition){
        String label = transition.action.toString();
        if(transition.has_entity()){
            // res indicate if the number 'transition.entity' reached his mission
            boolean res = label.contains("accept") || (label.contains("reject") && transition.index == dcs.k);
            boolean[] missionRow = (boolean[])transition.state.customFeaturesMatrix.get(0);
            if(res){
                missionRow[transition.entity] = true;
            }
            return missionRow[transition.entity];
        }
        return false;
    }
}

class InstanceDomainTA extends InstanceDomain{
    public int service;

    InstanceDomainTA(DirectedControllerSynthesisBlocking dcs, DCSFeatures dcsFeatures){
        super(dcs);
        this.service = 0;
        this.globalCustomFeatureSize = 1;
        this.compostateFeatureSize = 6;
        this.featureMatrix = newFeatureMatrix(globalCustomFeatureSize);
        this.featureMaker = dcsFeatures;

        customFeatureList.add(featureMaker.actions_TA_feature);
        customFeatureList.add(featureMaker.next_entity_TA_feature);
        //customFeatureList.add(featureMaker.current_service_TA_feature);
    }

    @Override
    public boolean missionFeature(ActionWithFeatures transition){
        String label = transition.action.toString();
        if(transition.has_entity()){
            int entity = transition.entity;
            boolean res = label.contains("purchase.succ") || label.contains("purchase.fail");
            boolean[] missionRow = (boolean[])transition.state.customFeaturesMatrix.get(0);
            if(res){
                missionRow[entity] = true;
            }
            return missionRow[entity];
        }
        return false;
    }

    @Override
    public void updateMatrixFeature(HAction action, Compostate newState){
        String label = action.toString();
        int entity = ActionWithFeatures.getNumber(label, 1);
        if(entity != -1) {
            if(!((boolean[]) featureMatrix.get(0))[entity] && label.contains("query")) {
                ((boolean[]) featureMatrix.get(0))[entity] = true;
                service += 1;
            }
        }
    }
}

class InstanceDomainAT extends InstanceDomain{
    InstanceDomainAT(DirectedControllerSynthesisBlocking dcs, DCSFeatures dcsFeatures){
        super(dcs);
        this.globalCustomFeatureSize = 5;
        this.compostateFeatureSize = 0;
        this.featureMatrix = newFeatureMatrix(globalCustomFeatureSize);
        this.featureMaker = dcsFeatures;

        customFeatureList.add(featureMaker.actions_AT_feature);
        customFeatureList.add(featureMaker.first_height_AT_feature);
    }

    @Override
    public boolean missionFeature(ActionWithFeatures transition){
        if(transition.has_entity()){
            int entity = transition.entity;
            // res indicate if the number 'transition.entity' reached his mission
            boolean res = transition.action.toString().contains("land");
            boolean[] missionRow = (boolean[])transition.state.customFeaturesMatrix.get(0);
            if(res){
                missionRow[entity] = true;
            }
            return missionRow[entity];
        }
        return false;
    }
}

class InstanceDomainDP extends InstanceDomain{
    int actualPhilosofer;

    InstanceDomainDP(DirectedControllerSynthesisBlocking dcs, DCSFeatures dcsFeatures){
        super(dcs);
        this.globalCustomFeatureSize = 3;
        this.compostateFeatureSize = 4;
        this.featureMatrix = newFeatureMatrix(globalCustomFeatureSize);
        this.actualPhilosofer = 0;
        this.featureMaker = dcsFeatures;

        this.customFeatureList.add(featureMaker.philosopher_took_DP_feature);
    }

    @Override
    public boolean missionFeature(ActionWithFeatures transition){
        if(transition.parent != null && transition.has_entity()){
            int entity = transition.entity;
            // res indicate if the number 'transition.entity' reached his mission
            boolean res = transition.action.toString().contains("release");
            if(res){
                ((boolean []) transition.state.customFeaturesMatrix.get(0))[entity] = true;
            }
            return ((boolean []) transition.state.customFeaturesMatrix.get(0))[entity];
        }
        return false;
    }

    @Override
    public void updateMatrixFeature(HAction action, Compostate newState){
        String label = action.toString();
        int entity = ActionWithFeatures.getNumber(label, 1);

        if(entity != -1 && label.contains("take")){
            ((boolean[])newState.customFeaturesMatrix.get(1))[entity] = true;
        }
    }
}

class InstanceDomainCM<State, Action> extends InstanceDomain{
    InstanceDomainCM(DirectedControllerSynthesisBlocking dcs, DCSFeatures dcsFeatures){
        super(dcs);
        this.globalCustomFeatureSize = 3; // TODO: modificar
        this.compostateFeatureSize = 0;
        this.featureMatrix = newFeatureMatrix(globalCustomFeatureSize);
        this.featureMaker = dcsFeatures;
    }

    @Override
    public boolean missionFeature(ActionWithFeatures transition){
        String label = transition.action.toString();
        int entity = transition.entity;
        if(transition.has_entity()){
            // res indicate if the number 'transition.entity' reached his mission
            boolean res = label.contains("mouse") && label.contains("move") && transition.index == dcs.k;
            boolean[] missionRow = (boolean[])featureMatrix.get(0);
            if(res){
                missionRow[entity] = true;
            }
            return missionRow[entity];
        }
        return false;
    }

    @Override
    public void updateMatrixFeature(HAction action, Compostate newState){

    }

    @Override
    public void computeCustomFeature(ActionWithFeatures transition, int i){

        /*if(label.contains("mouse") && label.contains("move") && index == dcs.k){
            state.customFeatures.get(action).get(0)[entity] = true;
        }

        for(int j=0 ; j<dcs.n ; j++){
            state.customFeatures.get(action).get(1)[j] = false;
            state.customFeatures.get(action).get(2)[j] = false;
        }

        if(label.contains("mouse") && label.contains("move")) {
            int newIndex = 2 * dcs.k - index;
            int oldIndex = parent.getLastentityIndex()[entity];
            state.entityIndexes.get(action)[entity] = newIndex;
            if (newIndex > oldIndex) {
                state.customFeatures.get(action).get(1)[entity] = true;
            }
            if (newIndex < oldIndex) {
                state.customFeatures.get(action).get(2)[entity] = true;
            }
        }*/
    }
}

class InstanceDomainTL<State, Action> extends InstanceDomain{
    int lastEntity;

    InstanceDomainTL(DirectedControllerSynthesisBlocking dcs, DCSFeatures dcsFeatures){
        super(dcs);
        this.lastEntity = -1;
        this.globalCustomFeatureSize = 3; // TODO: modificar
        this.compostateFeatureSize = 0;
        this.featureMatrix = newFeatureMatrix(globalCustomFeatureSize);
        this.featureMaker = dcsFeatures;
    }

    @Override
    public boolean missionFeature(ActionWithFeatures transition){
        String label = transition.action.toString();
        int entity = transition.entity;
        if(transition.entity != -1 && entity < dcs.n){
            // res indicate if the number 'transition.entity' reached his mission
            boolean res = label.contains("put");
            boolean[] missionRow = (boolean[])featureMatrix.get(0);
            if(res){
                missionRow[entity] = true;
            }
            return missionRow[entity];
        }
        return false;
    }

    @Override
    public void updateMatrixFeature(HAction action, Compostate newState){

    }

    @Override
    public void computeCustomFeature(ActionWithFeatures transition, int i){

        /*for(int j=0 ; j<dcs.n ; j++){
            // reset custom features
        }

        if(label.contains("put") && (entity < dcs.n)){
            state.customFeatures.get(action).get(0)[entity] = true;
        }

        boolean a = entity != 2 && label.contains("get") && (entity == (lastEntity + 1)) && dcs.lastExpandedAction.getAction().toString().contains("put");
        if(entity < dcs.n){
            state.customFeatures.get(action).get(1)[entity] = a;
        }

        boolean b = entity != 2 && label.contains("put") && (entity == (lastEntity + 1)) && dcs.lastExpandedAction.getAction().toString().contains("get");
        if(entity < dcs.n){
            state.customFeatures.get(action).get(2)[entity] = b;
        }

        boolean c = entity != 2 && label.contains("return") && (entity == (lastEntity + 1)) && dcs.lastExpandedAction.getAction().toString().contains("akjghfakjgfhn");
        if(entity < dcs.n){
            state.customFeatures.get(action).get(3)[entity] = c;
        }

        state.customFeatures.get(action).get(4)[0] = firstExpansion;

        lastEntity = entity;
        firstExpansion = false;*/
    }
}