package MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking;

import java.util.Arrays;
import java.util.LinkedList;

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
    public boolean[][] featureMatrix;

    /** List of Objets ComputeFeature, each object compute a set of features custom for each type of instance */
    public LinkedList<DCSFeatures.ComputeFeature<State, Action>> customFeatureList;

    /** DCSFeatures is the object responasable of calculate all the normal features and the custom features */
    public DCSFeatures featureMaker;

    InstanceDomain(DirectedControllerSynthesisBlocking dcs){
        this.dcs = dcs;
        this.firstExpansion = true;
        this.customFeatureList = new LinkedList<>();
    }

    public boolean[][] copyMatrix(Compostate parent){
        boolean[][] res = new boolean[compostateFeatureSize][dcs.n];
        for(int i=0 ; i<compostateFeatureSize ; i++){
            res[i] = Arrays.copyOf(parent.customFeaturesMatrix[i], dcs.n);
        }
        return res;
    }

    public boolean[][] newFeatureMatrix(int size) {
        boolean[][] res = new boolean[compostateFeatureSize][dcs.n];
        for(int i=0 ; i<size ; i++){
            res[i] = new boolean[dcs.n];
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
            return new InstanceDomainCM(dcs, dcsFeatures);
        } else if(domain.equals("TL")) {
            return new InstanceDomainTL(dcs, dcsFeatures);
        }
        return new InstanceDomain<>(dcs);
    }

    public void initCompostate(Compostate state, Compostate parent){}

    public int size(){return featureMaker.getCustomFeatureSize(customFeatureList); }

    public void computeCustomFeature(ActionWithFeatures transition, int i){
        featureMaker.runFeaturesOfListWith(customFeatureList, transition, i);
    }

    // Each new InstanceDomain should overwrite this functions
    public boolean missionFeature(ActionWithFeatures transition){return false;}
    public void updateMatrixFeature(HAction<Action> action, Compostate<State, Action> state){}
}

class InstanceDomainBW extends InstanceDomain{
    InstanceDomainBW(DirectedControllerSynthesisBlocking dcs, DCSFeatures dcsFeatures){
        super(dcs);
        this.globalCustomFeatureSize = 0;
        this.compostateFeatureSize = 4;
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
            boolean[] missionRow = transition.state.customFeaturesMatrix[0];
            if(label.contains("accept") || (label.contains("reject") && transition.index == dcs.k)){
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
            boolean[] missionRow = transition.state.customFeaturesMatrix[0];
            if(label.contains("purchase.succ") || label.contains("purchase.fail")){
                missionRow[entity] = true;
            }
            return missionRow[entity];
        }
        return false;
    }

    @Override
    public void updateMatrixFeature(HAction action, Compostate state){
        String label = action.toString();
        int entity = ActionWithFeatures.getNumber(label, 1);
        if(entity != -1) {
            if(!(featureMatrix[0][entity] && label.contains("query"))) {
                featureMatrix[0][entity] = true;
                service += 1;
            }
        }
    }
}

class InstanceDomainAT extends InstanceDomain{
    InstanceDomainAT(DirectedControllerSynthesisBlocking dcs, DCSFeatures dcsFeatures){
        super(dcs);
        this.globalCustomFeatureSize = 0;
        this.compostateFeatureSize = 3;
        this.featureMatrix = newFeatureMatrix(globalCustomFeatureSize);
        this.featureMaker = dcsFeatures;

        customFeatureList.add(featureMaker.actions_AT_feature);
        customFeatureList.add(featureMaker.first_height_AT_feature);
    }

    @Override
    public boolean missionFeature(ActionWithFeatures transition){
        if(transition.has_entity()){
            int entity = transition.entity;
            boolean[] missionRow = transition.state.customFeaturesMatrix[0];
            if(transition.action.toString().contains("land")){
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
            if(transition.action.toString().contains("release")){
                transition.state.customFeaturesMatrix[0][entity] = true;
            }

            return transition.state.customFeaturesMatrix[0][entity];
        }
        return false;
    }

    @Override
    public void updateMatrixFeature(HAction action, Compostate state){
        String label = action.toString();
        int entity = ActionWithFeatures.getNumber(label, 1);

        if(entity != -1 && label.contains("take")){
            state.customFeaturesMatrix[1][entity] = true;
        }
    }
}

class InstanceDomainCM extends InstanceDomain{
    InstanceDomainCM(DirectedControllerSynthesisBlocking dcs, DCSFeatures dcsFeatures){
        super(dcs);
        this.globalCustomFeatureSize = 0;
        this.compostateFeatureSize = 1;
        this.featureMatrix = newFeatureMatrix(globalCustomFeatureSize);
        this.featureMaker = dcsFeatures;

        this.customFeatureList.add(featureMaker.mouse_closer_CM_feature);
    }

    @Override
    public boolean missionFeature(ActionWithFeatures transition){
        if(transition.has_entity()){
            int entity = transition.entity;
            String label = transition.action.toString();
            boolean[] missionRow = transition.state.customFeaturesMatrix[0];
            if(label.contains("mouse") && label.contains("move") && transition.index == dcs.k){
                missionRow[entity] = true;
            }
            return missionRow[entity];
        }
        return false;
    }

    @Override
    public void updateMatrixFeature(HAction action, Compostate state){
        String label = action.toString();

        if(label.contains("mouse") && label.contains("move")){
            int entity = ActionWithFeatures.getNumber(label, 1);
            int index = ActionWithFeatures.getNumber(label, 1);
            state.entityPositions[entity] = (2*dcs.n - index);
        }
    }

    @Override
    public void initCompostate(Compostate state, Compostate parent){
        if(parent == null){
            state.entityPositions = new int[dcs.n];
        }else{
            state.entityPositions = Arrays.copyOf(parent.entityPositions, dcs.n);
        }
    }
}

class InstanceDomainTL extends InstanceDomain{
    InstanceDomainTL(DirectedControllerSynthesisBlocking dcs, DCSFeatures dcsFeatures){
        super(dcs);
        this.globalCustomFeatureSize = 0;
        this.compostateFeatureSize = 3;
        this.featureMatrix = newFeatureMatrix(globalCustomFeatureSize);
        this.featureMaker = dcsFeatures;

        this.customFeatureList.add(featureMaker.load_machine_TL_feature);
        this.customFeatureList.add(featureMaker.buffer_returned_TL_feature);
        this.customFeatureList.add(featureMaker.last_get_TL_feature);
    }

    @Override
    public boolean missionFeature(ActionWithFeatures transition){
        if(transition.parent != null && transition.has_entity()){
            int entity = transition.entity;
            String label = transition.action.toString();
            if(label.contains("return") || (label.contains("get") && entity == dcs.n)){
                entity -= 1;
            }

            if(label.contains("put")){
                entity -= 1;
                transition.state.customFeaturesMatrix[0][entity] = true;
            }
            return transition.state.customFeaturesMatrix[0][entity];
        }
        return false;
    }

    @Override
    public void updateMatrixFeature(HAction action, Compostate newState){
        String label = action.toString();
        int entity = ActionWithFeatures.getNumber(label, 1);

        if(entity != -1 && entity < dcs.n && label.contains("get")){
            newState.customFeaturesMatrix[1][entity] = true;
        }

        if(entity != -1 && label.contains("return")){
            newState.customFeaturesMatrix[2][entity-1] = true;
        }
    }
}