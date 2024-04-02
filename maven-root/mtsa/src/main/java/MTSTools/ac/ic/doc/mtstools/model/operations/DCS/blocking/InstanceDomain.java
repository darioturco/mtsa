package MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class InstanceDomain<State, Action> {
    public DirectedControllerSynthesisBlocking dcs;
    public boolean firstExpansion;
    public int customFSize;
    public int compostateFeatureSize;



    /** Matrix where each row is the memory used for a diferent feature
     * For example the first row is used to feature 'missionComplete' in that row is saved whether a entity completed his mission or don't */
    public List<boolean[]> featureMatrix;   // TODO: cambiar por un boolean[][]

    InstanceDomain(DirectedControllerSynthesisBlocking dcs){
        this.dcs = dcs;
        this.firstExpansion = true;
    }

    public List<boolean[]> copyMatrix(Compostate parent){
        List<boolean[]> res = new ArrayList();
        for(int i=0 ; i<compostateFeatureSize ; i++){
            res.add(Arrays.copyOf((boolean[]) parent.compostateCustomFeatures.get(i), dcs.n));
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

    public static InstanceDomain createInstanceDomain(DirectedControllerSynthesisBlocking dcs){
        String domain = dcs.instance;
        if(domain.equals("BW")){
            return new InstanceDomainBW(dcs);
        } else if(domain.equals("TA")) {
            return new InstanceDomainTA(dcs);
        } else if(domain.equals("AT")) {
            return new InstanceDomainAT(dcs);
        } else if(domain.equals("DP")) {
            return new InstanceDomainDP(dcs);
        } else if(domain.equals("CM")) {
            return new InstanceDomainCM<>(dcs);
        } else if(domain.equals("TL")) {
            return new InstanceDomainTL<>(dcs);
        }
        return new InstanceDomain<>(dcs);
    }

    public static float toFloat(boolean b) {
        return b ? 1.0f : 0.0f;
    }


    public void computeCustomFeature(ActionWithFeatures transition, int i){}
    public int size(){return 1 + customFSize;}
    public boolean missionFeature(ActionWithFeatures transition){return false;}
    public void updateMatrixFeature(HAction<Action> action, Compostate<State, Action> newState){}
}

class InstanceDomainBW extends InstanceDomain{
    InstanceDomainBW(DirectedControllerSynthesisBlocking dcs){
        super(dcs);
        this.customFSize = 3;
        this.compostateFeatureSize = 0;
        this.featureMatrix = newFeatureMatrix(size());
    }

    @Override
    public boolean missionFeature(ActionWithFeatures transition){
        String label = transition.action.toString();
        if(transition.has_entity()){
            // res indicate if the number 'transition.entity' reached his mission
            boolean res = label.contains("accept") || (label.contains("reject") && transition.index == dcs.k);
            boolean[] missionRow = (boolean[])featureMatrix.get(0);
            if(res){
                missionRow[transition.entity] = true;
            }
            return missionRow[transition.entity];
        }
        return false;
    }

    @Override
    public void updateMatrixFeature(HAction action, Compostate newState){
        String label = action.toString();
        int entity = ActionWithFeatures.getNumber(label, 1);
        if(entity != -1){
            if(label.contains("assign")){
                ((boolean[])featureMatrix.get(1))[entity] = true;
            } else if(label.contains("reject")) {
                ((boolean[])featureMatrix.get(2))[entity] = true;
            } else if(label.contains("accept")) {
                ((boolean[])featureMatrix.get(3))[entity] = true;
            }
        }
    }

    @Override
    public void computeCustomFeature(ActionWithFeatures transition, int i){
        int entity = transition.entity;

        boolean entity_was_assigned = transition.has_entity() && ((boolean[])featureMatrix.get(1))[entity];
        transition.featureVector[i] = toFloat(entity_was_assigned);

        boolean entity_was_rejected = transition.has_entity() && ((boolean[])featureMatrix.get(2))[entity];
        transition.featureVector[i+1] = toFloat(entity_was_rejected);

        boolean entity_was_accepted = transition.has_entity() && ((boolean[])featureMatrix.get(3))[entity];
        transition.featureVector[i+2] = toFloat(entity_was_accepted);
    }
}

class InstanceDomainTA extends InstanceDomain{
    public int service;

    InstanceDomainTA(DirectedControllerSynthesisBlocking dcs){
        super(dcs);
        this.service = 0;
        this.customFSize = 6;
        this.featureMatrix = newFeatureMatrix(size());
    }

    @Override
    public boolean missionFeature(ActionWithFeatures transition){
        String label = transition.action.toString();
        if(transition.has_entity()){
            int entity = transition.entity;
            // res indicate if the number 'transition.entity' reached his mission
            boolean res = label.contains("purchase.succ") || label.contains("purchase.fail");
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
        String label = action.toString();
        int entity = ActionWithFeatures.getNumber(label, 1);
        if(entity != -1) {
            if(!((List<boolean[]>)featureMatrix).get(1)[entity] && label.contains("query")) {
                ((boolean[]) featureMatrix.get(1))[entity] = true;
                service += 1;
            } else if(label.contains("committed") && !label.contains("un")){
                ((boolean[])featureMatrix.get(2))[entity] = true;
            } else if(label.contains("uncommitted")){
                ((boolean[])featureMatrix.get(3))[entity] = true;
            }
        }
    }

    @Override
    public void computeCustomFeature(ActionWithFeatures transition, int i){
        int entity = transition.entity;

        boolean entity_made_query = transition.has_entity() && ((boolean[])featureMatrix.get(1))[entity];
        transition.featureVector[i] = toFloat(entity_made_query);

        boolean entity_was_committed = transition.has_entity() && ((boolean[])featureMatrix.get(2))[entity];
        transition.featureVector[i+1] = toFloat(entity_was_committed);

        boolean entity_was_uncommitted = transition.has_entity() && ((boolean[])featureMatrix.get(3))[entity];
        transition.featureVector[i+2] = toFloat(entity_was_uncommitted);

        boolean entity_is_current_service = transition.has_entity() && service == entity;
        transition.featureVector[i+3] = toFloat(entity_is_current_service);

        boolean entity_is_next_service = transition.has_entity() && (service == entity+1);
        transition.featureVector[i+4] = toFloat(entity_is_next_service);

        boolean entity_is_greater_service = transition.has_entity() && (service > entity+1);
        transition.featureVector[i+5] = toFloat(entity_is_greater_service);
    }
}

class InstanceDomainAT extends InstanceDomain{
    InstanceDomainAT(DirectedControllerSynthesisBlocking dcs){
        super(dcs);
        this.customFSize = 3;
        this.compostateFeatureSize = 0;
        this.featureMatrix = newFeatureMatrix(size());
    }

    @Override
    public boolean missionFeature(ActionWithFeatures transition){
        // TODO: Repensar el mission feature
        if(transition.has_entity()){
            int entity = transition.entity;
            // res indicate if the number 'transition.entity' reached his mission
            boolean res = transition.action.toString().contains("land");
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
        String label = action.toString();
        int entity = ActionWithFeatures.getNumber(label, 1);
        if(entity != -1) {
            if(label.contains("land")){
                ((boolean[])featureMatrix.get(1))[entity] = true;
            }else if(label.contains("requestLand")){
                ((boolean[])featureMatrix.get(2))[entity] = true;
            } else if(label.contains("extendFlight")){
                ((boolean[])featureMatrix.get(3))[entity] = true;
            }
        }
    }

    @Override
    public void computeCustomFeature(ActionWithFeatures transition, int i){
        int entity = transition.entity;

        boolean entity_landed = transition.has_entity() && ((boolean[])featureMatrix.get(1))[entity];
        transition.featureVector[i] = toFloat(entity_landed);

        boolean entity_requested_land = transition.has_entity() && ((boolean[])featureMatrix.get(2))[entity];
        transition.featureVector[i+1] = toFloat(entity_requested_land);

        boolean entity_extended_flight = transition.has_entity() && ((boolean[])featureMatrix.get(3))[entity];
        transition.featureVector[i+2] = toFloat(entity_extended_flight);
    }
}

class InstanceDomainDP extends InstanceDomain{
    InstanceDomainDP(DirectedControllerSynthesisBlocking dcs){
        super(dcs);
        this.customFSize = 4;
        this.compostateFeatureSize = 1;
        this.featureMatrix = newFeatureMatrix(size());
    }

    @Override
    public boolean missionFeature(ActionWithFeatures transition){
        if(transition.has_entity()){
            int entity = transition.entity;
            // res indicate if the number 'transition.entity' reached his mission
            boolean res = transition.action.toString().contains("release");
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
        String label = action.toString();
        int entity = ActionWithFeatures.getNumber(label, 1);
        int index = ActionWithFeatures.getNumber(label, 2);
        if(entity != -1) {
            if (label.contains("take")) {
                ((boolean[])newState.compostateCustomFeatures.get(0))[index] = true;
            } else if (label.contains("release")) {
                ((boolean[])newState.compostateCustomFeatures.get(0))[index] = false;
            } else if(label.contains("eat")){
                ((boolean[])featureMatrix.get(1))[entity] = true;
            }
        }
    }

    @Override
    public void computeCustomFeature(ActionWithFeatures transition, int i){
        String label = transition.action.toString();

        boolean fork_state = transition.index != -1 && ((boolean[])transition.state.compostateCustomFeatures.get(0))[transition.index];
        transition.featureVector[i] = toFloat(fork_state);

        transition.featureVector[i+1] = 0.0f;
        transition.featureVector[i+2] = 0.0f;
        if(dcs.lastExpandedAction != null){
            if(label.contains("step") && dcs.lastExpandedAction.getAction().toString().contains("take")){
                transition.featureVector[i+1] = 1.0f;
            }
            if(label.contains("take") && dcs.lastExpandedAction.getAction().toString().contains("step")){
                transition.featureVector[i+2] = 1.0f;
            }
        }

        boolean entity_ate = transition.has_entity() && ((boolean[])featureMatrix.get(1))[transition.entity];
        transition.featureVector[i+3] = toFloat(entity_ate);
    }
}

class InstanceDomainCM<State, Action> extends InstanceDomain{
    InstanceDomainCM(DirectedControllerSynthesisBlocking dcs){
        super(dcs);
        this.customFSize = 3; // TODO: modificar
        this.compostateFeatureSize = 0;
        this.featureMatrix = newFeatureMatrix(size());
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

    InstanceDomainTL(DirectedControllerSynthesisBlocking dcs){
        super(dcs);
        this.lastEntity = -1;
        this.customFSize = 3; // TODO: modificar
        this.compostateFeatureSize = 0;
        this.featureMatrix = newFeatureMatrix(size());
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