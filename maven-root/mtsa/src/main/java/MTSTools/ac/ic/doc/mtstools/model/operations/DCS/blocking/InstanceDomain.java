package MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class InstanceDomain<State, Action> {
    public DirectedControllerSynthesisBlocking dcs;
    public int f;

    InstanceDomain(DirectedControllerSynthesisBlocking dcs){
        this.dcs = dcs;
        this.f = 1;
    }

    public List<boolean[]> copyMatrix(Compostate parent){
        List<boolean[]> res = new ArrayList();
        for(int i=0 ; i<f ; i++){
            res.add(Arrays.copyOf(parent.getLastMissions(i), dcs.n));
        }
        return res;
    }

    public static InstanceDomain createInstanceDomain(DirectedControllerSynthesisBlocking dcs){
        String domain = dcs.instance;
        if(domain.equals("BW")){
            return new InstanceDomainBW<>(dcs);
        } else if(domain.equals("TA")) {
            return new InstanceDomainTA<>(dcs);
        } else if(domain.equals("AT")) {
            return new InstanceDomainAT<>(dcs);
        } else if(domain.equals("DP")) {
            return new InstanceDomainDP<>(dcs);
        } else if(domain.equals("CM")) {
            return new InstanceDomainCM<>(dcs);
        } else if(domain.equals("TL")) {
            return new InstanceDomainTL<>(dcs);
        }
        return new InstanceDomain<>(dcs);
    }

    public List<boolean[]> newFeatureMatrix() {
        List<boolean[]> res = new ArrayList();
        for(int i=0 ; i<f ; i++){
            res.add(new boolean[dcs.n]);
        }
        return res;
    }



    public void computeCustomFeature(ActionWithFeatures transition){}
}

class InstanceDomainBW<State, Action> extends InstanceDomain{
    InstanceDomainBW(DirectedControllerSynthesisBlocking dcs){
        super(dcs);
        this.f = 3;
    }

    public void computeCustomFeature(ActionWithFeatures transition){
        HAction<Action> action = transition.action;
        String label = action.toString();
        Compostate<State, Action> state = transition.state;
        int entity = transition.entity;
        int index = transition.index;

        if(label.contains("accept") || (label.contains("reject") && index == dcs.k)){
            state.customFeatures.get(action).get(0)[entity] = true;
        }

        if(label.contains("assign")){
            state.customFeatures.get(action).get(1)[entity] = true;
        }

        if(label.contains("accept") && state.customFeatures.get(action).get(0)[entity]){
            state.customFeatures.get(action).get(1)[entity] = false;
        }

        if(label.contains("reject")){
            state.customFeatures.get(action).get(2)[entity] = true;
        }

        // Ver de mejorar y usar 3 features distintos:
        //      uno para ver si a esa entidad ya le asignaron el documento
        //      otro para ver si ya lo acepto
        //      otro para ver si ya lo rechazo k veces
    }
}

class InstanceDomainTA<State, Action> extends InstanceDomain{
    public int service;

    InstanceDomainTA(DirectedControllerSynthesisBlocking dcs){
        super(dcs);
        this.f = 8;
        this.service = -1;
    }

    public boolean next_entity_query(Compostate<State, Action> state, HAction<Action> action, int entity){
        if(entity+1 < dcs.n){
            return state.customFeatures.get(action).get(1)[entity+1];
        }else{
            return false;
        }
    }

    public void computeCustomFeature(ActionWithFeatures transition){
        HAction<Action> action = transition.action;
        String label = action.toString();
        Compostate<State, Action> state = transition.state;
        Compostate<State, Action> parent = transition.parent;
        int entity = transition.entity;
        int index = transition.index;

        if(label.contains("purchase.succ") || label.contains("purchase.fail")){
            state.customFeatures.get(action).get(0)[entity] = true;
        }

        if(!state.customFeatures.get(action).get(1)[entity] && label.contains("query")){
            state.customFeatures.get(action).get(1)[entity] = true;
            service += 1;
        }

        state.customFeatures.get(action).get(2)[entity] = next_entity_query(state, action, entity);

        state.customFeatures.get(action).get(3)[entity] = (service == entity);
        state.customFeatures.get(action).get(4)[entity] = (service+1 == entity);
        state.customFeatures.get(action).get(5)[entity] = (service+1 < entity);

        if(label.contains("committed") && !label.contains("un")){
            state.customFeatures.get(action).get(6)[entity] = true;
        }

        if(label.contains("uncommitted")){
            state.customFeatures.get(action).get(7)[entity] = true;
        }

        if(label.contains("steps")){
            int oldIndex = parent.getLastentityIndex()[entity];
            state.entityIndexes.get(action)[entity] = index+1;
            if(index > oldIndex){
                transition.upIndex = true;
            }
            if(index < oldIndex) {
                transition.downIndex = true;
            }
        }
    }
}

class InstanceDomainAT<State, Action> extends InstanceDomain{
    InstanceDomainAT(DirectedControllerSynthesisBlocking dcs){
        super(dcs);
        this.f = 3;
    }

    public void computeCustomFeature(ActionWithFeatures transition){
        HAction<Action> action = transition.action;
        String label = action.toString();
        Compostate<State, Action> state = transition.state;
        Compostate<State, Action> parent = transition.parent;
        int entity = transition.entity;
        int index = transition.index;

        if(label.contains("land")){
            state.customFeatures.get(action).get(0)[entity] = true;
        }

        if(label.contains("requestLand")){
            state.customFeatures.get(action).get(1)[entity] = true;
        }

        if(label.contains("extendFlight")){
            state.customFeatures.get(action).get(2)[entity] = true;
        }



        // Index entity and up/down index
        if(label.contains("descend")){
            int oldIndex = parent.getLastentityIndex()[entity];
            state.entityIndexes.get(action)[entity] = index+1;
            if(index > oldIndex){
                transition.upIndex = true;
            }
            if(index < oldIndex) {
                transition.downIndex = true;
            }
        }
    }
}

class InstanceDomainDP<State, Action> extends InstanceDomain{
    InstanceDomainDP(DirectedControllerSynthesisBlocking dcs){
        super(dcs);
        this.f = 2;
    }

    public void computeCustomFeature(ActionWithFeatures transition){
        HAction<Action> action = transition.action;
        String label = action.toString();
        Compostate<State, Action> state = transition.state;
        int entity = transition.entity;

        if(dcs.lastExpandedAction != null){
            if(label.contains("step") && dcs.lastExpandedAction.getAction().toString().contains("take")){
                state.customFeatures.get(action).get(1)[entity] = true;
            }
            if(label.contains("take") && dcs.lastExpandedAction.getAction().toString().contains("step")){
                state.customFeatures.get(action).get(1)[entity] = true;
            }
        }

        if(label.contains("release")){
            state.customFeatures.get(action).get(0)[entity] = true;
            state.entityIndexes.get(action)[entity] = 0;
            transition.downIndex = true;
        }
        if(label.contains("step")){
            state.entityIndexes.get(action)[entity] += 1;
            transition.upIndex = true;
        }
    }
}

class InstanceDomainCM<State, Action> extends InstanceDomain{
    InstanceDomainCM(DirectedControllerSynthesisBlocking dcs){
        super(dcs);
        this.f = 1;
    }

    public void computeCustomFeature(ActionWithFeatures transition){
        HAction<Action> action = transition.action;
        String label = action.toString();
        Compostate<State, Action> state = transition.state;
        Compostate<State, Action> parent = transition.parent;
        int entity = transition.entity;
        int index = transition.index;

        if(label.contains("mouse") && label.contains("move") && index == dcs.k){
            state.customFeatures.get(action).get(0)[entity] = true;
        }
        if(label.contains("mouse") && label.contains("move")) {
            int newIndex = 2 * dcs.k - index;
            int oldIndex = parent.getLastentityIndex()[entity];
            state.entityIndexes.get(action)[entity] = newIndex;
            if (newIndex > oldIndex) {
                transition.upIndex = true;
            }
            if (newIndex < oldIndex) {
                transition.downIndex = true;
            }
        }
    }
}

class InstanceDomainTL<State, Action> extends InstanceDomain{
    InstanceDomainTL(DirectedControllerSynthesisBlocking dcs){
        super(dcs);
        this.f = 1;
    }

    public void computeCustomFeature(ActionWithFeatures transition){

    }
}
