package MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking;

import MTSTools.ac.ic.doc.commons.relations.Pair;
import MTSTools.ac.ic.doc.mtstools.model.LTS;
import MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.abstraction.HeuristicMode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ActionWithFeatures<State, Action> {
    public HAction<Action> action;
    public Compostate<State, Action> state;
    public Compostate<State, Action> parent;
    public Compostate<State, Action> child;
    public List<State> childStates;
    public DirectedControllerSynthesisBlocking<State, Action> dcs;
    public boolean childMarked;
    public boolean missionComplete;
    public int entity;
    public int index;
    public boolean upIndex;
    public boolean downIndex;
    public int amountMissionComplete;
    public boolean enable;
    public int expansionStep;


    ActionWithFeatures(Compostate<State, Action> state, HAction<Action> action, Compostate<State, Action> parent) {
        this.state = state;
        this.action = action;
        this.parent = parent;
        this.dcs = state.dcs;
        this.childStates = state.actionChildStates.get(action);
        this.child = this.dcs.compostates.get(childStates);
        this.entity = getNumber(action.toString(), 1);
        this.index = getNumber(action.toString(), 2);
        this.upIndex = false;
        this.downIndex = false;
        this.enable = true;
        this.expansionStep = dcs.expansionStep;

        if(parent == null){
            state.missionsCompletes.put(action, new boolean[dcs.n]);
            state.entityIndexes.put(action, new int[dcs.n]);
        }else{
            state.missionsCompletes.put(action, Arrays.copyOf(parent.getLastMissions(), dcs.n));
            state.entityIndexes.put(action, Arrays.copyOf(parent.getLastentityIndex(), dcs.n));
            updateMissions();
        }

        this.childMarked = true;
        for (int lts = 0; childMarked && lts < this.dcs.ltssSize; ++lts)
            childMarked = this.dcs.defaultTargets.get(lts).contains(childStates.get(lts));

        missionComplete = getMissionValue();
        amountMissionComplete = countMissionComplete();
    }

    public int countMissionComplete(){
        int res = 0;
        for(boolean missionComplete : state.missionsCompletes.get(action)){
            if(missionComplete){
                res += 1;
            }
        }
        return res;
    }

    public Pair<Compostate<State, Action>, HAction<Action>> toPair(){
        return new Pair<>(state, action);
    }

    public String arrayBoolToString(boolean[] arr){
        String res = "[";
        for(boolean b : arr){
            res += b + ", ";
        }

        int l = res.length();
        return res.substring(0, l-2) + "]";
    }

    public String toString(){
        return state.toString() + " | " + action.toString() + " | " + arrayBoolToString(state.missionsCompletes.get(action));
    }

    public boolean getMissionValue(){
        // Tengo n entidades, como se sobre cual de las n entidades estoy hablando
        String label = action.toString();
        if(label.matches(".*\\d.*") && !dcs.instance.equals("")) {
            if (entity < dcs.n) {
                return state.missionsCompletes.get(action)[entity];
            }
        }
        return false;
    }

    public int getNumber(String label, int n){
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

    public void updateMissions() {
        if(parent != null && action.toString().matches(".*\\d.*")){
            dcs.instanceDomain.computeCustomFeature(this);
        }
    }
}

// Despues hacer esta clase publica
class InstanceDomain<State, Action> {
    public DirectedControllerSynthesisBlocking dcs;
    
    InstanceDomain(DirectedControllerSynthesisBlocking dcs){
        this.dcs = dcs;
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
        return null;
    }

    public void computeCustomFeature(ActionWithFeatures transition){}
}

class InstanceDomainBW<State, Action> extends InstanceDomain{
    InstanceDomainBW(DirectedControllerSynthesisBlocking dcs){
        super(dcs);
    }
    
    public void computeCustomFeature(ActionWithFeatures transition){
        HAction<Action> action = transition.action;
        String label = action.toString();
        Compostate<State, Action> state = transition.state;
        int entity = transition.entity;
        
        // The Document is acepted by a team or is rejected k times
        if(label.contains("assign")){
            state.missionsCompletes.get(action)[entity] = true;
        }

        if(label.contains("assign")){
            if(state.missionsCompletes.get(action)[entity]) {
                state.missionsCompletes.get(action)[entity] = false;
            }
        }
        // Ver de mejorar y usar 3 features distintos:
        //      uno para ver si a esa entidad ya le asignaron el documento
        //      otro para ver si ya lo acepto
        //      otro para ver si ya lo rechazo k veces
    }
}

class InstanceDomainTA<State, Action> extends InstanceDomain{
    InstanceDomainTA(DirectedControllerSynthesisBlocking dcs){
        super(dcs);
    }

    public void computeCustomFeature(ActionWithFeatures transition){
        HAction<Action> action = transition.action;
        String label = action.toString();
        Compostate<State, Action> state = transition.state;
        Compostate<State, Action> parent = transition.parent;
        int entity = transition.entity;
        int index = transition.index;

        if(label.contains("purchase.succ")){
            state.missionsCompletes.get(action)[entity] = true;
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
    }
    
    public void computeCustomFeature(ActionWithFeatures transition){
        HAction<Action> action = transition.action;
        String label = action.toString();
        Compostate<State, Action> state = transition.state;
        Compostate<State, Action> parent = transition.parent;
        int entity = transition.entity;
        int index = transition.index;

        if(label.contains("land")){
            state.missionsCompletes.get(action)[entity] = true;
        }
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
    }

    public void computeCustomFeature(ActionWithFeatures transition){
        HAction<Action> action = transition.action;
        String label = action.toString();
        Compostate<State, Action> state = transition.state;
        int entity = transition.entity;

        if(label.contains("release")){
            state.missionsCompletes.get(action)[entity] = true;
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
    }
    
    public void computeCustomFeature(ActionWithFeatures transition){
        HAction<Action> action = transition.action;
        String label = action.toString();
        Compostate<State, Action> state = transition.state;
        Compostate<State, Action> parent = transition.parent;
        int entity = transition.entity;
        int index = transition.index;

        if(label.contains("mouse") && label.contains("move") && index == dcs.k){
            state.missionsCompletes.get(action)[entity] = true;
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
    }
    
    public void computeCustomFeature(ActionWithFeatures transition){

    }
}