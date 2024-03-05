package MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking;

import MTSTools.ac.ic.doc.commons.relations.Pair;
import MTSTools.ac.ic.doc.mtstools.model.LTS;

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
            // TODO: In the OpenSetHeuristic there is the dictionary of defaultTargets by colors, should use that
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
        if(parent != null){
            String label = action.toString();

            // TODO: Remove this switch and use a customizable dictionary
            if(label.matches(".*\\d.*")){
                switch(dcs.instance){
                    case "AT":
                        if(label.contains("land")){
                            state.missionsCompletes.get(action)[entity] = true;
                        }
                        if(label.contains("descend")){
                            int oldIndex = parent.getLastentityIndex()[entity];
                            state.entityIndexes.get(action)[entity] = index+1;
                            if(index > oldIndex){
                                upIndex = true;
                            }
                            if(index < oldIndex) {
                                downIndex = true;
                            }
                        }
                        break;

                    case "BW":
                        // The Document is acepted by a team or is rejected k times
                        if(label.contains("accept") || (label.contains("reject") && index == dcs.k)){
                            state.missionsCompletes.get(action)[entity] = true;
                        }
                        if(label.contains("refuse") || label.contains("approve")){
                            state.entityIndexes.get(action)[entity] = 0;
                            downIndex = true;
                        }
                        if(label.contains("reject")){
                            state.entityIndexes.get(action)[entity] = index;
                            upIndex = true;
                        }
                        break;

                    case "CM":
                        if(label.contains("mouse") && label.contains("move") && index == dcs.k){
                            state.missionsCompletes.get(action)[entity] = true;
                        }
                        if(label.contains("mouse") && label.contains("move")) {
                            int newIndex = 2 * dcs.k - index;
                            int oldIndex = parent.getLastentityIndex()[entity];
                            state.entityIndexes.get(action)[entity] = newIndex;
                            if (newIndex > oldIndex) {
                                upIndex = true;
                            }
                            if (newIndex < oldIndex) {
                                downIndex = true;
                            }
                        }
                        break;

                    case "DP":
                        if(label.contains("release")){
                            state.missionsCompletes.get(action)[entity] = true;
                            state.entityIndexes.get(action)[entity] = 0;
                            downIndex = true;
                        }
                        if(label.contains("step")){
                            state.entityIndexes.get(action)[entity] += 1;
                            upIndex = true;
                        }
                        break;

                    case "TA":
                        if(label.contains("purchase.succ")){
                            state.missionsCompletes.get(action)[entity] = true;
                        }
                        if(label.contains("steps")){
                            int oldIndex = parent.getLastentityIndex()[entity];
                            state.entityIndexes.get(action)[entity] = index+1;
                            if(index > oldIndex){
                                upIndex = true;
                            }
                            if(index < oldIndex) {
                                downIndex = true;
                            }
                        }
                        break;

                    case "TL":
                        if(entity == -1){ // Should not enter
                            state.missionsCompletes.get(action)[entity] = true;
                        }
                        break;
                }
            }
        }
    }

    // Not used yet
    public LTS<State, Action> getLTSOfEntity(String label, int entity){
        // Uses the name of the LTS to get LTS that belong to the entity
        return state.dcs.ltss.get(1); // TODO: Fix
    }
}
