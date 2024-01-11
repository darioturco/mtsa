package MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking;

import MTSTools.ac.ic.doc.commons.relations.Pair;

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


    ActionWithFeatures(Compostate<State, Action> state, HAction<Action> action, Compostate<State, Action> parent) {
        this.state = state;
        this.action = action;
        this.parent = parent;
        this.dcs = state.dcs;
        this.childStates = state.actionChildStates.get(action);
        this.child = this.dcs.compostates.get(childStates);

        if(parent == null){
            state.missionsCompletes.put(action, new boolean[dcs.n]);
        }else{
            state.missionsCompletes.put(action, Arrays.copyOf(parent.missionVector, dcs.n));
            updateMissions();
        }

        this.childMarked = true;
        for (int lts = 0; childMarked && lts < this.dcs.ltssSize; ++lts)
            // TODO: In the OpenSetHeuristic there is the dictionary of defaultTargets by colors, should use that
            childMarked = this.dcs.defaultTargets.get(lts).contains(childStates.get(lts));

        missionComplete = getMissionValue();
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
        //return arrayBoolToString(state.missionsCompletes);
    }

    public boolean getMissionValue(){
        // Tengo n entidades, como se sobre cual de las n entidades estoy hablando
        String label = action.toString();
        if(label.matches(".*\\d.*") && !dcs.instance.equals("")) {
            int entity = getNumber(label, 1);
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

            if(label.matches(".*\\d.*")){
                int entity = getNumber(label, 1);

                switch(dcs.instance){
                    case "AT":
                        if(label.contains("land")){
                            state.missionsCompletes.get(action)[entity] = true;
                        }
                        break;

                    case "BW":
                        // The Document is acepted by a team or is rejected k times
                        if(label.contains("accept") || (label.contains("reject") && getNumber(label, 2) == dcs.k)){
                            state.missionsCompletes.get(action)[entity] = true;
                        }
                        break;

                    case "CM":
                        if(label.contains("mouse") && label.contains("move")){
                            int m = getNumber(label, 2);
                            if(dcs.k == m){
                                state.missionsCompletes.get(action)[entity] = true;
                            }
                        }
                        break;

                    case "DP":
                        if(label.contains("release")){
                            state.missionsCompletes.get(action)[entity] = true;
                        }
                        break;

                    case "TA":
                        if(label.contains("purchase.succ")){
                            state.missionsCompletes.get(action)[entity] = true;
                        }
                        break;

                    case "TL":
                        if(entity == -1){
                            state.missionsCompletes.get(action)[entity] = true;
                        }
                        break;
                }
            }
        }
    }
}
