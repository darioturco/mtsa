package MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking;

import MTSTools.ac.ic.doc.commons.relations.Pair;

import java.util.ArrayList;
import java.util.List;

public class ActionWithFeatures<State, Action> {
    public HAction<Action> action;
    public Compostate<State, Action> state;
    public Compostate<State, Action> child;
    public List<State> childStates;
    public DirectedControllerSynthesisBlocking<State, Action> dcs;
    public boolean childMarked;
    public boolean missionComplete;

    ActionWithFeatures(Compostate<State, Action> state, HAction<Action> action) {
        this.state = state;
        this.action = action;
        this.dcs = state.dcs;
        this.childStates = state.actionChildStates.get(action);
        this.child = this.dcs.compostates.get(childStates);


        this.childMarked = true;
        for (int lts = 0; childMarked && lts < this.dcs.ltssSize; ++lts)
            // TODO: In the OpenSetHeuristic there is the dictionary of defaultTargets by colors, should use that
            childMarked = this.dcs.defaultTargets.get(lts).contains(childStates.get(lts));

        missionComplete = checkMissionValue();
    }

    public Pair<Compostate<State, Action>, HAction<Action>> toPair(){
        return new Pair<>(state, action);
    }

    public String arrayBoolToString(boolean[] arr){
        String res = "[";
        for(boolean b : state.missionsCompletes){
            res += b + ", ";
        }

        int l = res.length();
        return res.substring(0, l-2) + "]";

    }

    public String toString(){
        return state.toString() + " | " + action.toString() + " | " + arrayBoolToString(state.missionsCompletes);
        //return arrayBoolToString(state.missionsCompletes);
    }

    public boolean checkMissionValue(){
        // Tengo n entidades, como se sobre cual de las n entidades estoy hablando
        String label = action.toString();
        if(label.matches(".*\\d.*")){
            int entity = state.getNumber(label, 1);
            return state.missionsCompletes[entity];
        }else{
            return false;
        }
    }
}
