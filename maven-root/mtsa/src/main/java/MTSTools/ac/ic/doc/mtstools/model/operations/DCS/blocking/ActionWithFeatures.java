package MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking;

import MTSTools.ac.ic.doc.commons.relations.Pair;

import java.util.Arrays;
import java.util.List;

/* TODO: Arreglar todo el codigo de los features porque esta muy feito y es imposible de extender:
        - Algunos features dependen de la etidad y otros no (actualmente solo se soportan features de la entidad)
        - Algunos features necesitan ser reseteados
        - Arreglar que el feature 0 es "especial" eso no deberia ser asi
 */

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
    public int f;

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
            state.customFeatures.put(action, dcs.instanceDomain.newFeatureMatrix());
            state.entityIndexes.put(action, new int[dcs.n]);
        }else{
            state.customFeatures.put(action, dcs.instanceDomain.copyMatrix(parent));
            state.entityIndexes.put(action, Arrays.copyOf(parent.getLastentityIndex(), dcs.n));
            updateMissions();
        }

        this.childMarked = true;
        for (int lts = 0; childMarked && lts < this.dcs.ltssSize; ++lts)
            childMarked = this.dcs.defaultTargets.get(lts).contains(childStates.get(lts));

        missionComplete = getMissionValue(0);
        amountMissionComplete = countMissionComplete();
    }

    public int countMissionComplete(){
        int res = 0;
        for(boolean missionComplete : state.customFeatures.get(action).get(0)){
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
        return state.toString() + " | " + action.toString();
        //return state.toString() + " | " + action.toString() + " | " + arrayBoolToString(state.customFeatures.get(action).get(1));
    }

    public boolean getMissionValue(int mission){
        // There are n entities, this actionWithFeature refers about the entity number ´this.entity´
        String label = action.toString();
        if(label.matches(".*\\d.*") && !dcs.instance.equals("")) {
            if (entity < dcs.n) {
                return state.customFeatures.get(action).get(mission)[entity];
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