package MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking;

import MTSTools.ac.ic.doc.commons.relations.Pair;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.*;

import MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.DirectedControllerSynthesisBlocking;
import MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.abstraction.DebuggingAbstraction;
import MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.abstraction.HEstimate;
import MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.abstraction.Recommendation;
import ai.onnxruntime.OrtException;
import ltsa.dispatcher.TransitionSystemDispatcher;
import ltsa.lts.*;
import org.junit.Test;

import static org.junit.Assert.*;

/** This class can be used from python with jpype */
public class DCSForPython {
    public FeatureBasedExplorationHeuristic<Long, String> heuristic;
    public DirectedControllerSynthesisBlocking<Long, String> dcs;
    public FloatBuffer input_buffer;
    DCSFeatures<Long, String> featureMaker;

    public boolean started_synthesis;
    public DCSForPython(String features_path, String labels_path, int max_frontier, CompositeState ltss_init){
        this.featureMaker = new DCSFeatures<>(features_path, labels_path, max_frontier, ltss_init);
        ByteBuffer bb = ByteBuffer.allocateDirect(featureMaker.n_features*4*max_frontier);
        bb.order(ByteOrder.nativeOrder());
        this.input_buffer = bb.asFloatBuffer();
        this.started_synthesis = false;
    }

    /** restarts synthesis for a given fsp */
    public void startSynthesis(String path) {
        String[] s = path.split("-");
        // Descomentar y revisar
        int n = 3;//Integer.parseInt(s[s.length - 2]);
        int k = 3;//Integer.parseInt(s[s.length - 1].split("\\.")[0]);

        Pair<CompositeState, LTSOutput> c = FeatureBasedExplorationHeuristic.compileFSP(path);
        // c.first es un objeto con la lista de automatas a componer y las goals
        // c.second es la salida en la que se escriben los errores (mucho no importa)

        DirectedControllerSynthesisBlocking<Long, String> dcs = TransitionSystemDispatcher.hcsInteractiveForBlocking(c.getFirst(), c.getSecond());
        if(dcs == null) fail("Could not start DCS for the given fsp");

        this.heuristic = new FeatureBasedExplorationHeuristic<>("python", featureMaker, false);
        this.dcs = dcs;
        this.dcs.heuristic = this.heuristic;

        this.heuristic.set_nk(n, k);
        this.heuristic.setFeaturesBuffer(this.input_buffer);

        this.heuristic.startSynthesis(this.dcs);
        this.dcs.abstraction = new DebuggingAbstraction<>();

        this.dcs.setupInitialState();
        this.heuristic.filterFrontier();
        this.heuristic.computeFeatures();
        this.started_synthesis = false;
    }

    public FloatBuffer getBuffer(){
        return this.input_buffer;
    }

    public double getSynthesisTime(){
        return this.dcs.getStatistics().getElapsed();
    }

    public int getExpandedTransitions(){
        return this.dcs.getStatistics().getExpandedTransitions();
    }

    public boolean isExceededBudget(){
        return this.dcs.getStatistics().isExceededBudget();
    }

    public int getExpandedStates(){
        return this.dcs.getStatistics().getExpandedStates();
    }

    public int frontierSize(){
        return this.heuristic.frontierSize();
    }

    public int getNumberOfFeatures(){
        return featureMaker.n_features;
    }

    public boolean isFinished(){
        return dcs.isFinished();
    }

    public Set<String> all_transition_labels(){
        if(!started_synthesis) System.out.println("Transition labels not computed yet, synthesis pending.");

        return (Set<String>) this.dcs.alphabet.actions;

    }
    public Set<String> all_transition_types(){
        return this.featureMaker.labels_idx.keySet();
    }
    public void expandAction(int idx){
        ActionWithFeatures<Long, String> stateAction = heuristic.removeFromFrontier(idx);
        stateAction.state.removeRecommendation(stateAction);
        //System.out.println(stateAction.state.toString() + " | " + stateAction.action);
        Recommendation<String> recommendation = new Recommendation(stateAction.action, new HEstimate(1));
        Compostate<Long, String> child = dcs.expand(stateAction.state, recommendation);
        if(!dcs.isFinished()){
            this.heuristic.filterFrontier();
            this.heuristic.computeFeatures();
        }
        this.heuristic.lastExpandedStateAction = stateAction;
        this.heuristic.expansionDone(stateAction.state, stateAction.action, child);

    }

    public int[] lastExpandedHashes(){
        int[] hashes= {this.heuristic.lastExpandedFrom.hashCode(),
                this.heuristic.lastExpandedStateAction.action.hashCode(),
                this.heuristic.lastExpandedTo.hashCode(),
                //this.heuristic.lastExpandedTo.marked? 1:0,
                this.heuristic.lastExpandedStateAction.action.isControllable()? 1:0
        };

        return hashes;
    }
    public String[] lastExpandedStringIdentifiers(){
        String[] hashes= {this.heuristic.lastExpandedFrom.toString(),
                this.heuristic.lastExpandedStateAction.action.toString(),
                this.heuristic.lastExpandedTo.toString(),
                //this.heuristic.lastExpandedTo.marked? "1":"0",
                this.heuristic.lastExpandedStateAction.action.isControllable()? "1":"0"};

        return hashes;
    }

    public int getIndexOfStateAction(Pair<Compostate<Long, String>, HAction<String>> pairStateAction){
        Compostate<Long, String> state = pairStateAction.getFirst();
        HAction<String> action = pairStateAction.getSecond();
        int idx = 0;
        for(ActionWithFeatures<Long, String> actionWF : heuristic.explorationFrontier){
            if(actionWF.action.toString().equals(action.toString()) && actionWF.state.toString().equals(state.toString())){
                return idx;
            }
            idx++;
        }


        return -1;
    }

    public static void main(String[] args) throws OrtException {
        //String FSP_path = "/home/dario/Documents/Tesis/mtsa/maven-root/mtsa/target/test-classes/Blocking/ControllableFSPs/GR1test1.lts"; // Falla porque tiene guiones
        //String FSP_path = "F:\\UBA\\Tesis\\mtsa\\maven-root\\mtsa\\target\\test-classes\\Blocking\\ControllableFSPs\\GR1Test10.lts";
        //String FSP_path = "F:\\UBA\\Tesis\\mtsa\\maven-root\\mtsa\\target\\test-classes\\Blocking\\NoControllableFSPs\\GR1Test11.lts";
        String FSP_path = "/home/dario/Documents/Tesis/Learning-Synthesis/fsp/Blocking/ControllableFSPs/GR1Test10.lts";
        //String FSP_path = "/home/dario/Documents/Tesis/Learning-Synthesis/fsp/DP/DP-2-2.fsp";



        // This main is for testing purposes only
        CompositeState ltss_init = FeatureBasedExplorationHeuristic.compileFSP(FSP_path).getFirst();
        DCSForPython env = new DCSForPython(null, null,10000, ltss_init);

        Random rand = new Random();
        List<Integer> list = Arrays.asList(0, 1, 1, 0, 0, 0, 0); // Lista para la intancia 10 Controlable
        //List<Integer> list = Arrays.asList(0, 1, 1); // Lista para la intancia 11 No Controlable
        int idx = 0;
        env.startSynthesis(FSP_path);
        int i = 0;
        while (!env.isFinished()) {
            System.out.println("----------------------------------: " + (i+1));
            for(Compostate<Long, String> c : env.dcs.open){
                System.out.println(c);
            }

            if(i < list.size()){
                idx = list.get(i);
            }else{
                idx = rand.nextInt(env.frontierSize());
            }

            System.out.println("Expandido: " + env.heuristic.explorationFrontier.get(idx));

            //env.expandAction(rand.nextInt(env.frontierSize()));
            env.expandAction(idx);
            i = i + 1;
        }
        System.out.println("End Run :)");
    }
}


