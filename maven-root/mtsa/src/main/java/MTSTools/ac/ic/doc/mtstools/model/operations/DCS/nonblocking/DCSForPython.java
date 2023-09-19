package MTSTools.ac.ic.doc.mtstools.model.operations.DCS.nonblocking;

import MTSTools.ac.ic.doc.commons.relations.Pair;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.Set;

import ai.onnxruntime.OrtException;
import ltsa.dispatcher.TransitionSystemDispatcher;
import ltsa.lts.*;
import org.junit.Test;

import static org.junit.Assert.*;

/** This class can be used from python with jpype */
public class DCSForPython {
    private FeatureBasedExplorationHeuristic<Long, String> heuristic;
    private DirectedControllerSynthesisNonBlocking<Long, String> dcs;
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
        int n = 3;//Integer.parseInt(s[s.length - 2]);
        int k = 3;//Integer.parseInt(s[s.length - 1].split("\\.")[0]);

        Pair<CompositeState, LTSOutput> c = FeatureBasedExplorationHeuristic.compileFSP(path);

        DirectedControllerSynthesisNonBlocking<Long, String> dcs = TransitionSystemDispatcher.hcsInteractive(c.getFirst(), c.getSecond());
        if(dcs == null) fail("Could not start DCS for the given fsp");

        this.heuristic = new FeatureBasedExplorationHeuristic<>("python", featureMaker, false);
        this.dcs = dcs;
        this.dcs.heuristic = this.heuristic;

        this.heuristic.set_nk(n, k);
        this.heuristic.setFeaturesBuffer(this.input_buffer);

        this.heuristic.startSynthesis(this.dcs); // inicializa la frontera como una lista vacia

        this.dcs.setupInitialState(); // llena la lista de la frontera(explorationFrontier)
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
        dcs.expand(stateAction.state, stateAction.action);
        if(!dcs.isFinished()){
            this.heuristic.filterFrontier();
            this.heuristic.computeFeatures();
        }
        this.heuristic.lastExpandedStateAction = stateAction;
    }

    public int[] lastExpandedHashes(){
        int[] hashes= {this.heuristic.lastExpandedFrom.hashCode(),
                this.heuristic.lastExpandedStateAction.action.hashCode(),
                this.heuristic.lastExpandedTo.hashCode(),
                this.heuristic.lastExpandedTo.marked? 1:0,
                this.heuristic.lastExpandedStateAction.action.isControllable()? 1:0
        };

        return hashes;
    }
    public String[] lastExpandedStringIdentifiers(){
        String[] hashes= {this.heuristic.lastExpandedFrom.toString(),
                this.heuristic.lastExpandedStateAction.action.toString(),
                this.heuristic.lastExpandedTo.toString(),
                this.heuristic.lastExpandedTo.marked? "1":"0",
                this.heuristic.lastExpandedStateAction.action.isControllable()? "1":"0"};

        return hashes;
    }

    public static void main(String[] args) throws OrtException {
        //String FSP_path = "/home/dario/Documents/Tesis/mtsa/maven-root/mtsa/target/test-classes/Blocking/ControllableFSPs/GR1test1.lts"; // Falla porque tiene guiones
        //String FSP_path = "/home/dario/Documents/Tesis/Learning-Synthesis/fsp/Blocking/ControllableFSPs/GR1test1.lts";
        String FSP_path = "/home/dario/Documents/Tesis/Learning-Synthesis/fsp/DP/DP-2-2.fsp";

        // This main is for testing purposes only
        CompositeState ltss_init = FeatureBasedExplorationHeuristic.compileFSP(FSP_path).getFirst();
        DCSForPython env = new DCSForPython(null, null,10000, ltss_init);

        Random rand = new Random();

        for(int i = 0; i < 10; i++){
            env.startSynthesis(FSP_path);
            while (!env.isFinished()) {
                /*for(ActionWithFeatures<Long, String> action : env.heuristic.explorationFrontier){
                    System.out.println(action);
                }*/
                env.expandAction(rand.nextInt(env.frontierSize()));
            }
        }
    }
}


