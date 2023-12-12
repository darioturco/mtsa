package MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking;

import MTSTools.ac.ic.doc.commons.relations.Pair;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.*;

import MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.DirectedControllerSynthesisBlocking;
import MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.abstraction.HEstimate;
import MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.abstraction.HeuristicMode;
import MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.abstraction.Recommendation;
import ltsa.dispatcher.TransitionSystemDispatcher;
import ltsa.lts.*;
import ltsa.ui.StandardOutput;
import org.junit.Test;

import static org.junit.Assert.*;

/** This class can be used from python with jpype */
public class DCSForPython {

    public DirectedControllerSynthesisBlocking<Long, String> dcs;
    public ExplorationHeuristic<Long, String> heuristic;
    public HeuristicMode heuristicMode;

    public boolean started_synthesis;
    // TODO: Rehacer el init, tiene muchos argunmentos que no se usan mas
    public DCSForPython(String features_path, String labels_path, int max_frontier, CompositeState ltss_init, String heuristicMode){
        this.started_synthesis = false;
        this.heuristicMode = HeuristicMode.valueOf(heuristicMode);
    }

    public static Pair<CompositeState, LTSOutput> compileFSP(String filename){
        String currentDirectory = null;
        try {
            currentDirectory = (new File(".")).getCanonicalPath();
        } catch (IOException e) {
            e.printStackTrace();
        }

        LTSInput input = new LTSInputString(readFile(filename));
        LTSOutput output = new StandardOutput();

        LTSCompiler compiler = new LTSCompiler(input, output, currentDirectory);
        compiler.compile();
        CompositeState c = compiler.continueCompilation("DirectedController");

        TransitionSystemDispatcher.parallelComposition(c, output);

        return new Pair<>(c, output);
    }

    public static String readFile(String filename) {
        String result = null;
        try {
            BufferedReader file = new BufferedReader(new FileReader(filename));
            String thisLine;
            StringBuffer buff = new StringBuffer();
            while ((thisLine = file.readLine()) != null)
                buff.append(thisLine+"\n");
            file.close();
            result = buff.toString();
        } catch (Exception e) {
            System.err.print("Error reading FSP file " + filename + ": " + e);
            System.exit(1);
        }
        return result;
    }

    public void startSynthesis(String path) {
        // c.first is an object that contains the list of all automaton to compose
        // c.second is the output where the errors are written (is not important)
        Pair<CompositeState, LTSOutput> c = DCSForPython.compileFSP(path);

        DirectedControllerSynthesisBlocking<Long, String> dcs = TransitionSystemDispatcher.hcsInteractiveForBlocking(c.getFirst(), c.getSecond());

        if(dcs == null) fail("Could not start DCS for the given fsp");
        this.dcs = dcs;

        this.heuristic = this.dcs.getHeuristic(this.heuristicMode);

        this.dcs.heuristic = this.heuristic;

        this.dcs.setupInitialState();
        this.heuristic.filterFrontier();
        this.started_synthesis = false;
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

    public boolean isFinished(){
        return dcs.isFinished();
    }

    public Set<String> all_transition_labels(){
        if(!started_synthesis) System.out.println("Transition labels not computed yet, synthesis pending.");
        return (Set<String>) this.dcs.alphabet.actions;
    }

    public void expandAction(int idx){
        ActionWithFeatures<Long, String> stateAction = heuristic.removeFromFrontier(idx);
        Compostate<Long, String> state = stateAction.state;
        HAction<String> action = stateAction.action;

        Compostate<Long, String> child = dcs.expand(state, action);
        if(!dcs.isFinished()){
            this.heuristic.filterFrontier();
        }

        this.heuristic.setLastExpandedStateAction(stateAction);
        this.heuristic.expansionDone(state, action, child);
    }
    public int getActionFronAuxiliarHeuristic(){
        return heuristic.getNextActionIndex();
    }

    public int getIndexOfStateAction(Pair<Compostate<Long, String>, HAction<String>> actionState){
        return heuristic.getIndexOfStateAction(actionState);
    }

    public ArrayList<Integer> getHeuristicOrder(){
        return heuristic.getOrder();
    }

    // This main is for testing purposes only
    public static void main(String[] args)  {
        //String FSP_path = "/home/dario/Documents/Tesis/mtsa/maven-root/mtsa/target/test-classes/Blocking/ControllableFSPs/GR1test1.lts"; // Falla porque tiene guiones
        //String FSP_path = "F:\\UBA\\Tesis\\mtsa\\maven-root\\mtsa\\target\\test-classes\\Blocking\\ControllableFSPs\\GR1Test10.lts";
        //String FSP_path = "F:\\UBA\\Tesis\\mtsa\\maven-root\\mtsa\\target\\test-classes\\Blocking\\NoControllableFSPs\\GR1Test11.lts";
        String FSP_path = "F:\\UBA\\Tesis\\mtsa\\MTSApy\\fsp\\TL\\TL-2-2.fsp";
        //String FSP_path = "/home/dario/Documents/Tesis/Learning-Synthesis/fsp/Blocking/ControllableFSPs/GR1Test10.lts";
        //String FSP_path = "/home/dario/Documents/Tesis/Learning-Synthesis/fsp/DP/DP-2-2.fsp";

        CompositeState ltss_init = DCSForPython.compileFSP(FSP_path).getFirst();

        String heuristicMode = "Ready";
        //String heuristicMode = "BFS";
        //String heuristicMode = "Debugging";
        DCSForPython env = new DCSForPython(null, null,10000, ltss_init, heuristicMode);

        Random rand = new Random();
        //List<Integer> list = Arrays.asList(0, 1, 1, 0, 0, 0, 0); // Lista para la intancia 10 Controlable
        //List<Integer> list = Arrays.asList(0, 1, 1); // Lista para la intancia 11 No Controlable
        //List<Integer> list = Arrays.asList(2);
        List<Integer> list = new ArrayList();
        int idx = 0;
        env.startSynthesis(FSP_path);
        int i = 0;
        while (!env.isFinished()) {
            System.out.println("----------------------------------: " + (i+1));
            env.heuristic.printFrontier();

            if(i < list.size()){
                idx = list.get(i);
            }else{
                //idx = rand.nextInt(env.frontierSize());
                if (env.getHeuristicOrder().contains(-1)) {
                    System.out.println("Error");
                    env.getHeuristicOrder();
                }

                System.out.println("Recomendation Order: " + env.getHeuristicOrder());
                idx = env.getActionFronAuxiliarHeuristic();
            }

            System.out.println("Expanded action: " + idx);

            //for(Integer j : env.getHeuristicOrder()){
            //    System.out.println("   " + j);
            //}

            env.expandAction(idx);
            i = i + 1;
        }
        System.out.println("End Run :)");
    }
}



