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
import MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.abstraction.ReadyAbstraction;
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
    public DCSForPython(String heuristicMode){
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
        this.dcs.load_data(path);
        this.dcs.setupInitialState();
        this.heuristic.filterFrontier();
        this.started_synthesis = true;
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

    public void setFlags(boolean CONSIDERAR_GOALS, boolean CONSIDERAR_ESTRUCTURA){
        ReadyAbstraction.CONSIDERAR_GOALS = CONSIDERAR_GOALS;
        ReadyAbstraction.CONSIDERAR_ESTRUCTURA = CONSIDERAR_ESTRUCTURA;
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
        if(!isFinished()) {
            this.heuristic.somethingLeftToExplore();
        }
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

    public static int syntetizeWithHeuristic(String FSP_path, String heuristic, int budget, boolean verbose){
        DCSForPython env = new DCSForPython(heuristic);
        env.setFlags(false, false);
        env.startSynthesis(FSP_path);

        int idx;
        int i = 0;
        while (!env.isFinished() && i < budget) {
            idx = env.getActionFronAuxiliarHeuristic();
            if(verbose){
                System.out.println("----------------------------------: " + (i+1));
                System.out.println("Expanded: " + env.heuristic.getFrontier().get(idx));
            }

            env.expandAction(idx);
            i = i + 1;
        }

        return i;
    }

    // This main is for testing purposes only
    public static void main(String[] args)  {

        /*
        for(int n=2;n<=15;n++){
            int res = 0;
            for(int k=2;k<=15 && res<15000;k++){
                res = DCSForPython.syntetizeWithHeuristic("F:\\UBA\\Tesis\\mtsa\\MTSApy\\fsp\\CM\\CM-" + n + "-" + k + ".fsp", "Ready", 15000, false);
                System.out.println("Result of CM-" + n + "-" + k +": " + res);
            }
        }*/


        String instance = "TA";

        //String FSP_path = "/home/dario/Documents/Tesis/mtsa/maven-root/mtsa/target/test-classes/Blocking/ControllableFSPs/GR1test1.lts"; // Falla porque tiene guiones
        //String FSP_path = "F:\\UBA\\Tesis\\mtsa\\maven-root\\mtsa\\target\\test-classes\\Blocking\\ControllableFSPs\\GR1Test43.lts";
        //String FSP_path = "F:\\UBA\\Tesis\\mtsa\\maven-root\\mtsa\\target\\test-classes\\Blocking\\NoControllableFSPs\\GR1Test11.lts";
        String FSP_path = "F:\\UBA\\Tesis\\mtsa\\MTSApy\\fsp\\" + instance + "\\" + instance + "-2-2.fsp";
        //String FSP_path = "/home/dario/Documents/Tesis/Learning-Synthesis/fsp/Blocking/ControllableFSPs/GR1Test10.lts";
        //String FSP_path = "/home/dario/Documents/Tesis/mtsa/MTSApy/fsp/CM/CM-2-2.fsp";

        //String heuristicMode = "Ready";
        String heuristicMode = "Complete";
        //String heuristicMode = "Random";
        //String heuristicMode = "Interactive";
        //String heuristicMode = "BFS";
        //String heuristicMode = "Debugging";
        //String heuristicMode = "CMHeuristic";
        DCSForPython env = new DCSForPython(heuristicMode);
        env.startSynthesis(FSP_path);

        //env.dcs.analyzer.printInformation();

        Random rand = new Random();
        //List<Integer> list = Arrays.asList(1, 1, 2); // Lista para la intancia 11 No Controlable
        List<Integer> list = Arrays.asList();
        int idx = 0;
        int i = 0;

        while (!env.isFinished()) {
            System.out.println("----------------------------------: " + (i+1));
            env.heuristic.printFrontier();

            if(i < list.size()){
                idx = list.get(i);
            }else{
                //idx = rand.nextInt(env.frontierSize());
                idx = env.getActionFronAuxiliarHeuristic();
            }

            System.out.println("Expanded: " + env.heuristic.getFrontier().get(idx));
            env.expandAction(idx);
            i = i + 1;
        }
        System.out.println("End Run :)");
    }
}



