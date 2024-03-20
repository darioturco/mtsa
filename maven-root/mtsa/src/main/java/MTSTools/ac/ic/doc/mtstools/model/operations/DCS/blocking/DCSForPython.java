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
    public boolean realizable;
    public boolean finished;
    public int lastEntityExpanded;
    public int lastEntityExpandedWithoutReset;

    public boolean started_synthesis;

    public DCSForPython(String heuristicMode){
        this.started_synthesis = false;
        this.realizable = false;
        this.finished = false;
        this.heuristicMode = HeuristicMode.valueOf(heuristicMode);
        this.lastEntityExpanded = -1;
        this.lastEntityExpandedWithoutReset = -1;
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

        this.dcs.load_data(path);
        this.heuristic = this.dcs.getHeuristic(this.heuristicMode);
        this.dcs.heuristic = this.heuristic;
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

    public void updateLastEntityExpanded(ActionWithFeatures<Long, String> stateAction){
        lastEntityExpanded = stateAction.entity;
        if(stateAction.entity >= 0)
            lastEntityExpandedWithoutReset = stateAction.entity;
    }

    public void expandAction(int idx){
        ActionWithFeatures<Long, String> stateAction = heuristic.removeFromFrontier(idx);
        updateLastEntityExpanded(stateAction);
        Compostate<Long, String> state = stateAction.state;
        HAction<String> action = stateAction.action;

        Compostate<Long, String> child = dcs.expand(state, action);
        if(!dcs.isFinished()){
            this.heuristic.filterFrontier();
        }

        this.heuristic.setLastExpandedStateAction(stateAction);
        this.heuristic.expansionDone(state, action, child);
        if(isFinished()) {
            finished = true;
            if(dcs.isGoal(dcs.initial))
                realizable = true;
        }else{
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
        env.setFlags(true, true);
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

    public static float mean(List<Integer> ls) {
        int total = 0;
        for (int i = 0; i < ls.size(); i++)
            total += ls.get(i);

        return total / ls.size();
    }

    public static double std(List<Integer> ls, float mean) {
        float squareSum = 0;
        for(int i = 0; i < ls.size(); i++)
            squareSum += Math.pow(ls.get(i) - mean, 2);

        if((ls.size() - 1) == 0){
            return 0;
        }else{
            return Math.sqrt((squareSum) / (ls.size() - 1));
        }
    }

    // Este metodo corre todas las instancias de una familia con una heuristica dada
    public static int testHeuristic(int budget, String instance, String heuristic, int repetitions, int verbose){
        int solvedInstances = 0;
        int res = 0;
        for(int n=1;n<=15;n++){

            for(int k=1;k<=15 && res<budget;k++){

                //String fsp_path = "F:\\UBA\\Tesis\\mtsa\\MTSApy\\fsp\\" + instance + "\\" + instance + "-" + n + "-" + k + ".fsp";
                String fsp_path = "/home/dario/Documents/Tesis/mtsa/MTSApy/fsp/" + instance + "/" + instance + "-" + n + "-" + k + ".fsp";

                List<Integer> res_list = new ArrayList<>();
                for(int i=0;i<repetitions;i++){
                    res = DCSForPython.syntetizeWithHeuristic(fsp_path, heuristic, budget, false);
                    if(res < 10000){
                        solvedInstances += 1;
                    }
                    res_list.add(res);
                }

                if(verbose == 0) {
                    System.out.println(res);
                } else if (verbose == 1) {
                    float m = mean(res_list);
                    System.out.println("Result of " + instance + "-" + n + "-" + k + ": " + m + " std: " + std(res_list, m));
                }
            }
        }
        System.out.println("Solved instances:  " + solvedInstances);
        return solvedInstances;
    }

    // This main is for testing purposes only
    public static void main(String[] args) {
        //DCSForPython.testHeuristic(10000, "CM", "BFS", 1, 1);     // Ejemplo de como correr todas las de CM con la heuristica BFS y un budget de 10000 expanciones

        // Acontinuacion un ejemplo de como se deberia usar DCSForPython
        String instance = "TL";

        //String FSP_path = "/home/dario/Documents/Tesis/mtsa/maven-root/mtsa/target/test-classes/Blocking/ControllableFSPs/GR1test1.lts"; // Falla porque tiene guiones
        //String FSP_path = "F:\\UBA\\Tesis\\mtsa\\maven-root\\mtsa\\target\\test-classes\\Blocking\\ControllableFSPs\\GR1Test43.lts";
        //String FSP_path = "F:\\UBA\\Tesis\\mtsa\\maven-root\\mtsa\\target\\test-classes\\Blocking\\NoControllableFSPs\\GR1Test11.lts";
        String FSP_path = "F:\\UBA\\Tesis\\mtsa\\MTSApy\\fsp\\" + instance + "\\" + instance + "-2-2.fsp";
        //String FSP_path = "/home/dario/Documents/Tesis/Learning-Synthesis/fsp/Blocking/ControllableFSPs/GR1Test10.lts";
        //String FSP_path = "/home/dario/Documents/Tesis/mtsa/MTSApy/fsp/" + instance + "/" + instance + "-2-2.fsp";

        String heuristicMode = "Ready";
        //String heuristicMode = "Random";
        //String heuristicMode = "Interactive";
        //String heuristicMode = "BFS";
        //String heuristicMode = "Debugging";
        //String heuristicMode = "CMHeuristic";
        //String heuristicMode = "BWHeuristic";
        DCSForPython env = new DCSForPython(heuristicMode);
        env.startSynthesis(FSP_path);

        List<Integer> list = Arrays.asList();
        int idx = 0;
        int i = 0;

        while (!env.isFinished()) {
            System.out.println("----------------------------------: " + (i+1));
            env.heuristic.printFrontier();

            if(i < list.size()){
                idx = list.get(i);
            }else{
                idx = env.getActionFronAuxiliarHeuristic();
            }

            System.out.println("Expanded(" + idx + "): " + env.heuristic.getFrontier().get(idx));
            env.expandAction(idx);
            i = i + 1;
        }

        System.out.println("Realizable: " + env.realizable);
        System.out.println("End Run :)");
    }
}