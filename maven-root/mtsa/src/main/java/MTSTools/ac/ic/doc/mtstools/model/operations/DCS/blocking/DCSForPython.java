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

    public boolean started_synthesis;
    // TODO: Rehacer el init, tiene muchos argunmentos que no se usan mas
    public DCSForPython(String heuristicMode){
        this.started_synthesis = false;
        this.realizable = false;
        this.finished = false;
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

    public static int testHeuristic(int budget, String instance, String heuristic, int repetitions){
        int solvedInstances = 0;
        for(int n=1;n<=15;n++){
            int res = 0;
            for(int k=1;k<=15 && res<budget;k++){
                //System.out.println("Result of " + instance + "-" + n + "-" + k);
                //String fsp_path = "F:\\UBA\\Tesis\\mtsa\\MTSApy\\fsp\\" + instance + "\\" + instance + "-" + n + "-" + k + ".fsp";
                String fsp_path = "/home/dario/Documents/Tesis/mtsa/MTSApy/fsp/" + instance + "/" + instance + "-" + n + "-" + k + ".fsp";
                res = DCSForPython.syntetizeWithHeuristic(fsp_path, heuristic, budget, false);
                if(res < 10000){
                    solvedInstances += 1;
                }
                //System.out.println("Result of " + instance + "-" + n + "-" + k +": " + res);
                System.out.println(res);
            }
        }
        System.out.println("Solved instances:  " + solvedInstances);
        return solvedInstances;
    }

    // This main is for testing purposes only
    public static void main(String[] args) {
        //DCSForPython.testHeuristic(10000, "AT", "Complete", 1);


        //String instance = "BW";

        //String FSP_path = "/home/dario/Documents/Tesis/mtsa/maven-root/mtsa/target/test-classes/Blocking/ControllableFSPs/GR1test1.lts"; // Falla porque tiene guiones
        //String FSP_path = "F:\\UBA\\Tesis\\mtsa\\maven-root\\mtsa\\target\\test-classes\\Blocking\\ControllableFSPs\\GR1Test43.lts";
        //String FSP_path = "F:\\UBA\\Tesis\\mtsa\\maven-root\\mtsa\\target\\test-classes\\Blocking\\NoControllableFSPs\\GR1Test11.lts";
        //String FSP_path = "F:\\UBA\\Tesis\\mtsa\\MTSApy\\fsp\\" + instance + "\\" + instance + "-2-2.fsp";
        //String FSP_path = "/home/dario/Documents/Tesis/Learning-Synthesis/fsp/Blocking/ControllableFSPs/GR1Test10.lts";
        String FSP_path = "/home/dario/Documents/Tesis/mtsa/MTSApy/fsp/BW/BW-2-2.fsp";

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
        //List<Integer> list = Arrays.asList(0, 3, 0, 1, 2, 3, 4, 1, 1, 6, 4, 12, 11, 5, 14, 9, 1, 12, 6, 13, 12, 18, 1, 11, 9, 11, 14, 30, 24, 25, 0, 26, 26, 3, 21, 34, 39, 39, 24, 8, 4, 30, 31, 16, 19, 17, 9, 12, 48, 50, 3, 26, 8, 8, 7, 16, 29, 8, 22, 29, 63, 57, 43, 13, 40, 47, 37, 14, 39, 10, 42, 50, 9, 10, 53, 17, 35, 28, 22, 29, 43, 8, 39, 27, 58, 24, 64, 17, 45, 38, 28, 45, 58, 57, 63, 10, 13, 29, 39, 66, 5, 1, 12, 73, 50, 3, 43, 9, 80, 80, 15, 0, 32, 7, 44, 66, 43, 69, 24, 16, 82, 17, 29, 62, 80, 65, 32, 76, 42, 14, 63, 88, 37, 19, 3, 17, 52, 58, 36, 69, 77, 19, 57, 20, 94, 95, 5, 45, 72, 83, 45, 26, 34, 68, 64, 77, 7, 40, 1, 70, 12, 9, 106, 26, 65, 48, 104, 96, 88, 50, 27, 84, 3, 111, 10, 122, 64, 50, 50, 45, 39, 110, 12, 120, 34, 0, 72, 39, 42, 67, 118, 79, 67, 129, 43, 91, 62, 55, 41, 128, 20, 132, 93, 32, 7, 41, 45, 106, 94, 120, 95, 18, 26, 60, 7, 42, 99, 19, 104, 7, 74, 43, 57, 78, 90, 34, 82, 114, 117, 132, 132, 13, 7, 155, 96, 144, 118, 6, 118, 6, 129, 11, 12, 91, 30, 102, 25, 32, 134, 39, 61, 159, 91, 50, 161, 70, 65, 139, 153, 103, 79, 24, 53, 19, 58, 42, 59, 8, 79, 118, 76, 122, 13, 138, 60, 21, 120, 115, 21, 56, 170, 73, 53, 30, 144, 28, 90, 88, 92, 72, 19, 158, 153, 34, 18, 163, 88, 151, 20, 152, 36, 100, 174, 120, 59, 56, 165, 19, 59, 10, 28, 57, 29, 7, 83, 10, 44, 104, 95, 80, 82, 25, 147, 41, 151, 50, 121, 94, 169, 30, 17, 172, 73, 9, 53, 81, 52, 109, 34, 17, 155, 152, 73, 107, 154, 52, 34, 64, 156, 104, 83, 134, 25, 56, 4, 11, 21, 50, 33, 139, 119, 43, 156, 105, 19, 69, 25, 63, 122, 113, 45, 154, 50, 111, 158, 117, 115, 117, 11, 16, 169, 42, 76, 59, 132, 83, 7, 113, 36, 143, 16, 147, 181, 178, 112, 28, 50, 54, 183, 124, 115, 70, 129, 55, 8, 182, 134, 10, 139, 163, 142, 59, 44, 86, 150, 76, 2, 143, 41, 72, 111, 165, 88, 36, 34, 94, 131, 164, 164, 121, 183, 111, 164, 101, 140, 163, 64, 21, 114, 134, 61, 111, 109, 9, 180, 21, 106, 67, 154, 134, 155, 25, 38, 63, 41, 100, 79, 42, 57, 144, 100, 131, 113, 62, 36, 30, 130, 163, 40, 161, 145, 90, 143, 160, 160, 14, 95, 84, 139, 51, 99, 45, 19, 152, 91, 11, 117, 7, 11, 104, 22, 95, 35, 74, 7, 80, 10, 155, 19, 64, 54, 143, 4, 19, 107, 81, 50, 57, 36, 59, 36, 6, 11, 78, 40, 42, 30, 148, 69, 130, 25, 114, 74, 36, 102, 75, 99, 2, 115, 29, 93, 116, 143, 85, 8, 131, 9, 99, 146, 4, 145, 35, 106, 4, 61, 125, 117, 105, 20, 136, 115, 39, 93, 78, 78, 31, 132, 110, 50, 6, 139, 121, 41, 5, 63, 90, 101, 128, 106, 106, 81, 146, 125, 19, 22, 69, 9, 116, 142, 97, 57, 75, 58, 155, 32, 41, 10, 112, 110, 81, 78, 53, 91, 78, 30, 10, 59, 68, 71, 81, 66, 82, 134, 116, 98, 52, 96, 8, 93, 81, 88, 49, 83, 57, 133, 30, 26, 105, 29, 128, 5, 17, 94, 117, 100, 46, 13, 134, 51, 4, 33, 81, 83, 103, 68, 21, 26, 124, 121, 15, 65, 116, 113, 0, 40, 13, 73, 49, 47, 6, 113, 55, 7, 39, 103, 15, 22, 87, 49, 78, 122, 113, 2, 108, 26, 113, 116, 11, 117, 59, 135, 20, 39, 41, 116, 99, 49, 18, 24, 39, 79, 90, 114, 34, 127, 126, 27, 60, 71, 36, 99, 61, 99, 84, 31, 29, 59, 54, 35, 31, 128, 126, 58, 114, 115, 5, 14, 125, 131, 120, 129, 131, 133, 102, 41, 111, 91, 89, 112, 37, 131, 109, 17, 52, 7, 30, 102, 8, 67, 115, 67, 92, 103, 158, 10, 12, 13, 59, 108, 25, 109, 84, 171, 40, 157, 12, 173, 49, 44, 161, 43, 24, 178, 112, 39, 125, 32, 32, 166, 15, 145, 153, 9, 1, 12, 153, 181, 129, 141, 1, 9, 111, 69, 176, 124, 105, 19, 140, 100, 2, 119, 2, 26, 110, 85, 79, 58, 3, 28, 63, 155, 44, 174, 139, 114, 116, 96, 3, 22, 116, 121, 96, 169, 163, 82, 25, 24, 101, 59, 128, 128, 137, 22, 119, 119, 166, 13, 29, 45, 34, 68, 59, 58, 120, 126, 20, 138, 137, 97, 7, 23, 136, 88, 140, 134, 127, 121, 49, 30, 110, 124, 4, 132, 76, 66, 54, 34, 32, 12, 17, 10, 17, 11, 72, 120, 83, 131, 114, 164, 110, 34, 148, 121, 28, 34, 15, 37, 106, 130, 129, 25, 75, 146, 148, 124, 97, 134, 55, 106, 130, 81, 142, 73, 160, 45, 69, 92, 111, 154, 59, 36, 124, 154, 82, 68, 25, 7, 14, 44, 117, 14, 137, 135, 20, 36, 42, 26, 95, 37, 27, 1, 77, 51, 1, 39, 142, 12, 34, 44, 40, 16, 30, 133, 5, 78, 7, 135, 78, 10, 16, 125, 91, 84, 128, 56, 109, 44, 93, 117, 109, 126, 45, 45, 48, 71, 117, 76, 21, 66, 126, 119, 67, 89, 1, 80, 119, 107, 62, 21, 5, 62, 50, 33, 4, 47, 2, 27, 122, 92, 95, 9, 76, 24, 45, 14, 73, 79, 28, 12, 12, 27, 74, 71, 26, 19, 46, 5, 14, 81, 94, 11, 64, 77, 23, 104, 102, 49, 81, 14, 35, 99, 45, 18, 31, 8, 78, 40, 95, 41, 68, 91, 101, 56, 68, 98, 29, 17, 79, 73, 13, 10, 8, 42, 61, 55, 8, 48, 57, 33, 4, 71, 41, 80, 60, 72, 51, 30, 43, 12, 30, 28, 39, 18, 41, 83, 25, 76, 40, 64, 85, 52, 27, 55, 41, 96, 92, 102, 2, 81, 71, 19, 81, 78, 14, 110, 19, 99, 17, 19, 15, 111, 111, 28, 61, 106, 24, 110, 111, 6, 18, 7, 9, 106, 8, 22, 53, 49, 103, 92, 36, 70, 73, 85, 95, 85, 27, 22, 39, 62, 99, 25, 17, 37, 58, 43, 26, 11, 72, 33, 35, 21, 31, 32, 92, 68, 12, 16, 14, 17, 85, 37, 58, 6, 29, 43, 36, 89, 82, 89, 47, 60, 0, 63, 76, 79, 48, 8, 32, 16, 25, 82, 34, 81, 23, 47, 71, 25, 17, 76, 81, 4, 8, 50, 10, 74, 34, 46, 30, 67, 66, 45, 40, 59, 44, 67, 49, 15, 29, 3, 30, 61, 21, 49, 27, 59, 26, 25, 4, 46, 46, 27, 58, 49, 40, 29, 26, 46, 36, 47, 36, 53, 36, 2, 31, 46, 0, 30, 44, 35, 47, 4, 5, 39, 27, 42, 22, 29, 32, 3, 29, 40, 12, 12, 5, 8, 23, 50, 49, 34, 16, 24, 11, 30, 0, 48, 12, 0, 6, 28, 31, 9, 23, 43, 6, 50, 12, 27, 6, 38, 3, 2, 39, 42, 6, 54, 20, 14, 24, 16, 12, 52, 31, 40, 11, 0, 36, 48, 33, 4, 16, 33, 0, 18, 55, 28, 12, 3, 12, 2, 23, 10, 18, 7, 15, 51, 47, 2, 30, 14, 19, 6, 14, 21, 5, 21, 25, 2, 14, 46, 15, 28, 47, 24, 24, 61, 49, 38, 23, 38, 76, 54, 35, 14, 15, 81, 4, 71, 48, 74, 26, 15, 44, 77, 18, 86, 50, 19, 18, 52, 81, 20, 70, 18, 38, 71, 100, 100, 24, 32, 28, 76, 49, 33, 46, 1, 26, 70, 103, 99, 34, 35, 50, 40, 60, 116, 35, 35, 38, 107, 106, 99, 102, 40, 106, 110, 10, 107, 99, 11, 96, 99, 25, 51, 18, 77, 96, 111, 63, 18, 102, 67, 80, 31, 82, 44, 50, 31, 112, 26, 19, 9, 92, 50, 104, 14, 50, 63, 3, 16, 119, 80, 11, 129, 36, 102, 131, 111, 85, 111, 11, 88, 54, 37, 48, 9, 90, 17, 109, 6, 21, 126, 8, 27, 136, 22, 31, 26, 58, 128, 27, 17, 105, 16, 26, 22, 138, 136, 11, 102, 60, 50, 154, 80, 111, 10, 140, 55, 37, 52, 124, 111, 100, 148, 59, 80, 124, 85, 9, 43, 129, 154, 45, 106, 9, 52, 106, 125, 43, 133, 136, 102, 17, 3, 21, 37, 10, 83, 153, 165, 129, 16, 94, 0, 71, 79, 108, 42, 166, 3, 138, 162, 102, 90, 25, 189, 196, 119, 190, 90, 46, 39, 148, 46, 156, 134, 50, 37, 146, 140, 7, 124, 195, 11, 3, 89, 52, 57, 144, 80, 59, 171, 26, 67, 84, 96, 163, 85, 50, 80, 65, 188, 97, 19, 95, 125, 178, 50, 122, 88, 200, 44, 210, 186, 116, 90, 154, 119, 172, 172, 59, 163, 156, 52, 57, 161, 107, 56, 24, 157, 166, 56, 156, 116, 161, 56, 136, 147, 162, 91, 66, 36, 177, 216, 39, 125, 12, 163, 230, 208, 113, 193, 218, 170, 5, 123, 220, 138, 119, 79, 89, 145, 173, 25, 123, 210, 15, 96, 6, 101, 167, 107, 15, 2, 113, 72, 120, 222, 111, 147, 110, 96, 147, 85, 9, 226, 61, 60, 227, 120, 174, 139, 189, 191, 0, 103, 5, 38, 256, 242, 208, 186, 43, 127, 181, 155, 43, 259, 261, 176, 202, 235, 162, 162, 222, 175, 52, 95, 6, 43, 52, 202, 12, 135, 257, 258, 62, 169, 49, 101, 38, 122, 25, 94, 90, 54, 54, 83, 209, 8, 98, 82, 92, 225, 224, 222, 133, 13, 15, 234, 171, 236, 207, 129, 146, 188, 76, 246, 0, 191, 107, 249, 13, 142, 203, 76, 16, 95, 82, 83, 250, 198, 222, 202, 168, 166, 200, 213, 186, 202, 58, 107, 188, 15, 40, 26, 16, 119, 119, 192, 160, 203, 148, 61, 67, 232, 131, 212, 214, 136, 212, 14, 224, 257, 195, 85, 185, 89, 258, 229, 17, 182, 29, 191, 13, 214, 78, 259, 257, 67, 142, 117, 110, 128, 111, 43, 120, 158, 235, 76, 115, 108, 226, 124, 145, 26, 76, 145, 49, 169, 188, 16, 169, 268, 271, 157, 43, 103, 132, 265, 170, 78, 51, 136, 157, 101, 78, 196, 18, 16, 234, 230, 204, 253, 230, 265, 226, 240, 243, 161, 54, 248, 74, 186, 21, 163, 159, 137, 261, 171, 85, 103, 156, 229, 2, 241, 20, 181, 58, 19, 231, 19, 164, 19, 164, 154, 65, 231, 139, 185, 64, 64, 134, 159, 270, 212, 147, 112, 23, 40, 23, 223, 31, 168, 244, 184, 114, 96, 244, 110, 162, 201, 154, 199, 255, 48, 171, 247, 1, 247, 204, 100, 141, 156, 156, 232, 127, 24, 100, 31, 118, 238, 84, 117, 147, 57, 223, 239, 101, 45, 22, 87, 102, 143, 214, 233, 83, 155, 202, 244, 61, 114, 210, 195, 41, 258, 12, 268, 253, 13, 195, 90, 210, 91, 265, 65, 22, 73, 163, 206, 12, 102, 170, 61, 74, 11, 5, 84, 5, 185, 73, 236, 147, 173, 5, 116, 25, 36, 149, 281, 236, 217, 115, 224, 118, 281, 122, 244, 149, 242, 118, 18, 27, 271, 35, 12, 113, 206, 234, 176, 84, 80, 29, 200, 141, 156, 82, 134, 106, 19, 106, 113, 241, 135, 80, 13, 129, 59, 202, 127, 229, 71, 214, 105, 148, 242, 187, 0, 215, 126, 41, 239, 18, 261, 246, 252, 188, 79, 152, 223, 161, 238, 217, 149, 10, 219, 259, 231, 208, 172, 29, 108, 220, 102, 10, 174, 200, 169, 136, 90, 169, 151, 111, 143, 220, 25, 233, 173, 130, 72, 89, 223, 41, 155, 170, 84, 118, 172, 32, 232, 9, 238, 178, 180, 25, 56, 24, 116, 95, 95, 196, 247, 8, 190, 216, 135, 242, 189, 1, 65, 83, 165, 189, 63, 2, 135, 138, 198, 73, 143, 177, 152, 1, 53, 223, 18, 61, 153, 143, 153, 86, 222, 71, 155, 90, 56, 152, 168, 44, 145, 55, 85, 70, 65, 224, 110, 56, 175, 220, 192, 203, 77, 70, 215, 64, 145, 194, 19, 206, 177, 78, 46, 19, 21, 31, 198, 155, 140, 116, 49, 23, 122, 56, 85, 55, 156, 133, 104, 119, 126, 5, 176, 199, 47, 63, 47, 85, 102, 71, 19, 78, 35, 145, 179, 138, 58, 132, 113, 55, 29, 41, 165, 19, 34, 17, 118, 91, 162, 66, 168, 94, 143, 133, 54, 133, 131, 172, 104, 38, 95, 73, 173, 206, 50, 191, 139, 205, 83, 65, 2, 109, 96, 85, 201, 141, 91, 84, 90, 15, 22, 16, 161, 108, 131, 195, 51, 56, 51, 180, 171, 182, 140, 71, 129, 61, 127, 158, 131, 178, 71, 92, 189, 51, 78, 130, 158, 112, 46, 46, 8, 127, 107, 150, 107, 61, 79, 12, 54, 17, 135, 68, 135, 131, 68, 4, 113, 162, 144, 12, 158, 155, 61, 98, 116, 19, 25, 29, 6, 25, 22, 17, 132, 159, 144, 95, 157, 98, 20, 2, 18, 130, 44, 142, 153, 161, 78, 75, 65, 145, 137, 122, 26, 70, 63, 78, 13, 23, 63, 159, 104, 32, 122, 17, 18, 13, 38, 139, 24, 139, 18, 139, 122, 139, 119, 33, 68, 120, 142, 46, 119, 113, 95, 2, 113, 21, 75, 23, 65, 96, 55, 139, 34, 74, 86, 119, 126, 47, 134, 140, 130, 39, 130, 12, 41, 71, 61, 54, 75, 131, 12, 115, 80, 21, 126, 103, 92, 72, 51, 49, 41, 46, 114, 9, 17, 7, 117, 90, 98, 49, 122, 115, 74, 82, 102, 112, 86, 34, 29, 79, 59, 97, 55, 29, 108, 9, 8, 104, 5, 59, 20, 55, 80, 101, 30, 6, 25, 21, 57, 21, 6, 89, 43, 50, 57, 88, 73, 1, 11, 50, 57, 15, 89, 57, 42, 10, 17, 0, 2, 90, 21, 91, 17, 43, 17, 22, 89, 7, 36, 80, 7, 22, 40, 57, 34, 86, 47, 16, 27, 7, 37, 0, 28, 41, 46, 64, 15, 35, 73, 21, 33, 72, 0, 15, 74, 16, 10, 38, 23, 11, 41, 3, 45, 56, 9, 13, 65, 18, 27, 72, 58, 39, 45, 9, 46, 64, 29, 58, 60, 41, 47, 52, 11, 10, 33, 58, 24, 58, 55, 16, 55, 28, 27, 32, 31, 42, 8, 10, 4, 47, 0, 27, 1, 20, 39, 30, 22, 8, 25, 22, 42, 23, 39, 23, 7, 4, 4, 24, 24, 4, 3, 4, 19, 4, 3, 3, 20, 10, 20, 23, 23, 17, 15, 3, 8, 10, 13, 0, 1, 12, 1, 0, 7, 4, 0, 5, 7, 1, 4, 11, 4, 8);
        List<Integer> list = Arrays.asList();
        int idx = 0;
        int i = 0;

        while (!env.isFinished()) {
            System.out.println("----------------------------------: " + (i+1));
            env.heuristic.printFrontier();

            if(i < list.size()){
                idx = list.get(i);
            }else{
                idx = rand.nextInt(env.frontierSize());
                //idx = env.getActionFronAuxiliarHeuristic();
            }

            System.out.println("Expanded: " + env.heuristic.getFrontier().get(idx));
            env.expandAction(idx);
            i = i + 1;
        }
        System.out.println("Realizable: " + env.realizable);

        System.out.println("End Run :)");





    }
}



