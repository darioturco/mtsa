package MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking;

import MTSTools.ac.ic.doc.commons.relations.Pair;

import java.io.*;
import java.util.*;

import MTSTools.ac.ic.doc.mtstools.model.LTS;
import MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.abstraction.HeuristicMode;
import MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.abstraction.ReadyAbstraction;
import ai.onnxruntime.OrtException;
import jargs.gnu.CmdLineParser;
import ltsa.dispatcher.TransitionSystemDispatcher;
import ltsa.lts.*;
import ltsa.ui.StandardOutput;

import static org.junit.Assert.*;



/** This class can be used from python with jpype */
public class DCSForPython {
    public DirectedControllerSynthesisBlocking<Long, String> dcs;
    public ExplorationHeuristic<Long, String> heuristic;
    public HeuristicMode heuristicMode;
    public String featureGroup;     // TODO: Cambiar a un enumerate
    public String modelPath;
    public boolean realizable;
    public boolean finished;
    public int lastEntityExpanded;
    public int lastEntityExpandedWithoutReset;
    public boolean started_synthesis;

    static String testHeader = "Instance,N,K,Name,Transitions\n";
    static String selectionHeader = "Instance,Model,Solved,Expansions\n";

    public DCSForPython(String heuristicMode){
        this.started_synthesis = false;
        this.realizable = false;
        this.finished = false;
        this.heuristicMode = HeuristicMode.valueOf(heuristicMode);
        this.lastEntityExpanded = -1;
        this.lastEntityExpandedWithoutReset = -1;
        this.featureGroup = "";
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
        this.heuristic = this.dcs.getHeuristic(this.heuristicMode, featureGroup);
        if(this.heuristicMode == HeuristicMode.RL){
            try {
                ((RLExplorationHeuristic) this.heuristic).loadModelFromPath(modelPath);
            }catch (OrtException e){e.printStackTrace();}
        }
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

    public LTS<Long, String> buildControler(){
        if(realizable){
            return dcs.buildController();
        }else{
            return null;
        }
    }

    public void setRLParameters(String featureGroup, String modelPath){
        this.featureGroup = featureGroup;
        this.modelPath = modelPath;
    }

    public static int syntetizeWithHeuristic(String FSP_path, String heuristic, String featuresGroup, String modelPath, int budget, boolean verbose){
        DCSForPython env = new DCSForPython(heuristic);
        env.setRLParameters(featuresGroup, modelPath);
        env.setFlags(true, true);
        env.startSynthesis(FSP_path);

        int idx;
        int i = 0;
        while (!env.isFinished() && i < budget) {

            idx = env.getActionFronAuxiliarHeuristic();
            if(verbose){
                System.out.println("----------------------------------: " + (i+1));
                env.heuristic.printFrontier();
                System.out.println("Expanded(" + idx + "): " + env.heuristic.getFrontier().get(idx));
            }

            env.expandAction(idx);
            i = i + 1;
        }

        env.dcs.notify_end_synthesis();



        return i;
    }

    // Este metodo corre todas las instancias de una familia con una heuristica dada
    public static Pair<Integer, Integer> testHeuristic(int budget, String instance, String heuristic, String featuresGroup, String modelPath, boolean save, int minSize, int maxSize, int verbose){
        int solvedInstances = 0;
        int totalExpansions = 0;

        if(verbose >= 1){
            System.out.println("Testing model: " + modelPath);
        }

        for(int n=minSize;n<=maxSize;n++){
            int res = 0;
            for(int k=minSize;k<=maxSize;k++){
                if(res < budget){
                    String fsp_path = "./fsp/" + instance + "/" + instance + "-" + n + "-" + k + ".fsp";

                    res = DCSForPython.syntetizeWithHeuristic(fsp_path, heuristic, featuresGroup, modelPath, budget, false);

                    if(res < budget){
                        solvedInstances += 1;
                    }else{
                        res = budget;
                    }
                }
                totalExpansions += res;

                if(verbose == 1) {
                    System.out.println(res);
                } else if (verbose == 2) {
                    System.out.println("Result of " + instance + "-" + n + "-" + k + ": " + res);
                }

                if(save){
                    String csvPath = "./results/csv/" + featuresGroup + "-" + instance + ".csv";
                    String[] data = {instance, String.valueOf(n), String.valueOf(k), modelPath, String.valueOf(res)};
                    writeCSV(csvPath, data, testHeader);
                }
            }
        }
        return new Pair<>(solvedInstances, totalExpansions);
    }
    public static void writeCSV(String csvPath, String[] data, String header){
        File file = new File(csvPath);
        boolean exists = file.exists();

        try{
            FileWriter outputFile = new FileWriter(file, true);
            StringBuilder line = new StringBuilder();

            if(!exists && !header.isEmpty()){
                line.append(header);
                outputFile.write(line.toString());
                line = new StringBuilder();
            }

            for (int i = 0; i < data.length; i++) {
                line.append(data[i]);
                if (i != data.length - 1) {
                    line.append(',');
                }
            }
            line.append("\n");

            outputFile.write(line.toString());
            outputFile.close();
        }catch (IOException e){
            e.printStackTrace();
        }
    }

    // Esta funcion levanta todos los modelos de un experimento para una instancia y los testea con el budget dado
    public static void selectRL(String instance, String experimentName, int budget, int startInModel){
        String folderPath = "./results/models/" + instance + "/" + experimentName + "/";

        File folder = new File(folderPath);
        Set<File> setOfFiles = new HashSet<>(Arrays.asList(folder.listFiles()));

        Set<String> setTestedModels = readCSVColumn("./results/selection/" + experimentName + "-" + instance + ".csv");
        List<String> listModels = new ArrayList<>();
        for(File f : setOfFiles){
            if(f.isFile()){
                String fileName = folderPath + f.getName();
                if(fileName.contains(".onnx") && !setTestedModels.contains(fileName)){
                    listModels.add(fileName);
                }
            }
        }
        Collections.shuffle(listModels);

        int bestValue = budget * 225 + 1;
        String bestModel = "";

        System.out.println("Starting selection...");
        for (String modelName : listModels) {
            System.out.println("Testing model: " + modelName);
            Pair<Integer, Integer> res = testHeuristic(budget, instance, "RL", experimentName, modelName, false, 2, 9, 0);
            int solvedInstances = res.getFirst();
            int totalExpansions = res.getSecond();

            if(totalExpansions < bestValue){
                bestModel = modelName;
                bestValue = totalExpansions;
            }

            System.out.println("Model " + modelName + ": " + totalExpansions);
            String csvPath = "./results/selection/" + experimentName + "-" + instance + ".csv";
            String[] data = {instance, modelName, String.valueOf(solvedInstances), String.valueOf(totalExpansions)};

            writeCSV(csvPath, data, selectionHeader);
        }
        System.out.println("Best Model: " + bestModel);
    }

    public static Set<String> readCSVColumn(String folderPath){
        Set<String> res = new HashSet<>();

        try{
            BufferedReader br = new BufferedReader(new FileReader(folderPath));
            String line = br.readLine();
            System.out.println(line);
            while (line != null) {
                String[] cols = line.split(","); // use comma as separator
                if(cols[1].contains(".onnx")){
                    res.add(cols[1]);
                }
                line = br.readLine();
            }
        } catch (Exception e) {}
        return res;
    }

    public static void printHelp(){
        // TODO: completar
        System.out.println("Esta es la ayuda... completar");
        
    }

    // This function is for testing purposes only
    public static void testExample(){
        //String modelPath = "F:\\UBA\\Tesis\\mtsa\\MTSApy\\TA-2-2-10-partial.onnx";
        //String modelPath = "F:\\UBA\\Tesis\\mtsa\\MTSApy\\DP-2-2-4460-partial.onnx";
        //String modelPath = "F:\\UBA\\Tesis\\mtsa\\MTSApy\\results\\models\\DP\\2-2\\DP-2-2-690-partial.onnx";
        //String modelPath = "";

        selectRL("AT", "2-2", 1000, 0);

        //DCSForPython.testHeuristic(1000, "DP", "RL", "2-2", modelPath, false, 1, 15, 2);

        //String fsp_path = "F:\\UBA\\Tesis\\mtsa\\MTSApy\\fsp\\DP\\DP-2-2.fsp";
        //DCSForPython.syntetizeWithHeuristic(fsp_path, "RL", "CRL", modelPath, 1000, true);



        /*
        // Acontinuacion un ejemplo de como se deberia usar DCSForPython
        String instance = "BW";

        //String FSP_path = "/home/dario/Documents/Tesis/mtsa/maven-root/mtsa/target/test-classes/Blocking/ControllableFSPs/GR1test1.lts"; // Falla porque tiene guiones
        //String FSP_path = "F:\\UBA\\Tesis\\mtsa\\maven-root\\mtsa\\target\\test-classes\\Blocking\\ControllableFSPs\\GR1Test43.lts";
        //String FSP_path = "F:\\UBA\\Tesis\\mtsa\\maven-root\\mtsa\\target\\test-classes\\Blocking\\NoControllableFSPs\\GR1Test11.lts";
        String FSP_path = "F:\\UBA\\Tesis\\mtsa\\MTSApy\\fsp\\" + instance + "\\" + instance + "-5-3.fsp";
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
            //env.heuristic.printFrontier();

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
        if(env.realizable){
            LTS<Long, String> director = env.buildControler();
            System.out.println("Director's Transitions: " + director.getTransitions().size());
        }
        System.out.println("End Run :)");
        */
    }


    public static void main(String[] args) {
        //testExample();

        CmdLineParser cmdParser = new CmdLineParser();
        CmdLineParser.Option selection_opt = cmdParser.addBooleanOption('s', "selection");
        CmdLineParser.Option help_opt = cmdParser.addBooleanOption('h', "help");
        CmdLineParser.Option instance_opt = cmdParser.addStringOption('i', "instance");
        CmdLineParser.Option experiment_opt = cmdParser.addStringOption('e', "experiment");
        CmdLineParser.Option budget_opt = cmdParser.addIntegerOption('b', "budget");
        CmdLineParser.Option start_opt = cmdParser.addIntegerOption('r', "startModel");

        CmdLineParser.Option model_opt = cmdParser.addStringOption('m', "model");

        try {
            cmdParser.parse(args);
        } catch (CmdLineParser.OptionException e) {
            System.out.println("Invalid args: " + e.getMessage() + "\n");
            DCSForPython.printHelp();
            System.exit(0);
        }

        Boolean selection_b = (Boolean)cmdParser.getOptionValue(selection_opt);
        boolean selection = (selection_b != null && selection_b);

        Boolean help_b = (Boolean)cmdParser.getOptionValue(help_opt);
        boolean help = (help_b != null && help_b);

        if(help){
            DCSForPython.printHelp();
        }else{
            String instance = (String)cmdParser.getOptionValue(instance_opt);
            String experiment = (String)cmdParser.getOptionValue(experiment_opt);
            int budget = (int)cmdParser.getOptionValue(budget_opt);

            if(selection){
                Object startModelObj = cmdParser.getOptionValue(start_opt);
                int startModel = 0;
                if(startModelObj != null){
                    startModel = (int)startModelObj;
                }
                selectRL(instance, experiment, budget, startModel);
            }else {
                String modelPath = (String) cmdParser.getOptionValue(model_opt);
                DCSForPython.testHeuristic(budget, instance, "RL", experiment, modelPath, true, 1, 15, 2);
            }
        }
    }
}