package IntegrationTests;

import FSP2MTS.ac.ic.doc.mtstools.test.util.TestLTSOuput;

import java.util.*;

import MTSSynthesis.ar.dc.uba.util.FormulaToMarkedLTS;
import MTSSynthesis.controller.model.ControllerGoal;
import MTSTools.ac.ic.doc.commons.relations.Pair;
import MTSTools.ac.ic.doc.mtstools.model.LTS;
import MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.*;
import MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.abstraction.Abstraction;
import MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.abstraction.HeuristicMode;
import MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking.abstraction.Recommendation;
import MTSTools.ac.ic.doc.mtstools.model.operations.impl.WeakAlphabetMergeBuilder;
import ltsa.control.ControllerGoalDefinition;
import ltsa.dispatcher.TransitionSystemDispatcher;
import ltsa.lts.*;
import ltsa.lts.ltl.AssertDefinition;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.io.File;
import java.io.IOException;

import static org.junit.Assert.*;
import static org.junit.Assert.fail;


@RunWith(Parameterized.class)
public class TestBlockingDCSForPythonRL {

    private static final String[] RESOURCE_FOLDERS = {"Blocking/ControllableFSPs","Blocking/NoControllableFSPs"};
    private static final String FSP_NAME = "DirectedController";
    private File ltsFile;

    public TestBlockingDCSForPythonRL(File getFile) {
        ltsFile = getFile;
    }


    @Parameterized.Parameters(name = "{index}: {0}")
    public static List<File> controllerFiles() throws IOException {
        List<File> allFiles = new ArrayList();
        for (String resource_folder:RESOURCE_FOLDERS) {
            allFiles.addAll(LTSTestsUtils.getFiles(resource_folder));
        }
        return allFiles;
    }


    @Test
    public void testControllability() throws Exception {
        String FSP_Path = ltsFile.toString();
        Pair<CompositeState, LTSOutput> c = DCSForPython.compileFSP(FSP_Path);
        CompositeState compositeState = c.getFirst();
        LTSOutput output = c.getSecond();
        ControllerGoal<String> goal = compositeState.goal;

        if(TransitionSystemDispatcher.checkGuaranteesAndAssumptions(goal, output)){
            return;
        }

        // ltss
        Pair<List<LTS<Long, String>>, Set<String>> p = TransitionSystemDispatcher.getLTSs(compositeState);
        List<LTS<Long, String>> ltss = p.getFirst();
        Set<String> actions = p.getSecond();

        // goal (marked or blocking)
        FormulaToMarkedLTS ftm = new FormulaToMarkedLTS();
        Pair<HashMap<Integer, Integer>, HashMap<Integer, Integer>> pairGuaranteesAndAssumptions = TransitionSystemDispatcher.getGuaranteesAndAssumptions(goal, ltss, actions, output);
        HashMap<Integer, Integer> guarantees =  pairGuaranteesAndAssumptions.getFirst();
        HashMap<Integer, Integer> assumptions = pairGuaranteesAndAssumptions.getSecond();

        boolean hasInvalidAction = TransitionSystemDispatcher.filterActions(compositeState, ltss, assumptions, guarantees, output);
        if(hasInvalidAction){
            return;
        }

        DirectedControllerSynthesisBlocking<Long,String> dcs = new DirectedControllerSynthesisBlocking<>();

        String heuristicModeString = "Ready";
        HeuristicMode heuristicMode = HeuristicMode.valueOf(heuristicModeString);
        ExplorationHeuristic<Long, String> heuristic = dcs.getHeuristic(heuristicMode);
        dcs.heuristic = heuristic;


        //System.out.println("***********************************************************************************");
        //System.out.println("Synthesizing controller by DCS...");

        dcs.synthesize(
                ltss,
                goal.getControllableActions(),
                goal.isReachability(),
                guarantees,
                assumptions
        );

        List<Pair<Compostate<Long, String>, HAction<String>>> expansionList = dcs.synthesizeTrace;
        //System.out.println("------------------------------------------------");

        DCSForPython env = new DCSForPython(heuristicModeString);
        env.startSynthesis(FSP_Path);
        int idx;
        Iterator<Pair<Compostate<Long, String>, HAction<String>>> it = expansionList.iterator();

        try {
            while (!env.isFinished()) {
                idx = env.getIndexOfStateAction(it.next());
                env.expandAction(idx);
            }
        } catch (Exception e) {
            fail("Error expanding action.");
        }




        // The exploration shold finish at this point
        assertTrue("The exploration hasn't finished.", env.dcs.isFinished());

        // The answer from this instance is correct
        String folder = ltsFile.getParentFile().getName();
        boolean isControllable = env.dcs.isGoal(env.dcs.initial);
        if(folder.equals("ControllableFSPs")){
            assertTrue("The FSP is controllable", isControllable);
        } else {
            assertFalse("The FSP is not controllable", isControllable);
        }
    }
}


