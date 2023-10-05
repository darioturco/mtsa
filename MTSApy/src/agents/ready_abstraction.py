import random
from src.agents.agent import Agent

class ReadyAbstraction(Agent):

    # En java se inicializa con la list de LTSs, el conjunto de estados marcados y el alfabeto
    def __init__(self, env=None):
        super().__init__(env)
        self.clear_sets()
        self.javaEnv = self._env.context.composition.javaEnv
        self.ltss = self.javaEnv.dcs.ltss
        self.alphabet = self.javaEnv.dcs.alphabet

    def train(self, *args, **kwargs):
        pass

    def get_action(self, state, *args, **kwargs):
        # state es una lista de ActionWithFeatures (java)
        for a in state:
            self.clear_sets()
            self.buildRA(a)
            self.evaluateRA()



        # Parte random (Borrar)
        n_actions = len(state)
        return random.randint(0, n_actions - 1)

    def buildRA(self, action):
        compostate = action.state

        for lts in self.ltss:
            s = buildHState(lts, compostate.getStates().get(lts))
            x = 3 + 5

        return x
    """
    for (int lts = 0; lts < ltss.size(); ++lts) {
        HState<State, Action> s = buildHState(lts, compostate.getStates().get(lts));
        for (Pair<Action,State> transition : s.getTransitions()) {
            HAction<Action> action = alphabet.getHAction(transition.getFirst());
            if (!s.state.equals(transition.getSecond())) { // !s.isSelfLoop(action)
                readyInLTS.get(action).add(lts);
                vertices.add(action);
            }
        }
    }
    """

    def clear_sets(self):
        self.vertices = {}
        self.edges = {}
        self.estimates = {}
        self.shortest = {}
        self.fresh = {}
        self.readyInLTS = {}
        self.gapCache = {}



    #eval(Compostate <State, Action> compostate){
    #    if (!compostate.isEvaluated()) {
    #        clear();                               // Limpia el grafo, deja vacios todos sus conjuntos
    #        buildRA(compostate);                   // Construye el grafo (primero agraga los vertices y lugo losconecta como e devido)
    #        evaluateRA(compostate);                // Construye la tabla de estiamtes
    #        extractRecommendations(compostate);    //
    #    }
    #}

    # Hay dos opciones para correr las instancias con la heuristica RA:
    #  - La primera es usar subprocess con el comando ["timeout", timeout, "java", "-Xmx8g", "-classpath", "mtsa.jar",
    #                    "ltsa.ui.LTSABatch", "-c", "DirectedController", "-r", "-i", fsp_path(self.problem, n, k)]
    #  - La segunda es hacer mi propia adaptacion de RA para DCS For Python. El problema de esto es que la heuristica de RA usa
    #                    los LTS individuales y el alfabeto los cuales no son faciles de usar desde python. Puedo probar y ver que tan lento es...

# timeout 10h java -Xmx8g -classpath mtsa.jar ltsa.ui.LTSABatch -c DirectedController -r -i /home/dario/Documents/Tesis/mtsa/MTSApy/fsp/TA/TA-2-2.fsp # Comando para invocar la heristica RA (el flag bolenao -r es el que indica que se use esa heuristica)
# timeout 10h java -Xmx8g -XX:MaxDirectMemorySize=512m -classpath mtsa.jar MTSTools.ac.ic.doc.mtstools.model.operations.DCS.nonblocking.FeatureBasedExplorationHeuristic -m self.heuristic_name() -f str(self.max_frontier)
