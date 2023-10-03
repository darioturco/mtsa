import random
from src.agents.agent import Agent

class ReadyAbstraction(Agent):

    # En java se inicializa con la list de LTSs, el conjunto de estados marcados y el alfabeto
    def __init__(self, env=None):
        super().__init__(env)
        # Podria sacar eso del ambiente

    def train(self, *args, **kwargs):
        pass

    def get_action(self, state, *args, **kwargs):
        # state es una lista de HActions (java)


        return 0


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
    #  - La segunda es hacer mi propia adaptacion de RA para DCS For Python. Eso es algo mas complicado y desafiente pero
    #                    corremos el riego de que salga mal ademas de que vamos a gastar mucho tiempo

# timeout 10h java -Xmx8g -classpath mtsa.jar ltsa.ui.LTSABatch -c DirectedController -r -i fsp_path(self.problem, n, k)
