import random
from src.agents.agent import Agent

class DebuggingAbstraction(Agent):

    # En java se inicializa con la list de LTSs, el conjunto de estados marcados y el alfabeto
    def __init__(self, env=None):
        super().__init__(env)
        self.javaEnv = self._env.context.composition.javaEnv
        self.ltss = self.javaEnv.dcs.ltss
        self.alphabet = self.javaEnv.dcs.alphabet
        self.open = [self.javaEnv.dcs.initial]

    def train(self, *args, **kwargs):
        pass

    def get_action(self, state, *args, **kwargs):
        # state es una lista de ActionWithFeatures (java)



        compostate = self.remove_from_open()
        ### TODO: Deberia chequar que la primer trancision del open este en la lista de states

        best_action, _ = self.eval(compostate) # best_action es el string de la mejor accion
        self.doOpen(compostate)

        # Obtengo el indice de la mejor transicion (best_action) en la lista de states
        return self.get_index_in_state(state, best_action, compostate)


    def add_to_open(self, compostate):
        compostate.inOpen = True
        return self.open.append(compostate)

    def remove_from_open(self):
        ### TODO: Deberia assertar que open tenga al menos un elemento
        return self.open.pop(0)

    # Debugging Abstraction avaluation:
    #    it order the transation lexicograficaly using its 3 first letters
    def eval(self, compostate):
        # Le da un valor a cada trancision del compostate
        # devuelve el python string de la mejor
        best_action = ""
        best_value = -1
        for action in compostate.getTransitions():
            action_str = str(action.toString())
            value = ord(action_str[0])
            if len(action_str) >= 3:
                value += ord(action_str[1]) + ord(action_str[2])

            if value > best_value:
                best_value = value
                best_action = action

        return best_action, best_value

    """
    private Recommendation<Action> doOpen(Compostate<State, Action> compostate) {
        compostate.inOpen = false;
        if (!compostate.isLive() || compostate.getStates() == null) {
            return null;
        }
        assertTrue("compostate to open doesn't have valid recommendation", compostate.hasValidRecommendation());

        Recommendation<Action> recommendation = compostate.nextRecommendation();
        //System.out.println(compostate + " | " + recommendation.getAction());
        expand(compostate, recommendation);
        if (compostate.isControlled() && compostate.hasValidRecommendation() && compostate.isStatus(Status.NONE))
            compostate.open();
        return recommendation;
    }
    """
    def doOpen(self, compostate):
        compostate.inOpen = false
        pass

    def open(self, compostate):
        # Actualiza la cola
        result = False
        compostate.live = True
        if not compostate.inOpen:
            if compostate.hasStatusChildNone():
                result = True
                self.add_to_open(compostate)
            else:
                for transition in compostate.getExploredChildren():
                    child = transition.getSecond()
                    # chequear isLive()
                    if child.isStatusNone():
                        if not child.isLive():
                            result = result or self.open(child)

                if (not result) or compostate.isControlled():
                    result = True
                    self.add_to_open(compostate)

        return result

    def get_index_in_state(self, state, action, compostate):

        res = 0
        for action_with_features in state:
            if action_with_features.state == compostate and action_with_features.action == action:
                return res
            res += 1
            #if action_with_features.action == compostate.action and best_action == action_with_features.action

        # TODO: assert, it should not reach this part of the code
        # Parte random (Borrar)
        n_actions = len(state)
        return random.randint(0, n_actions - 1)