import sys
import os
import random

class InstanceMaker():
    def __init__(self, c, nc, lts, substates, or_p, assumptions, guaranties, p):
        assert c>0
        assert nc>0
        assert 1.0 >= p > 0.0
        assert 1.0 >= or_p >= 0.0

        self.c = c
        self.nc = nc
        self.lts = lts
        self.substates = substates
        self.or_p = or_p
        self.assumptions = assumptions
        self.guaranties = guaranties
        self.p = p
        self.letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    def make_alphabet(self):
        res = []
        for i in range(self.c):
            res.append(self.letters[i])

        for i in range(self.nc):
            res.append("_" + self.letters[i])

        return res

    def make_fluets(self, amount, alphabet):
        fluents = []

        for _ in range(amount):
            fluent = []
            while len(fluent) == 0:
                for letter in alphabet:
                    if(random.random() < self.p):
                        fluent.append(letter)

            fluent = tuple(fluent)
            if(fluent not in fluents):
                fluents.append(fluent)

        return [set(f) for f in fluents]

    def random_from(self, alphabet, restric=None):
        res = alphabet[random.randint(0, len(alphabet)-1)]
        while res == restric and len(alphabet) > 2:
            res = alphabet[random.randint(0, len(alphabet) - 1)]

        return res

    def make_ltss(self, alphabet):
        ltss = []
        for l in range(self.lts):
            substates = [f"F{l}S{i}" for i in range(self.substates)]
            lts = f"{substates[0]},\n"
            for substate in substates:
                lts += f"{substate} = ({self.random_from(alphabet)} -> {self.random_from(substates, substate)}"

                while random.random() > self.or_p:
                    lts += f" | {self.random_from(alphabet)} -> {self.random_from(substates)}"

                lts += f"),\n"

            ltss.append(lts[:-2] + '.')

        return ltss

    def get_used_alphabet(self, ltss, alphabet):
        new_alphabet = []
        for c in alphabet:
            for lts in ltss:
                if c in lts:
                    new_alphabet.append(c)

        return new_alphabet

    def make(self, name):
        # TODO: Save a comment with all the parameters of creation
        res = ''

        alphabet = self.make_alphabet()
        ltss = self.make_ltss(alphabet)
        alphabet = self.get_used_alphabet(ltss, alphabet)
        fluents_assumptions = self.make_fluets(self.assumptions, alphabet)
        fluents_guaranties = self.make_fluets(self.guaranties, alphabet)

        for i, lts in enumerate(ltss):
            res += f"LTS{i} = " + lts + "\n\n"

        lts_names = "".join([f"LTS{i} || " for i, _ in enumerate(ltss)])
        res += f"||Plant = ({lts_names[:-4]}).\n"
        res += f"set Allactions = {str(set(alphabet))}\n"

        for i, f in enumerate(fluents_assumptions + fluents_guaranties):
            res += f"fluent F{i} = <{str(f)}, Allactions\{str(f)}>\n"

        for i in range(self.assumptions + self.guaranties):
            res += f"assert A{i} = F{i}\n"

        controlable = f"{str({self.letters[i] for i in range(self.c)})}"
        assumption_str = "" if self.assumptions == 0 else "     assumption = " + str({f"A{i}" for i in range(self.assumptions)}) + "\n"
        liveness_str = "" if self.guaranties == 0 else "     liveness = " + str({f"A{i}" for i in range(self.assumptions, self.assumptions + self.guaranties)}) + "\n"
        res += "controllerSpec Goal = {\n     controllable = " + controlable + "\n" + assumption_str + liveness_str + "}\n"

        res += "heuristic ||DirectedController = Plant~{Goal}.\n"

        res += f"assert Check = ("
        for i in range(self.guaranties):
            res += f'[]<>A{i} && '
        res = res[:-4] + f")\n"
        res += f"||Sys = (DirectedController || Plant)."
        res = res.replace("'", "")

        with open(name, 'w') as f:
            f.write(res)

        return res



