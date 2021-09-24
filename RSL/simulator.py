import random
from tarski.grounding.errors import ReachabilityLPUnsolvable
from tarski.syntax.transform.action_grounding import ground_schema
import tarski.evaluators
from tarski.grounding.lp_grounding import LPGroundingStrategy
from tarski.theories import Theory
from tarski.syntax import *
from tarski.syntax.formulas import unwrap_conjunction_or_atom, land
from tarski.io import PDDLReader
from tarski.grounding.lp_grounding import ground_problem_schemas_into_plain_operators
from tarski.search import GroundForwardSearchModel, BreadthFirstSearch
from tarski.search.model import progress
from tarski.fstrips import AddEffect, DelEffect
import numpy as np
import copy
import time

class Simulator():
    def __init__(self, domainFile, instanceFile, state_mutexes=None, seed=1337):
        random.seed(seed)
        self.domainFile = domainFile
        self.reader = PDDLReader(raise_on_error=True)
        self.reader.parse_domain(domainFile)
        self.reader.parse_instance(instanceFile)
        self.problem = self.reader.problem
        self.model = GroundForwardSearchModel(self.problem, ground_problem_schemas_into_plain_operators(self.problem))
        self.opsToInt = {}
        self.intToOps = {}
        self.atomToInt = {}
        self.intToAtom = {}
        self.set_state_mutexes(state_mutexes)
        nOp = 0
        for op in self.model.operators:
            self.opsToInt[str(op)] = nOp
            self.intToOps[nOp] = op
            nOp += 1
        self.maxBranchingFactor = nOp
        self.state = self.problem.init
        self.grounder = LPGroundingStrategy(self.reader.problem, ground_actions=False)
        lpvariables = self.grounder.ground_state_variables()

        nAtoms = 0
        # Initialise atoms to ints
        for atom_index, atoms in lpvariables.enumerate():
            atom = Atom(atoms.symbol, atoms.binding)
            print("\'{}\'".format(atom))
            self.atomToInt[str(atom)] = nAtoms
            self.intToAtom[nAtoms] = atom
            nAtoms += 1

        self.numAtoms = nAtoms

        self.goalSet = set(unwrap_conjunction_or_atom(self.problem.goal))

    def set_state_mutexes(self, mutexes):
        self.state_mutexes_string = mutexes
        if mutexes is not None:
            self.state_mutexes = []
            maxLength = 0
            for mutexGroup in mutexes:
                group = []
                maxLength = max(maxLength, len(mutexGroup))
                for atom in mutexGroup:
                    if atom != '<none of those>':
                        group.append(self.atomToInt[atom])
                self.state_mutexes.append(group)
            for group_indx in range(len(self.state_mutexes)):
                while len(self.state_mutexes[group_indx]) < maxLength:
                    self.state_mutexes[group_indx].append(-1)
            self.state_mutexes = np.array(self.state_mutexes, dtype=np.int16)
        else:
            self.state_mutexes = None


    def getGroundedDicts(self):
        return self.opsToInt, self.atomToInt

    def fwd_step (self, op):
        assert not self.model.is_goal(self.state)  # Should not call step if already at goal.
        op = self.intToOps[op]
        self.state = progress(self.state, op) # TODO Need to check operator is applicable
        #print("operator {} new state {}".format(op, self.state))
        reward = -1
        done = self.model.is_goal(self.state)
        return self.binaryState(self.state), reward, done

    def applicable_ops_and_successors(self, state=None):
        if state is None:
            state = self.state
        ops = self.fwd_applicable_ops(state)
        opsucstates = []
        for op in ops:
            opsucstates.append((op, progress(state, op)))
        return opsucstates

    def binaryState(self, stateIn):
        state = np.zeros(self.numAtoms)
        for atom in stateIn.as_atoms():
            if str(atom) in self.atomToInt.keys():
                state[self.atomToInt[str(atom)]] = 1  # Setting each true atom.
        return state

    def binaryState2(self, stateIn):
        state = np.zeros(self.numAtoms)
        for atom in stateIn.as_atoms():
            state[self.atomToInt[str(atom)]] = 1  # Setting each true atom.
        return state

    def binaryStateToState(self, binState):
        state = []
        for key, val in enumerate(binState):
            if val == 1:
                state.append(self.intToAtom[key])
        return state

    def fwd_applicable_ops(self, state=None):
        if state is None:
            state = self.state
        ops = []
        opsBin = []
        for op in self.model.applicable(state):
            ops += [op]
        return ops

    def reset_to_initial_state(self):
        self.state = self.problem.init
        return self.binaryState(self.state)

    def getBinaryStringState(self, state):
        binaryString = ""
        try:
            atoms = state.as_atoms()
        except:
            atoms = state # Not a model object and just a list of atoms
        for atom in atoms:
            if binaryString == "":
                binaryString = "{}".format(atom)
            else:
                binaryString = binaryString + " & {}".format(atom)
        return binaryString

    def preimage_set(self, formula, operator):
        operator = self.intToOps[self.opsToInt[operator]] # Get operator object if string

        # Delete add effects
        for eff in operator.effects:
            if isinstance(eff, AddEffect):
                formula.discard(eff.atom)

        # Add preconditions
        for atom in unwrap_conjunction_or_atom(operator.precondition):
            formula.add(atom)
        return formula

    def check_for_state_invariant(self, atom, formula, notIncludingAtoms = []):
        if atom in unwrap_conjunction_or_atom(formula):
            return False
        atomString = str(atom).replace(",", ", ")
        formulaAtoms = []
        for formAtom in unwrap_conjunction_or_atom(formula):
            formulaAtoms.append((formAtom, str(formAtom).replace(",", ", ")))

        for mutexGroup in self.state_mutexes_string:
            if atomString in mutexGroup:
                for formAtom, formAtomStr in formulaAtoms:
                    if formAtom not in notIncludingAtoms:
                        if formAtomStr in mutexGroup:
                            return True
        return False

    def check_for_state_invariant_atoms(self, atom, atom2):
        for mutexGroup in self.state_mutexes_string:
            if atom in mutexGroup and atom2 in mutexGroup:
                return True
        return False

    def check_for_state_invariant_atoms_list(self, atom, listOfAtoms):
        for mutexGroup in self.state_mutexes_string:
            if atom in mutexGroup and any(atom_b in mutexGroup for atom_b in listOfAtoms):
                return True
        return False

    def get_state_mutexes_in_set(self, setOfAtoms):
        mutexes = set()
        for mutexGroup in self.state_mutexes_string:
            if len(setOfAtoms.intersection(mutexGroup)) > 0:
                for atom in mutexGroup:
                    if atom not in setOfAtoms and atom not in mutexes:
                        mutexes.add(atom)
        return mutexes

    def check_for_state_mutexes_in_a_set_of_atoms(self, listOfAtoms):
        if np.max(np.sum(np.isin(self.state_mutexes, listOfAtoms), axis=1)) > 1:
            return True
        return False

    def return_sets_of_mutexes(self, listOfAtoms):
        is_in_atoms = np.isin(self.state_mutexes, listOfAtoms)
        sumOfGroups = np.sum(is_in_atoms, axis=1)
        list_of_mutex_sets = []
        print(is_in_atoms.shape[0])
        for i in range(is_in_atoms.shape[0]):
            if sumOfGroups[i] > 1:
                atomsInGroup = list(self.state_mutexes[i][np.nonzero(is_in_atoms[i])])
                list_of_mutex_sets.append(atomsInGroup)
        print(list_of_mutex_sets)
        return list_of_mutex_sets

    # returns random valid action for use by the explicit search
    def check_for_valid_reg_operators_according_to_SING_def(self, state, operatorsAlreadyTried):
        operatorNums = list(range(self.maxBranchingFactor))
        random.shuffle(operatorNums)
        for operatorNum in operatorNums:
            operator = self.intToOps[operatorNum]
            if operator in operatorsAlreadyTried:
                continue
            addEffects = set()
            delEffects = set()
            preconidtions = set()
            for eff in operator.effects:
                if isinstance(eff, AddEffect):  # Add effects that cause invariant
                    strAtom = str(eff.atom)
                    addEffects.add(self.atomToInt[strAtom])

                if isinstance(eff, DelEffect):
                    strAtom = str(eff.atom)
                    delEffects.add(self.atomToInt[strAtom])

            for atom in unwrap_conjunction_or_atom(operator.precondition):
                if str(atom) in self.atomToInt.keys(): # if non static fluent
                    preconidtions.add(self.atomToInt[str(atom)])
            trueAtoms = set()
            for nonzeroval in np.nonzero(state)[0]:
                trueAtoms.add(nonzeroval)
            if trueAtoms.issuperset(preconidtions.union(addEffects).difference(delEffects)):
                return operator
        return None

    # explicit state search as explained by Yu et al
    def applyOperatorSING(self, state, operator_selected):
        for eff in operator_selected.effects:
            if isinstance(eff, AddEffect):  # Del add effects
                strAtom = str(eff.atom)
                state[self.atomToInt[strAtom]] = 0

            if isinstance(eff, DelEffect): # Add del effects
                strAtom = str(eff.atom)
                state[self.atomToInt[strAtom]] = 1
        return state


    def get_random_regression_plan(self, planLength, regression_method):
        formula = copy.deepcopy(self.problem.goal)
        plan = []
        planFormulas = [copy.deepcopy(formula)]
        atomsInPlan = set()
        atomsAlwaysInPlan = set()
        for atom in unwrap_conjunction_or_atom(formula):
            atomsAlwaysInPlan.add(str(atom))
        print("goal formula {}".format(formula))
        for step in range(planLength):
            actionsToSelectFrom = []
            startTimeactions = time.perf_counter()
            atomsInFormulaStr = set()
            for atom in unwrap_conjunction_or_atom(formula):
                atomsInFormulaStr.add(str(atom))
            current_mutexes_with_formula = self.get_state_mutexes_in_set(atomsInFormulaStr)
            maxTime1 = 0
            maxTime2 = 0
            maxTime3 =0
            print("branching factor is {}".format(self.maxBranchingFactor))
            for operatorNum in range(self.maxBranchingFactor):
                operator = self.intToOps[operatorNum]
                operator_is_consistent = None
                addEffects = set()
                startTime = time.perf_counter()
                for eff in operator.effects:
                    if isinstance(eff, AddEffect):  # Add effects that cause invariant
                        strAtom = str(eff.atom)
                        addEffects.add(strAtom)
                        if operator_is_consistent is None:
                            operator_is_consistent = False
                        if strAtom in atomsInFormulaStr:
                            operator_is_consistent = True
                    if isinstance(eff, DelEffect):
                        if str(eff.atom) in atomsInFormulaStr:  # not in original definition but makes stronger
                            operator_is_consistent = False
                            break
                maxTime1 += time.perf_counter() - startTime
                startTime = time.perf_counter()

                if operator_is_consistent != False and len(addEffects.intersection(current_mutexes_with_formula)) > 0:
                    operator_is_consistent = False
                maxTime2 += time.perf_counter() - startTime
                startTime = time.perf_counter()

                if operator_is_consistent is None:
                    operator_is_consistent = True

                # Check preconditions don't cause invariant
                if operator_is_consistent:
                    # print("operator {} is consistent before checking pre conditions".format(operatorNum))
                    forCheckingMutexes = set()
                    for atom in unwrap_conjunction_or_atom(operator.precondition):
                        if str(atom) in self.atomToInt.keys():
                            forCheckingMutexes.add(self.atomToInt[str(atom)])
                    for atom in unwrap_conjunction_or_atom(formula):
                        if str(atom) not in addEffects:
                            forCheckingMutexes.add(self.atomToInt[str(atom)])
                    if self.check_for_state_mutexes_in_a_set_of_atoms(list(forCheckingMutexes)):
                        operator_is_consistent = False
                maxTime3 += time.perf_counter() - startTime

                if not operator_is_consistent:
                    # print("operator {} is not consistent".format(operatorNum))
                    continue
                actionsToSelectFrom.append(operator)
            # print("end time to get relevant actions {}".format(time.perf_counter() - startTimeactions))
            # print("a {}, b {}, c {}".format(maxTime1, maxTime2, maxTime3))
            startTime = time.perf_counter()

            candidateOpsNFormulas = []
            if len(actionsToSelectFrom) > 0:
                random.shuffle(actionsToSelectFrom)
                formulaS = set()
                for atom in unwrap_conjunction_or_atom(formula):
                    formulaS.add(str(atom))
                for operator in actionsToSelectFrom:
                    # Delete add effects
                    atomsForNewFormula = formulaS.copy()
                    for eff in operator.effects:
                        if isinstance(eff, AddEffect):
                            atomsForNewFormula.discard(str(eff.atom))

                    # Add preconditions
                    for atom in unwrap_conjunction_or_atom(operator.precondition):
                        if str(atom) in self.atomToInt.keys():
                            atomsForNewFormula.add(str(atom))
                    candidateOpsNFormulas.append((operator, atomsForNewFormula))
            print("end time to preimage actions {}".format(time.perf_counter() - startTime))
            print("number of canidate actions {}".format(len(candidateOpsNFormulas)))
            startTime = time.perf_counter()
            if len(candidateOpsNFormulas) > 0:
                max_unique_atoms = None
                bestOp = None
                best_formula = None
                for operator, temp_formula in candidateOpsNFormulas:
                    num_unique_atoms = 0
                    if regression_method == "countAdds" or regression_method == "countBoth":
                        for atom in temp_formula:
                            if atom not in atomsInPlan:
                                num_unique_atoms += 1
                    if regression_method == "countDels" or regression_method == "countBoth":
                        for atom in atomsAlwaysInPlan:
                            if atom not in temp_formula:
                                num_unique_atoms += 1 # atom has never been deleted in plan thus far
                    if bestOp is None:
                        bestOp = operator
                        best_formula = temp_formula
                        max_unique_atoms = num_unique_atoms
                    elif max_unique_atoms < num_unique_atoms:
                        bestOp = operator
                        best_formula = temp_formula
                        max_unique_atoms = num_unique_atoms
                best_formula_atoms = []
                for atomS in best_formula:
                    best_formula_atoms.append(self.intToAtom[self.atomToInt[atomS]])
                formula = land(*best_formula_atoms, flat=True)
                print("op {}, unique atoms {}".format(bestOp, max_unique_atoms, formula))
                planFormulas.append(copy.deepcopy(formula))
                plan.append(str(bestOp))
                for atom in unwrap_conjunction_or_atom(formula):
                    atomsInPlan.add(str(atom))
                atomsAlwaysInPlan = set(atom for atom in atomsAlwaysInPlan if self.intToAtom[self.atomToInt[atom]] in unwrap_conjunction_or_atom(formula))
            else:
                break
            print("end time to select best action {}".format(time.perf_counter() - startTime))
        return plan
