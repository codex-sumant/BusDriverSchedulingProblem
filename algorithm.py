import random
import time
import logging
import math
from neighborhood import Neighborhood
from copy import deepcopy
from abc import ABC, abstractmethod
from xmlrpc.client import Boolean
from neighborhood import Shift

from data import Solution 
from typing import List
import csv

global_store = []

class Algorithm(ABC):
    """Abstract algorithm class."""
    
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)  

    def apply(self, solution: Solution=None, timeout:float=None,  start_time:float=None,algorithm:str = None,nestedCall:bool = False,instance:str = None):
        """Apply the algorithm to the given solution.

        Parameters
        ----------
        solution : Solution, optional
            the initial solution, by default None
        timeout : float, optional
            maximum time allowed for running the algorithm, by default None
        start_time : float, optional
            starting_time (sometimes it's necessary to evaluate the runtime), by default None

        """
        self.instance = instance
        self.start = time.perf_counter() if start_time is None else start_time
        if not nestedCall:
            self.time_limit = timeout
        else:
            self.time_limit = timeout - self.start
        if not nestedCall:
            solution = self._do_apply(solution, None if timeout is None else self.start + float(timeout), algorithm = algorithm)
        else:
            solution = self._do_apply(solution, None if timeout is None else float(timeout), algorithm = algorithm)
        self.total_time = time.perf_counter() - self.start
        return solution

    @abstractmethod
    def _do_apply(self, solution, timeout=None, algorithm = None):
        pass

    def check_timeout(self, timeout: float) -> bool:
        """Check if current time > timeout

        Parameters
        ----------
        timeout : float
            maximum CPU time allowed

        Returns
        -------
        Boolean
            True if current time < timeout. False, otherwise.
        """
        if timeout is not None and time.perf_counter() > timeout:
            return True
        return False
    def writeToCsv(self):
        with open('output.csv', 'a')as f:
        # using csv.writer method from CSV package
            write = csv.writer(f)
            write.writerow(global_store)

    def path_relinking(self, initial_solution: Solution, guiding_solution: Solution,timeout:float = None) -> Solution:
        best_val = 10**(20)
        best_move = None
        #solution = deepcopy(initial_solution)
        best_sol = None
        count = 0
        for move in self.connectingPath(initial_solution,guiding_solution,timeout):
            if self.check_timeout(timeout):
                break
            if len(move) == 0:
                continue
            solution = deepcopy(initial_solution)
            
            solution.update(move)
            #solution = TabuSearch(neighborhood=Shift(),tabu_tenure=12,no_improvement=48).apply(solution,timeout=None)
            count += 1
            if solution.value < best_val: 
                best_val = solution.value
                best_move = move
                best_sol = solution
            #initial_solution.revert(move)
        if best_move is None:
            logging.debug("No improving move found")
            return None
        logging.debug(F"Found better solution in {count} candidates")
        #solution.update(best_move)
        return best_sol
    
    def connectingPath(self,initial_solution: Solution, guiding_solution: Solution,timeout:float = None) -> List[tuple]:
        n = len(initial_solution.employees)
        #rm = min(4,math.floor(math.sqrt(n)))
        rm = math.floor(math.sqrt(n))
        N = 50
        for i in range(N):
            if self.check_timeout(timeout):
                    break
            possible_employees = random.sample(initial_solution.employees.items(),rm)
            moves = []
            for id,emp in possible_employees:
                if self.check_timeout(timeout):
                    break
                additionalLegs = []
                m = len(emp.bus_legs)
                flag = False
                if m > 0 and id not in guiding_solution.employees:
                    [additionalLegs.append(leg) for leg in emp.bus_legs]
                    flag = True
                
                emp1 = guiding_solution.employees[id]
                n = len(emp1.bus_legs)
                
                i , j = 0 , 0
                while i < m and j < n and not flag:
                    leg1 = emp.bus_legs[i]
                    leg2 = emp1.bus_legs[j]
                    matched =  False
                    if leg1.id == leg2.id:
                        i += 1
                        j += 1
                        matched = True
                    else:
                        if leg1.start > leg2.start:
                            j += 1
                        else:
                            additionalLegs.append(leg1)
                            i += 1
                    if i >= m:
                        i = m-1
                        if not matched:
                            j += 1
                    elif j >= n:
                        j = n-1
                        if not matched:
                            i += 1
                
                for leg in additionalLegs:
                    if self.check_timeout(timeout):
                        break
                    found = False
                    for emps in guiding_solution.employees.values():
                        if emps.id == id:
                            continue
                        for leg1 in emps.bus_legs:
                            if leg.id == leg1.id:
                                set_2 = [leg2 for leg2 in initial_solution.employees[emps.id].bus_legs if ((leg2.end >= leg.start and leg2.end <= leg.end) or (leg2.start <= leg.start and leg2.end >= leg.end) or (leg2.start >= leg.start and leg2.start <= leg.end))]    
                                if len(moves) == 0:
                                    moves = [(id, emps.id, leg)] + [(emps.id, id, leg2) for leg2 in set_2]
                                else:
                                    moves += [(id, emps.id, leg)] + [(emps.id, id, leg2) for leg2 in set_2]
                                #set_1 = [leg2 for leg2 in emp.bus_legs if leg2.start >= leg.start ]    
                                #set_2 = [leg2 for leg2 in initial_solution.employees[emps.id].bus_legs if leg2.start >= leg.start ]    
                                #moves = [(id, emps.id, leg2) for leg2 in set_1] + [(emps.id, id, leg2) for leg2 in set_2]
                    
                                found = True
                                break
                            if leg1.start >= leg.start:
                                break
                        if found:
                            break
            yield moves




class TabuSearch(Algorithm):
    
    def __init__(self, neighborhood:Neighborhood, tabu_tenure, no_improvement = -1) -> None:
        """Initialise the Tabu Search class.
            :param neighborhood: neighborhood object used for the tabu search  
            :param tabu_tenure: is the length of the tabu list.
            :param tabu_list: Is a matrix number_employee x number_legs. 
                The number tabu_list[i][j] says the iteration in which the leg j was inserted in employee i.
            :param max_iter: maximum number of iterations without improvement.
        """    
        super().__init__()
        self.neighborhood = neighborhood
        self.alpha = tabu_tenure
        self.tabu_tenure = tabu_tenure
        self.tabu_list = []
        self.stop = no_improvement

    def _do_apply(self, solution: Solution, timeout=None, algorithm = None) -> Solution:

        """ Apply the tabu_search method as long as the timeout is not reached.
            The number iter counts the iteration in which we are. This is used when tabu_list is updated.

            :param solution: the initial solution
            :param timeout: the maximum CPU time, set as the stopping criterion
        """
        no_improvement = 0
        success = True
        best_value = solution.value
        if algorithm is not None:
            global_store.clear()
            global_store.append(solution.instance+'_'+algorithm)
        global_store.append(best_value)
        best_solution = deepcopy(solution)
        number_of_bus_legs = len(solution.employees[1].instance.legs)
        number_of_employees = len(solution.employees)
        tt = math.ceil(self.alpha*math.sqrt(number_of_bus_legs)/10)
        initialSolution = deepcopy(solution)
        self.tabu_tenure = tt
        self.logger.debug(f'Calculated tabu tenure: {tt}')
        self.tabu_list = [[-self.tabu_tenure for _ in range(number_of_bus_legs)] for _ in range(number_of_employees)]    
        iter = 1   
        while (not self.check_timeout(timeout)) and (no_improvement != self.stop):
            #previous_sol = deepcopy(sol)
            success = self.neighborhood.tabu_search(solution, best_value, self.tabu_tenure, self.tabu_list, iter)
            if not success:
                break
            no_improvement += 1
            if solution.value < best_value:
                best_value = solution.value
                best_solution = deepcopy(solution)
                no_improvement = 0
            iter += 1
            dist = initialSolution.calculateDistance(solution)
            self.logger.debug(f'iteration {iter}, objective: {solution.value} distance from initial: {dist}')
            global_store.append(solution.value)
        if algorithm is not None:
            self.writeToCsv()
        dist = initialSolution.calculateDistance(best_solution)
        self.logger.debug(f'iteration {iter}, objective: {best_value} distance from initial: {dist}')
        return best_solution
    

class TabuSearchPR(Algorithm):
    
    def __init__(self, neighborhood:Neighborhood, tabu_tenure, no_improvement = -1) -> None:
        """Initialise the Tabu Search class.
            :param neighborhood: neighborhood object used for the tabu search  
            :param tabu_tenure: is the length of the tabu list.
            :param tabu_list: Is a matrix number_employee x number_legs. 
                The number tabu_list[i][j] says the iteration in which the leg j was inserted in employee i.
            :param max_iter: maximum number of iterations without improvement.
        """    
        super().__init__()
        self.neighborhood = neighborhood
        self.alpha = tabu_tenure
        self.tabu_tenure = tabu_tenure
        self.tabu_list = []
        self.stop = no_improvement

    def _do_apply(self, solution: Solution, timeout=None, algorithm = None) -> Solution:

        """ Apply the tabu_search method as long as the timeout is not reached.
            The number iter counts the iteration in which we are. This is used when tabu_list is updated.

            :param solution: the initial solution
            :param timeout: the maximum CPU time, set as the stopping criterion
        """
        no_improvement = 0
        success = True
        best_value = solution.value
        timeout = self.start + self.time_limit*0.9
        if algorithm is not None:
            global_store.clear()
            global_store.append(solution.instance+'_'+algorithm)
        global_store.append(best_value)
        best_solution = deepcopy(solution)
        number_of_bus_legs = len(solution.employees[1].instance.legs)
        number_of_employees = len(solution.employees)
        tt = math.ceil(self.alpha*math.sqrt(number_of_bus_legs)/10)
        startSolution = deepcopy(solution)
        self.tabu_tenure = tt
        self.logger.debug(f'Calculated tabu tenure: {tt}')
        self.tabu_list = [[-self.tabu_tenure for _ in range(number_of_bus_legs)] for _ in range(number_of_employees)]    
        iter = 1   
        while (not self.check_timeout(timeout)) and (no_improvement != self.stop):
            #previous_sol = deepcopy(sol)
            success = self.neighborhood.tabu_search(solution, best_value, self.tabu_tenure, self.tabu_list, iter)
            if not success:
                break
            no_improvement += 1
            if solution.value < best_value:
                best_value = solution.value
                best_solution = deepcopy(solution)
                no_improvement = 0
            iter += 1
            dist = startSolution.calculateDistance(solution)
            self.logger.debug(f'iteration {iter}, objective: {solution.value} distance from initial: {dist}')
            global_store.append(solution.value)
            if iter % 30 == 0 and no_improvement > 0:
                timeout1 = time.perf_counter() + self.time_limit*0.05
                check_sol = deepcopy(solution)
                bestPR = solution

                while not self.check_timeout(timeout1):
                    self.logger.debug(f'Begining path relinking with initial solution value {check_sol.value} and guiding solution value {best_value} and distance {best_solution.calculateDistance(check_sol)}')
                    check_sol = self.path_relinking(check_sol,best_solution,timeout)
                    if check_sol is None:
                        break
                    if bestPR is None or check_sol.value < bestPR.value:
                        bestPR = check_sol
                d = bestPR.calculateDistance(solution)
                self.logger.debug(f'Path relinking found solution with value {bestPR.value}, distance from {d}')    
                if bestPR.value < solution.value:
                    solution = bestPR
        if algorithm is not None:
            self.writeToCsv()
        return best_solution
    

class GradualSearch(Algorithm):
    """Variable neighborhood descent with a list of neighborhoods that are used with first improvement."""

    def __init__(self, neighborhoods: Neighborhood) -> None:
        """Initialize the algorithm with the list of neighborhoods to use."""
        super().__init__()
        self.local = neighborhoods

    def _do_apply(self, solution, timeout = None,algorithm = None):
        """ Apply the VND method.

        :param solution: the initial solution
        :param timeout: the timeout 
         """
        N = len(self.local)
        level = 0
        best = solution.value
        best_sol = deepcopy(solution)
        if algorithm is not None:
            global_store.clear()
            global_store.append(solution.instance+'_'+algorithm)
        global_store.append(best)
        initial_solution = deepcopy(solution)
        while level < N and not self.check_timeout(timeout):
            timeout1 = self.start + math.ceil(((level + 1)/N)*self.time_limit)
            solution = TabuSearch(neighborhood = self.local[level], tabu_tenure = 12).apply(solution, timeout1,nestedCall = True)
            global_store.append(solution.value)
            logging.debug(f'Solution at level {level + 1} with value {solution.value}, distance from initial {initial_solution.calculateDistance(solution)}')
            level += 1 
            if abs(solution.value) < abs(best):
                logging.debug(f'Found better solution at level {level} with value {solution.value}')
                best = solution.value
                best_sol = deepcopy(solution)
        logging.debug('No improving move found')               
        return best_sol

class VND(Algorithm):
    """Variable neighborhood descent with a list of neighborhoods that are used with first improvement."""

    def __init__(self, neighborhoods: Neighborhood) -> None:
        """Initialize the algorithm with the list of neighborhoods to use."""
        super().__init__()
        self.local = neighborhoods

    def _do_apply(self, solution, timeout = None,algorithm = None):
        """ Apply the VND method.

        :param solution: the initial solution
        :param timeout: the timeout 
         """
        level = 0
        best = solution.value
        best_sol = deepcopy(solution)
        if algorithm is not None:
            global_store.clear()
            global_store.append(solution.instance+'_'+algorithm)
        global_store.append(best)
        initial_solution = deepcopy(solution)
        while level < len(self.local) and not self.check_timeout(timeout):
            solution = TabuSearch(neighborhood = self.local[level], tabu_tenure = 12,no_improvement = 48).apply(solution, timeout,nestedCall = True)
            global_store.append(solution.value)
            logging.debug(f'Solution at level {level + 1} with value {solution.value}, distance from initial {initial_solution.calculateDistance(solution)}')
            level += 1 
            if abs(solution.value) < abs(best):
                logging.debug(f'Found better solution at level {level} with value {solution.value}')
                best = solution.value
                best_sol = deepcopy(solution)
                level = 0
            else:
                solution = deepcopy(best_sol)
        logging.debug('No improving move found')               
        return best_sol

class VNDPR(Algorithm):
    """Variable neighborhood descent with a list of neighborhoods that are used with first improvement."""

    def __init__(self, neighborhoods: Neighborhood) -> None:
        """Initialize the algorithm with the list of neighborhoods to use."""
        super().__init__()
        self.local = neighborhoods

    def _do_apply(self, solution, timeout = None,algorithm = None,):
        """ Apply the VND method.

        :param solution: the initial solution
        :param timeout: the timeout 
         """
        level = 0
        timeout = self.start + 0.9*self.time_limit
        best = solution.value
        best_sol = deepcopy(solution)
        guidingSol = None
        guidingVal = 0 
        if algorithm is not None:
            global_store.clear()
            global_store.append(solution.instance+'_'+algorithm)
        global_store.append(best)
        initial_solution = deepcopy(solution)
        while level < len(self.local) and not self.check_timeout(timeout):
            solution = TabuSearch(neighborhood = self.local[level], tabu_tenure = 12,no_improvement = 48).apply(solution, timeout,nestedCall = True)
            global_store.append(solution.value)
            logging.debug(f'Solution at level {level + 1} with value {solution.value}, distance from initial {initial_solution.calculateDistance(solution)}')
            level += 1 
            if abs(solution.value) < abs(best):
                logging.debug(f'Found better solution at level {level} with value {solution.value}')
                if guidingSol is None or guidingVal < best:
                    guidingSol = deepcopy(best_sol)
                    guidingVal = best
                best = solution.value
                best_sol = deepcopy(solution)
                level = 0
            else:
                if guidingSol is None or guidingVal < solution.value:
                    guidingSol = deepcopy(solution)
                    guidingVal = solution.value
                solution = deepcopy(best_sol)
        
        timeout = time.perf_counter() + 0.1*self.time_limit
        initial_solution = deepcopy(best_sol)
        while (not self.check_timeout(timeout)) and guidingSol is not None:
            self.logger.debug(f'Begining path relinking with initial solution value {initial_solution.value} and guiding solution value {guidingVal}')
            initial_solution = self.path_relinking(initial_solution,guidingSol,timeout)
            if initial_solution is None:
                break
            self.logger.debug(f'Path relinking found solution with value {initial_solution.value}')
            #initial_solution = TabuSearch(neighborhood=Shift(),tabu_tenure=12,no_improvement=48).apply(initial_solution,timeout=None)
            if best > initial_solution.value:
                best = initial_solution.value
                best_sol = deepcopy(initial_solution)
        logging.debug('No improving move found')               
        
        if best > guidingVal and guidingSol is not None:
            return guidingSol
        return best_sol

 
class GVNS(Algorithm):
    """General VNS using a list of neighborhoods for shaking and a given VND algorithm."""

    def __init__(self, neighborhoods, vnd, no_improvement = -1) -> None:
        """Initialize the algorithm with the list of neighborhoods for shaking and the VND.

        Runs until timeout or no_improvement many shaking cycles without finding an improving solution.
        :param neighborhoods: list of neighborhood
        """
        super().__init__()
        self.neighborhoods = neighborhoods
        self.vnd = vnd
        self.stop = no_improvement

    def _do_apply(self, solution, timeout=None,algorithm = None):
        level = 0
        best = solution.value
        best_sol = deepcopy(solution)
        no_improvement = 0
        if algorithm is not None:
            global_store.clear()
            global_store.append(solution.instance+'_'+algorithm)
        global_store.append(best)        
        while no_improvement != self.stop and not self.check_timeout(timeout):            
            solution = self.vnd.apply(solution, timeout,nestedCall = True)  
            level = (level + 1) % len(self.neighborhoods)
            no_improvement += 1
            if abs(solution.value) < abs(best):
                logging.info(f'Improving after {no_improvement} at level {level} with value {solution.value}')
                best = solution.value
                global_store.append(best)
                best_sol = deepcopy(solution)
                level = 0
                no_improvement = 0
            else:
                solution = deepcopy(best_sol)
            self.random_move(solution)
        solution.set(best_sol)
        if algorithm is not None:
            self.writeToCsv()
        return best_sol

    def random_move(self, solution):
        possible_employees = []
        for k, v in solution.employees.items():
            if (v.bus_legs is not None and len(v.bus_legs) > 0):
                possible_employees.append(k)
        n = math.floor(math.sqrt(len(possible_employees)))
        i = 0
        while i < n:
            try:
                e_1, e_2, e_3 = random.sample(possible_employees, k = 3)
                leg_1 = random.choice(solution.employees[e_1].bus_legs)
                leg_2 = random.choice(solution.employees[e_2].bus_legs)
                solution.update([[e_1, e_2, leg_1],[e_2, e_3, leg_2]])
                i += 1
            except Exception as e:
                logging.info(str(e))
        logging.info(f'Making {n} random moves')


class LocalSearch(Algorithm):
    """Local search with a single neighborhood and different search modes."""

    FIRST_IMPROVEMENT = 0  # apply first improvement method
    BEST_IMPROVEMENT = 1  # apply best improvement method
   
    def __init__(self, neighborhood, mode) -> None:
        super().__init__()
        self.neighborhood = neighborhood
        self.mode = mode

    def _do_apply(self, solution: Solution, timeout=None,algorithm = ''):
        success = True
        while not self.check_timeout(timeout):
            if self.mode == self.FIRST_IMPROVEMENT:
                success = self.neighborhood.first_improvement(solution)
            if self.mode == self.BEST_IMPROVEMENT:
                success = self.neighborhood.best_improvement(solution)
            if not success:
                break
        return solution


class IteratedLocalSearch(Algorithm):
    """Applies the Iterated Local Search algorithm:
        1 Perturbate the solution s* --> s'
        2 Perform a local search from s', finding s''
        3 s* = AcceptanceCriterion()
        4 Start again from 1. 
    """
    BETTER = 0 # Acceptance criterion: better.
    LSMC = 1 # Acceptance criterion: Large Step Markov Chain.
    RW = 2 # Acceptance criterion: Random Walk, always accept the new solution.
    KICK = -1
    RANDOM = -2
    CROSSOVER = -3
    TEMPERATURE = 2408.37
    RANDOM_MOVES = 2

    def __init__(self,  local_search, accept_criterion, perturbation, no_improvement = -1) -> None:
        super().__init__()
        self.local_search = local_search
        self.accept_crit = accept_criterion
        self.pert = perturbation
        self.stop = no_improvement

    def _do_apply(self, solution, timeout=None,algorithm = None) -> Solution:
        solution = self.local_search.apply(solution, self.stop)
        best = solution.copy()
        global_store.clear()
        global_store.append(solution.instance+'_'+algorithm)
        global_store.append(solution.value)
        no_improvement = 0
        while not self.check_timeout(timeout):
            global_store.append(solution.value)
            for i in range(self.RANDOM_MOVES):
                new_solution = self.perturb(solution)
                self.logger.debug(f'Applied perturbation {i}: solution with value {new_solution.value}')
            new_solution = self.local_search.apply(new_solution, self.stop)
            self.logger.debug(f'Local Search: solution with value {new_solution.value}')
            solution = self.accept(new_solution, solution)
            no_improvement += 1
            if solution.value < best.value: 
                self.logger.info(f'New best solution after {no_improvement} with value {solution.value}')
                best = solution.copy()
                no_improvement = 0
        self.writeToCsv()
        return best


    def perturb(self, solution: Solution) -> Solution:
        """Apply a perturbation on the solution
 
        Parameters
        ----------
        solution : Solution
            the solution to perturb

        Returns
        -------
        Solution
            the perturbed solution
        """

        if self.pert == self.KICK:
            possible_employees = [k for k, v in solution.employees.items() if len(v.bus_legs) > 0] 
            e_1, e_2, e_3 = random.choices(possible_employees, k = 3)
            leg_1 = random.choice(solution.employees[e_1].bus_legs)
            leg_2 = random.choice(solution.employees[e_2].bus_legs)
            move_1 = [e_1, e_2, leg_1]
            move_2 = [e_2, e_3, leg_2]
            solution.update([move_1, move_2])
            return solution
        elif self.pert == self.RANDOM:
            possible_employees = [k for k, v in solution.employees.items() if len(v.bus_legs) > 0] 
            e_1, e_2 = random.choices(possible_employees, k = 2)
            leg_1 = random.choice(solution.employees[e_1].bus_legs)
            move = [e_1, e_2, leg_1]
            solution.update([move])
            return solution
        elif self.pert == self.CROSSOVER:
            possible_employees = [k for k, v in solution.employees.items() if len(v.bus_legs) > 0] 
            e_1, e_2 = random.choices(possible_employees, k = 2)
            time = random.uniform(min(solution.employees[e_1].state.start_shift, solution.employees[e_2].state.start_shift), 
                                  max(solution.employees[e_1].state.end_shift, solution.employees[e_2].state.end_shift))  
            set_1 = [leg for leg in solution.employees[e_1].bus_legs if leg.start > time] 
            set_2 = [leg for leg in solution.employees[e_2].bus_legs if leg.start > time]    
            moves = [(e_1, e_2, leg) for leg in set_1] + [(e_2, e_1, leg) for leg in set_2]
            solution.update(moves)
            return solution


    def accept(self, solution: Solution, old_solution: Solution) -> Solution:
        """Apply the acceptance criterion to the solution

        Parameters
        ----------
        solution : Solution
            the new solution 
        old_solution : Solution
            the old solution

        Returns
        -------
        Solution
            this can be either the old solution or the new one, depending on the acceptance criterion result.
        """

        if self.accept_crit == self.BETTER:
            if solution.value < old_solution.value:
                return solution
            else:
                return old_solution
        elif self.accept_crit == self.LSMC:
            prob = math.exp((old_solution.value - solution.value)/self.TEMPERATURE)
            random_number = random.random()
            if prob < random_number:
                return solution
            else:
                return old_solution
        elif self.accept_crit == self.RW:
            return solution


class AdaptiveIteratedLocalSearch(Algorithm):
    """Applies the Iterated Local Search algorithm: """

    MIN_NUMBER = 1
    INITIAL_NUMBER = 2
    MAX_NUMBER = 50

    def __init__(self,  local_search, no_improvement=-1) -> None:
        super().__init__()
        self.local_search = local_search
        self.stop = no_improvement

    def _do_apply(self, solution, timeout=None,algorithm = ''):
        solution = self.local_search.apply(solution, self.stop)
        best = solution.copy()
        no_improvement = 0
        number_random_moves = self.INITIAL_NUMBER
        while not self.check_timeout(timeout):
            for i in range(number_random_moves):
                new_solution = self.perturb(solution, number_random_moves)
                self.logger.debug(f'Applied perturbation {i}: solution with value {new_solution.value}')
            self.logger.debug(f'Moving towards solution with value {new_solution.value}')
            new_solution = self.local_search.apply(new_solution, self.stop)
            no_improvement += 1
            if solution.value < best.value: 
                self.logger.info(f'New best solution after {no_improvement} with value {solution.value}')
                best = solution.copy()
                no_improvement = 0
                number_random_moves = max(number_random_moves-1, self.MIN_NUMBER)
                self.logger.debug(f'Number of random moves decreased: {number_random_moves}')
            else:
                no_improvement += 1
                number_random_moves = min(number_random_moves+1, self.MAX_NUMBER)
                self.logger.debug(f'Number of random moves increased: {number_random_moves}')
        return best


    def perturb(self, solution, number_random_moves) -> Solution:
        """ Perturb the solution.
        """
        
        for _ in range(number_random_moves):
            possible_employees = [k for k, v in solution.employees.items() if len(v.bus_legs) > 0] 
            e_1, e_2 = random.choices(possible_employees, k = 2)
            leg_1 = random.choice(solution.employees[e_1].bus_legs)
            move = [e_1, e_2, leg_1]
            solution.update([move])
        return solution
