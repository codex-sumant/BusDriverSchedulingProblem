import logging
import math
from typing import List
from abc import ABC, abstractmethod
from data import Solution
import random

class Neighborhood(ABC):
    """ This class is an Abstract class.
    We introduce methods related to the neighborhood structure. The goal is to improve the initial solution """

    def __init__(self)-> None:
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def _get(self, solution: Solution):
        pass

    def first_improvement(self, solution: Solution) -> bool:
        """Apply the first found improvement in the neighborhood.

         Parameters
        ----------
        solution : Solution
            Initial solution

        Returns
        -------
        bool
            False if no improvement was found, otherwise True.
        """
        for move in self._get(solution):
            initial_value = solution.value
            solution.update(move)
            if solution.value < initial_value:
                self.logger.debug(f'New solution found: {solution.value}')
                return True
            solution.revert(move)
        return False


    def tabu_search(self, solution: Solution, best_value: float, tabu_tenure: int, tabu_list: list, iter: int) -> bool:
        """The initial solution, that has to be improved.

            We set tabu_list[j][leg] = iter in order to prohibite the move (i, j, leg) for every i 
            for the next (tabu_tenure - tabu_list[j][leg]) iterations

        Parameters
        ----------
        solution : Solution
            The initial solution, that has to be improved.
        best_value : float
            best objective function value value so far
        tabu_tenure : int
            The length of the tabu list.
        tabu_list : list
            Is a matrix n x l. The number tabu_list[i][j] says the iteration in which
                leg j was inserted in employee i.

        iter : int
            The number of iteration in which we are.

        Returns
        -------
        bool
            False if no improvement was found, otherwise True.
        """
        best_NT_val = 10**(20)
        best_T_val = 10**(20)
        best_move = None
        best_T_move = None
        best_NT_move = None
        # moves = list(self._get(solution))
        for move in self._get(solution):
            Tabu = False
            value = solution.update(move)
            for _, e_2, leg in move:
                # Check if enough iterations are passed for the move to not be considered tabu anymore.
                if iter - tabu_list[e_2-1][leg.id-1] < tabu_tenure:
                    Tabu = True
                    break
            if Tabu:
                if value < best_T_val:
                    best_T_val = value
                    best_T_move = move
            else:
                if value < best_NT_val:
                    best_NT_val = value
                    best_NT_move = move
            solution.revert(move)
        # Check if the tabu solution is the best one so far. In this case, accept it.
        if best_T_val < min(best_NT_val, best_value):
            logging.debug("Aspiration criteria")
            best_move = best_T_move
            # Update the tabu list in relation with the insertions in the move.
            for insertion in best_T_move:
                tabu_list[insertion[0]-1][insertion[2].id-1] = iter
        else:
            # Update the tabu list in relation with the insertions in the move.
            if best_NT_move is None:
                logging.debug('Every move is tabu!')
                return False
            for insertion in best_NT_move:
                tabu_list[insertion[0]-1][insertion[2].id-1] = iter
            best_move = best_NT_move
        if best_move is None:
            logging.debug("No improving move found")
            return False
        solution.update(best_move)
        return True


class Shift(Neighborhood):
    """ Basic, elementary insertion move.

        (i, j, leg) means that leg (of employee i) is inserted in employee j
    """

    def _get(self, solution: Solution) -> List[tuple]:
        for i, employee_1 in solution.employees.items():
            for leg in employee_1.bus_legs:
                for j, employee_2 in solution.employees.items():
                    if employee_1 == employee_2:
                        continue
                    yield [(i, j, leg)]


class Swap(Neighborhood):
    """ Swap move: sequence of two elementary insertions that swap  leg_1 and leg_2. 

        [(i, j, leg_1), (j, i, leg_2)] means that:
            -) leg_1 (of employee i) is inserted in employee j
            -) leg_2 (of employee j) is inserted in employee i
    """

    def _get(self, solution: Solution) -> List[tuple]:
        for i, e_1 in solution.employees.items():
            for leg_1 in e_1.bus_legs:
                for j, e_2 in solution.employees.items():
                    if e_1.id <= e_2.id:
                        continue
                    for leg_2 in e_2.bus_legs:
                        yield [(i, j, leg_1), (j, i, leg_2)]   

class Kick(Neighborhood):

    def _get(self, solution: Solution) -> List[tuple]:
        L = len(solution.employees[1].instance.legs)
        n = len(solution.employees)
        limit = math.ceil(L*(n-1)/2)
        counter = 0
        possible_employees = [k for k, v in solution.employees.items() if len(v.bus_legs) > 0] 
        while counter < limit:
            e_1, e_2, e_3 = random.sample(possible_employees, k = 3)
            leg_1 = random.choice(solution.employees[e_1].bus_legs)
            leg_2 = random.choice(solution.employees[e_2].bus_legs)
          
            yield [(e_1, e_2, leg_1), (e_2, e_3, leg_2)]
            counter += 1
            
            if counter % 2 == 0:
                yield [(e_2, e_1, leg_2)]
                counter += 1
            

class Rotate(Neighborhood):

    def _get(self, solution: Solution) -> List[tuple]:
        L = len(solution.employees[1].instance.legs)
        n = len(solution.employees)
        limit = math.ceil(L*(n-1)/2)
        counter = 0
        possible_employees = [k for k, v in solution.employees.items() if len(v.bus_legs) > 0] 
        while counter < limit:
            e_1, e_2, e_3 = random.sample(possible_employees, k = 3)
            leg_1 = random.choice(solution.employees[e_1].bus_legs)
            leg_2 = random.choice(solution.employees[e_2].bus_legs)
            leg_3 = random.choice(solution.employees[e_3].bus_legs)
                
            yield [(e_1, e_2, leg_1), (e_2, e_3, leg_2),(e_3,e_1,leg_3)]
            counter += 1

class Explore(Neighborhood):
    """ Combination move: essentially it combines a Swap and Shift move. 
    

    def _get(self, solution: Solution) -> List[tuple]:
        emps = list(solution.employees.items())
        n = len(emps)
        x = 0
        for i, e_1 in emps[0:n-2]:
            for leg_1 in e_1.bus_legs:
                for j, e_2 in emps[x+1:n-1]:
                    for leg_2 in e_2.bus_legs:
                        for k, e_3 in emps[x+2:n]:
                            yield [(i, j, leg_1), (j, k, leg_2)]
            x += 1
    """
    def _get(self, solution: Solution) -> List[tuple]:
        L = len(solution.employees[1].instance.legs)
        n = len(solution.employees)
        limit = math.ceil(L*(n-1)/2)
        counter = 0
        possible_employees = [k for k, v in solution.employees.items() if len(v.bus_legs) > 0] 
        while counter < limit:
            e_1, e_2, e_3 = random.sample(possible_employees, k = 3)
            leg_1 = random.choice(solution.employees[e_1].bus_legs)
            leg_2 = random.choice(solution.employees[e_2].bus_legs)
            
            if counter % 2 == 0:
                yield [(e_2, e_1, leg_2)]
                counter += 1
                
            yield [(e_1, e_2, leg_1), (e_2, e_3, leg_2)]
            counter += 1

                            
              
                             

class Combination(Neighborhood):
    """ Combination move: essentially it combines a Swap and Shift move. 
    """

    def _get(self, solution: Solution) -> List[tuple]:
        for i, e_1 in solution.employees.items():
            for leg_1 in e_1.bus_legs:
                for j, e_2 in solution.employees.items():
                    if e_1 == e_2:
                        continue
                    yield[(i, j, leg_1)]
                    if e_1.id <= e_2.id:
                        continue
                    for leg_2 in e_2.bus_legs:
                        yield [(i, j, leg_1), (j, i, leg_2)]   

