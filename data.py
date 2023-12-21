from __future__ import annotations
import csv
from typing import List
from sortedcontainers import SortedList
from employee import Employee

class Instance:
    """This class represents the instance of the BDSP problem.
    """

    def __init__(self, legs, distance_matrix, start_work: float, end_work: float, instance = '') -> None:
        self.legs = legs
        self.distance_matrix = distance_matrix
        self.start_work = start_work
        self.end_work = end_work
        self.instance = instance 

    @staticmethod
    def from_file(file: str) -> Instance:
        """Read from file

        Parameters
        ----------
        file : str
            Input file in the form "realistic_m_n.csv"

        Returns
        -------
        Instance
            Instance returned.
        """
        with open(f'files/busdriver_instances/{file}.csv', 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',',
                                    quoting=csv.QUOTE_NONNUMERIC)
            bus_legs = SortedList()
            line_counter = 1
            next(csv_file)
            for row in csv_reader:
                bus_legs.add(BusLeg(line_counter, int(row[0]), int(row[1]),
                                       int(row[2]), int(row[3]), int(row[4])))
                line_counter += 1

        with open(f'files/busdriver_instances/{file}_dist.csv', 'r') as f:
            csv_reader = csv.reader(f, delimiter=',',
                                    quoting=csv.QUOTE_NONNUMERIC)
            distance_matrix = list(csv_reader)

        with open(f'files/busdriver_instances/{file}_extra.csv', 'r') as csv_file_extra:
            csv_reader = csv.reader(csv_file_extra, delimiter=',',
                                    quoting=csv.QUOTE_NONNUMERIC)
            start_work = next(csv_reader)
            start_work = [int(x) for x in start_work]
            end_work = next(csv_reader)
            end_work = [int(x) for x in end_work]
        return Instance(bus_legs, distance_matrix, start_work, end_work,file)


class BusLeg:
    """The bus leg class.
    """

    def __init__(self, id: int, tour: int, start: float, end: float, start_pos: int, end_pos: int) -> None:
        self.id = id
        self.tour = tour
        self.start = start
        self.end = end
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.start_shift = start
        self.end_shift = end
        self.name = id
        self.name = ''
        

    def __str__(self) -> str:
        return str(self.id)

    def __repr__(self) -> str:
        return str(self.id)

    def __hash__(self):
        return hash(self.id)


    def __getitem__(self, item):
        return self.item
    
    @property
    def drive(self) -> int:
        return self.end - self.start

    def __eq__(self, other):
        if isinstance(other, BusLeg):
            return self.id == other.id

    def __le__(self, other):
        if isinstance(other, BusLeg):
            return self.id == other.id
        return (self.start < other.start) or (self.start == other.start and self.id < other.id)

    def __lt__(self, other):
        if (self.id is None) or (other is None):
            return self.start < other.start
        return (self.start < other.start or (self.start == other.start and self.id < other.id))

    def __gt__(self, other):
        if (self.id is None) or (other is None):
            return self.start > other.start
        return (self.start > other.start or (self.start == other.start and self.id > other.id))


class Solution:
    """Solution class, represented by a dictionary of employees
    """

    def __init__(self, employees: List[Employee],instance = '') -> None:
        self.employees = {employee.id: employee for employee in employees}
        self.value = 0
        self.change = 0
        self.changing_employees = set()
        self.instance = instance

    def copy(self) -> Solution:
        """Copy the current solution

        Returns
        -------
        Solution
            Solution copied
        """
        employees_copy = [e.copy() for e in self.employees.values()]
        output = Solution(employees_copy)
        output.value = self.value
        output.instance = self.instance
        return output


    def set(self, solution: Solution) -> None:
        """Set the solution to the solution given as the argument

        Parameters
        ----------
        solution : Solution
            New solution.
        """
        self.employees = solution.employees    
        self.value = solution.value
        self.change = solution.change
    
    def evaluate(self) -> float:
        """Evaluate every employee of the solution

        Returns
        -------
        float
            The objective function of the solution
        """
        self.value = 0
        for _, employee in self.employees.items():
            self.value += employee.evaluate()
        return self.value

   
    def print_to_file(self, file: str) -> None:
        """  Print the solution into the given file.  
             
             The output format is a binary matrix n x l where:
                n is the number of employee
                l is the number of bus legs (ordered by start time)
                the element (i,j) is 1 if leg j is assigned to employee i, 0 otherwise.

        Parameters
        ----------
        file : str
            Desired output  file
            
        """
        l = len(self.employees[1].instance.legs)
        n = len(self.employees)
        data = [[0 for _ in range(l)] for _ in range(n)] 
        with open(f'solutions/{file}', 'w', newline='') as f:
            writer = csv.writer(f)
            for key, value in enumerate(self.employees.items()):
                for _, leg in enumerate(value[1].bus_legs):
                    j = self.employees[1].instance.legs.index(leg)
                    data[key][j] = 1
                writer.writerow(data[key])


    @staticmethod
    def from_file(instance: Instance, file: str) -> Solution:
        """Read a solution from file

        Parameters
        ----------
        instance : Instance
            Instance used to read the solution
        file : str
            name of the file, in the form "realistic_m_n_solution.csv"

        Returns
        -------
        Solution
            Solution readed.
        """
        employees = []
        counter = 0
        with open(file, 'r') as f:    
            f = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
            for row in f:
                row_legs =[index for index, value in enumerate(row) if value == 1] 
                if len(row_legs) == 0:
                    continue
                counter += 1
                employees.append(Employee(counter, instance))
                employees[counter-1].name = 'E' + str(counter)
                for leg in row_legs:
                    employees[counter -1].bus_legs.add(instance.legs[leg])
        employees = sorted(employees, key=lambda x: x.bus_legs[0].start) 
        solution = Solution(employees,instance.instance)
        return solution

    def update(self, moves: List) -> float:
        """Execute a list of moves.

            Each move contains a triple (e_1, e_2, leg), where
                - e_1 is the id of initial employee
                - e_2 is the id of final employee
                - leg is the id of a leg of the initial employee

        Parameters
        ----------
        moves : List
            List of triples (e_1, e_2, leg) that have to be performed

        Returns
        -------
        float
            The objective function value, after the execution of the move
        """
        self.changing_employees.clear()
        for move in moves:
            self.changing_employees.add(move[0])
            self.changing_employees.add(move[1])
            try:
                self.employees[move[0]].bus_legs.remove(move[2])
                self.employees[move[1]].bus_legs.add(move[2])
            except ValueError:
                pass
        self.change = 0
        for employee in self.changing_employees:
            self.change += - self.employees[employee].objective + self.employees[employee].evaluate() 
        self.value += self.change
        return self.value

    def revert(self, moves: List) -> float:
        """Return the solution to the previous state

        Parameters
        ----------
        moves : List
            List of triples previously performed

        Returns
        -------
        float
            old objective function value
        """
        for move in moves:
            self.employees[move[0]].bus_legs.add(move[2])
            self.employees[move[1]].bus_legs.remove(move[2])
        for employee in self.changing_employees:
            self.employees[employee].revert()
        self.value -= self.change
        return self.value


    def removeEmptyEmployees(self) -> Solution:
        """Remove all the employees tha have empty bus legs

        Returns
        -------
        Solution
            solution without empty employees.
        """
        employees = []
        for _, employee in self.employees.items():
            if employee.objective > 0:
                employees.append(employee)
        employees = sorted(employees, key=lambda x: x.bus_legs[0].start)  
        for i, employee in enumerate(employees):
            employee.id = i+1  
            employee.name = 'E' + str(employee.id)
        outputSolution = Solution(employees)
        return outputSolution

    def calculateDistance(self,neighbour:Solution):
        hammingDistance = 0
        workDistance = 0
        for id,emp in self.employees.items():
            emp1 = neighbour.employees[id]
            #workDistance += abs(emp.state.work_time - emp1.state.work_time)
            m = len(emp.bus_legs)
            n = len(emp1.bus_legs)
            if m == 0 or n == 0:
                continue
            i , j = 0 , 0
            while i < m and j < n:
                leg1 = emp.bus_legs[i]
                leg2 = emp1.bus_legs[j]
                if leg1.id == leg2.id:
                    i += 1
                    j += 1
                elif leg1.start < leg2.start:
                    hammingDistance += 1
                    i += 1
                elif leg1.start > leg2.start:
                    hammingDistance += 1
                    j += 1
                else:
                    i += 1
                    j += 1
                    hammingDistance += 2
                if i >= m:
                    j += 1
                    i = m -1
                elif j >= n:
                    i += 1
                    j = n-1
        return hammingDistance + workDistance