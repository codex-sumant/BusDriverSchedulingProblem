import time
import logging
import math
import sys
import csv

from data import Instance, Solution
from algorithm import TabuSearch, LocalSearch, IteratedLocalSearch, GVNS, VND, GradualSearch, TabuSearchPR, VNDPR

from neighborhood import Swap, Shift, Combination, Kick, Explore, Rotate


def main(algorithm: str, instance_file: str, timeout: float=None, solution_file: str=None) -> tuple:
    """  Execute the specified algorithm on the specified instance file.

    Parameters
    ----------
    algorithm : str
        specified algorithm
    instance_file : str
        instance file's name
    timeout : float, optional
        maximum runtime, by default None
    solution_file : str, optional
        specified solution file's name, by default None

    Returns
    -------
    tuple
        it contains data such as initial and final objective function values.
    """
    initial_solution_file = f'files/solutions/initial/{instance_file}_initial.csv'
    instance = Instance.from_file(instance_file)
    solution = Solution.from_file(instance, initial_solution_file) 
    solution.evaluate()
    initial_value = solution.value
    initial_employees = len(solution.employees)
    
    algorithms = {'TS_shift': TabuSearch(Shift(), tabu_tenure = 12)                  ,
                  'TS_swap': TabuSearch(Swap(),tabu_tenure = 12) ,
                  'TS_combination': TabuSearch(Combination(), tabu_tenure = 12) ,
                  'TS_kick': TabuSearch(Kick(), tabu_tenure = 12) ,
                  'TS_explore': TabuSearch(Explore(), tabu_tenure = 12) ,
                  'TS_rotate': TabuSearch(Rotate(), tabu_tenure = 12) ,
                  'TSPR_shift': TabuSearchPR(Shift(), tabu_tenure = 12) ,
                    # ITERATED LOCAL SEARCH WITH TABU                                                                                                   
                  'ILS_TS_better_crossover': IteratedLocalSearch(TabuSearch(Shift(),  tabu_tenure = 12),                                                       
                                                        IteratedLocalSearch.BETTER, 
                                                        IteratedLocalSearch.CROSSOVER,
                                                        no_improvement=48),
                  'ILS_TS_LSMC_crossover': IteratedLocalSearch(TabuSearch(Shift(),  tabu_tenure = 12),                                                            
                                                        IteratedLocalSearch.LSMC ,
                                                        IteratedLocalSearch.CROSSOVER),                                                    
                  'ILS_TS_better_kick': IteratedLocalSearch(TabuSearch(Shift(),  tabu_tenure = 12),                                                          
                                                        IteratedLocalSearch.BETTER, 
                                                        IteratedLocalSearch.KICK ,
                                                        no_improvement=48),
                  'ILS_TS_LSMC_kick': IteratedLocalSearch(TabuSearch(Shift(),  tabu_tenure = 12),                                                          
                                                        IteratedLocalSearch.LSMC, 
                                                        IteratedLocalSearch.KICK,
                                                        no_improvement=48),    
                  'ILS_TS_better_random': IteratedLocalSearch(TabuSearch(Shift(),  tabu_tenure = 12),                                                          
                                                        IteratedLocalSearch.BETTER, 
                                                        IteratedLocalSearch.RANDOM,
                                                        no_improvement=48),            
                  'ILS_TS_RW_random': IteratedLocalSearch(TabuSearch(Shift(),  tabu_tenure = 12),                                                          
                                                        IteratedLocalSearch.RW, 
                                                        IteratedLocalSearch.RANDOM,
                                                        no_improvement=48),
                    'ILS_TS_RW_crossover': IteratedLocalSearch(TabuSearch(Shift(),  tabu_tenure = 12),                                                          
                                                        IteratedLocalSearch.RW, 
                                                        IteratedLocalSearch.CROSSOVER,
                                                        no_improvement=48),                                          
                  'ILS_TS_LSMC_random': IteratedLocalSearch(TabuSearch(Shift(),  tabu_tenure = 12),                                                          
                                                        IteratedLocalSearch.LSMC, 
                                                        IteratedLocalSearch.RANDOM,
                                                        no_improvement=48),
                'ILS_TS_RW_kick': IteratedLocalSearch(TabuSearch(Shift(),  tabu_tenure = 12),                                                          
                                                        IteratedLocalSearch.RW, 
                                                        IteratedLocalSearch.KICK,
                                                        no_improvement=48),
                'VNDPR1' : VNDPR([Shift(),Kick(),Combination(),Rotate()]),
                'VNDPR2' : VNDPR([Kick(),Shift(),Combination(),Rotate()]),
                'VNDPR3' : VNDPR([Shift(),Kick()]),
                'VND1' : VND([Shift(),Kick(),Rotate(),Combination()]),
                'VND2' : VND([Rotate(),Combination(),Kick(),Shift()]),
                'VND3' : VND([Kick(),Shift(),Combination(),Rotate()]),
                'VND4' : VND([Shift(),Rotate(),Combination(),Kick()]),
                'VND2' : VND([Rotate(),Kick(),Shift()]),
                'VNS' : GVNS(neighborhoods = [Kick(),Shift()],vnd = VND([Kick(),Shift()]), no_improvement = 48),
                'VNS1' : GVNS(neighborhoods = [Kick(),Shift(),Kick(),Shift()],vnd = VND([Kick(),Shift(),Kick(),Shift()]), no_improvement = 48),
                'VNS2' : GVNS(neighborhoods = [Kick(),Shift(),Rotate(),Combination()],vnd = VND([Kick(),Shift(),Swap(),Combination()]), no_improvement = 48),
                'VNS3' : GVNS(neighborhoods = [Kick(),Shift(),Combination()],vnd = VND([Kick(),Shift(),Combination()]), no_improvement = 48),
                'VNS4' : GVNS(neighborhoods = [Rotate(),Shift(),Combination()],vnd = VND([Rotate(),Shift(),Combination()]), no_improvement = 48),
                'VNS5' : GVNS(neighborhoods = [Rotate(),Kick(),Shift(),Combination()],vnd = VND([Rotate(),Kick(),Shift(),Combination()]), no_improvement = 48),
                'VNS6' : GVNS(neighborhoods = [Shift(),Kick(),Combination(),Rotate()],vnd = VND([Shift(),Kick(),Combination(),Rotate()]), no_improvement = 48),
                'VNTS' : GVNS(neighborhoods = [Shift(),Kick(),Rotate(),Combination()],vnd = VND([Shift(),Kick(),Rotate(),Combination()]), no_improvement = 48),
                'VNS8' : GVNS(neighborhoods = [Shift(),Combination(),Rotate()],vnd = VND([Shift(),Combination(),Rotate()]), no_improvement = 48),
                'GS1' : GradualSearch(neighborhoods = [Rotate(),Kick(),Shift(),Combination()]),  
                'GS2' : GradualSearch(neighborhoods = [Kick(),Shift(),Combination()]),        
                'GS3' : GradualSearch(neighborhoods = [Rotate(),Kick(),Shift()]),  
                'GS4' : GradualSearch(neighborhoods = [Rotate(),Shift(),Combination()]),
                'GS5' : GradualSearch(neighborhoods = [Rotate(),Kick(),Combination(),Shift()])    
                }
    logger = logging.getLogger(__name__)              
    logger.info(f'Executing {algorithm} on {instance_file}')
    logger.info(f'Starting with initial solution: {solution.value} with {len(solution.employees)} employees')
    start = time.perf_counter()
    solution = algorithms[algorithm].apply(solution, timeout,None,algorithm = algorithm,instance=instance_file)
    duration = time.perf_counter() - start
    solution.removeEmptyEmployees()
    logger.info(f'Finished execution in {duration} with objective function {solution.value} and {len(solution.employees)} employees')
    
    if solution_file is not None:
        solution.print_to_file(solution_file)
    return (instance_file, initial_employees, initial_value, algorithm, duration, len(solution.employees), solution.value)

if __name__ == '__main__':
    f = '%(asctime)s|%(levelname)s|%(name)s|%(message)s'
    logging.basicConfig(level=logging.DEBUG, format=f)
    if len(sys.argv) == 4:
        main(algorithm = sys.argv[1], instance_file = sys.argv[2], timeout = int(sys.argv[3]))
    else:
        main(algorithm='TS_shift',instance_file='realistic_100_50', timeout=600)