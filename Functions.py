import numpy as np
import math
# from problem_base import RealValuedOptimizationProblem

# Base class for real-valued optimization problems
class RealValuedOptimizationProblem:
    '''
    topic 5.1: Write a class which represents real-valued optimisation problems of different numbers
    of dimensions.
    '''
    def __init__(self, dimensions):
        self.dimensions = dimensions
        
    def evaluate(self, solution):
        """
        Evaluate the solution. This should be overridden by derived classes.
        """
        raise NotImplementedError("This method should be implemented by derived classes.")
        
    def get_dimensions(self):
        return self.dimensions

# Derived class implementing the Ackley function
class AckleyFunction(RealValuedOptimizationProblem):
    '''
    topic 5.1: Write a derived class that implements the Ackley function with variable
    dimension and boundaries from Lecture 4, which is a benchmark problem for optimisers like PSO
    '''
    def __init__(self, dimensions):
        super().__init__(dimensions)
        self.a = 20
        self.b = 0.2
        self.c = 2 * math.pi
        
    def evaluate(self, solution):
        if len(solution) != self.dimensions:
            raise ValueError("The solution size does not match the problem dimensions.")
            
        sum1 = sum([x**2 for x in solution])
        sum2 = sum([math.cos(self.c * x) for x in solution])
        
        term1 = -self.a * math.exp(-self.b * math.sqrt(sum1 / self.dimensions))
        term2 = -math.exp(sum2 / self.dimensions)
        
        return term1 + term2 + self.a + math.exp(1)
    

# if __name__ == "__main__":
#     # For demonstration purposes, you can optimize the Ackley function using the PSOOptimizer
#     ackley_function = AckleyFunction(2)
#     pso_optimizer = PSOOptimizer(ackley_function.evaluate, dimensions=2)
#     best_position, best_value = pso_optimizer.optimize()
#     print(best_position, best_value)
