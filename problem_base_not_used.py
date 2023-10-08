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