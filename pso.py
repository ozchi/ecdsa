import numpy as np

class TopologyStrategy:
    def get_best_position_value(self, particles, particle_index):
        raise NotImplementedError

class GBestStrategy(TopologyStrategy):
    def get_best_position_value(self, particles, particle_index):
        best_particle = min(particles, key=lambda p: p.pbest_value)
        return best_particle.pbest_position, best_particle.pbest_value
    
class LBestStrategy(TopologyStrategy):
    def get_best_position_value(self, particles, particle_index):
        # Determine neighbour indices in a ring topology
        left_neighbour_index = (particle_index - 1) % len(particles)
        right_neighbour_index = (particle_index + 1) % len(particles)

        # Candidates are the particle and its two neighbours
        candidates = [particles[particle_index], particles[left_neighbour_index], particles[right_neighbour_index]]

        # Find the best particle among the candidates
        best_particle = min(candidates, key=lambda p: p.pbest_value)

        return best_particle.pbest_position, best_particle.pbest_value
    
class StarStrategy(TopologyStrategy):
    def get_best_position_value(self, particles, particle_index):
        # The best particle in a star topology is simply the one with the global best value
        best_particle = min(particles, key=lambda p: p.pbest_value)
        
        return best_particle.pbest_position, best_particle.pbest_value

import random

class RandomNeighborStrategy(TopologyStrategy):
    def __init__(self, num_neighbors=3):
        self.num_neighbors = num_neighbors
    
    def get_best_position_value(self, particles, particle_index):
        # Choose random neighbors
        neighbors = random.sample(particles, self.num_neighbors)
        
        # Ensure the particle itself is in the candidate list
        if particles[particle_index] not in neighbors:
            neighbors.pop()
            neighbors.append(particles[particle_index])
        
        # Find the best particle among the neighbours
        best_particle = min(neighbors, key=lambda p: p.pbest_value)
        
        return best_particle.pbest_position, best_particle.pbest_value


class Particle:
    '''
    ramdomly initialize the position and velocity of the particle
    this is what topic 5.3 asks us to do
    '''
    # Initialize a particle with random position and velocity
    def __init__(self, dimensions, bounds):
        self.position = np.array([np.random.uniform(bounds[0], bounds[1]) for _ in range(dimensions)])
        self.velocity = np.array([np.random.uniform(-abs(bounds[1] - bounds[0]), abs(bounds[1] - bounds[0])) for _ in range(dimensions)])
        self.pbest_position = np.copy(self.position)
        self.pbest_value = float('inf')

class PSOOptimizer:
    '''
    A class which implements the PSO algorithm for real-valued optimisation problems
    '''
    def __init__(self, objective_function, dimensions, strategy,
                bounds=(-32.768, 32.768), num_particles=30, w=0.5, c1=1.5, c2=1.5, max_iterations=200):
        # Objective function to be optimized
        self.objective_function = objective_function
        # Number of dimensions
        self.dimensions = dimensions
        # Bounds for each dimension, any particle outside these bounds will be clipped
        self.bounds = bounds
        # Number of particles in the swarm
        self.num_particles = num_particles
        # Inertia weight
        self.w = w
        # acceleration coefficients
        self.c1 = c1
        self.c2 = c2
        # Maximum number of iterations
        self.max_iterations = max_iterations
        # random initialize the global best position with no prior knowledge about the global minimum
        # this is done for each dimension
        self.gbest_position = np.array([np.random.uniform(bounds[0], bounds[1]) for _ in range(dimensions)])
        # initialize the global best value of the objective function
        self.gbest_value = float('inf')
        # initialize the particles
        self.particles = [Particle(dimensions, bounds) for _ in range(num_particles)]
        #strategy to use for finding the best position in the swarm
        self.topology_strategy = strategy
        
    def evaluate(self, particle):
        return self.objective_function(particle.position)
    
    def update_pbest_gbest(self):
        for i, particle in enumerate(self.particles):
            fitness = self.evaluate(particle)

            # Update pbest
            if fitness < particle.pbest_value:
                particle.pbest_position = np.copy(particle.position)
                particle.pbest_value = fitness

            # Update gbest using topology strategy
            best_position, best_value = self.topology_strategy.get_best_position_value(self.particles, i)
            if best_value < self.gbest_value:
                self.gbest_position = np.copy(best_position)
                self.gbest_value = best_value
    
    def update_velocities_positions(self):
        '''
        function to update the velocities and positions of the particles based on the PSO algorithm
        this is where the PSO algorithm is implemented
        also where the randomization of the velocity and position comes into play
        '''
        for particle in self.particles:
            inertia = self.w * particle.velocity
            personal_attraction = self.c1 * np.random.random() * (particle.pbest_position - particle.position)
            social_attraction = self.c2 * np.random.random() * (self.gbest_position - particle.position)
            
            particle.velocity = inertia + personal_attraction + social_attraction
            particle.position += particle.velocity
            
            # Clip position values to be within bounds
            particle.position = np.clip(particle.position, self.bounds[0], self.bounds[1])
    
    def _compute_com_distance(self):
        """
        Compute the distance of the center of mass to the global optimum.
        """
        global_optimum = np.zeros(self.dimensions)  # Assuming global optimum is at [0, 0, ..., 0]
        center_of_mass = np.mean([particle.position for particle in self.particles], axis=0)
        return np.linalg.norm(center_of_mass - global_optimum)

    def _compute_position_stddev(self):
        """
        Compute the standard deviation of particle positions around the center of mass.
        """
        center_of_mass = np.mean([particle.position for particle in self.particles], axis=0)
        return np.mean([np.linalg.norm(particle.position - center_of_mass) for particle in self.particles])

    def _compute_mean_velocity(self):
        """
        Compute the mean length of the velocity vectors of particles.
        """
        return np.mean([np.linalg.norm(particle.velocity) for particle in self.particles])

    def optimize(self, best_fitness_values, com_distances, position_stddevs, mean_velocities):
        for _ in range(self.max_iterations):
            self.update_pbest_gbest()
            self.update_velocities_positions()

            # Compute metrics and store them
            best_fitness_values.append(self.gbest_value)
            com_distances.append(self._compute_com_distance())
            position_stddevs.append(self._compute_position_stddev())
            mean_velocities.append(self._compute_mean_velocity())

        return self.gbest_position, self.gbest_value, best_fitness_values, com_distances, position_stddevs, mean_velocities