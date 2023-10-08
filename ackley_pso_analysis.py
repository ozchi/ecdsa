import numpy as np
import matplotlib.pyplot as plt
from pso import PSOOptimizer, GBestStrategy
from Functions import AckleyFunction

dimensions = 10
bounds = (-30, 30)
iterations = 1000
num_runs = 10

ackley_function = AckleyFunction(dimensions)

results = []
best_fitness_values_list = []
com_distances_list = []
position_stddevs_list = []
mean_velocities_list = []

for _ in range(num_runs):
    pso_optimizer = PSOOptimizer(ackley_function.evaluate, dimensions, GBestStrategy(), bounds, max_iterations=iterations)
    
    best_fitness_values = []
    com_distances = []
    position_stddevs = []
    mean_velocities = []
    
    _, best_value, best_fitness_values, com_distances, position_stddevs, mean_velocities = pso_optimizer.optimize(best_fitness_values, com_distances, position_stddevs, mean_velocities)
    
    results.append(best_value)
    best_fitness_values_list.append(best_fitness_values)
    com_distances_list.append(com_distances)
    position_stddevs_list.append(position_stddevs)
    mean_velocities_list.append(mean_velocities)

# Plotting the results
plt.plot(results, marker='o')
plt.title('PSO Runs on Ackley Function')
plt.xlabel('Run')
plt.ylabel('Best Value')
plt.grid(True)
plt.savefig('results/metrics std pso.png')
plt.show()

# Saving the analysis
with open('results/analysis std pso.txt', 'w') as f:
    f.write("Analysis of PSO runs on Ackley Function:\n")
    f.write(f"Average best value: {np.mean(results)}\n")
    f.write(f"Standard deviation: {np.std(results)}\n")
    
print("completed")
