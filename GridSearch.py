from pso import PSOOptimizer, GBestStrategy, LBestStrategy, StarStrategy, RandomNeighborStrategy
from Functions import AckleyFunction
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from main_with_plot import plot_metrics

'''
Hyperparameters settings:
'''
num_runs = 1
ackley_function = AckleyFunction(2)
w_values = [0.2]
c1_values = [2.5]
c2_values = [2.5]
num_particles_values = [30] 
results = []

'''
save the hyperparameters to a file, so that we can keep track of them
'''

with open("run_number.txt", 'r') as f:
    run_number = int(f.read())

if not os.path.exists(f"hyperparameter_settings/GridSearch_{run_number}th_run_hyperparameters.txt"):
    # Creating a file to store hyperparameters
    with open(f"hyperparameter_settings/GridSearch_{run_number}th_run_hyperparameters.txt", 'w') as f:
        f.write(f"hyperparameters for {run_number}th run\n")
else:
    print(f"hyperparameter_settings/GridSearch_{run_number}th_run_hyperparameters.txt already exists. THIS SHOULDNT HAPPEN.")
    print(f"Please delete this file: hyperparameter_settings/GridSearch_{run_number}th_run_hyperparameters.txt and run the program again.")
    print("Exiting...")
    sys.exit(1)


with open(f"hyperparameter_settings/GridSearch_{run_number}th_run_hyperparameters.txt", 'w') as f:
    f.write(f"w_values = {w_values}\n")
    f.write(f"c1_values = {c1_values}\n")
    f.write(f"c2_values = {c2_values}\n")
    f.write(f"num_particles_values = {num_particles_values}\n")
    f.write(f"num_runs = {num_runs}\n")
    f.write("\n")

'''
Running PSO for each hyperparameter possible combination
'''

for w in w_values:
    for c1 in c1_values:
        for c2 in c2_values:
            if c1 == c2 == 3.0:
                continue
            for num_particles in num_particles_values:
                run_results = []
                for run in range(num_runs):
                    pso_optimizer = PSOOptimizer(ackley_function.evaluate, dimensions=2, w=w, c1=c1, c2=c2, num_particles=num_particles, strategy=RandomNeighborStrategy())
                    _, best_value, metrics = pso_optimizer.optimize()
                    run_results.append(best_value)
                    print("num_run = {}\tw = {}\tc1 = {}\tc2 = {}\tparticles = {}\tbest_value = {}".format(run, w, c1, c2, num_particles, best_value))
                
                results.append({
                    'w': w,
                    'c1': c1,
                    'c2': c2,
                    'num_particles': num_particles,
                    'results': run_results,
                    'mean': np.mean(run_results),
                    'std': np.std(run_results)
                })


plot_metrics(metrics, title="PSO Optimization Progress")

'''
Saving results to a file
'''
 
# Extracting means and standard deviations for plotting
means = [res['mean'] for res in results]
stds = [res['std'] for res in results]
labels = [f"w={res['w']} c1={res['c1']} c2={res['c2']} particles={res['num_particles']}" for res in results]

#extract the best hyperparameters and the best value
best_result = results[np.argmin(means)]
best_hyperparameters = f"w={best_result['w']} c1={best_result['c1']} c2={best_result['c2']} particles={best_result['num_particles']}"
best_value = np.min(means)

# Saving results to a file, in a nice format
with open(f'results/results_run_{run_number}.txt', 'w') as f:
    f.write("############# BEST RESULT ###############\n")
    f.write(f"best hyperparameters: {best_hyperparameters}\n")
    f.write(f"best value: {best_value}\n")
    f.write("############# ALL VALUES ###############\n")
    for res in results:
        f.write(f"w={res['w']} c1={res['c1']} c2={res['c2']} particles={res['num_particles']}\n")
        f.write(f"mean={res['mean']}\n")
        f.write("\n")

'''
plotting the results and save the plot as a png file
'''

# Plotting
plt.figure(figsize=(20,12), dpi=200)  # Adjust width as needed

# Plot the bars
bars = plt.bar(range(len(means)), means, yerr=stds, align='center', alpha=0.7, ecolor='black', capsize=10)

# Adjusting x-tick labels using bar positions
positions = [bar.get_x() + bar.get_width() / 2 for bar in bars]
xticks, xticklabels = plt.xticks(positions, labels, rotation=45)

# Adjust the horizontal alignment for each xtick label
for xticklabel in xticklabels:
    xticklabel.set_horizontalalignment('right')

plt.ylabel('Best Value', fontsize=20)
plt.title('PSO Performance over Multiple Runs', fontsize=20)
plt.tight_layout()

# Save the plot as a png file
plt.savefig(f'graphs/PSO_run_{run_number}.png')

# Updating the run number
with open("run_number.txt", 'w') as f:
    f.write(str(run_number + 1))
