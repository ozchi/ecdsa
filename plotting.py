import matplotlib.pyplot as plt
import numpy as np

def plot_metrics(metrics, title="PSO Optimization Progress"):
    """
    Plots the optimization progress across 4 different metrics.

    Parameters:
        metrics (tuple of lists): Contains the optimization metrics across iterations. 
                                  Each element of the tuple should be a list of lists, where each inner list 
                                  represents a run and contains the metric values across iterations.
        title (str): The title of the plot.
    """

    best_fitness, distance_to_optimum, swarm_std_dev, mean_velocity_length = metrics

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    fig.suptitle(title)

    axs[0, 0].set_xlabel('Iterations')
    axs[0, 0].set_ylabel('Best Fitness')

    axs[0, 1].set_xlabel('Iterations')
    axs[0, 1].set_ylabel('Distance to Global Optimum')

    axs[1, 0].set_xlabel('Iterations')
    axs[1, 0].set_ylabel('Swarm Standard Deviation')

    axs[1, 1].set_xlabel('Iterations')
    axs[1, 1].set_ylabel('Mean Velocity Length')

    num_runs = len(best_fitness)

    for iter in range(num_runs):
        print(f"Run {iter+1}:")
        print(f"Best fitness: {best_fitness[iter]}")
        # Subplot 1: Best Fitness Value
        axs[0, 0].semilogy(best_fitness[iter], label=f"Run {iter+1}")
        axs[0, 0].title.set_text("Best Fitness")
        axs[0, 0].legend()

        # Subplot 2: Distance to Global Optimum
        axs[0, 1].semilogy(distance_to_optimum[iter], label=f"Run {iter+1}")
        axs[0, 1].title.set_text("Distance to Global Optimum")
        axs[0, 1].legend()

        # Subplot 3: Standard Deviation of Particle Positions
        axs[1, 0].semilogy(swarm_std_dev[iter], label=f"Run {iter+1}")
        axs[1, 0].title.set_text("Swarm Standard Deviation")
        axs[1, 0].legend()

        # Subplot 4: Mean Length of Velocity Vectors
        axs[1, 1].semilogy(mean_velocity_length[iter], label=f"Run {iter+1}")
        axs[1, 1].title.set_text("Mean Velocity Length")
        axs[1, 1].legend()

    plt.show()
