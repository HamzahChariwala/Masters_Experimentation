import numpy as np
import time
import random
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
from dataclasses import dataclass
from tqdm import tqdm
import copy


@dataclass
class OptimizationResult:
    """Class to store optimization results."""
    best_params: np.ndarray
    best_value: float
    all_values: List[float]
    all_params: List[np.ndarray]
    runtime: float
    evaluations: int
    success: bool
    message: str


class Optimizer(ABC):
    """Base class for optimization algorithms."""
    
    def __init__(self, 
                 bounds: np.ndarray,
                 max_evals: int = 1000,
                 minimize: bool = True,
                 name: str = "Optimizer"):
        """
        Initialize the optimizer.
        
        Args:
            bounds: Bounds for each parameter, shape (n_params, 2)
            max_evals: Maximum number of function evaluations
            minimize: If True, minimize the objective function, otherwise maximize
            name: Name of the optimizer
        """
        self.bounds = np.array(bounds)
        self.n_params = len(bounds)
        self.lower_bounds = self.bounds[:, 0]
        self.upper_bounds = self.bounds[:, 1]
        self.max_evals = max_evals
        self.minimize = minimize
        self.name = name
        
        # Track optimization progress
        self.evaluations = 0
        self.best_params = None
        self.best_value = np.inf if minimize else -np.inf
        self.all_values = []
        self.all_params = []
        
    def _eval_objective(self, params: np.ndarray, objective_func: Callable) -> float:
        """
        Evaluate the objective function and update tracking variables.
        
        Args:
            params: Parameters to evaluate
            objective_func: Objective function
            
        Returns:
            Objective function value
        """
        value = objective_func(params)
        
        # Update tracking variables
        self.evaluations += 1
        
        # Store for visualization
        self.all_values.append(value)
        self.all_params.append(params.copy())
        
        # Update best value if better
        if (self.minimize and value < self.best_value) or (not self.minimize and value > self.best_value):
            self.best_value = value
            self.best_params = params.copy()
            
        return value
    
    def _clip_to_bounds(self, params: np.ndarray) -> np.ndarray:
        """Clip parameters to bounds."""
        return np.clip(params, self.lower_bounds, self.upper_bounds)
    
    @abstractmethod
    def _optimize(self, objective_func: Callable) -> None:
        """
        Perform optimization.
        
        Args:
            objective_func: Objective function to optimize
        """
        pass
    
    def optimize(self, objective_func: Callable) -> OptimizationResult:
        """
        Optimize the objective function.
        
        Args:
            objective_func: Objective function to optimize
            
        Returns:
            OptimizationResult: Results of optimization
        """
        # Reset optimization state
        self.evaluations = 0
        self.best_params = None
        self.best_value = np.inf if self.minimize else -np.inf
        self.all_values = []
        self.all_params = []
        
        # Start timer
        start_time = time.time()
        
        # Perform optimization
        try:
            self._optimize(objective_func)
            success = True
            message = f"Optimization completed with {self.evaluations} evaluations."
        except Exception as e:
            success = False
            message = f"Optimization failed: {str(e)}"
        
        # End timer
        runtime = time.time() - start_time
        
        # Return results
        return OptimizationResult(
            best_params=self.best_params,
            best_value=self.best_value,
            all_values=self.all_values,
            all_params=self.all_params,
            runtime=runtime,
            evaluations=self.evaluations,
            success=success,
            message=message
        )


class RandomSearch(Optimizer):
    """Random search optimizer."""
    
    def __init__(self, 
                 bounds: np.ndarray,
                 max_evals: int = 1000,
                 minimize: bool = True):
        """
        Initialize the random search optimizer.
        
        Args:
            bounds: Bounds for each parameter, shape (n_params, 2)
            max_evals: Maximum number of function evaluations
            minimize: If True, minimize the objective function, otherwise maximize
        """
        super().__init__(bounds, max_evals, minimize, name="Random Search")
    
    def _optimize(self, objective_func: Callable) -> None:
        """Perform random search optimization."""
        for _ in tqdm(range(self.max_evals), desc=self.name):
            # Generate random parameters
            params = self.lower_bounds + np.random.random(self.n_params) * (self.upper_bounds - self.lower_bounds)
            
            # Evaluate objective function
            self._eval_objective(params, objective_func)
            
            # Check if max evaluations reached
            if self.evaluations >= self.max_evals:
                break


class SimulatedAnnealing(Optimizer):
    """Simulated Annealing optimizer."""
    
    def __init__(self, 
                 bounds: np.ndarray,
                 max_evals: int = 1000,
                 minimize: bool = True,
                 init_temp: float = 100.0,
                 cooling_rate: float = 0.95,
                 step_size: float = 0.1):
        """
        Initialize the simulated annealing optimizer.
        
        Args:
            bounds: Bounds for each parameter, shape (n_params, 2)
            max_evals: Maximum number of function evaluations
            minimize: If True, minimize the objective function, otherwise maximize
            init_temp: Initial temperature
            cooling_rate: Cooling rate for temperature reduction
            step_size: Initial step size as a fraction of the parameter range
        """
        super().__init__(bounds, max_evals, minimize, name="Simulated Annealing")
        self.init_temp = init_temp
        self.cooling_rate = cooling_rate
        self.step_size = step_size
        
    def _optimize(self, objective_func: Callable) -> None:
        """Perform simulated annealing optimization."""
        # Initialize current solution
        current_params = self.lower_bounds + np.random.random(self.n_params) * (self.upper_bounds - self.lower_bounds)
        current_value = self._eval_objective(current_params, objective_func)
        
        # Initialize temperature
        temp = self.init_temp
        
        # Calculate step size in absolute units
        param_range = self.upper_bounds - self.lower_bounds
        step_sizes = param_range * self.step_size
        
        # Main loop
        while self.evaluations < self.max_evals:
            # Update progress bar every 10 iterations
            if self.evaluations % 10 == 0:
                tqdm.write(f"{self.name}: Evaluation {self.evaluations}/{self.max_evals}, Temp: {temp:.4f}, Best: {self.best_value:.6f}")
            
            # Generate new candidate solution
            candidate_params = current_params + np.random.uniform(-1, 1, self.n_params) * step_sizes
            candidate_params = self._clip_to_bounds(candidate_params)
            
            # Evaluate candidate
            candidate_value = self._eval_objective(candidate_params, objective_func)
            
            # Calculate acceptance probability
            delta = candidate_value - current_value
            if (self.minimize and delta < 0) or (not self.minimize and delta > 0):
                # Always accept better solutions
                accept_prob = 1.0
            else:
                # Accept worse solutions with a probability that decreases with temperature
                if self.minimize:
                    accept_prob = np.exp(-delta / temp)
                else:
                    accept_prob = np.exp(delta / temp)
            
            # Accept or reject candidate
            if np.random.random() < accept_prob:
                current_params = candidate_params
                current_value = candidate_value
            
            # Cool temperature
            temp *= self.cooling_rate
            
            # Stop if temperature is very low
            if temp < 1e-8:
                break


class GeneticAlgorithm(Optimizer):
    """Genetic Algorithm optimizer."""
    
    def __init__(self, 
                 bounds: np.ndarray,
                 max_evals: int = 1000,
                 minimize: bool = True,
                 pop_size: int = 50,
                 elite_size: int = 5,
                 crossover_prob: float = 0.7,
                 mutation_prob: float = 0.2,
                 mutation_scale: float = 0.1):
        """
        Initialize the genetic algorithm optimizer.
        
        Args:
            bounds: Bounds for each parameter, shape (n_params, 2)
            max_evals: Maximum number of function evaluations
            minimize: If True, minimize the objective function, otherwise maximize
            pop_size: Population size
            elite_size: Number of elite individuals to keep
            crossover_prob: Probability of crossover
            mutation_prob: Probability of mutation
            mutation_scale: Scale of mutation as a fraction of the parameter range
        """
        super().__init__(bounds, max_evals, minimize, name="Genetic Algorithm")
        self.pop_size = pop_size
        self.elite_size = min(elite_size, pop_size)
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.mutation_scale = mutation_scale
    
    def _optimize(self, objective_func: Callable) -> None:
        """Perform genetic algorithm optimization."""
        # Initialize population
        population = []
        param_range = self.upper_bounds - self.lower_bounds
        
        for _ in range(self.pop_size):
            # Generate random individual
            params = self.lower_bounds + np.random.random(self.n_params) * param_range
            value = self._eval_objective(params, objective_func)
            population.append((params, value))
        
        # Main loop
        generation = 0
        with tqdm(total=self.max_evals, desc=self.name) as pbar:
            pbar.update(self.pop_size)  # Update for initial population
            
            while self.evaluations < self.max_evals:
                generation += 1
                
                # Sort population by fitness
                population.sort(key=lambda x: x[1] if self.minimize else -x[1])
                
                # Create new population
                new_population = []
                
                # Add elite individuals
                new_population.extend(population[:self.elite_size])
                
                # Fill the rest of the population with offspring
                while len(new_population) < self.pop_size and self.evaluations < self.max_evals:
                    # Select parents using tournament selection
                    parent1 = self._tournament_selection(population)
                    parent2 = self._tournament_selection(population)
                    
                    # Crossover with probability
                    if np.random.random() < self.crossover_prob:
                        offspring = self._crossover(parent1[0], parent2[0])
                    else:
                        offspring = parent1[0].copy()
                    
                    # Mutation with probability
                    if np.random.random() < self.mutation_prob:
                        offspring = self._mutate(offspring, param_range)
                    
                    # Ensure within bounds
                    offspring = self._clip_to_bounds(offspring)
                    
                    # Evaluate offspring
                    value = self._eval_objective(offspring, objective_func)
                    new_population.append((offspring, value))
                    
                    # Update progress bar
                    pbar.update(1)
                
                # Update population
                population = new_population
                
                # Update progress message
                pbar.set_description(f"{self.name}: Gen {generation}, Best: {self.best_value:.6f}")
    
    def _tournament_selection(self, population: List[Tuple[np.ndarray, float]], tournament_size: int = 3) -> Tuple[np.ndarray, float]:
        """
        Select an individual using tournament selection.
        
        Args:
            population: List of (params, value) tuples
            tournament_size: Number of individuals in each tournament
            
        Returns:
            Selected individual as (params, value)
        """
        # Select random individuals for tournament
        tournament = random.sample(population, min(tournament_size, len(population)))
        
        # Return the best individual in the tournament
        if self.minimize:
            return min(tournament, key=lambda x: x[1])
        else:
            return max(tournament, key=lambda x: x[1])
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """
        Perform crossover between two parents.
        
        Args:
            parent1: First parent parameters
            parent2: Second parent parameters
            
        Returns:
            Offspring parameters
        """
        # Uniform crossover
        mask = np.random.random(self.n_params) < 0.5
        offspring = np.where(mask, parent1, parent2)
        return offspring
    
    def _mutate(self, params: np.ndarray, param_range: np.ndarray) -> np.ndarray:
        """
        Mutate parameters.
        
        Args:
            params: Parameters to mutate
            param_range: Range of each parameter
            
        Returns:
            Mutated parameters
        """
        # Apply Gaussian mutation
        mutation = np.random.normal(0, 1, self.n_params) * param_range * self.mutation_scale
        
        # Apply mutation only to some parameters
        mask = np.random.random(self.n_params) < 0.3  # Mutate ~30% of parameters
        mutation = mutation * mask
        
        return params + mutation


class ParticleSwarmOptimization(Optimizer):
    """Particle Swarm Optimization (PSO) algorithm."""
    
    def __init__(self, 
                 bounds: np.ndarray,
                 max_evals: int = 1000,
                 minimize: bool = True,
                 pop_size: int = 30,
                 inertia_weight: float = 0.7,
                 cognitive_coef: float = 1.5,
                 social_coef: float = 1.5):
        """
        Initialize the PSO optimizer.
        
        Args:
            bounds: Bounds for each parameter, shape (n_params, 2)
            max_evals: Maximum number of function evaluations
            minimize: If True, minimize the objective function, otherwise maximize
            pop_size: Swarm size (number of particles)
            inertia_weight: Inertia weight for velocity update
            cognitive_coef: Cognitive coefficient (personal best influence)
            social_coef: Social coefficient (global best influence)
        """
        super().__init__(bounds, max_evals, minimize, name="Particle Swarm Optimization")
        self.pop_size = pop_size
        self.inertia_weight = inertia_weight
        self.cognitive_coef = cognitive_coef
        self.social_coef = social_coef
    
    def _optimize(self, objective_func: Callable) -> None:
        """Perform particle swarm optimization."""
        # Initialize swarm particles, velocities, and personal bests
        param_range = self.upper_bounds - self.lower_bounds
        
        # Particle positions and velocities
        positions = np.zeros((self.pop_size, self.n_params))
        velocities = np.zeros((self.pop_size, self.n_params))
        
        # Initialize personal best positions and values
        pbest_pos = np.zeros((self.pop_size, self.n_params))
        pbest_val = np.inf * np.ones(self.pop_size) if self.minimize else -np.inf * np.ones(self.pop_size)
        
        # Initialize global best position and value
        gbest_pos = np.zeros(self.n_params)
        gbest_val = np.inf if self.minimize else -np.inf
        
        # Initialize particles with random positions and velocities
        for i in range(self.pop_size):
            # Random position within bounds
            positions[i] = self.lower_bounds + np.random.random(self.n_params) * param_range
            
            # Random initial velocity (between -range and +range)
            velocities[i] = np.random.uniform(-1, 1, self.n_params) * param_range * 0.1
            
            # Evaluate particle
            value = self._eval_objective(positions[i], objective_func)
            
            # Update personal best
            pbest_pos[i] = positions[i].copy()
            pbest_val[i] = value
            
            # Update global best
            if (self.minimize and value < gbest_val) or (not self.minimize and value > gbest_val):
                gbest_pos = positions[i].copy()
                gbest_val = value
        
        # Main loop
        iteration = 0
        with tqdm(total=self.max_evals, desc=self.name) as pbar:
            pbar.update(self.pop_size)  # Update for initial evaluations
            
            while self.evaluations < self.max_evals:
                iteration += 1
                
                for i in range(self.pop_size):
                    if self.evaluations >= self.max_evals:
                        break
                    
                    # Update velocity
                    cognitive = self.cognitive_coef * np.random.random(self.n_params) * (pbest_pos[i] - positions[i])
                    social = self.social_coef * np.random.random(self.n_params) * (gbest_pos - positions[i])
                    velocities[i] = self.inertia_weight * velocities[i] + cognitive + social
                    
                    # Limit velocity to prevent explosion
                    velocities[i] = np.clip(velocities[i], -param_range, param_range)
                    
                    # Update position
                    positions[i] += velocities[i]
                    positions[i] = self._clip_to_bounds(positions[i])
                    
                    # Evaluate particle
                    value = self._eval_objective(positions[i], objective_func)
                    
                    # Update personal best
                    if (self.minimize and value < pbest_val[i]) or (not self.minimize and value > pbest_val[i]):
                        pbest_pos[i] = positions[i].copy()
                        pbest_val[i] = value
                        
                        # Update global best
                        if (self.minimize and value < gbest_val) or (not self.minimize and value > gbest_val):
                            gbest_pos = positions[i].copy()
                            gbest_val = value
                    
                    # Update progress bar
                    pbar.update(1)
                
                # Update progress message
                pbar.set_description(f"{self.name}: Iter {iteration}, Best: {self.best_value:.6f}")


class DifferentialEvolution(Optimizer):
    """Differential Evolution optimizer."""
    
    def __init__(self, 
                 bounds: np.ndarray,
                 max_evals: int = 1000,
                 minimize: bool = True,
                 pop_size: int = 50,
                 f: float = 0.8,
                 cr: float = 0.7,
                 strategy: str = "rand/1/bin"):
        """
        Initialize the differential evolution optimizer.
        
        Args:
            bounds: Bounds for each parameter, shape (n_params, 2)
            max_evals: Maximum number of function evaluations
            minimize: If True, minimize the objective function, otherwise maximize
            pop_size: Population size
            f: Mutation scale factor
            cr: Crossover probability
            strategy: DE strategy ('rand/1/bin', 'best/1/bin', 'rand/2/bin', 'best/2/bin')
        """
        super().__init__(bounds, max_evals, minimize, name="Differential Evolution")
        self.pop_size = max(pop_size, 4)  # Need at least 4 individuals
        self.f = f
        self.cr = cr
        self.strategy = strategy
        
        # Validate strategy
        valid_strategies = ["rand/1/bin", "best/1/bin", "rand/2/bin", "best/2/bin"]
        if strategy not in valid_strategies:
            raise ValueError(f"Strategy must be one of {valid_strategies}")
    
    def _optimize(self, objective_func: Callable) -> None:
        """Perform differential evolution optimization."""
        # Initialize population
        population = np.zeros((self.pop_size, self.n_params))
        values = np.zeros(self.pop_size)
        
        # Initialize randomly within bounds
        for i in range(self.pop_size):
            population[i] = self.lower_bounds + np.random.random(self.n_params) * (self.upper_bounds - self.lower_bounds)
            values[i] = self._eval_objective(population[i], objective_func)
        
        # Main loop
        generation = 0
        with tqdm(total=self.max_evals, desc=self.name) as pbar:
            pbar.update(self.pop_size)  # Update for initial evaluations
            
            while self.evaluations < self.max_evals:
                generation += 1
                
                # Process each individual in the population
                for i in range(self.pop_size):
                    if self.evaluations >= self.max_evals:
                        break
                    
                    # Create trial vector using the selected strategy
                    trial = self._create_trial(population, values, i)
                    
                    # Ensure trial is within bounds
                    trial = self._clip_to_bounds(trial)
                    
                    # Evaluate trial vector
                    trial_value = self._eval_objective(trial, objective_func)
                    
                    # Selection: keep the better solution
                    if (self.minimize and trial_value <= values[i]) or (not self.minimize and trial_value >= values[i]):
                        population[i] = trial
                        values[i] = trial_value
                    
                    # Update progress bar
                    pbar.update(1)
                
                # Update progress message
                pbar.set_description(f"{self.name}: Gen {generation}, Best: {self.best_value:.6f}")
    
    def _create_trial(self, population: np.ndarray, values: np.ndarray, idx: int) -> np.ndarray:
        """
        Create a trial vector using the specified strategy.
        
        Args:
            population: Current population
            values: Current population values
            idx: Index of current individual
            
        Returns:
            Trial vector
        """
        # Parse strategy
        parts = self.strategy.split("/")
        base_vector_type = parts[0]  # 'rand' or 'best'
        n_difference_vectors = int(parts[1])  # '1' or '2'
        crossover_type = parts[2]  # 'bin'
        
        # Select base vector
        if base_vector_type == "rand":
            # Random individual (excluding current)
            candidates = list(range(self.pop_size))
            candidates.remove(idx)
            base_idx = np.random.choice(candidates)
            base = population[base_idx]
        else:  # "best"
            # Best individual
            if self.minimize:
                best_idx = np.argmin(values)
            else:
                best_idx = np.argmax(values)
            base = population[best_idx]
        
        # Select random indices for difference vectors
        candidates = list(range(self.pop_size))
        candidates.remove(idx)
        if base_vector_type == "rand":
            candidates.remove(base_idx)
        
        # Need 2 or 4 distinct individuals
        if n_difference_vectors == 1:
            r1, r2 = np.random.choice(candidates, 2, replace=False)
            diff_term = self.f * (population[r1] - population[r2])
        else:  # n_difference_vectors == 2
            r1, r2, r3, r4 = np.random.choice(candidates, 4, replace=False)
            diff_term = self.f * (population[r1] - population[r2] + population[r3] - population[r4])
        
        # Create mutant vector
        mutant = base + diff_term
        
        # Perform crossover
        if crossover_type == "bin":
            # Binomial crossover
            mask = np.random.random(self.n_params) <= self.cr
            
            # Ensure at least one dimension is taken from the mutant
            if not np.any(mask):
                j_rand = np.random.randint(0, self.n_params)
                mask[j_rand] = True
            
            # Combine current individual and mutant
            trial = np.where(mask, mutant, population[idx])
        else:
            raise ValueError(f"Unsupported crossover type: {crossover_type}")
        
        return trial


def run_optimization(objective_func: Callable,
                     bounds: np.ndarray,
                     method: str = "all",
                     max_evals: int = 1000,
                     minimize: bool = True,
                     verbose: bool = True,
                     plot_results: bool = True,
                     save_plot: Optional[str] = None,
                     **kwargs) -> Dict[str, OptimizationResult]:
    """
    Run optimization with the specified method(s).
    
    Args:
        objective_func: Objective function to optimize
        bounds: Bounds for each parameter, shape (n_params, 2)
        method: Optimization method (pso, sa, ga, de, random, all)
        max_evals: Maximum number of function evaluations per method
        minimize: If True, minimize the objective function, otherwise maximize
        verbose: If True, print detailed results
        plot_results: If True, plot convergence curves
        save_plot: If provided, save the plot to this filename
        **kwargs: Additional parameters for optimizers
        
    Returns:
        Dictionary of optimization results for each method
    """
    # Available optimization methods
    optimizers = {
        "random": lambda: RandomSearch(bounds, max_evals, minimize),
        "sa": lambda: SimulatedAnnealing(
            bounds, max_evals, minimize,
            init_temp=kwargs.get("init_temp", 100.0),
            cooling_rate=kwargs.get("cooling_rate", 0.95),
            step_size=kwargs.get("step_size", 0.1)
        ),
        "ga": lambda: GeneticAlgorithm(
            bounds, max_evals, minimize,
            pop_size=kwargs.get("pop_size", 50),
            elite_size=kwargs.get("elite_size", 5),
            crossover_prob=kwargs.get("crossover_prob", 0.7),
            mutation_prob=kwargs.get("mutation_prob", 0.2),
            mutation_scale=kwargs.get("mutation_scale", 0.1)
        ),
        "pso": lambda: ParticleSwarmOptimization(
            bounds, max_evals, minimize,
            pop_size=kwargs.get("pop_size", 30),
            inertia_weight=kwargs.get("inertia_weight", 0.7),
            cognitive_coef=kwargs.get("cognitive_coef", 1.5),
            social_coef=kwargs.get("social_coef", 1.5)
        ),
        "de": lambda: DifferentialEvolution(
            bounds, max_evals, minimize,
            pop_size=kwargs.get("pop_size", 50),
            f=kwargs.get("f", 0.8),
            cr=kwargs.get("cr", 0.7),
            strategy=kwargs.get("strategy", "rand/1/bin")
        )
    }
    
    # Select methods to run
    if method.lower() == "all":
        methods = list(optimizers.keys())
    else:
        if method.lower() not in optimizers:
            raise ValueError(f"Method {method} not recognized. Available methods: {list(optimizers.keys())}")
        methods = [method.lower()]
    
    # Run selected methods
    results = {}
    for method_name in methods:
        print(f"\nRunning {method_name.upper()}...")
        optimizer = optimizers[method_name]()
        result = optimizer.optimize(objective_func)
        results[method_name] = result
        
        if verbose:
            print(f"Best value: {result.best_value:.6f}")
            print(f"Best parameters: {result.best_params}")
            print(f"Number of evaluations: {result.evaluations}")
            print(f"Runtime: {result.runtime:.2f} seconds")
            print(f"Status: {'Success' if result.success else 'Failed'}")
            print(f"Message: {result.message}")
    
    # Plot convergence curves
    if plot_results and results:
        plt.figure(figsize=(12, 7))
        for method_name, result in results.items():
            # Get x and y values for plotting
            y_values = result.all_values
            x_values = np.arange(1, len(y_values) + 1)
            
            # Plot convergence curve
            plt.plot(x_values, y_values, label=f"{method_name.upper()}")
        
        # Add labels and title
        plt.xlabel("Function Evaluations")
        plt.ylabel("Objective Value")
        plt.title("Optimization Convergence" if minimize else "Optimization Progress")
        plt.legend()
        plt.grid(True)
        
        # Use log scale if values span multiple orders of magnitude
        if any(result.all_values for result in results.values()):
            first_values = results[list(results.keys())[0]].all_values
            if max(first_values) / (min(first_values) + 1e-10) > 100:
                plt.yscale("log")
        
        # Save plot if requested
        if save_plot:
            plt.savefig(save_plot, dpi=300, bbox_inches="tight")
        
        plt.show()
    
    return results


def example_usage():
    """Example usage of optimization methods."""
    # Example: Minimize the Rosenbrock function
    def rosenbrock(x):
        return sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    
    # Example: Minimize the Ackley function
    def ackley(x):
        a = 20
        b = 0.2
        c = 2 * np.pi
        n = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(c * x))
        term1 = -a * np.exp(-b * np.sqrt(sum1 / n))
        term2 = -np.exp(sum2 / n)
        return term1 + term2 + a + np.exp(1)
    
    # Example optimization
    print("Optimizing Rosenbrock function (2D)...")
    bounds = np.array([[-5, 5], [-5, 5]])  # 2D bounds
    results = run_optimization(
        rosenbrock, bounds, method="all", max_evals=2000, minimize=True,
        verbose=True, plot_results=True, save_plot="rosenbrock_optimization.png"
    )
    
    print("\nOptimizing Ackley function (5D)...")
    bounds = np.array([[-5, 5]] * 5)  # 5D bounds
    results = run_optimization(
        ackley, bounds, method="all", max_evals=2000, minimize=True,
        verbose=True, plot_results=True, save_plot="ackley_optimization.png"
    )


if __name__ == "__main__":
    example_usage() 