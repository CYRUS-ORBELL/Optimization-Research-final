import numpy as np
import ioh
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
import pyswarms as ps
from cmaes import CMA
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

# ── Settings ──────────────────────────────────────────────

N_PARTICLES = 20
N_ITERATIONS = 100
K_CLUSTERS = 5
BBOB_FUNCTIONS = range(1,25)    
N_RUNS = 30                              
DIMENSION = 30
BOUNDS_LOW, BOUNDS_HIGH = -5.0, 5.0
OUTPUT_DIR = f"dataset_{DIMENSION}_unimodal"
# ──────────────────────────────────────────────────────────

def get_fingerprint(trajectory):
    """Convert trajectory (iters, particles, dims) to fingerprint matrix (iters, K)."""
    n_iterations = trajectory.shape[0]
    fingerprint = np.zeros((n_iterations, K_CLUSTERS))
    for t in range(n_iterations):
        particles = trajectory[t]
        kmeans = KMeans(n_clusters=K_CLUSTERS, random_state=0, n_init='auto')
        labels = kmeans.fit_predict(particles)
        for k in range(K_CLUSTERS):
            fingerprint[t, k] = np.sum(labels == k) / len(labels)
    return fingerprint

def save_image(fingerprint, path):
    """Save fingerprint matrix as grayscale PNG."""
    plt.figure(figsize=(2, 2))
    plt.imshow(fingerprint.T, aspect='auto', cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=64, bbox_inches='tight', pad_inches=0)
    plt.close()

# ── Algorithm runners ──────────────────────────────────────

def run_pso(problem):
    trajectory = []
    def objective(particles):
        trajectory.append(particles.copy())
        return np.array([problem(p) for p in particles])
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    optimizer = ps.single.GlobalBestPSO(
        n_particles=N_PARTICLES,
        dimensions=DIMENSION,
        options=options,
        bounds=(np.full(DIMENSION, BOUNDS_LOW), np.full(DIMENSION, BOUNDS_HIGH))
    )
    optimizer.optimize(objective, iters=N_ITERATIONS, verbose=False)
    return np.array(trajectory)

def run_random_search(problem):
    trajectory = []
    for _ in range(N_ITERATIONS):
        particles = np.random.uniform(BOUNDS_LOW, BOUNDS_HIGH, (N_PARTICLES, DIMENSION))
        trajectory.append(particles.copy())
        for p in particles:
            problem(p)
    return np.array(trajectory)

def run_de(problem):
    trajectory = []
    population = np.random.uniform(BOUNDS_LOW, BOUNDS_HIGH, (N_PARTICLES, DIMENSION))
    F, CR = 0.8, 0.9
    fitness = np.array([problem(ind) for ind in population])
    for _ in range(N_ITERATIONS):
        trajectory.append(population.copy())
        for i in range(N_PARTICLES):
            idxs = [j for j in range(N_PARTICLES) if j != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), BOUNDS_LOW, BOUNDS_HIGH)
            cross = np.random.rand(DIMENSION) < CR
            trial = np.where(cross, mutant, population[i])
            trial_fit = problem(trial)
            if trial_fit < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fit
    return np.array(trajectory)

def run_cmaes(problem):
    trajectory = []
    optimizer = CMA(mean=np.zeros(DIMENSION), sigma=2.0)
    for _ in range(N_ITERATIONS):
        popsize = optimizer.population_size  # use CMA's own popsize
        solutions = []
        particles = []
        for _ in range(popsize):
            x = optimizer.ask()
            particles.append(x.copy())
            solutions.append((x, problem(x)))
        # pad trajectory to N_PARTICLES so all images are same size
        particles_arr = np.array(particles)
        if len(particles) < N_PARTICLES:
            pad = np.tile(particles_arr, (N_PARTICLES // len(particles) + 1, 1))[:N_PARTICLES]
            particles_arr = pad
        trajectory.append(particles_arr[:N_PARTICLES])
        optimizer.tell(solutions)
    return np.array(trajectory)

def run_simulated_annealing(problem):
    trajectory = []
    current = np.random.uniform(BOUNDS_LOW, BOUNDS_HIGH, DIMENSION)
    current_fit = problem(current)
    temp = 1.0
    
    for _ in range(N_ITERATIONS):
        # Generate N_PARTICLES candidate solutions around current
        particles = current + np.random.randn(N_PARTICLES, DIMENSION) * temp
        particles = np.clip(particles, BOUNDS_LOW, BOUNDS_HIGH)
        trajectory.append(particles.copy())
        
        # Move to best candidate if better, or with some probability if worse
        fits = np.array([problem(p) for p in particles])
        best_idx = np.argmin(fits)
        delta = fits[best_idx] - current_fit
        if delta < 0 or np.random.rand() < np.exp(-delta / (temp + 1e-10)):
            current = particles[best_idx]
            current_fit = fits[best_idx]
        
        temp *= 0.95  # cool down
    
    return np.array(trajectory)

# ── Main generation loop ───────────────────────────────────

ALGORITHMS = {
    'PSO': run_pso,
    'RandomSearch': run_random_search,
    'DE': run_de,
    'CMAES': run_cmaes,
    'SimulatedAnnealing': run_simulated_annealing,  
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
for algo_name, algo_fn in ALGORITHMS.items():
    os.makedirs(f'{OUTPUT_DIR}/{algo_name}', exist_ok=True)

total = len(ALGORITHMS) * len(BBOB_FUNCTIONS) * N_RUNS
done = 0

for algo_name, algo_fn in ALGORITHMS.items():
    for func_id in BBOB_FUNCTIONS:
        for run in range(N_RUNS):
            problem = ioh.get_problem(func_id, dimension=DIMENSION, instance=run+1)
            trajectory = algo_fn(problem)
            fingerprint = get_fingerprint(trajectory)
            filename = f'{OUTPUT_DIR}/{algo_name}/{algo_name}_f{func_id}_run{run:02d}.png'
            save_image(fingerprint, filename)
            done += 1
            print(f'[{done}/{total}] {filename}')

print('Dataset generation complete!')
print(f'Total images: {done}')