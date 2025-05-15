import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter


# Ackley Function (2D)
def ackley(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    return term1 + term2 + a + np.exp(1)

# PSO Parameters
num_particles = 30
dimensions = 2
iters = 100
bounds = (-5, 5)
w = 0.7

# NEW CHANGE: Modified PSO weights and parameters
c1 = 1.5  # Weight for personal best (Pbest) influence
c2 = 1.0  # Weight for global best (Gbest) influence - reduced from 1.5 to balance with Lbest
c3 = 1.5  # NEW: Weight for local best (Lbest) influence
k = 3     # NEW: Number of neighbors on each side in ring topology (total 3k neighbors per particle)

# Initialize Particles
position = np.random.uniform(bounds[0], bounds[1], (num_particles, dimensions))
velocity = np.zeros((num_particles, dimensions))
pbest = position.copy()
pbest_val = np.array([ackley(p) for p in pbest])
gbest = pbest[np.argmin(pbest_val)]
gbest_val = np.min(pbest_val)

# NEW FUNCTION: Calculate local best using ring topology neighborhood
def get_lbest(particle_idx, pbest, pbest_val):
    # Create list of neighbor indices in ring structure
    neighbors = []
    for i in range(-k, k + 1):
        # Use modulo to wrap around the array (creating ring structure)
        neighbor_idx = (particle_idx + i) % num_particles
        neighbors.append(neighbor_idx)
    
    # Find the neighbor with best fitness value
    neighbor_best_idx = neighbors[np.argmin(pbest_val[neighbors])]
    return pbest[neighbor_best_idx]

# Store positions for animation
positions_over_time = []

# PSO Loop
for i in range(iters):
    r1 = np.random.rand(num_particles, dimensions)
    r2 = np.random.rand(num_particles, dimensions)
    r3 = np.random.rand(num_particles, dimensions)  # NEW: Random weights for Lbest
    
    # NEW CHANGE: Modified velocity update to include Lbest influence
    for j in range(num_particles):
        lbest = get_lbest(j, pbest, pbest_val)  # Get local best for current particle
        velocity[j] = (w * velocity[j] +                                # Inertia
                      c1 * r1[j] * (pbest[j] - position[j]) +         # Personal best influence
                      c2 * r2[j] * (gbest - position[j]) +            # Global best influence
                      c3 * r3[j] * (lbest - position[j]))            # NEW: Local best influence
    
    position += velocity
    position = np.clip(position, bounds[0], bounds[1])

    scores = np.array([ackley(p) for p in position])
    improved = scores < pbest_val
    pbest[improved] = position[improved]
    pbest_val[improved] = scores[improved]

    if np.min(pbest_val) < gbest_val:
        gbest_val = np.min(pbest_val)
        gbest = pbest[np.argmin(pbest_val)]

    positions_over_time.append(position.copy())

    if i % 10 == 0 or i == iters - 1:
        print(f"Iteration {i}: gBest = {gbest}, value = {gbest_val:.6f}")

# Create meshgrid for contour plot
x = np.linspace(bounds[0], bounds[1], 200)
y = np.linspace(bounds[0], bounds[1], 200)
X, Y = np.meshgrid(x, y)

# Evaluate Ackley function on the grid
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = ackley(np.array([X[i, j], Y[i, j]]))

# Set up the plot
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
particles, = ax.plot([], [], 'ro', markersize=4)

# Animation functions
def init():
    particles.set_data([], [])
    return particles,

def update(frame):
    pos = positions_over_time[frame]
    particles.set_data(pos[:, 0], pos[:, 1])
    ax.set_title(f"Iteration {frame}")
    return particles,

ani = animation.FuncAnimation(
    fig, update, frames=iters,
    init_func=init, blit=True, interval=100, repeat=False
)

plt.colorbar(contour)
plt.xlabel("x")
plt.ylabel("y")
plt.title("PSO on Ackley Function")
plt.show()

# Save animation as GIF
ani.save("pso_ackley.gif", writer=PillowWriter(fps=10))
print("GIF saved as pso_ackley.gif")