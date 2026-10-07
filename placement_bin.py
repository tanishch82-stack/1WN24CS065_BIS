import numpy as np

# Set random seed for identical project layouts
np.random.seed(42)

# ==========================================
# 1. SETUP INTERIOR CAMPUS DATA
# ==========================================
num_classrooms = 20
num_can_spots = 12
MAX_WALKING_CAP = 30.0
BIN_CAPACITY = 40.0

# Setup Coordinates [Floor, X, Y]
building_classrooms = np.zeros((num_classrooms, 3), dtype=int)
building_classrooms[:, 0] = np.random.randint(0, 3, size=num_classrooms)
building_classrooms[:, 1] = np.random.randint(0, 51, size=num_classrooms)
building_classrooms[:, 2] = np.random.randint(0, 31, size=num_classrooms)

building_candidates = np.zeros((num_can_spots, 3), dtype=int)
building_candidates[:, 0] = np.random.randint(0, 3, size=num_can_spots)
building_candidates[:, 1] = np.random.randint(0, 51, size=num_can_spots)
building_candidates[:, 2] = np.random.randint(0, 31, size=num_can_spots)

classroom_trash_volume = np.random.randint(5, 15, size=num_classrooms)

# Calculate Manhattan Distance Matrix
stair_penalty = 15
distance_matrix = np.zeros((num_classrooms, num_can_spots))
for i in range(num_classrooms):
    for j in range(num_can_spots):
        floor_diff = abs(building_classrooms[i, 0] - building_candidates[j, 0])
        x_diff = abs(building_classrooms[i, 1] - building_candidates[j, 1])
        y_diff = abs(building_classrooms[i, 2] - building_candidates[j, 2])
        distance_matrix[i, j] = x_diff + y_diff + (floor_diff * stair_penalty)

# ==========================================
# 2. BALANCED FITNESS FUNCTION
# ==========================================
def compute_fitness(chromosome):
    active_bin_indices = np.where(chromosome == 1)[0]

    if len(active_bin_indices) == 0:
        return 99999.0

    # Calculate Distances
    active_distances = distance_matrix[:, active_bin_indices]
    min_distances = np.min(active_distances, axis=1)

    penalty = 0.0

    # Constraint 1: Max Walking Cap Distance
    for dist in min_distances:
        if dist > MAX_WALKING_CAP:
            penalty += (dist - MAX_WALKING_CAP) * 100

    # Constraint 2: Dustbin Capacity Limits
    closest_active_idx = np.argmin(active_distances, axis=1)
    nearest_bin_global_idx = active_bin_indices[closest_active_idx]
    bin_loadings = np.zeros(num_can_spots)
    for room_id, bin_id in enumerate(nearest_bin_global_idx):
        bin_loadings[bin_id] += classroom_trash_volume[room_id]
    for load in bin_loadings:
        if load > BIN_CAPACITY:
            penalty += (load - BIN_CAPACITY) * 150

    # Constraint 3: Strict Balanced Floor Rule
    bins_on_floor = {0: 0, 1: 0, 2: 0}
    for b_idx in active_bin_indices:
        floor = building_candidates[b_idx, 0]
        bins_on_floor[floor] += 1

    # Enforce exactly your suggested strategy: 1 on Floor 0, 3 on Floor 1, 2 on Floor 2
    if bins_on_floor[0] != 1: penalty += 500
    if bins_on_floor[1] != 3: penalty += 500
    if bins_on_floor[2] != 2: penalty += 500

    return len(active_bin_indices) + penalty

# GA Operators
def create_population(pop_size):
    return [np.random.randint(0, 2, size=num_can_spots) for _ in range(pop_size)]

def crossover(parent1, parent2):
    cut = np.random.randint(1, num_can_spots)
    return np.concatenate((parent1[:cut], parent2[cut:]))

def mutate(chromosome, mutation_rate=0.15):
    for i in range(num_can_spots):
        if np.random.rand() < mutation_rate:
            chromosome[i] = 1 - chromosome[i]
    return chromosome

# ==========================================
# 3. EVOLUTIONARY EXECUTION LOOP
# ==========================================
pop_size, generations = 40, 100
population = create_population(pop_size)

for gen in range(generations):
    population = sorted(population, key=compute_fitness)
    new_population = population[:4]
    while len(new_population) < pop_size:
        p1, p2 = np.random.choice(len(population) // 2, 2, replace=False)
        child = mutate(crossover(population[p1], population[p2]))
        new_population.append(child)
    population = new_population

# ==========================================
# 4. GENERATING REPORT OUTPUT WITH SCHEMATICS
# ==========================================
best_layout = sorted(population, key=compute_fitness)[0]
final_bin_spots = np.where(best_layout == 1)[0]

active_distances = distance_matrix[:, final_bin_spots]
min_distances = np.min(active_distances, axis=1)

print("="*55)
print("     COLLEGE PROJECT: BALANCED DUSTBIN PLACEMENT      ")
print("="*55)
print(f"Total Optimal Bins: {len(final_bin_spots)}")
print(f"New Average Walking Distance: {np.mean(min_distances):.2f} meters")

print("\n--- NEW BALANCED BLUEPRINT ---")
for i, spot_id in enumerate(final_bin_spots):
    floor, x, y = building_candidates[spot_id]
    print(f"Dustbin {i+1} [Spot #{spot_id}]: Installed on Floor {floor} (X: {x}m, Y: {y}m)")

print("\n--- TEXT-BASED FLOOR MAP SCHEMATIC ---")
for f in range(3):
    print(f"\n[FLOOR {f}] Layout Grid:")
    grid = np.full((7, 13), '.') # Map sizing layout grid

    # Plot Classrooms on the grid as 'C'
    for r_id, (floor, x, y) in enumerate(building_classrooms):
        if floor == f:
            grid[int(y/5), int(x/4)] = 'C'

    # Plot Placed Active Dustbins on the grid as '★'
    for b_id in final_bin_spots:
        floor, x, y = building_candidates[b_id]
        if floor == f:
            grid[int(y/5), int(x/4)] = '★'

    for row in grid:
        print(" ".join(row))

print("\nLegend: [ C ] = Classroom / [ ★ ] = Optimized Dustbin Location")
print("="*55)

