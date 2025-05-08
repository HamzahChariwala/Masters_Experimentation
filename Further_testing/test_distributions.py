#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import os
from SpawnDistributions.spawn_distributions import DistributionMap

# Create output directory
output_dir = "distribution_samples"
os.makedirs(output_dir, exist_ok=True)

# Grid dimensions for testing
width, height = 10, 10

# Test different distribution types
print("Generating distribution samples...")

# 1. Uniform distribution
uniform_dist = DistributionMap(width, height)
uniform_dist.uniform_distribution()
uniform_dist.plot(title="Uniform Distribution", 
                  show=False, 
                  save_path=os.path.join(output_dir, "uniform.png"))

# 2. Poisson distribution (from goal)
goal_pos = (8, 8)  # Example goal position
poisson_near = DistributionMap(width, height)
poisson_near.poisson_from_point(goal_pos[0], goal_pos[1], lambda_param=1.0)
poisson_near.plot(title="Poisson Distribution (Near Goal)", 
                  show=False, 
                  save_path=os.path.join(output_dir, "poisson_near.png"))

# 3. Inverted Poisson (far from goal)
poisson_far = DistributionMap(width, height)
poisson_far.poisson_from_point(goal_pos[0], goal_pos[1], lambda_param=1.0)
poisson_far.invert()
poisson_far.plot(title="Poisson Distribution (Far from Goal)", 
                show=False, 
                save_path=os.path.join(output_dir, "poisson_far.png"))

# 4. Gaussian distribution
gaussian_near = DistributionMap(width, height)
gaussian_near.gaussian_from_point(goal_pos[0], goal_pos[1], sigma=2.0)
gaussian_near.plot(title="Gaussian Distribution (Near Goal)", 
                  show=False, 
                  save_path=os.path.join(output_dir, "gaussian_near.png"))

# 5. Inverted Gaussian
gaussian_far = DistributionMap(width, height)
gaussian_far.gaussian_from_point(goal_pos[0], goal_pos[1], sigma=2.0)
gaussian_far.invert()
gaussian_far.plot(title="Gaussian Distribution (Far from Goal)", 
                 show=False, 
                 save_path=os.path.join(output_dir, "gaussian_far.png"))

# 6. Distance-based (near goal)
distance_near = DistributionMap(width, height)
distance_near.distance_based_from_point(goal_pos[0], goal_pos[1], favor_near=True, power=1)
distance_near.plot(title="Distance-Based Distribution (Near Goal)", 
                  show=False, 
                  save_path=os.path.join(output_dir, "distance_near.png"))

# 7. Distance-based (far from goal)
distance_far = DistributionMap(width, height)
distance_far.distance_based_from_point(goal_pos[0], goal_pos[1], favor_near=False, power=1)
distance_far.plot(title="Distance-Based Distribution (Far from Goal)", 
                 show=False, 
                 save_path=os.path.join(output_dir, "distance_far.png"))

# 8. Multi-point distribution (simulating lava)
lava_positions = [(2, 3), (3, 7), (7, 2)]
multi_point = DistributionMap(width, height)
multi_point.multi_point_distribution(lava_positions, distribution_type="gaussian", params={"sigma": 1.5})
multi_point.plot(title="Multi-Point Gaussian Distribution", 
                show=False, 
                save_path=os.path.join(output_dir, "multi_point.png"))

# 9. With cell masking (simulate walls/obstacles)
mask = np.ones((height, width))
# Add some "walls"
mask[3:6, 3:6] = 0  # Center block
mask[0:3, 8:10] = 0  # Corner block
mask[7:10, 0:3] = 0  # Another corner block

# Apply mask to distance distribution
masked_dist = DistributionMap(width, height)
masked_dist.distance_based_from_point(goal_pos[0], goal_pos[1], favor_near=False)
masked_dist.mask_cells(mask)
masked_dist.plot(title="Masked Distribution (with walls)", 
                show=False, 
                save_path=os.path.join(output_dir, "masked.png"))

# 10. Temporal interpolation example (start to end)
start_dist = DistributionMap(width, height)
start_dist.poisson_from_point(1, 1, lambda_param=1.0)  # Start near (1,1)

end_dist = DistributionMap(width, height)
end_dist.poisson_from_point(8, 8, lambda_param=1.0)    # End near (8,8)

# Create interpolation steps
for step in range(6):
    progress = step / 5.0  # 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
    interp_dist = DistributionMap(width, height)
    interp_dist.from_existing_distribution(start_dist.probabilities)
    interp_dist.temporal_interpolation(end_dist, progress)
    interp_dist.plot(title=f"Temporal Interpolation (Progress: {progress:.1f})", 
                    show=False, 
                    save_path=os.path.join(output_dir, f"temporal_{step}.png"))

print(f"Distribution samples generated in {output_dir}/")
print("Generated the following visualization samples:")
print(" - uniform.png: Uniform distribution")
print(" - poisson_near.png: Poisson distribution (high probability near goal)")
print(" - poisson_far.png: Inverted Poisson (high probability far from goal)")
print(" - gaussian_near.png: Gaussian distribution near goal")
print(" - gaussian_far.png: Inverted Gaussian (far from goal)")
print(" - distance_near.png: Distance-based near goal")
print(" - distance_far.png: Distance-based far from goal")
print(" - multi_point.png: Multiple point distribution (e.g., for lava)")
print(" - masked.png: Distribution with masked cells (walls/obstacles)")
print(" - temporal_*.png: Temporal interpolation sequence") 