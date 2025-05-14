# Performance Degradation Analysis

## Issue
During training, there is a gradual decrease in frames per second (FPS) over time. Initially, the FPS is as high as 12,000 but continuously drops as training progresses.

## Potential Causes (Prioritized)

### 1. Unbounded Data Accumulation in PerformanceTracker (High Priority)
The `PerformanceTracker` class continually accumulates metrics in arrays without any limit, which can lead to memory bloat and slower operations as training progresses.

**Location:** `Agent_Training/TrainingTooling/Tooling.py`

**Impact:** High - As training progresses, these arrays can grow to thousands of entries, increasing memory pressure and slowing down any operations that use these arrays.

### 2. Frequent Matplotlib Plot Generation (High Priority)
After each evaluation, the code generates and saves multiple plots, which is a CPU-intensive operation.

**Location:** `Agent_Training/TrainingTooling/TerminalCondition.py` and `Agent_Training/TrainingTooling/Tooling.py`

**Impact:** High - Plot generation is one of the most expensive operations and happens regularly during training.

### 3. Thread Creation for Evaluations (Medium Priority)
The evaluation process creates new threads each time an evaluation is performed, which adds overhead.

**Location:** `Agent_Training/TrainingTooling/TerminalCondition.py`

**Impact:** Medium - Thread creation itself isn't extremely expensive, but it adds up when evaluations are frequent.

### 4. Increasing Complexity of Environment Processing (Medium Priority)
As the agent learns, it may explore more complex states in the environment, which can make processing of environment steps more expensive.

**Impact:** Medium - This depends on the specific environment and agent behavior.

### 5. Growing Replay Buffer and Model Updates (Low Priority)
As the replay buffer fills and the neural network parameters are updated more frequently, it may cause more garbage collection cycles.

**Impact:** Low - Modern systems handle these operations efficiently, but they could contribute to the slowdown.

## Solutions

### 1. Limit Data Storage in PerformanceTracker

Modify the `PerformanceTracker` class to maintain a fixed-size buffer for metrics:

```python
def update_train_metrics(self, timestep, rewards, lengths, episode_count):
    """Update training metrics with limited history"""
    # Limit arrays to a fixed size (e.g., 100 entries)
    max_history = 100
    
    self.train_timesteps.append(timestep)
    self.train_rewards.append(np.mean(rewards) if len(rewards) > 0 else 0)
    self.train_lengths.append(np.mean(lengths) if len(lengths) > 0 else 0)
    self.train_episode_counts.append(episode_count)
    
    # Trim arrays if they exceed max_history
    if len(self.train_timesteps) > max_history:
        self.train_timesteps = self.train_timesteps[-max_history:]
        self.train_rewards = self.train_rewards[-max_history:]
        self.train_lengths = self.train_lengths[-max_history:]
        self.train_episode_counts = self.train_episode_counts[-max_history:]
```

### 2. Reduce Frequency of Plot Generation

Modify the evaluation function to generate plots less frequently:

```python
# In TerminalCondition.py, _run_evaluation method
def _run_evaluation(self):
    # ... existing code ...
    
    # Generate performance plots only every N evaluations or based on time
    if self.evaluation_count % 5 == 0:  # Generate plots every 5 evaluations
        self.performance_tracker.plot_performance(save=True, show=False)
        if self.verbose > 0:
            print(f"Performance plots saved to {self.log_dir}")
    
    # ... rest of the method ...
```

### 3. Use a Thread Pool for Evaluations

Instead of creating new threads for each evaluation, use a thread pool:

```python
# Import ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor

# Initialize a thread pool in the constructor
def __init__(self, ...):
    # ... existing code ...
    self.executor = ThreadPoolExecutor(max_workers=1)

# Use the thread pool in _run_evaluation
def _run_evaluation(self):
    # ... existing code ...
    
    # Submit the evaluation task to the thread pool
    future = self.executor.submit(self._evaluate_agent)
    
    # Wait for the task to complete with timeout
    start_wait = time.time()
    while not self.evaluation_complete and (time.time() - start_wait) < self.eval_timeout:
        time.sleep(0.1)
    
    # ... rest of the method ...
```

### 4. Adaptive Evaluation Frequency

Reduce evaluation frequency as training progresses:

```python
def _on_step(self):
    # ... existing code ...
    
    # Adaptive check frequency - increase as training progresses
    current_check_freq = self.check_freq
    if self.num_timesteps > 1000000:
        current_check_freq = self.check_freq * 2
    if self.num_timesteps > 5000000:
        current_check_freq = self.check_freq * 4
    
    # Only check at the adaptively determined frequency
    if self.num_timesteps % current_check_freq != 0:
        return True
    
    # ... rest of the method ...
```

### 5. Profile Memory Usage

Add memory profiling to identify specific leaks:

```python
# Install memory_profiler: pip install memory_profiler
from memory_profiler import profile

# Add profiling to key methods
@profile
def learn(self, total_timesteps, ...):
    # ... existing code ...
```

## Implementation Priority

1. **First Fix**: Limit data accumulation in PerformanceTracker (easiest and highest impact)
2. **Second Fix**: Reduce frequency of plot generation (high impact, easy implementation)
3. **Third Fix**: Implement adaptive evaluation frequency (good balance of impact and effort)
4. **If Needed**: Use a thread pool for evaluations and/or add memory profiling

## Expected Outcomes

After implementing these changes, you should observe:
- More stable FPS throughout the training process
- Lower memory usage
- Faster overall training time
- Minimal impact on the quality of the trained agent

## Monitoring the Changes

After implementing each change, monitor the FPS over time to see if the degradation is reduced. The fixes can be implemented incrementally to measure the impact of each change. 