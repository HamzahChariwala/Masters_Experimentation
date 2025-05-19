import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import pickle
import yaml

class StateActionDataset(Dataset):
    """Dataset for state-action pairs."""
    def __init__(self, states, actions):
        """
        Args:
            states: Array of environment states 
            actions: Array of corresponding actions
        """
        self.states = torch.tensor(states, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.float32)
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return {
            'state': self.states[idx],
            'action': self.actions[idx]
        }

class VAE(nn.Module):
    """Variational Autoencoder for state-action pairs."""
    def __init__(self, state_dim, action_dim, latent_dim=32, hidden_dim=128):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            latent_dim: Dimension of latent space
            hidden_dim: Dimension of hidden layers
        """
        super(VAE, self).__init__()
        
        # Input dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.input_dim = state_dim + action_dim
        self.latent_dim = latent_dim
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.input_dim)
        )
    
    def encode(self, state, action):
        """Encode state-action pair to latent distribution parameters."""
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)
        
        # Encode
        h = self.encoder(x)
        
        # Get mean and log variance
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """Reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """Decode latent vector to state-action pair."""
        output = self.decoder(z)
        
        # Split output into state and action
        state_recon = output[:, :self.state_dim]
        action_recon = output[:, self.state_dim:]
        
        return state_recon, action_recon
    
    def forward(self, state, action):
        """Full forward pass."""
        # Encode
        mu, log_var = self.encode(state, action)
        
        # Sample from latent distribution
        z = self.reparameterize(mu, log_var)
        
        # Decode
        state_recon, action_recon = self.decode(z)
        
        return state_recon, action_recon, mu, log_var
    
    def get_latent(self, state, action):
        """Get latent vector for a state-action pair."""
        with torch.no_grad():
            mu, log_var = self.encode(state, action)
            z = self.reparameterize(mu, log_var)
        return z

def vae_loss(state_recon, action_recon, state, action, mu, log_var, beta=1.0):
    """
    VAE loss function: reconstruction loss + KL divergence.
    
    Args:
        state_recon: Reconstructed state
        action_recon: Reconstructed action
        state: Original state
        action: Original action
        mu: Mean of latent distribution
        log_var: Log variance of latent distribution
        beta: Weight of KL divergence term
    """
    # State reconstruction loss (MSE)
    state_recon_loss = F.mse_loss(state_recon, state, reduction='sum')
    
    # Action reconstruction loss (MSE)
    action_recon_loss = F.mse_loss(action_recon, action, reduction='sum')
    
    # Total reconstruction loss
    recon_loss = state_recon_loss + action_recon_loss
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # Total loss
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

def collect_rollout_data(env, policy, n_episodes=100, max_steps=1000):
    """
    Collect state-action pairs from environment rollouts.
    
    Args:
        env: Gymnasium environment
        policy: Policy function or model that maps states to actions
        n_episodes: Number of episodes to collect
        max_steps: Maximum steps per episode
        
    Returns:
        states: List of states
        actions: List of actions
    """
    import gymnasium as gym
    
    states = []
    actions = []
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            # Get action from policy
            action, _ = policy.predict(state, deterministic=False)
            
            # Store state and action
            states.append(state)
            actions.append(action)
            
            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            steps += 1
            
    return np.array(states), np.array(actions)

def train_vae(vae, dataloader, optimizer, device, epochs=100, beta=1.0):
    """Train the VAE model."""
    vae.train()
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        recon_loss_epoch = 0
        kl_loss_epoch = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Get batch data
            state = batch['state'].to(device)
            action = batch['action'].to(device)
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Forward pass
            state_recon, action_recon, mu, log_var = vae(state, action)
            
            # Compute loss
            loss, recon_loss, kl_loss = vae_loss(state_recon, action_recon, state, action, mu, log_var, beta)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # Accumulate loss
            epoch_loss += loss.item()
            recon_loss_epoch += recon_loss.item()
            kl_loss_epoch += kl_loss.item()
        
        # Average losses
        avg_loss = epoch_loss / len(dataloader)
        avg_recon_loss = recon_loss_epoch / len(dataloader)
        avg_kl_loss = kl_loss_epoch / len(dataloader)
        
        losses.append(avg_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Recon Loss: {avg_recon_loss:.4f}, KL Loss: {avg_kl_loss:.4f}")
    
    return losses

def visualize_latent_space(vae, dataloader, device, perplexity=30):
    """Visualize the latent space using t-SNE."""
    latent_vectors = []
    all_actions = []
    
    vae.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating latent vectors"):
            state = batch['state'].to(device)
            action = batch['action'].to(device)
            
            # Get latent vectors
            mu, _ = vae.encode(state, action)
            latent_vectors.append(mu.cpu().numpy())
            all_actions.append(action.cpu().numpy())
    
    # Concatenate
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    
    # Use t-SNE for dimensionality reduction to 2D
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    latent_2d = tsne.fit_transform(latent_vectors)
    
    # Convert one-hot actions to indices if needed
    if all_actions.shape[1] > 1:
        action_indices = np.argmax(all_actions, axis=1)
    else:
        action_indices = all_actions.flatten()
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=action_indices, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, label='Action')
    plt.title('t-SNE Visualization of Latent Space')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig('latent_space_visualization.png')
    plt.close()

def save_model(vae, optimizer, epoch, loss, filename):
    """Save model and optimizer state."""
    state = {
        'epoch': epoch,
        'model_state_dict': vae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(state, filename)
    print(f"Model saved to {filename}")

def load_model(vae, optimizer, filename):
    """Load model and optimizer state."""
    checkpoint = torch.load(filename)
    vae.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Model loaded from {filename}, epoch {epoch}")
    return epoch, loss

def load_config(config_path='vae_config.yml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_device():
    """Get the best available device (MPS for Mac, CPU otherwise)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def load_data(data_path, data_format="pickle"):
    """
    Load data from file in specified format.
    
    Args:
        data_path: Path to data file
        data_format: Format of data file (pickle, numpy, etc.)
        
    Returns:
        states: Array of states
        actions: Array of actions
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    if data_format.lower() == "pickle":
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict):
            # Check for required keys
            if 'states' not in data or 'actions' not in data:
                raise ValueError("Data dictionary must contain 'states' and 'actions' keys")
            return data['states'], data['actions']
        else:
            raise ValueError(f"Expected dictionary in pickle file, got {type(data)}")
    
    elif data_format.lower() == "numpy":
        # Assumes two .npy files: {data_path}_states.npy and {data_path}_actions.npy
        states_path = f"{os.path.splitext(data_path)[0]}_states.npy"
        actions_path = f"{os.path.splitext(data_path)[0]}_actions.npy"
        
        if not os.path.exists(states_path) or not os.path.exists(actions_path):
            raise FileNotFoundError(f"States or actions file not found: {states_path}, {actions_path}")
        
        states = np.load(states_path)
        actions = np.load(actions_path)
        return states, actions
    
    else:
        raise ValueError(f"Unsupported data format: {data_format}")

def main(config_path='vae_config.yml'):
    # Load configuration
    config = load_config(config_path)
    
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Set up directories
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Load data from file (primary method)
    if config.get('data_path'):
        try:
            print(f"Loading data from {config['data_path']}")
            states, actions = load_data(
                config['data_path'], 
                data_format=config.get('data_format', 'pickle')
            )
        except Exception as e:
            print(f"Error loading data: {e}")
            if config.get('env_id'):
                print("Falling back to collecting data from environment (not recommended)")
                states, actions = collect_data_from_env(config)
            else:
                raise
    else:
        # Fall back to collecting data (not recommended)
        print("No data_path provided. Collecting data from environment (not recommended)")
        states, actions = collect_data_from_env(config)
    
    # Determine state and action dimensions
    state_dim = states.shape[1]
    if len(actions.shape) == 1:
        # Convert scalar actions to one-hot if discrete
        action_dim = int(max(actions) + 1)
        actions_one_hot = np.zeros((len(actions), action_dim))
        actions_one_hot[np.arange(len(actions)), actions.astype(int)] = 1
        actions = actions_one_hot
    else:
        action_dim = actions.shape[1]
    
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    
    # Create dataset and dataloader
    dataset = StateActionDataset(states, actions)
    dataloader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config.get('num_workers', 0)
    )
    
    # Create VAE model
    vae = VAE(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=config['latent_dim'],
        hidden_dim=config['hidden_dim']
    ).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(vae.parameters(), lr=config['learning_rate'])
    
    # Start or resume training
    start_epoch = 0
    if config.get('resume') and os.path.exists(config['resume']):
        start_epoch, _ = load_model(vae, optimizer, config['resume'])
    
    # Train VAE
    losses = train_vae(
        vae=vae,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        epochs=config['epochs'],
        beta=config['beta']
    )
    
    # Save final model
    save_model(
        vae=vae,
        optimizer=optimizer,
        epoch=start_epoch + config['epochs'],
        loss=losses[-1],
        filename=os.path.join(config['output_dir'], 'vae_model.pth')
    )
    
    # Visualize latent space
    if config.get('visualize'):
        visualize_latent_space(vae, dataloader, device, perplexity=config.get('perplexity', 30))
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses)
    plt.title('VAE Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(config['output_dir'], 'vae_loss.png'))
    plt.close()
    
    print("Training complete!")
    print(f"Final model saved to {os.path.join(config['output_dir'], 'vae_model.pth')}")
    if config.get('visualize'):
        print(f"Visualization saved to {os.path.join(config['output_dir'], 'latent_space_visualization.png')}")
    print(f"Loss curve saved to {os.path.join(config['output_dir'], 'vae_loss.png')}")


def collect_data_from_env(config):
    """Helper function to collect data from environment (fallback method)."""
    import gymnasium as gym
    
    # Load environment and policy
    env = gym.make(config['env_id'])
    
    # Check if we need to load a policy or use random
    if config.get('policy_path') and os.path.exists(config['policy_path']):
        from stable_baselines3 import PPO
        policy = PPO.load(config['policy_path'])
    else:
        # Define a random policy
        class RandomPolicy:
            def __init__(self, action_space):
                self.action_space = action_space
            
            def predict(self, state, deterministic=False):
                return self.action_space.sample(), None
        
        policy = RandomPolicy(env.action_space)
    
    # Collect data
    states, actions = collect_rollout_data(
        env, policy, 
        n_episodes=config.get('episodes', 100), 
        max_steps=config.get('max_steps', 1000)
    )
    
    # Save data if needed
    if config.get('save_data'):
        data_path = os.path.join(config.get('output_dir', '.'), 'rollout_data.pkl')
        with open(data_path, 'wb') as f:
            pickle.dump({'states': states, 'actions': actions}, f)
        print(f"Data saved to {data_path}")
    
    return states, actions


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main() 