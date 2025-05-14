# Exit script if any command fails
set -e

# Print each command before executing it
set -x

echo "Starting command sequence..."

echo "Training 0.025 penalty..."
python Agent_Training/train.py --path LavaTests/NoDeath/0.025_penalty

echo "Training 0.050 penalty..."
python Agent_Training/train.py --path LavaTests/NoDeath/0.050_penalty

echo "Training 0.075 penalty..."
python Agent_Training/train.py --path LavaTests/NoDeath/0.075_penalty

echo "Training 0.100 penalty..."
python Agent_Training/train.py --path LavaTests/NoDeath/0.100_penalty

echo "Training 0.125 penalty..."
python Agent_Training/train.py --path LavaTests/NoDeath/0.125_penalty

echo "Training 0.150 penalty..."
python Agent_Training/train.py --path LavaTests/NoDeath/0.150_penalty

echo "Training 0.175 penalty..."
python Agent_Training/train.py --path LavaTests/NoDeath/0.175_penalty

echo "Training 0.200 penalty..."
python Agent_Training/train.py --path LavaTests/NoDeath/0.200_penalty

echo "Training standard..."
python Agent_Training/train.py --path LavaTests/Standard

echo "Training 2 penalty..."
python Agent_Training/train.py --path LavaTests/EpisodeEnd/Linear/2_penalty

echo "Training 3 penalty..."
python Agent_Training/train.py --path LavaTests/EpisodeEnd/Linear/3_penalty

echo "Training 4 penalty..."
python Agent_Training/train.py --path LavaTests/EpisodeEnd/Linear/4_penalty

echo "Training 5 penalty..."
python Agent_Training/train.py --path LavaTests/EpisodeEnd/Linear/5_penalty

echo "Training 6 penalty..."
python Agent_Training/train.py --path LavaTests/EpisodeEnd/Linear/6_penalty

echo "Training standard2..."
python Agent_Training/train.py --path LavaTests/Standard2