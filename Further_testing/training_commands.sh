# Exit script if any command fails
set -e

# Print each command before executing it
set -x

echo "Starting command sequence..."

# echo "Training default..."
# python Agent_Training/train.py --path SpawnTests/default

# echo "Training biased..."
# python Agent_Training/train.py --path SpawnTests/biased

# echo "Training uniform..."
# python Agent_Training/train.py --path SpawnTests/uniform

echo "Training size 3..."
python Agent_Training/train.py --path WindowSizing/size_3

echo "Training size 5..."
python Agent_Training/train.py --path WindowSizing/size_5

echo "Training size 7..."
python Agent_Training/train.py --path WindowSizing/size_7

echo "Training size 9..."
python Agent_Training/train.py --path WindowSizing/size_9

echo "Training size 11..."
python Agent_Training/train.py --path WindowSizing/size_11

echo "Training size 13..."
python Agent_Training/train.py --path WindowSizing/size_13

echo "Training size 15..."
python Agent_Training/train.py --path WindowSizing/size_15

echo "Tuning hyperparameters..."
python Agent_Training/hyperparam.py --base-config Agent_Storage/Hyperparameters/example_config.yaml --tuning-config Agent_Storage/Hyperparameters/optuna_config.yaml --method bayesian --samples 20