import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from training.train_mappo import train

if __name__ == "__main__":
    train()
