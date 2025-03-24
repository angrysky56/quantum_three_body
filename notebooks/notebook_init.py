
# Add project root to path
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

# Import GPU configuration
try:
    from src.gpu_accelerator import configure_gpu
    configure_gpu(enable=True)  # Enable GPU acceleration
except ImportError:
    print("GPU acceleration not available")

# Common imports for notebooks
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import HTML

# Initialize plotting style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
