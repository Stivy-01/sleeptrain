#!/usr/bin/env python
"""
Quick Run Script for LoRA Hyperparameter Sweep

Usage:
    # Set your Gemini API key
    export GEMINI_API_KEY="your_key_here"
    
    # Run quick test (4 experiments)
    python run_sweep.py --quick
    
    # Run full sweep (16 experiments)
    python run_sweep.py
    
    # Run custom number of experiments
    python run_sweep.py --max-experiments 8
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sweep import main

if __name__ == "__main__":
    main()

