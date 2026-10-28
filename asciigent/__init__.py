"""ASCIIgent: training-free agentic ASCII art generation."""

from .env import load_env

__version__ = "0.1.0"

# Importing the package makes keys in .env available, so every entry point --
# scripts, tests, notebooks -- sees the same credentials without setup.
load_env()
