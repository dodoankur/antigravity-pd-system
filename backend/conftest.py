# conftest.py — pytest configuration for PD backend tests
# This file ensures the backend root is on sys.path so imports resolve correctly.
import sys
import os

# Add backend root to path for all tests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
