# conftest.py: Root-level path setup so tests can import from src/.
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
