"""Engine regression = the full benchmark.   python tests/test_engine.py"""
import os, subprocess, sys
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.exit(subprocess.run([sys.executable, "evaluate.py"], cwd=root).returncode)
