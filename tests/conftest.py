import sys
from pathlib import Path

# Get the absolute path to the trajectory_tracer directory
project_root = Path(__file__).parent.parent.absolute()

# Add to path
sys.path.insert(0, str(project_root))
