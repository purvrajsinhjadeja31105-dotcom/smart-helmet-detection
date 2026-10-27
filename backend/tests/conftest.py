import sys
from pathlib import Path

# make "import app" work when pytest is started from the backend folder or the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
