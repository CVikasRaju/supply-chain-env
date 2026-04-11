import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())

try:
    from graders.grader_baseline import grade as grade_baseline
    print("Successfully imported graders.grader_baseline:grade")
except ImportError as e:
    print(f"Failed to import graders.grader_baseline:grade: {e}")

try:
    from graders.grader_easy import grade as grade_easy

    print("Successfully imported graders.grader_easy:grade")
except ImportError as e:
    print(f"Failed to import graders.grader_easy:grade: {e}")

try:
    from graders.grader_medium import grade as grade_medium
    print("Successfully imported graders.grader_medium:grade")
except ImportError as e:
    print(f"Failed to import graders.grader_medium:grade: {e}")

try:
    from graders.grader_hard import grade as grade_hard
    print("Successfully imported graders.grader_hard:grade")
except ImportError as e:
    print(f"Failed to import graders.grader_hard:grade: {e}")
