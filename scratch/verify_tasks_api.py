import json
import sys
import os
sys.path.insert(0, os.getcwd())
from main import get_tasks

tasks_resp = get_tasks()
print(f"Number of tasks: {len(tasks_resp['tasks'])}")
for task in tasks_resp['tasks']:
    print(f"- {task['id']}: {task['grader']}")

# Test if the graders exist in the file system as specified
for task in tasks_resp['tasks']:
    module_path, func_name = task['grader'].split(':')
    try:
        # Simple import test
        parts = module_path.split('.')
        module = __import__(parts[0])
        for part in parts[1:]:
            module = getattr(module, part)
        func = getattr(module, func_name)
        print(f"Verified {task['id']} grader: {func}")
    except Exception as e:
        print(f"Error verifying {task['id']} grader: {e}")
