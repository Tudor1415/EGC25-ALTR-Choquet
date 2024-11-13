# maven_utils.py
import subprocess
import os
import sys

def get_maven_classpath(project_root):
    print("Getting classpath from Maven...")
    # mvn_dependency_command = ['mvn', 'dependency:build-classpath', '-Dmdep.outputFile=classpath.txt']
    # try:
    #     subprocess.run(mvn_dependency_command, check=True, cwd=project_root)
    # except subprocess.CalledProcessError as e:
    #     print(f"Failed to generate classpath using Maven: {e}")
    #     sys.exit(1)
    
    classpath_file = os.path.join(project_root, 'classpath.txt')
    if not os.path.exists(classpath_file):
        print("Classpath file not found after Maven execution.")
        sys.exit(1)
    
    with open(classpath_file, 'r') as f:
        maven_classpath = f.read().strip()
    
    return maven_classpath
