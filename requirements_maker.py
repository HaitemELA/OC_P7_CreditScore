import os
import ast
import subprocess

def extract_libraries_from_file(file_path):
    """
    Extract imported libraries from a Python file.

    Parameters:
    - file_path (str): Path to the Python file.

    Returns:
    - set: Set of imported libraries.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        tree = ast.parse(file.read(), filename=file_path)

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module)

    return imports

def main():
    # Set the root directory
    root_directory = '.'  # Change this to your desired root directory

    # Create a set to store unique libraries
    all_libraries = set()

    # Loop over subdirectories
    for dirpath, _, filenames in os.walk(root_directory):
        for filename in filenames:
            if filename.endswith('.py'):
                file_path = os.path.join(dirpath, filename)
                libraries = extract_libraries_from_file(file_path)
                all_libraries.update(libraries)

    # Write the libraries to the requirements.txt file
    with open('requirements.txt', 'w', encoding='utf-8') as req_file:
        for library in sorted(all_libraries):
            # Use subprocess to get the installed version of each library
            try:
                version = subprocess.check_output(['pip', 'show', library],
                                                 stderr=subprocess.STDOUT,
                                                 text=True).splitlines()
                version = next((line.split(': ')[1] for line in version if line.startswith('Version:')), None)
                req_file.write(f"{library}=={version}\n")
            except subprocess.CalledProcessError:
                req_file.write(f"{library}\n")

    print("Requirements file generated: requirements.txt")

if __name__ == "__main__":
    main()
