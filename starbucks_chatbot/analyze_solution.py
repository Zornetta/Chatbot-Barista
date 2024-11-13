import os
import sys
import traceback
import random
import json
import re
import argparse

def get_all_files_recursive(directory, extension):
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(extension):
                full_path = os.path.join(root, filename)
                files.append(full_path)
    return files

def should_exclude_file(file_path, exclude_patterns):
    if not exclude_patterns:
        return False
    return any(re.search(pattern['path'], file_path) for pattern in exclude_patterns)

def read_file_content(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            # Agregar el path completo como comentario antes del contenido
            relative_path = os.path.relpath(file_path)
            return f"# File: {relative_path}\n\n{content}\n\n"
    except Exception as e:
        return f"# Error reading file {file_path}: {str(e)}\n"

def write_content_to_file(content, output_file):
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(content)
        return True
    except Exception as e:
        print(f"Error writing to {output_file}: {str(e)}")
        return False

def process_layer(directory, exclude_patterns=None):
    content = ''
    files = get_all_files_recursive(directory, '.py')

    # Ordenar archivos para tener un orden consistente
    files.sort()

    for file_path in files:
        if not should_exclude_file(file_path, exclude_patterns):
            content += read_file_content(file_path)

    return content

def process_directory(directory, exclude_patterns=None):
    content = ''
    if os.path.exists(directory):
        files = get_all_files_recursive(directory, '.py')
        files.sort()

        for file_path in files:
            if not should_exclude_file(file_path, exclude_patterns):
                content += read_file_content(file_path)

    return content

def process_init_file(directory):
    """Procesa el archivo __init__.py de un directorio si existe"""
    init_path = os.path.join(directory, '__init__.py')
    if os.path.exists(init_path):
        return read_file_content(init_path)
    return ''

def analyze_project_structure(base_path, definition):
    error_log = []
    processed_directories = []

    # Definir las rutas base y resultados
    results_base = os.path.join(base_path, 'analysis_results')

    # Estructura de directorios del proyecto
    project_dirs = {
        'src': {
            'path': os.path.join(base_path, 'src'),
            'layers': definition['layers'],
            'exclude_key': 'src',
            'prefix': 'src_'
        },
        'tests': {
            'path': os.path.join(base_path, 'tests'),
            'layers': definition['layers'],
            'exclude_key': 'tests',
            'prefix': 'tests_'
        },
        'data': {
            'path': os.path.join(base_path, 'data'),
            'layers': None,
            'exclude_key': 'data',
            'prefix': ''
        },
        'models': {
            'path': os.path.join(base_path, 'models'),
            'layers': None,
            'exclude_key': 'models',
            'prefix': ''
        },
        'scripts': {
            'path': os.path.join(base_path, 'scripts'),
            'layers': None,
            'exclude_key': 'scripts',
            'prefix': ''
        },
        'docs': {
            'path': os.path.join(base_path, 'docs'),
            'layers': None,
            'exclude_key': 'docs',
            'prefix': ''
        }
    }

    # Procesar cada directorio del proyecto
    for dir_name, dir_config in project_dirs.items():
        if os.path.exists(dir_config['path']):
            results_dir = os.path.join(results_base, dir_name)
            os.makedirs(results_dir, exist_ok=True)

            # Procesar __init__.py del directorio principal si existe
            init_content = process_init_file(dir_config['path'])
            if init_content:
                init_output_file = os.path.join(results_dir, '__init__.py')
                if write_content_to_file(init_content, init_output_file):
                    print(f"Created {init_output_file}")
                    processed_directories.append(f"{dir_name}/__init__.py")
                else:
                    error_log.append(f"Error writing __init__.py for {dir_name}")

            if dir_config['layers']:  # Directorios con capas (src y tests)
                for layer in dir_config['layers']:
                    layer_path = os.path.join(dir_config['path'], layer)
                    if os.path.exists(layer_path):
                        try:
                            exclude_patterns = definition.get('exclude', {}).get(dir_config['exclude_key'], {}).get(layer, [])
                            content = process_layer(layer_path, exclude_patterns)

                            # Add prefix to the output filename
                            output_filename = f"{dir_config['prefix']}{layer}.py"
                            output_file = os.path.join(results_dir, output_filename)

                            if write_content_to_file(content, output_file):
                                print(f"Created {output_file}")
                                processed_directories.append(f"{dir_name}/{output_filename}")
                            else:
                                error_log.append(f"Error writing combined file for {dir_name}/{output_filename}")
                        except Exception as e:
                            error_message = f"Error processing {dir_name}/{layer}: {str(e)}\n{traceback.format_exc()}"
                            print(error_message)
                            error_log.append(error_message)
            else:  # Directorios sin capas
                try:
                    exclude_patterns = definition.get('exclude', {}).get(dir_config['exclude_key'], [])
                    content = process_directory(dir_config['path'], exclude_patterns)

                    if content:  # Solo crear archivo si hay contenido
                        output_file = os.path.join(results_dir, f'{dir_name}_combined.py')
                        if write_content_to_file(content, output_file):
                            print(f"Created {output_file}")
                            processed_directories.append(dir_name)
                        else:
                            error_log.append(f"Error writing combined file for {dir_name}")
                except Exception as e:
                    error_message = f"Error processing {dir_name}: {str(e)}\n{traceback.format_exc()}"
                    print(error_message)
                    error_log.append(error_message)

    # Procesar archivos en la raíz
    root_files = [f for f in os.listdir(base_path) if f.endswith('.py') and
                 os.path.isfile(os.path.join(base_path, f)) and
                 f != 'analyze_solution.py']

    if root_files:
        for file in root_files:
            content = read_file_content(os.path.join(base_path, file))
            output_file = os.path.join(results_base, file)
            if write_content_to_file(content, output_file):
                print(f"Created {output_file}")
                processed_directories.append(file)
            else:
                error_log.append(f"Error writing file {file}")

    print("\nProcessing Summary:")
    print(f"Results saved in: {results_base}")
    print("\nDirectory structure created:")
    print("  analysis_results/")
    print("    ├── src/")
    print("    │   ├── __init__.py")
    for layer in definition['layers']:
        print(f"    │   ├── src_{layer}.py")
    print("    ├── tests/")
    print("    │   ├── __init__.py")
    for layer in definition['layers']:
        print(f"    │   ├── tests_{layer}.py")
    for dir_name in ['data', 'models', 'scripts', 'docs']:
        print(f"    ├── {dir_name}/")
        print(f"    │   ├── __init__.py")
        print(f"    │   └── {dir_name}_combined.py")
    if root_files:
        for file in root_files:
            print(f"    ├── {file}")

    print("\nSuccessfully processed:")
    for directory in processed_directories:
        print(f"- {directory}")

    if error_log:
        print("\nErrors encountered:")
        for error in error_log:
            print(f"- {error}")
    else:
        print("\nNo errors encountered during processing.")

def load_definition(definition_path):
    default_definition = {
        "src_path": "./",
        "layers": [
            "domain",
            "infrastructure",
            "application",
            "interfaces"
        ],
        "exclude": {
            "src": {
                "domain": [
                    # { "path": "pattern" }
                ],
                "infrastructure": [
                    # { "path": "pattern" }
                ],
                "application": [
                    # { "path": "pattern" }
                ],
                "interfaces": [
                    # { "path": "pattern" }
                ]
            },
            "tests": {
                "domain": [
                    # { "path": "pattern" }
                ],
                "infrastructure": [
                    # { "path": "pattern" }
                ],
                "application": [
                    # { "path": "pattern" }
                ],
                "interfaces": [
                    # { "path": "pattern" }
                ]
            },
            "data": [
                # { "path": "pattern" }
            ],
            "models": [
                # { "path": "pattern" }
            ],
            "scripts": [
                # { "path": "pattern" }
            ],
            "docs": [
                # { "path": "pattern" }
            ]
        }
    }

    if definition_path:
        try:
            with open(definition_path, 'r') as f:
                user_definition = json.load(f)
                # Merge user definition with default, preserving user-defined values
                for key, value in user_definition.items():
                    if key in default_definition and isinstance(default_definition[key], dict):
                        default_definition[key].update(value)
                    else:
                        default_definition[key] = value
                return default_definition
        except Exception as e:
            print(f"Error loading definition file: {str(e)}")
            print("Using default definition.")

    return default_definition

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Python project structure")
    parser.add_argument("--base_path", default="./", help="Path to the project base directory")
    parser.add_argument("--definition_path", help="Path to the JSON definition file")
    args = parser.parse_args()

    definition = load_definition(args.definition_path)
    base_path = args.base_path or definition['src_path']

    print(f"Analyzing project in: {base_path}")
    analyze_project_structure(base_path, definition)