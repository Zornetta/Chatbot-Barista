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

def analyze_source_and_tests(base_path, definition):
    error_log = []
    processed_layers = []
    
    # Definir las rutas base
    src_base = os.path.join(base_path, 'src')
    tests_base = os.path.join(base_path, 'tests')
    results_base = os.path.join(base_path, 'analysis_results')
    
    # Crear directorios de resultados
    results_src = os.path.join(results_base, 'src')
    results_tests = os.path.join(results_base, 'tests')
    os.makedirs(results_src, exist_ok=True)
    os.makedirs(results_tests, exist_ok=True)

    # Procesar cada capa tanto para src como para tests
    for layer in definition['layers']:
        # Procesar archivos de src
        src_layer_path = os.path.join(src_base, layer)
        if os.path.exists(src_layer_path):
            try:
                # Obtener patrones de exclusión para src
                exclude_patterns = definition.get('exclude', {}).get('src', {}).get(layer, [])
                content = process_layer(src_layer_path, exclude_patterns)
                
                output_file = os.path.join(results_src, f'{layer}.py')
                if write_content_to_file(content, output_file):
                    print(f"Created {output_file}")
                    processed_layers.append(f"src/{layer}")
                else:
                    error_log.append(f"Error writing combined file for src/{layer}")
            except Exception as e:
                error_message = f"Error processing src/{layer}: {str(e)}\n{traceback.format_exc()}"
                print(error_message)
                error_log.append(error_message)

        # Procesar archivos de tests
        test_layer_path = os.path.join(tests_base, layer)
        if os.path.exists(test_layer_path):
            try:
                # Obtener patrones de exclusión para tests
                exclude_patterns = definition.get('exclude', {}).get('tests', {}).get(layer, [])
                content = process_layer(test_layer_path, exclude_patterns)
                
                output_file = os.path.join(results_tests, f'{layer}.py')
                if write_content_to_file(content, output_file):
                    print(f"Created {output_file}")
                    processed_layers.append(f"tests/{layer}")
                else:
                    error_log.append(f"Error writing combined file for tests/{layer}")
            except Exception as e:
                error_message = f"Error processing tests/{layer}: {str(e)}\n{traceback.format_exc()}"
                print(error_message)
                error_log.append(error_message)

    print("\nProcessing Summary:")
    print(f"Results saved in: {results_base}")
    print("Directory structure created:")
    print("  analysis_results/")
    print("    ├── src/")
    for layer in definition['layers']:
        print(f"    │   ├── {layer}.py")
    print("    └── tests/")
    for layer in definition['layers']:
        print(f"        ├── {layer}.py")
    
    print("\nSuccessfully processed:")
    for layer in processed_layers:
        print(f"- {layer}")
    
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
            "src": {},
            "tests": {}
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
    analyze_source_and_tests(base_path, definition)