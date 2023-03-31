from pathlib import Path
import sys
import os
import glob
import ast
import argparse


def get_python_files(path):
    for file in Path(path).glob("*.py"):
        if file.is_file() and not file.name.startswith("combine"):
            yield str(file)

def parse_files(files):
    imports = set()
    class_definitions = set()
    node_class_mappings = set()
    functions = set()

    for file in files:
        # read file as lines
        with open(file, "r") as f:
            lines = f.readlines()

        # remove comments
        lines = [line for line in lines if not line.startswith("#")]

        num_lines = len(lines)
        i = 0
        while i < num_lines:
            line = lines[i]
            if line.startswith("import") or line.startswith("from"):
                imports.add(line.strip())

            elif line.startswith("class"):
                class_info = line
                j = i + 1
                while not lines[j].startswith("NODE_CLASS_MAPPINGS"):
                    class_info += lines[j]
                    j += 1
                class_definitions.add(class_info)
                i = j - 1

            elif line.startswith("NODE_CLASS_MAPPINGS"):
                node_class_mappings.add(lines[i+1])

            elif line.startswith("def"):
                function_info = line
                j = i + 1
                while j < num_lines and not lines[j].startswith("NODE_CLASS_MAPPINGS") and not lines[j].startswith("def"):
                    function_info += lines[j]
                    j += 1
                functions.add(function_info)
                i = j - 1

            i += 1

    return imports, class_definitions, node_class_mappings, functions

def write_combined(imports: list[str], class_definitions: list[str], node_class_mappings: list[str], functions: list[str], output_file: str):
    with open(output_file, "w") as f:
        # write imports
        for line in imports:
            f.write(line + "\n")

        # write 2 blank lines
        f.write("\n\n")

        # write class definitions
        for line in class_definitions:
            f.write(line)

        # write functions
        for line in functions:
            f.write(line + "\n")

        # write node class mappings
        f.write("NODE_CLASS_MAPPINGS = {\n")
        for line in node_class_mappings:
            if not line.endswith(",\n"):
                if line.endswith("\n"):
                    line = line[:-1] + ",\n"
                else:
                    line += ",\n"

            f.write(line)
        f.write("}\n")


def main():
    parser = argparse.ArgumentParser(description="Collect unique imports from Python files")
    parser.add_argument("--all", action="store_true", help="Include all Python files in the specified directory")
    parser.add_argument("--files", nargs="+", help="Specify Python files to parse")
    parser.add_argument("--output", default="combined_nodes.py", help="Specify the output file name")
    args = parser.parse_args()

    if args.all:
        args.path = "."
        files = get_python_files(args.path)
        imports, class_definitions, node_class_mappings, functions = parse_files(files)
        write_combined(imports, class_definitions, node_class_mappings, functions, args.output)
    else:
        if args.files is None:
            print("No files specified")
            return

        imports, class_definitions, node_class_mappings, functions = parse_files(args.files)
        write_combined(imports, class_definitions, node_class_mappings, functions, args.output)

if __name__ == "__main__":
    main()