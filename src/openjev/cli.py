"""Small command entry points. Training and serving load their own dependencies."""

import argparse
import importlib.util
import json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="openjev")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    validate = commands.add_parser("validate-artifact")
    validate.add_argument("path")
    args = parser.parse_args(argv)
    if args.command == "doctor":
        import platform
        print(json.dumps({"python": platform.python_version(), "available": {
            name: importlib.util.find_spec(name) is not None for name in ["torch", "transformers", "safetensors", "verl"]}}, indent=2))
    elif args.command == "validate-artifact":
        from openjev.artifacts import load_manifest
        result = load_manifest(args.path, verify_hashes=True)
        print(json.dumps({"stage": result["stage"], "files": len(result["files"]), "verified": True}))


if __name__ == "__main__":
    main()
