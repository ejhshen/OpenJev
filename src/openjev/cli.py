"""Small command entry points. Training and serving load their own dependencies."""

import argparse
import importlib.util
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="openjev")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--model", required=True)
    prepare.add_argument("--backend")
    prepare.add_argument("--revision")
    prepare.add_argument("--output", required=True)
    jevify = commands.add_parser("initialize")
    jevify.add_argument("--backbone", required=True)
    jevify.add_argument("--config")
    jevify.add_argument("--output", required=True)
    jevify.add_argument("--device", default="cpu")
    jevify.add_argument("--dtype", default="bfloat16")
    validate = commands.add_parser("validate-artifact")
    validate.add_argument("path")
    args = parser.parse_args(argv)
    if args.command == "doctor":
        import platform
        print(json.dumps({"python": platform.python_version(), "available": {
            name: importlib.util.find_spec(name) is not None for name in ["torch", "transformers", "safetensors", "verl"]}}, indent=2))
    elif args.command == "prepare":
        from openjev.artifacts import read_json
        from openjev.models.registry import resolve_source
        spec = resolve_source(read_json(Path(args.model) / "config.json"), args.backend)
        result = spec.prepare_source(args.model, args.output, revision=args.revision)
        print(json.dumps({"stage": result["stage"], "parameter_count": result["parameter_count"]}))
    elif args.command == "initialize":
        from openjev.config import OpenJevConfig
        from openjev.models.jevify import jevify as assemble
        config = None
        if args.config:
            text = Path(args.config).read_text()
            if Path(args.config).suffix == ".json":
                raw = json.loads(text)
            else:
                import yaml
                raw = yaml.safe_load(text)
            config = OpenJevConfig.from_dict(raw)
        result = assemble(args.backbone, args.output, config=config, device=args.device, dtype=args.dtype)
        print(json.dumps({"stage": result["stage"], "backend_id": result["backend_id"]}))
    elif args.command == "validate-artifact":
        from openjev.artifacts import load_manifest
        result = load_manifest(args.path, verify_hashes=True)
        print(json.dumps({"stage": result["stage"], "files": len(result["files"]), "verified": True}))


if __name__ == "__main__":
    main()
