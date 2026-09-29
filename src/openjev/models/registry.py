"""Explicit, lazy registration. No model weights or CUDA initialization here."""

from importlib import import_module

_BUILTINS = {"qwen35": "openjev.models.qwen35.spec:BACKEND_SPEC"}
_REGISTERED = {}


def register_backend(spec):
    previous = _REGISTERED.get(spec.backend_id)
    if previous is not None and previous != spec:
        raise ValueError(f"conflicting registration: {spec.backend_id}")
    if spec.backend_id in _BUILTINS and previous is None:
        expected = _load_builtin(spec.backend_id)
        if spec != expected:
            raise ValueError(f"cannot override builtin backend: {spec.backend_id}")
    _REGISTERED[spec.backend_id] = spec


def _load_builtin(name):
    module, attribute = _BUILTINS[name].split(":")
    return getattr(import_module(module), attribute)


def backend_ids():
    return sorted(set(_BUILTINS) | set(_REGISTERED))


def get_backend(name, adapter_api_version=1):
    if name not in _REGISTERED:
        if name not in _BUILTINS:
            raise ValueError(f"unknown backend {name!r}; supported: {backend_ids()}")
        _REGISTERED[name] = _load_builtin(name)
    spec = _REGISTERED[name]
    if spec.adapter_api_version != adapter_api_version:
        raise ValueError(f"adapter API mismatch for {name}: artifact={adapter_api_version}, code={spec.adapter_api_version}")
    return spec
