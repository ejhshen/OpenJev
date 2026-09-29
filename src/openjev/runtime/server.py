"""Serve a frozen decision artifact using the installed FastAPI/uvicorn."""

import argparse
from contextlib import asynccontextmanager


def create_app(engine_factory):
    from fastapi import FastAPI, HTTPException
    from openjev.api.typesafe import InvalidDecisionRequest

    @asynccontextmanager
    async def lifespan(app):
        app.state.engine = engine_factory()
        yield
        del app.state.engine

    app = FastAPI(title="OpenJev", lifespan=lifespan)

    @app.get("/health")
    def health():
        engine = getattr(app.state, "engine", None)
        if engine is None:
            raise HTTPException(503, "model is loading")
        return {"status": "ready", "model": engine.model_id}

    @app.get("/v1/models")
    def models():
        engine = app.state.engine
        return {"object": "list", "data": [{"id": engine.model_id, "object": "model", "owned_by": "openjev"},
                                            {"id": engine.alias, "object": "model", "root": engine.model_id, "owned_by": "openjev"}]}

    @app.post("/v1/systemone")
    def systemone(body: dict):
        try:
            return app.state.engine.predict(body)
        except InvalidDecisionRequest as exc:
            raise HTTPException(422, str(exc)) from exc

    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("float32", "bfloat16"))
    parser.add_argument("--alias", default="OpenJev-4B")
    parser.add_argument("--branch-microbatch-size", type=int, default=8)
    parser.add_argument("--execution-mode", choices=("auto", "shared_prefix", "expanded"), default="expanded",
                        help="auto uses shared prefixes on supported backends; expanded is the reference path")
    args = parser.parse_args(argv)
    import uvicorn
    from .engine import DecisionEngine

    app = create_app(lambda: DecisionEngine(args.artifact, device=args.device, dtype=args.dtype,
                                            alias=args.alias, branch_microbatch_size=args.branch_microbatch_size,
                                            execution_mode=args.execution_mode))
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
