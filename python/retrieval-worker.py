#!/usr/bin/env python3
"""One offline model process hosting retrieval and the unchanged FP32 decider."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

_spec = importlib.util.spec_from_file_location("retrieval", Path(__file__).with_name("retrieval.py"))
retrieval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(retrieval)
boundary = retrieval.boundary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--decision-artifact", type=Path, required=True)
    args = parser.parse_args()
    try:
        decision = retrieval.load_decider_manifest(args.decision_artifact)
        retriever = retrieval.Retriever(args.artifact)
        boundary.require(retriever.manifest["provenance"]["contextual_binding"] ==
                         decision["provenance"]["contextual_binding"])
        decider = boundary.Router(args.decision_artifact)
    except Exception:
        print("backend_unavailable", file=sys.stderr)
        return 1
    while True:
        try:
            request = boundary.read_request(sys.stdin.buffer)
            if request is None:
                return 0
            boundary.validate_identity(request)
        except (ValueError, UnicodeError, OSError):
            return 2
        try:
            boundary.validate_request(request)
            scorer = retriever if request["selectorDigest"] == retriever.manifest["selector_sha256"] else decider
            reply = scorer.route(request)
        except boundary.InvalidInput:
            reply = boundary.response(request, "uncertain", "invalid_response")
        except Exception:
            reply = boundary.response(request, "uncertain", "backend_unavailable")
        try:
            print(json.dumps(reply, separators=(",", ":"), allow_nan=False), flush=True)
        except (BrokenPipeError, OSError):
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
