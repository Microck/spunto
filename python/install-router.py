"""Prepare an inert local router, then explicitly enable it for new OMP sessions.

The default source is Microck/Spunto-Router-23M on Hugging Face. A Hub install
fetches config.json first, pins its resolved commit and verifies every file hash.
Local --decision and --shortlist bundles remain available for offline installs.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = Path.home() / ".omp/agent/extensions"
LOADER_NAME = "skill-router.ts"
DEFAULT_HUB = "Microck/Spunto-Router-23M"
SOURCES = ["extensions/skill-router.ts", "src/catalog.ts", "src/protocol.ts", "src/router.ts", "src/session.ts",
           "src/worker.ts", "package.json", "bun.lock", "python/router-worker.py", "python/pair-windows.py",
           "python/retrieval.py", "python/retrieval-worker.py", "requirements/inference.lock"]
EMPTY_POLICY = {"version": 1, "rules": [],
                "unsupported": ["Natural-language instruction hierarchy has not been compiled"]}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return "sha256:" + hasher.hexdigest()


def hub_bundles(repo_id, revision=None, *, download=None):
    """Return verified snapshot directories. The injected file provider keeps tests offline."""
    require(revision is None or re.fullmatch(r"[0-9a-f]{40}", revision),
            "--revision must be a full 40-character Hub commit SHA")
    if download is None:
        from huggingface_hub import hf_hub_download
        download = hf_hub_download
    # Do not use snapshot_download or a raw URL here. The normal config request
    # is the Hub's download-count entry point, including its usual cache behavior.
    config_path = Path(download(repo_id=repo_id, filename="config.json", revision=revision or "main"))
    commit = config_path.parent.name
    require(re.fullmatch(r"[0-9a-f]{40}", commit), "Hub cache did not return a pinned snapshot")
    require(revision is None or revision == commit, "Hub returned a different revision")
    config = json.loads(config_path.read_text())
    require(config.get("format") == "spunto-router-v1" and
            config.get("bundles") == {"decision": "decision-bundle", "shortlist": "shortlist-bundle"},
            "Unsupported router config format")
    require(isinstance(config.get("files"), list) and config["files"], "Router file inventory is missing")
    seen = set()
    for entry in config["files"]:
        require(isinstance(entry, dict) and set(entry) == {"path", "bytes", "sha256"}, "Invalid file inventory")
        name = entry["path"]
        require(isinstance(name, str) and re.fullmatch(r"(?:decision|shortlist)-bundle/[A-Za-z0-9_.-]+", name)
                and PurePosixPath(name).name not in (".", "..") and name not in seen, "Unsafe or duplicate bundle path")
        seen.add(name)
        require(type(entry["bytes"]) is int and entry["bytes"] >= 0 and
                isinstance(entry["sha256"], str) and re.fullmatch(r"sha256:[0-9a-f]{64}", entry["sha256"]),
                "Invalid file hash or size")
        path = Path(download(repo_id=repo_id, filename=name, revision=commit))
        require(path.absolute() == (config_path.parent / name).absolute(), "Hub file is outside the pinned snapshot")
        require(path.is_file() and path.stat().st_size == entry["bytes"] and file_hash(path) == entry["sha256"],
                "Bundle file hash or size mismatch: " + name)
    require({"decision-bundle/manifest.json", "shortlist-bundle/manifest.json"} <= seen, "Bundle manifests are missing")
    return (config_path.parent / "decision-bundle", config_path.parent / "shortlist-bundle",
            {"repo_id": repo_id, "revision": commit, "config_sha256": file_hash(config_path)})


def run(*argv, cwd=None):
    subprocess.run(argv, cwd=cwd, check=True)


def preflight(decision, shortlist):
    """Verify both closed bundle inventories before creating an installation."""
    spec = importlib.util.spec_from_file_location("installer_retrieval", ROOT / "python/retrieval.py")
    retrieval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(retrieval)
    manifests = [retrieval.load_decider_manifest(decision), retrieval.load_manifest(shortlist)[0]]
    require(manifests[0]["provenance"]["contextual_binding"] ==
            manifests[1]["provenance"]["contextual_binding"], "Bundles belong to different training releases")
    return manifests


def prepare(args):
    target = args.target.resolve()
    require(not target.exists(), "Target already exists; choose a new directory to preserve rollback")
    require((args.decision is None) == (args.shortlist is None), "Provide both --decision and --shortlist")
    require(not (args.decision and (args.from_hub or args.revision)), "Local bundles cannot use --from-hub or --revision")
    source = None
    if args.decision is None:
        decision, shortlist, source = hub_bundles(args.from_hub or DEFAULT_HUB, args.revision)
    else:
        decision, shortlist = args.decision, args.shortlist
    preflight(decision, shortlist)
    runtime = json.loads(subprocess.check_output([str(args.python), "-c",
        "import json,platform,sys; print(json.dumps([sys.version_info[:2],sys.platform,platform.machine()]))"]))
    require(runtime == [[3, 11], "linux", "aarch64"],
            "The bundled inference lock requires CPython 3.11 on Linux aarch64")
    target.mkdir(parents=True, mode=0o700)
    for relative in SOURCES:
        (target / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target / relative)
    shutil.copytree(ROOT / "schemas", target / "schemas")
    shutil.copytree(decision, target / "decision-bundle")
    shutil.copytree(shortlist, target / "shortlist-bundle")
    run(str(args.python), "-m", "venv", str(target / "venv"))
    run(str(target / "venv/bin/python"), "-m", "pip", "install", "--require-hashes", "--only-binary=:all:",
        "--no-cache-dir", "--disable-pip-version-check", "-r", str(target / "requirements/inference.lock"))
    run(str(args.bun), "install", "--production", "--ignore-scripts", "--frozen-lockfile", cwd=target)
    (target / "policy.json").write_text(json.dumps(EMPTY_POLICY, indent=2) + "\n")
    if source is not None:
        (target / "hub.json").write_text(json.dumps(source, indent=2) + "\n")
    (target / "config.json").write_text(json.dumps({
        "taskRoot": str(target), "pythonPath": str(target / "venv/bin/python"),
        "workerPath": str(target / "python/retrieval-worker.py"), "bundlePath": str(target / "decision-bundle"),
        "shortlistBundlePath": str(target / "shortlist-bundle"), "policyPath": str(target / "policy.json"),
        "bodyBudgetBytes": 65536, "minAvailableMiB": 1024}, indent=2) + "\n")
    (target / "omp-loader.ts").write_text(
        "// Installed by python/install-router.py; remove this file to disable the skill router.\n"
        f'import {{ loadSkillRouter }} from {json.dumps(str(target / "extensions/skill-router.ts"))};\n'
        f'export default loadSkillRouter({json.dumps(str(target / "config.json"))});\n')
    print(f"Prepared {target}. Run enable --target {target} to load it in new OMP sessions.")


def enable(args):
    loader = args.target.resolve() / "omp-loader.ts"
    destination = EXTENSIONS / LOADER_NAME
    require(loader.is_file(), "Prepared loader is missing; run prepare first")
    require(not destination.exists(), "A router loader already exists; disable it before switching")
    EXTENSIONS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(loader, destination)
    print(f"Enabled {destination}. New OMP sessions load the router.")


def disable(_args):
    destination = EXTENSIONS / LOADER_NAME
    require(destination.exists(), "No router loader is enabled")
    os.remove(destination)
    print(f"Disabled: removed {destination}. Restart OMP to unload the router.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    item = sub.add_parser("prepare")
    item.add_argument("--target", type=Path, required=True)
    item.add_argument("--from-hub", help=f"Model repository, default: {DEFAULT_HUB}")
    item.add_argument("--revision", help="Full Hub commit SHA; default resolves main once")
    item.add_argument("--decision", type=Path, help="Local public-format FP32 decider bundle")
    item.add_argument("--shortlist", type=Path, help="Local public-format FP32 retrieval bundle")
    item.add_argument("--python", type=Path, default=Path(shutil.which("python3.11") or sys.executable))
    item.add_argument("--bun", type=Path, default=Path(shutil.which("bun") or "bun"))
    item = sub.add_parser("enable")
    item.add_argument("--target", type=Path, required=True)
    sub.add_parser("disable")
    args = parser.parse_args()
    try:
        {"prepare": prepare, "enable": enable, "disable": disable}[args.command](args)
        return 0
    except KeyboardInterrupt:
        print("Installation interrupted", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Installation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
