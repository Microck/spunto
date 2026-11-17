"""Exercise the Hub file contract using a local file provider, without network or mocks."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("installer", Path(__file__).resolve().parents[1] / "python/install-router.py")
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)
COMMIT = "a" * 40


class SnapshotProvider:
    def __init__(self, root, *, bad_hash=False, bad_path=False):
        self.root = root / "snapshots" / COMMIT
        self.root.mkdir(parents=True)
        self.calls = []
        files = []
        for folder in ("decision-bundle", "shortlist-bundle"):
            path = self.root / folder / "manifest.json"
            path.parent.mkdir()
            path.write_bytes(b"{}\n")
            files.append({"path": folder + "/manifest.json", "bytes": 3,
                          "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()})
        if bad_hash:
            files[0]["sha256"] = "sha256:" + "0" * 64
        if bad_path:
            files[0]["path"] = "../escape"
        (self.root / "config.json").write_text(json.dumps({
            "format": "spunto-router-v1", "bundles": {
                "decision": "decision-bundle", "shortlist": "shortlist-bundle"}, "files": files}))

    def __call__(self, *, repo_id, filename, revision):
        self.calls.append((repo_id, filename, revision))
        return str(self.root / filename)


class HubInstallTests(unittest.TestCase):
    def test_config_download_is_first_and_all_payloads_pin_the_resolved_commit(self):
        with tempfile.TemporaryDirectory() as temporary:
            provider = SnapshotProvider(Path(temporary))
            decision, shortlist, source = installer.hub_bundles("example/router", download=provider)
            self.assertEqual(provider.calls, [
                ("example/router", "config.json", "main"),
                ("example/router", "decision-bundle/manifest.json", COMMIT),
                ("example/router", "shortlist-bundle/manifest.json", COMMIT)])
            self.assertEqual((decision, shortlist), (provider.root / "decision-bundle", provider.root / "shortlist-bundle"))
            self.assertEqual(source, {"repo_id": "example/router", "revision": COMMIT,
                                     "config_sha256": installer.file_hash(provider.root / "config.json")})

    def test_explicit_revision_pins_even_the_config_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            provider = SnapshotProvider(Path(temporary))
            installer.hub_bundles("example/router", COMMIT, download=provider)
            self.assertTrue(all(call[2] == COMMIT for call in provider.calls))

    def test_hash_mismatch_and_unsafe_paths_fail_closed(self):
        for bad_hash, bad_path, message in ((True, False, "hash or size mismatch"), (False, True, "Unsafe")):
            with self.subTest(bad_hash=bad_hash), tempfile.TemporaryDirectory() as temporary:
                provider = SnapshotProvider(Path(temporary), bad_hash=bad_hash, bad_path=bad_path)
                with self.assertRaisesRegex(ValueError, message):
                    installer.hub_bundles("example/router", download=provider)
                self.assertEqual(len(provider.calls), 2 if bad_hash else 1)

    def test_moving_refs_and_wrong_commits_are_not_accepted_as_pins(self):
        with tempfile.TemporaryDirectory() as temporary:
            provider = SnapshotProvider(Path(temporary))
            with self.assertRaisesRegex(ValueError, "full 40-character"):
                installer.hub_bundles("example/router", "main", download=provider)
            self.assertEqual(provider.calls, [])
            with self.assertRaisesRegex(ValueError, "different revision"):
                installer.hub_bundles("example/router", "b" * 40, download=provider)
            self.assertEqual(len(provider.calls), 1)


if __name__ == "__main__":
    unittest.main()
