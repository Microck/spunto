# Spunto

Spunto scores conversation and skill-description pairs to help agents select relevant `SKILL.md`-style instructions. Its two-stage model runs on a local CPU and can abstain when evidence is insufficient. The [Spunto-Router-23M model bundles](https://huggingface.co/Microck/Spunto-Router-23M) are agent-agnostic; this repository ships an OMP integration that reads the live catalog and injects selected skill bodies.

## How it works

The extension handles OMP's `context` hook. It sends the active request and eligible skill descriptions to one local Python worker. An `all-MiniLM-L6-v2` bi-encoder ranks the catalog and returns eight candidates. A fine-tuned MiniLM cross-encoder of about 22.7 million parameters with a projection and recurrent decision head scores those candidates. The extension inserts the selected skill bodies into the outgoing context and checks that required instructions are present before a tool call.

The model can select multiple skills, return no match, or abstain. Explicit skill requests and reviewed policy rules remain separate from learned choices. Skill IDs are join keys, not fixed output classes. The worker embeds descriptions from your catalog and caches them in memory. The model download contains no bundled skill bodies or precomputed personal catalog.

## Integrations

OMP (oh-my-pi) is the first shipped integration. Other integrations are welcome; none are included in this release. The model scores pairs rather than OMP-specific skill classes. The supplied extension provides OMP catalog discovery, policy handling, and body injection.

## Requirements

- OMP **18.18.0**. The extension checks this exact version because it uses OMP's host API.
- CPython **3.11** and Bun **1.4.4**.
- Linux **aarch64** for the supplied hash-locked inference wheels. The installer rejects other platforms rather than silently resolving different dependencies.
- At least 1 GiB of available host memory before worker startup. Model files occupy about 183 MB, before the Python environment and JavaScript dependencies.
- Internet access for installation. Routing uses local files only and has no remote inference fallback.

## Install and enable

Clone this repository and enter it, then create a small environment for the installer:

```sh
git clone https://github.com/Microck/spunto.git
cd spunto
python3.11 -m venv .venv
.venv/bin/python -m pip install --require-hashes --only-binary=:all: -r requirements/inference.lock
.venv/bin/python python/install-router.py prepare \
  --target "$HOME/.local/share/spunto" \
  --from-hub Microck/Spunto-Router-23M
.venv/bin/python python/install-router.py enable \
  --target "$HOME/.local/share/spunto"
```

Start a new OMP session after enabling. `prepare` does not change your OMP extensions. `enable` writes only `~/.omp/agent/extensions/skill-router.ts` and refuses to replace an existing loader.

The Hub source is the default, so `--from-hub` is optional. The installer downloads the top-level `config.json` through `hf_hub_download`, resolves one immutable Hub commit, and verifies every bundle file's SHA-256 and byte count before installing it. This follows [Hugging Face's download-counting path](https://huggingface.co/docs/hub/models-download-stats), without a separate analytics request. It records the resolved commit in the installation's `hub.json`.

For a reproducible install, add `--revision` with the full 40-character commit SHA from the model's Hugging Face history. For an offline model source, pass `--decision /path/to/decision-bundle --shortlist /path/to/shortlist-bundle` instead. Local bundles must use this repository's public manifest format. Dependency installation still needs package access or a populated package cache.

## Disable, upgrade, or roll back

```sh
.venv/bin/python python/install-router.py disable
```

Restart OMP to unload the extension. Disabling removes the loader, not your prepared installation or skills.

Prepare an upgrade in a different directory. Keep the previous directory until you have tested the replacement, disable the existing loader, then enable the replacement directory. To roll back, disable it and enable the previous directory again. Never overwrite a prepared directory in place.

## Configure routing

The installation's `config.json` contains absolute paths to its worker, bundles, and policy file. It also sets `bodyBudgetBytes` to 65,536 and `minAvailableMiB` to 1,024. Supported body budgets range from 1,024 to 262,144 bytes; the minimum memory admission setting is 1,024 MiB.

The generated `policy.json` starts with no rules. You can add reviewed required or excluded skill IDs, scoped to an actor and optional tool name. For example:

```json
{
  "version": 1,
  "rules": [{
    "id": "review-before-write",
    "provenance": "Reviewed local policy",
    "scope": {"actorId": null, "toolName": "write"},
    "requiredIds": ["code-review"],
    "excludedIds": []
  }],
  "unsupported": []
}
```

Use IDs that exist in your catalog. A required/excluded conflict or missing required body fails closed. Natural-language rules in other files are not automatically compiled into this policy. The generated policy records that limitation in `unsupported`.

The supported hook covers the main agent and supported child agents. Built-in advisors are outside its scope. The router does not change their configuration, native permissions, or your existing skill files.

## Evaluation

These are historical measurements of the released weights on an authored, private evaluation set. They are not public benchmark scores or guarantees for another catalog.

| Fresh set, 41 requests | Released router | Previous router |
| --- | ---: | ---: |
| Top-1 precision, lower label bound | 0.80 | 0.30 |
| Coverage, lower label bound | 0.69 | 0.31 |
| False fires on known-no-skill requests | 2 | 12 |
| Warm CPU route latency, p95 | 344 ms | Not reported here |

Precision counts known-correct top-1 choices divided by fired routes. Unknown labels do not count as correct in the lower bound. Coverage counts any fire on a known-needed request, including a wrong fire, so it is **not recall of correct skills**. Bounds describe incomplete labels, not statistical confidence intervals.

The router did not meet the project's original 0.95 precision target. It also missed a later 0.75 fresh-coverage floor and zero-false-fire requirement. The main observed limitations were retrieval misses outside the eight-candidate shortlist and confusion about the latest active request, especially when earlier work was quoted or cancelled. More sibling examples did not remove those errors. The [model card](https://huggingface.co/Microck/Spunto-Router-23M) includes development results and training details.

Latency excludes cold model loading and catalog preparation. Public bundles start with an empty description cache. The worker prepares your live catalog under a separate 10-second readiness budget before the 750 ms route budget starts. Large catalogs or long descriptions can exceed readiness and cause abstention. The historical p95 is not a measurement of cold public-package installation.

## Retrain or evaluate your own router

See [the retraining guide](docs/retraining.md) and [source-data contract](docs/contextual-source-v2.md). The repository includes dataset validation, split freezing, training, ONNX export, retrieval export, and offline evaluation tools. It does not include the original training data, review judgments, private skill catalog, cloud account orchestration, or experiment logs. You can train on your own admitted data, but cannot reproduce these weights byte for byte from this repository alone.

## Develop

```sh
bun install --frozen-lockfile --ignore-scripts
bun test test/*.test.ts
bun run typecheck
.venv/bin/python -m pip install jsonschema referencing
.venv/bin/python -B -m unittest discover -s test -p 'test_*.py'
```

The default tests use authored, disposable fixtures and do not require model weights or private run outputs. One optional tiny-model export test needs the training dependencies and otherwise reports an explicit skip. There is no configured lint command.

## Limitations and privacy

- Treat routing as a convenience, not an authorization or safety boundary. A relevant skill can be missed, and an irrelevant skill can fire.
- The decider retains at most 96 recent intent tokens, up to 96 action tokens, and the beginning of each description within a 320-token payload. Important earlier context may be lost. Oversized structured actions cause abstention.
- The base encoders are English-oriented. Some training examples used Spanish, Portuguese, or mixed language, but this is not evidence of broad multilingual reliability.
- Training and review used LLMs. Reviewer agreement is not human ground truth, and correlated reviewer bias remains possible.
- The extension reads local skill bodies for injection. Trust the skills you install. Model selection does not make an untrusted body safe.
- Model and dependency downloads contact their package hosts. Inference has no network access path in the worker protocol and does not send requests or skill text to a service.

## License

The code and model release use [Apache-2.0](LICENSE). Both base models are Apache-2.0. See [NOTICE](NOTICE) and the linked base-model cards for attribution.
