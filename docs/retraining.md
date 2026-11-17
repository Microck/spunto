# Retrain on your own catalog

Retraining requires your own reviewed examples, a CPython 3.11 training environment, and a host with enough memory for PyTorch and the selected teacher. Do not run a training job on a live OMP worker. This repository does not provision cloud accounts or start paid compute.

The released data is not public. The scripts retain strict source, split, and review checks rather than substituting unreviewed examples. The authored fixtures under `test/` demonstrate the data structures, but are not training data or evidence of model quality.

## Prepare and review examples

1. Capture a catalog in the format of `schemas/catalog.schema.json`. An enabled extension exposes `/skill-router-export-catalog` with a new absolute output path inside its installation directory. The export contains your local catalog, so review it before sharing.
2. Author contexts and labels using `schemas/dataset.schema.json` and `schemas/contextual-annotation-v2.schema.json`. Follow [the source-data contract](contextual-source-v2.md). Preserve the active request, structured action, source groups, explicit exclusions, and unresolved judgments.
3. Obtain independent reviews and a pinned approval for the source set. Keep review evidence locally with its sealed `proof/` inventory. The validation code checks exact file hashes and schema bindings. Approval is an input from the operator, not a value the trainer invents.
4. Freeze group, temporal, and held-out-family splits before fitting a model. The dataset script's temporal boundaries are explicit constants in `python/build-dataset.py`; review them for your corpus before constructing the source authority. Never change them after inspecting held-out results.

The dataset tool separates authoring, validation, freezing, and export:

```sh
python3.11 python/build-dataset.py extract --help
python3.11 python/build-dataset.py validate --help
python3.11 python/build-dataset.py freeze --help
python3.11 python/build-dataset.py export --help
```

`extract` checks pending authored records without assigning labels. `validate` checks the catalog, schema, review evidence, and externally pinned approval. `freeze` writes the fixed partitioned dataset. `export` writes admitted training pairs, catalog, source references, and their inventories into a new directory. Its pair-context table avoids duplicating long contexts for every skill pair.

The optional `inventory` command indexes local OMP v3 lineage files without exporting raw message text. It is not required for synthetic sources. Keep its local key and index private.

## Set up a training host

`requirements/training.lock` records the Linux x86_64 CPython 3.11 CUDA training closure used by the training pipeline. It is separate from the Linux aarch64 inference lock. Inspect your hardware and wheel compatibility before installing it:

```sh
python3.11 -m venv training-env
training-env/bin/python -m pip install --require-hashes --only-binary=:all: \
  -r requirements/training.lock
```

`requirements/training.in` lists the direct dependency intent. If you need another platform, resolve and test a complete version/hash lock there, then pass that lock to the trainer. Do not edit hashes merely to make installation pass.

## Train and export

From an admitted export directory, supply its exact approval digest and the hash of the current worker. The following command expects your export at `data/export` and your approval digest in the `APPROVAL_SHA256` environment variable:

```sh
WORKER_SHA256="sha256:$(sha256sum python/router-worker.py | cut -d ' ' -f 1)"
training-env/bin/python python/train-router.py \
  --data data/export/audited-pairs.jsonl \
  --dataset-manifest data/export/manifest.json \
  --approval-sha256 "$APPROVAL_SHA256" \
  --runtime-revision "$WORKER_SHA256" \
  --dependency-lock requirements/training.lock \
  --output runs/model-01 \
  --base cross-encoder/ms-marco-MiniLM-L6-v2 \
  --target-precision 0.95 --target-negative-precision 0.95 \
  --min-calibration-support 10 --min-coverage 0.5
```

The trainer validates the admitted export before importing ML dependencies. It uses the pinned base revision, trains the projection and recurrent head with the encoder, fits thresholds on validation only, and exports an unchanged pretrained control plus the tuned FP32 bundle. An optional INT8 export has its own acceptance result. A failed required export remains a visible rejection, not a deployable placeholder.

The default objective is reviewed hard-label binary cross-entropy. To use the pinned Qwen distillation recipe, add all six flags together:

```sh
--teacher-model Qwen/Qwen3-Reranker-0.6B \
--teacher-revision e61197ed45024b0ed8a2d74b80b4d909f1255473 \
--teacher-instruction-sha256 sha256:651211a426279ad97834d82147afa6f150251760ac30a80b630c0b559e2ff4ab \
--teacher-candidate-rule train-reviewed-union-biencoder-top8-qwen-top16-seeded-negative8-v2 \
--distillation-alpha 1.0 --distillation-temperature 2.0
```

Distillation requires `--device cuda`. It constructs candidates from admitted training contexts only, excludes held-out families, and builds the pinned unchanged pretrained retriever in the output directory. Teacher weights are downloaded at a fixed revision. This consumes substantially more memory and compute than inference.

`--resume` accepts only the same output directory, recipe, inputs, runtime, and dependency lock. `--init-weights` with its matching `--init-sha256` supports an explicit safetensors initialization. Neither option silently changes the pretrained control.

## Export the retriever separately

For a hard-label-only run, build a matching retrieval bundle using the same admitted catalog and export manifest:

```sh
training-env/bin/python python/build-retrieval-bundle.py \
  --catalog data/export/catalog.json \
  --plan data/export/manifest.json \
  --output runs/shortlist-bundle
```

This builder is offline. Cache the exact pinned `sentence-transformers/all-MiniLM-L6-v2` snapshot first. It verifies PyTorch/ONNX parity and writes content-addressed embeddings for your catalog. Review the embedding table before publishing a bundle. It contains skill IDs and hashes even when it contains no skill body or description text. A public distribution can use an empty table because the runtime prepares the live descriptions before routing.

## Evaluate before enabling

```sh
training-env/bin/python python/evaluate-router.py --help
```

The evaluator requires an explicit evaluation design and bound observations. `src/router-evaluation.ts` supplies request-level routing metrics and replay checks. Keep thresholds frozen on temporal, unseen-family, and fresh cases. Report unknown labels, abstentions, false fires, warm latency, and cold catalog preparation separately. Pair-level export parity is not end-to-end routing quality.

Only install bundles whose complete inventories and shared source binding verify. Prepare them into a new inert directory with the installer's local `--decision` and `--shortlist` options. Evaluate through that installation before enabling its OMP loader.

## Check a prepared installation

The release probes are authored examples, not benchmark cases. Run both stages and
the host routing path without enabling an OMP loader:

```sh
python3.11 scripts/verify-model.py --model-root /path/to/prepared-installation \
  --runtime-root /path/to/prepared-installation \
  --python /path/to/prepared-installation/venv/bin/python
bun scripts/verify-install.ts /path/to/prepared-installation
```

The probes report actual scores and selections rather than asserting that every
choice is correct. In particular, the review-only release probe can still select
the deployment skill. Keep the distinction between numerical reproducibility and
correct routing.
