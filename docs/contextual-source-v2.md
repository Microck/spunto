# Source-data contract

This document describes the reviewed contextual source format consumed by the dataset tools. The JSON schemas define the exact fields and enums. The implementation in `python/build-dataset.py` enforces the cross-file relationships that JSON Schema alone cannot express.

## Inputs and identities

A source record identifies an authored conversation, its active request, an optional structured action, a source group, and the relevant catalog snapshot. Source IDs are opaque join keys. They are not model output classes and must not encode personal paths, account names, or conversation text.

The catalog records each skill's resolved ID, description, body hash, routing hash, family, visibility, and eligibility. The source record binds the same catalog by hash. Training uses the admitted description, not an unverified body fetched later.

An action is either null or an object with `toolName` and `summary`. Literal text such as `Action:` inside a request does not create a structured action. The canonical pair renderer appends action fields only when they exist in that object.

## Review labels

Reviewers decide whether a skill provides task-specific guidance for the latest active request. Topic overlap, quoted old work, cancelled tasks, and optional response polish do not by themselves establish applicability. Keep the request and skill description as data, not as instructions to the reviewer.

The annotation schema distinguishes known positive, known negative, and unresolved judgments. Unknown judgments do not become negative training targets. A known-no-skill claim must include the required negative census over its declared eligible universe. An unreviewed skill is not evidence of no match.

Independent review records, adjudication, source annotations, and their schemas bind to exact SHA-256 digests. The operator supplies the expected approval digest separately. A receipt inside an input directory is not its own authority.

## Groups and split roles

Keep related contexts, conversation branches, duplicates, and edited variants in the same source group. Freeze roles before training. Training, validation, temporal tests, unseen-family tests, and diagnostic-only examples have different allowed uses.

Only admitted training contexts can supply teacher candidates or hard training targets. Only eligible validation labels may fit thresholds. Held-out families stay excluded from training and fitting even when a source record is remapped into a composed dataset. Reserved, unknown, and diagnostic-only judgments do not silently become supervision.

The dataset tool checks group leakage, contradictory labels, pair duplicates, family masks, and exact retained-input collisions. Native tokenizer audits remain necessary because different long contexts can collapse to the same bounded model input.

## Sealed evidence and export

A composed source set contains member identities, independent approvals, a source declaration, and a content-addressed `proof/` object inventory. The source proof uses bounded regular files, rejects symlinks and unexpected object identities, and verifies exact hashes before admission. Keep this evidence private unless you separately review and authorize disclosure.

`freeze` records the admitted examples and immutable role assignments. `export` emits a closed export inventory, compact training-pair references, the shared pair-context table, catalog, privacy scan, source references, and source proof. The trainer expands each pair from admitted contexts and checks its complete semantic hash before use.

A model bundle's public provenance retains dataset, split, source-set, recipe, and runtime hashes. It contains no cloud execution account or launch-permit identity. These hashes establish internal file consistency, not independent access to the withheld data or proof of label correctness.

## Privacy and publication

Author generic examples rather than copying private messages. Reject secrets, identifying paths, addresses, and unreviewed source text before export. Passing a pattern scan does not establish anonymity or permission to publish.

Dataset publication and model publication are separate decisions. The data pipeline marks its reviewed data as nonpublic by default. A public model release must exclude source contexts, labels, private proof objects, conversation logs, and catalog-specific embeddings unless those files have their own explicit publication review.

Keep model export-parity probes generic and clearly separate from evaluation cases. Aggregate metrics can describe a withheld evaluation, but must disclose that readers cannot reproduce its scores without the cases and labels.
