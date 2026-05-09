# Symphony Agent Files

This directory is the place for repo-local Symphony and future-agent guidance.
The repository is also used directly by humans, so keep agent-specific files in
`symphony/` instead of adding root-level instruction files.

- `WORKFLOW.md` is the canonical future-agent prompt and workflow file.
- Keep local credentials in `symphony/.env`; it is ignored by Git.
- Keep temporary agent artifacts under `symphony/.scratch/`, `symphony/tmp/`, or
  `symphony/logs/`; these are ignored by Git.

Do not commit raw datasets, checkpoints, experiment logs, or large generated
artifacts from agent runs.

Detached experiment callbacks must keep Linear comments below the API limit.
Use `scripts/linear_experiment_callback.py` instead of hand-built callback
posts; it caps log excerpts and the final comment body while leaving full log
paths in the comment for inspection.
