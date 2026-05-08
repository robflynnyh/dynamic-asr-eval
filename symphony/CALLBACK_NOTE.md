# Callback Note

ROB-51 completed its progressive-bottom eval, but the Linear completion
callback failed after the run succeeded. The failure was not in the experiment:
the wrapper had changed directory before exit, then called
`scripts/linear_experiment_callback.py` as a cwd-relative path, so Python looked
for the callback helper under the wrong directory.

ROB-55 had the same failure mode in its running wrapper. The live run was
rescued by adding `lcasr/scripts/linear_experiment_callback.py` as a small
compatibility wrapper that normalizes the log/results paths and delegates to the
real repo-level `scripts/linear_experiment_callback.py`. The ROB-55 launcher was
also patched so future runs `cd "${REPO_ROOT}"` inside `on_exit` before invoking
the callback.

Future-agent rule: always smoke test the callback before queueing any detached
GPU run. Run the callback helper with `--dry-run` from the same cwd the wrapper
will have when its `EXIT` trap fires, using the exact `--log` and `--results`
arguments the wrapper will pass. Do this even if the experiment smoke test
already passed.
