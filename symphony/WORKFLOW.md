---
tracker:
  kind: linear
  project_slug: "dynamic-asr-eval-6cd4f0af50e0"
  api_key: $LINEAR_API_KEY
  active_states:
    - Todo
    - In Progress
  terminal_states:
    - Closed
    - Cancelled
    - Canceled
    - Duplicate
    - Done
polling:
  interval_ms: 30000
workspace:
  root: /exp/exp4/acp21rjf/symphony-workspaces-dynamic-asr-eval
hooks:
  timeout_ms: 120000
  after_create: |
    if [ -f /exp/exp4/acp21rjf/dynamic-asr-eval/symphony/.env ]; then
      set -a
      . /exp/exp4/acp21rjf/dynamic-asr-eval/symphony/.env
      set +a
    fi
    git clone --depth 1 "$SOURCE_REPO_URL" .
    if [ -n "${SOURCE_REF:-}" ]; then
      git fetch --depth 1 origin "$SOURCE_REF"
      git checkout -B symphony-source FETCH_HEAD
    fi
agent:
  max_concurrent_agents: 1
  max_turns: 20
codex:
  command: /home/acp21rjf/.npm-global/bin/codex --config shell_environment_policy.inherit=all app-server
  approval_policy: never
  thread_sandbox: danger-full-access
  turn_sandbox_policy:
    type: dangerFullAccess
---

You are working on Linear issue {{ issue.identifier }} for the
dynamic-asr-eval repository.

Title: {{ issue.title }}
Current status: {{ issue.state }}
URL: {{ issue.url }}

Description:
{% if issue.description %}
{{ issue.description }}
{% else %}
No description provided.
{% endif %}

Linear discussion context:
- Before planning or editing, use the `linear_graphql` tool to fetch this
  issue's recent comments, newest last.
- Use this query shape with `{{ issue.id }}`:
  ```graphql
  query IssueComments($id: String!) {
    issue(id: $id) {
      comments(first: 20) {
        nodes {
          body
          createdAt
          user {
            name
          }
        }
      }
    }
  }
  ```
- Treat recent human comments as part of the current task context, especially
  comments made after the latest completion or blocker comment.
- If a recent human comment asks a question or requests clarification rather
  than implementation, answer it in a Linear comment and do not make code
  changes or move the issue to `Done`.
- If recent comments request rework on an existing PR, inspect the PR or branch
  referenced in the comments before editing.
- In your plan, explicitly state which recent comments changed or constrained
  the task.

Repository orientation:
- `lcasr/` is the main active path for this project. Prefer it unless the issue
  explicitly points to another model family.
- `lcasr/lib.py` contains shared dynamic-evaluation and adaptation helpers and
  loads `../paths.yaml`.
- `lcasr/run_*_eval.py` and `lcasr/enc_dec_*_test.py` are the main Python entry
  points for current evaluation and adaptation workflows.
- `lcasr/launch_scripts/` contains shell launchers for sweeps and repeated
  experiments. Read the relevant launcher before changing a runner or launching
  a job.
- `lcasr/results/*/` contains existing experiment writeups, aggregation scripts,
  plotting scripts, and durable result summaries. Recompute results from
  artifacts rather than hand-editing metrics.
- `lcasr/test_cases/` contains durable regression coverage. The existing tests
  use real checkpoints and real audio when equivalence is the point.
- `lcasr_nemo/`, `nvidia_ctc/`, and `wav2vec2/` are sibling or older model paths.
  Inspect them only when the issue specifically mentions them or a shared
  behavior crosses model families.

Local configuration and artifacts:
- Copy `paths_template.yaml` to `paths.yaml` for local dataset and checkpoint
  paths. `paths.yaml` is ignored and must not be committed.
- Treat `/store/...` datasets, audio, checkpoints, and language-model files as
  read-only inputs unless the issue explicitly says otherwise.
- Do not commit raw audio, checkpoints, pickles, logs, W&B output, credentials,
  or large generated artifacts.
- Keep Symphony-specific instructions and runtime files under `symphony/`.
  The repository is also used by humans, so do not add root-level agent files
  unless an issue explicitly asks for them.

Before editing:
- Inspect the repository state and task context first.
- Make a concise plan.
- Identify validation for the specific change.
- If the issue description includes a line like `Branch/ref: <name>`, treat
  that as the base branch for the work. Fetch and check out that branch/ref
  before making edits.
- Confirm the checked-out commit with `git status`,
  `git rev-parse --abbrev-ref HEAD`, and `git rev-parse HEAD`.
- Create a working branch named `symphony/{{ issue.identifier }}-<short-slug>`
  from the checked-out base branch. Do not commit directly to the base branch.

During work:
- Keep edits narrowly scoped to the issue.
- Prefer existing `lcasr/` patterns, launchers, result directories, and helper
  functions over new abstractions.
- Use structured parsers for structured data when reasonable.
- Record exact commands, configs, checkpoint paths, output paths, and validation
  outcomes for experiment or result changes.
- If a comparison is partial or still running, label it as a snapshot instead of
  presenting it as a final result.
- Do not silently relax regression checks. If an equivalence test fails,
  explain the contract difference.

Experiment launching:
- Do not launch long-running GPU work unless the issue asks for a run.
- Before launching, report the exact command, target GPU, output path, and log
  path.
- Check GPU availability before launch. Do not use a GPU with active compute
  processes.
- Use detached `screen` sessions with log files for long runs.
- If a run crashes, diagnose the log and fix the concrete issue before rerunning
  the same command.

Validation:
- Run the most targeted command or test that demonstrates the task is complete.
- For documentation-only changes, run `git diff --check` and inspect the diff.
- For code changes, run the narrowest relevant script or test case. Prefer real
  checkpoint/audio fixtures when the behavior depends on model equivalence.
- If validation cannot run, document the exact blocker and the command that
  should be run later.

GitHub handoff:
- Commit completed changes on the issue branch.
- Push the branch to `origin`.
- Open a GitHub pull request using
  `/exp/exp4/acp21rjf/scripts/github-create-pr.sh`, using the issue
  `Branch/ref` as the PR base when provided, otherwise the repository default
  branch. Example:
  `/exp/exp4/acp21rjf/scripts/github-create-pr.sh --base <base-branch> --head <pushed-branch> --title "<PR title>" --body-file <pr-body.md>`.
- Include the PR URL in the Linear completion comment.
- If pushing or PR creation fails, do not move the issue to `Done`; post a
  blocker comment with the exact failing command and error.

Linear handoff:
- Use the `linear_graphql` tool for Linear updates.
- Post one completion comment summarizing files changed, validation, output
  paths if any, GitHub PR URL, and residual risk.
- Move the issue to `Done` only when the task is complete and the GitHub handoff
  has succeeded.
- Do not move the issue to `Done` if the requested work is incomplete, blocked,
  not pushed, or missing a PR. In that case, post a blocker comment explaining
  exactly what is missing or failing.
- Before ending, verify with `linear_graphql` that the expected comment exists
  and, for completed work, that the issue state is `Done`.

Final response:
- Summarize what changed.
- Include validation commands and outcomes.
- State blockers clearly.
