# Agent Eval Harness

An extensible evaluation platform for language models and tool-using agents.

`agent-eval-harness` is being built to answer the questions that ordinary unit tests cannot:

- Did the model produce the right answer in the required structure?
- Did the agent choose the right tools and arguments?
- Was its trajectory efficient, grounded, and free of loops?
- Did a prompt or model change improve quality without unacceptable latency or cost?
- Should a pull request pass the AI quality gate?

## Project status

This repository is under incremental development. The architecture is defined, and implementation will land in focused pull requests with tests and reproducible validation at each milestone.

See [DESIGN.md](DESIGN.md) for the goals, system boundaries, contracts, metric strategy, and delivery plan.

## Planned capabilities

- Model and agent evaluation through distinct evaluators
- Observable agent trajectory tracing
- Exact-match and structured-output checks
- Tool precision, recall, argument accuracy, and unknown-tool detection
- Loop, failure-rate, and step-efficiency metrics
- Configurable LLM-as-a-Judge evaluation
- Repeated runs with score variance and latency percentiles
- Token usage and estimated cost tracking
- Baseline-versus-candidate regression comparison
- Terminal, Markdown, and JSON reporting
- Deterministic good-agent and bad-agent examples
- Docker-based local execution
- GitHub Actions quality gates that do not require paid model credentials

## Architecture at a glance

```text
dataset + configuration
          |
          v
  asynchronous runner
      /           \
model evaluator   agent evaluator -> observable trace
      \           /
       metric engine
            |
 aggregation + regression comparison
            |
 terminal / Markdown / JSON / CI gate
```

Providers and target agents will produce normalized execution results. Metrics will evaluate those results independently, keeping provider SDKs and agent frameworks out of the core evaluation logic.

## Evaluation philosophy

The harness prioritizes deterministic evidence whenever ground truth exists. LLM judges are useful for open-ended qualities such as faithfulness and goal completion, but their scores are probabilistic signals—not objective truth.

Agent evaluation uses observable events such as tool calls, arguments, results, errors, retries, timing, and final answers. It does not depend on hidden chain-of-thought.

## Planned command-line interface

The following commands describe the intended developer experience and will be enabled as their milestones land:

```bash
python cli.py model-eval \
  --dataset datasets/model_golden_dataset.json \
  --model openai/gpt-4.1-mini \
  --runs 3

python cli.py agent-eval \
  --dataset datasets/agent_golden_dataset.json \
  --agent mock-good \
  --threshold 0.85 \
  --json-output artifacts/current.json

python cli.py compare \
  --baseline artifacts/baseline.json \
  --candidate artifacts/current.json
```

The CLI will use stable exit codes:

- `0`: quality gate passed
- `1`: quality gate failed
- `2`: configuration or execution error

## Delivery milestones

1. Architecture and initial documentation
2. Python package and container skeleton
3. Typed schemas and deterministic metrics
4. Async runner, trajectory tracer, and mock-agent evaluation
5. Model providers and LLM-as-a-Judge
6. Regression engine, reporters, and CLI
7. CI quality gates and documentation hardening

Each milestone will be delivered through a focused pull request. Tests and validation will grow with the implementation.

## Guiding constraints

- No paid API key is required for the default test and CI path.
- One failed case must not terminate a complete evaluation run.
- Missing cost or usage data must remain explicit rather than silently becoming zero.
- Stored evaluation artifacts must be versioned and reproducible.
- Provider and agent integrations must depend on stable interfaces.

## Contributing

The project is currently establishing its core architecture. Keep changes focused, typed, testable, and provider-neutral. New behavior should include deterministic tests, and architectural tradeoffs should be recorded in `DESIGN.md`.

## License

A license will be selected before the first stable release.
