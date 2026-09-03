# Contributing to LoongSage

[中文](CONTRIBUTING_zh.md)

👍🎉 First off, thanks for taking the time to contribute! 🎉👍

Please read the [Apache Code of Conduct](https://www.apache.org/foundation/policies/conduct.html) before getting started.

We welcome contributions of every size — bug reports, new algorithms, agent integrations, documentation, and ideas. LoongSage is built so that most contributions land as plugins rather than as edits to the framework trunk, so please read [Extending LoongSage](#extending-loongsage) before you start writing code.

## Ways to contribute

- **Report a bug** or **request a feature** through GitHub Issues.
- **Add an algorithm, agent, reward, or sandbox** through the existing extension points — usually no trunk changes required.
- **Improve documentation.** LoongSage ships parallel Chinese and English docs; improvements to either are welcome.
- **Share results or ideas.** Convergence curves, throughput numbers, and reproduction reports on new models are as valuable as code.

## Issues

We use GitHub Issues to track bugs, feature requests, and public discussion.

### Search existing issues first

Before opening a new issue, please search open and closed issues for a similar report. This avoids duplicates and keeps discussion in one place.

### Reporting a new issue

Reinforcement learning post-training fails in ways that depend heavily on the execution mode and the parallelism layout, so please include as much of the following as you can:

- The command you ran, including the config name and any `++key=value` overrides.
- **Parallelism configuration** — TP / PP / EP / CP sizes, and the number of GPUs. This is usually the single most useful detail.
- **Execution mode** — colocated or disaggregated, synchronous or fully asynchronous.
- Versions of Megatron-Core, Megatron-Bridge, SGLang, and Ray, or the container image tag.
- The full traceback, and the log of the rank that actually failed rather than rank 0 only.
- For convergence or accuracy problems: reward and loss curves, and the step at which behaviour started to diverge.

## Pull requests

We strongly welcome pull requests. All of them are reviewed by maintainers, and automated checks run as part of the review. Once checks pass and the review is approved, the pull request will be merged; merging into `main` may be subject to scheduling and is not always immediate.

### Step 1 — Fork and clone

Fork the repository on GitHub, then:

```bash
git clone https://github.com/your-name/LoongSage.git
cd LoongSage

# Track the official repository as upstream
git remote add upstream https://github.com/baidu-baige/LoongSage.git
```

### Step 2 — Create a development branch

```bash
git checkout main
git pull upstream main
git checkout -b feature/your-feature-name
```

### Step 3 — Set up the environment

The container image is the supported development environment; it pins the CUDA, Megatron-Core, and SGLang versions LoongSage is tested against, and applies the SGLang patch during the build.

```bash
docker build -t loongsage/loongsage:latest docker/
```

### Step 4 — Develop, test, and commit

```bash
bash build.sh test                        # full unit test suite with branch coverage
bash build.sh test tests/ut/algorithms    # a single directory or file
```

Coverage reports are written to `output/coverage_html/`. Please add unit tests under `tests/ut/<subpackage>/` for any behaviour you add or change.

Commit messages follow `<type>(<scope>): <description>`:

```bash
git commit -m "feat(algorithms): add token-level GSPO advantage estimator"
```

- **type** — `feat`, `fix`, `refactor`, `perf`, `docs`, `test`, `chore`, `ci`
- **scope** — the subpackage you touched: `agentflow`, `algorithms`, `backends`, `controller`, `data_factory`, `resource_scheduler`, `transfer_mesh`, `agent`, `reward`, `utils`, `conf`, `docs`, `tests`

### Step 5 — Sync with upstream and push

```bash
git pull --rebase upstream main
git push -u origin feature/your-feature-name
```

### Step 6 — Open the pull request

Open a pull request from `your-name/LoongSage:feature/xxx` to `baidu-baige/LoongSage:main`. Describe what changed and why, how you verified it, and — for anything affecting training behaviour — the configuration you validated it with.

## Extending LoongSage

Most contributions should not touch the framework trunk. LoongSage exposes extension points for agents, rewards, sandboxes, advantage estimators, policy losses, divergences, router middleware, data filters, and asynchronous scheduling policies. Adding an implementation to `coda/custom/` makes it selectable by name from configuration — no trunk changes and no registration boilerplate.

Start from the extension points listed in the [README](README.md), then follow the relevant tutorial:

- [Custom extensions](docs/en/custom-extensions.md) · [中文](docs/zh/custom-extensions.md)
- [Custom agent](docs/en/custom-agent.md) · [中文](docs/zh/custom-agent.md)
- [Custom reward](docs/en/custom-reward.md) · [中文](docs/zh/custom-reward.md)
- [Custom sandbox](docs/en/custom-sandbox.md) · [中文](docs/zh/custom-sandbox.md)

If what you need has no extension point, please open an issue to discuss it before implementing. Adding a new extension point is a welcome contribution in itself, and is usually a better outcome than a special case in the trunk.

## Pre-submission checklist

Before opening a pull request, please confirm that:

1. Your branch is based on the latest `main`.
2. `bash build.sh test` passes, and new behaviour is covered by unit tests.
3. New capabilities are added through an extension point rather than by modifying the framework trunk, where possible.
4. Configuration keys you introduce are documented in [`conf/default.yaml`](conf/default.yaml) with a comment explaining their effect.
5. Documentation is updated in **both** `docs/zh/` and `docs/en/` when you change public behaviour. If you cannot write both, say so in the pull request and a maintainer will help.
6. No secrets, internal hostnames, cluster addresses, or absolute paths from your environment are included.
7. No large or generated files are committed — checkpoints, logs, core dumps, or coverage output.
8. Files derived from third-party projects retain their original copyright and license notices, with modification notices added where appropriate.

## License

By contributing to LoongSage, you agree that your original contributions are licensed under the [Apache License 2.0](LICENSE).

Some files in this repository are derived from third-party open-source projects. For those files, contributors must retain the upstream copyright, license, and attribution notices, and add modification notices where required. See the individual file headers for details.
