# 为 LoongSage 贡献代码

[English](CONTRIBUTING.md)

👍🎉 首先，感谢你愿意花时间参与贡献！🎉👍

在开始之前，请先阅读 [Apache 行为准则](https://www.apache.org/foundation/policies/conduct.html)。

我们欢迎任何规模的贡献 —— 缺陷报告、新算法、智能体接入、文档改进，或者只是一个想法。LoongSage 的设计目标之一是让大部分贡献以插件形式落地，而不是修改框架主干，因此在动手写代码前请先阅读[扩展 LoongSage](#扩展-loongsage)。

## 贡献方式

- 通过 GitHub Issues **报告缺陷**或**提出需求**。
- 通过已有扩展点**新增算法、智能体、奖励或沙箱** —— 通常无需改动主干。
- **改进文档**。LoongSage 维护中英双语文档，改进任意一侧都欢迎。
- **分享结果与想法**。收敛曲线、吞吐数据、在新模型上的复现报告，与代码同样有价值。

## Issues

我们使用 GitHub Issues 跟踪缺陷、需求与公开讨论。

### 先搜索已有 Issue

新建之前请先在已开启和已关闭的 Issue 中搜索类似问题，避免重复，也便于讨论集中在一处。

### 提交新 Issue

强化学习后训练的问题往往与执行模式和并行切分强相关，请尽可能提供以下信息：

- 你执行的命令，包括配置名与所有 `++key=value` 覆盖项。
- **并行配置** —— TP / PP / EP / CP 大小与 GPU 数量。这通常是最关键的一条信息。
- **执行模式** —— 同卡还是分离部署，同步还是全异步。
- Megatron-Core、Megatron-Bridge、SGLang、Ray 的版本，或所用镜像的 tag。
- 完整堆栈，以及真正报错的那个 rank 的日志（不要只提供 rank 0）。
- 若是收敛或精度问题：请附上 reward 与 loss 曲线，以及从第几步开始出现偏离。

## Pull Request

我们非常欢迎 Pull Request。所有 PR 都会由维护者评审，并在评审过程中运行自动检查。检查通过且评审通过后 PR 会被合入；合入 `main` 可能受排期影响，不一定立即完成。

### 第 1 步 —— Fork 并克隆

在 GitHub 上 fork 本仓库，然后：

```bash
git clone https://github.com/your-name/LoongSage.git
cd LoongSage

# 将官方仓库添加为 upstream
git remote add upstream https://github.com/baidu-baige/LoongSage.git
```

### 第 2 步 —— 创建开发分支

```bash
git checkout main
git pull upstream main
git checkout -b feature/your-feature-name
```

### 第 3 步 —— 准备环境

推荐使用容器镜像作为开发环境：它固定了 LoongSage 实际测试过的 CUDA、Megatron-Core 与 SGLang 版本，并在构建时应用 SGLang 补丁。

```bash
docker build -t loongsage/loongsage:latest docker/
```

### 第 4 步 —— 开发、测试、提交

```bash
bash build.sh test                        # 全量单测与分支覆盖率
bash build.sh test tests/ut/algorithms    # 指定目录或文件
```

覆盖率报告输出到 `output/coverage_html/`。新增或修改的行为请在 `tests/ut/<子包>/` 下补充单测。

提交信息遵循 `<type>(<scope>): <description>`：

```bash
git commit -m "feat(algorithms): add token-level GSPO advantage estimator"
```

- **type** —— `feat`、`fix`、`refactor`、`perf`、`docs`、`test`、`chore`、`ci`
- **scope** —— 改动所在子包：`agentflow`、`algorithms`、`backends`、`controller`、`data_factory`、`resource_scheduler`、`transfer_mesh`、`agent`、`reward`、`utils`、`conf`、`docs`、`tests`

### 第 5 步 —— 同步 upstream 并推送

```bash
git pull --rebase upstream main
git push -u origin feature/your-feature-name
```

### 第 6 步 —— 发起 Pull Request

从 `your-name/LoongSage:feature/xxx` 向 `baidu-baige/LoongSage:main` 发起 PR。请说明改了什么、为什么改、如何验证；若涉及训练行为，请一并给出验证时所用的配置。

## 扩展 LoongSage

大多数贡献不需要改动框架主干。LoongSage 为智能体、奖励、沙箱、优势估计、策略损失、散度、路由中间件、数据过滤、异步调度策略等提供了扩展点：把实现放进对应目录，即可在配置中按名称选用，无需改动主干，也不需要额外的注册样板代码。

请先看 [README](README.md) 中列出的扩展点，再参考对应教程：

- [自定义智能体](docs/zh/custom-agent.md) · [English](docs/en/custom-agent.md)
- [自定义奖励](docs/zh/custom-reward.md) · [English](docs/en/custom-reward.md)
- [自定义沙箱](docs/zh/custom-sandbox.md) · [English](docs/en/custom-sandbox.md)

如果你需要的能力没有对应扩展点，请先开 Issue 讨论再实现。新增一个扩展点本身就是受欢迎的贡献，通常也比在主干里加特例更好。

## 提交前检查清单

发起 PR 之前，请确认：

1. 分支基于最新的 `main`。
2. `bash build.sh test` 通过，且新增行为有单测覆盖。
3. 新能力尽可能通过扩展点接入，而非修改框架主干。
4. 新增的配置项已在 [`conf/default.yaml`](conf/default.yaml) 中列出，并有注释说明其作用。
5. 涉及对外行为变化时，`docs/zh/` 与 `docs/en/` **两侧**都已更新。若你无法同时提供双语，请在 PR 中说明，维护者会协助补齐。
6. 不包含任何密钥、内网域名、集群地址或你本地环境的绝对路径。
7. 不提交大文件或生成物 —— checkpoint、日志、core dump、覆盖率产物等。
8. 源自第三方项目的文件保留了原有版权与许可声明，并在需要处补充了修改说明。

## 许可证

向 LoongSage 贡献代码，即表示你同意你的原创贡献以 [Apache License 2.0](LICENSE) 授权。

本仓库部分文件源自第三方开源项目。对这些文件，贡献者必须保留上游的版权、许可与署名声明，并在需要处补充修改说明。具体见各文件头部。
