# 贡献指南

感谢你对 Anime Gen Pipeline 的关注！欢迎任何形式的贡献。

## 如何贡献

### 报告 Bug

1. 在 [Issues](https://github.com/yourusername/anime-gen-pipeline/issues) 中搜索是否已有相似问题
2. 如果没有, 新建 Issue, 包含:
   - 环境信息 (OS, Python 版本, VRAM, `pip list`)
   - 复现步骤
   - 期望行为 vs 实际行为
   - 错误日志

### 提交 PR

1. Fork 本仓库
2. 创建功能分支: `git checkout -b feat/your-feature`
3. 安装开发依赖: `pip install -e ".[dev]"`
4. 编写代码, 确保:
   ```bash
   ruff check src/   # Lint
   black src/        # 格式化
   ```
5. 提交: `git commit -m "feat: 描述你的改动"`
6. Push 并创建 PR

### 代码风格

- Python 3.10+ 语法
- 函数/类必须有 docstring (Google 风格)
- 类型注解 (type hints)
- 简洁优先

### Commit 规范

使用 [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` 新功能
- `fix:` 修复 Bug
- `docs:` 文档更新
- `perf:` 性能优化
- `refactor:` 重构
