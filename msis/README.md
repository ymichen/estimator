# Dilithium MSIS Security Estimator

基于“遗忘 q‑向量”模型（Forgotten q‑Vectors）的 BKZ 攻击代价估算工具，用于评估 CRYSTALS‑Dilithium 数字签名方案中 MSIS / SelfTargetMSIS 问题的经典安全性。

支持三种安全场景的对比：
- **UF‑CMA 无压缩**（公钥包含完整 $\mathbf{t}$）
- **UF‑CMA 带压缩**（仅公开 $\mathbf{t}_1$，符合 Dilithium 实际使用）
- **SUF‑CMA 强不可伪造**（碰撞产生 SIS 解）

根据 Dilithium Round 3 规范（Section 6.2.1, C.3）实现。

---

## 快速开始

### 环境要求
- Python ≥ 3.8
- 无需安装额外依赖（仅使用标准库 `math`, `argparse`, `json`, `importlib`）



# 单个安全等级
```
python -m src.cli --level 2        # 评估 Level 2
python -m src.cli --level 3        # Level 3
python -m src.cli --level 5        # Level 5
```

# 评估所有内置参数（含挑战集 1--, 1-, 5+, 5++）
```python -m src.cli --all
