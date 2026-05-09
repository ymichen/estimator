# ============================================================================
# 项目: 结构格困难问题安全性评估技术 / MSIS / LWE 评估工具
# 用途: 清理、打包、检查交付内容
# ============================================================================

.PHONY: help clean clean-all dist test-check

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} + > /dev/null 2>&1 || true
	@find . -type f -name "*.pyc" -delete > /dev/null 2>&1 || true
	@find . -type f -name "*.pyo" -delete > /dev/null 2>&1 || true
	@find . -type f -name "*.so" -delete > /dev/null 2>&1 || true
	@find . -type f -name "*.o" -delete > /dev/null 2>&1 || true
	@find . -type f -name "*~" -delete > /dev/null 2>&1 || true
	@find . -type f -name "*.swp" -delete > /dev/null 2>&1 || true
	@find . -type f -name "*.bak" -delete > /dev/null 2>&1 || true
