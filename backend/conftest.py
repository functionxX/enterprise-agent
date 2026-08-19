"""
conftest.py — pytest 全局配置

作用：把 backend/ 加入 sys.path，保证 `import app.*` 在任意
目录执行 pytest 都能解析（CI 从仓库根目录跑测试时的保险）。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
