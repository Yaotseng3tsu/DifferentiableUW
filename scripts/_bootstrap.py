"""把仓库根目录加入 sys.path，使 scripts/ 下的脚本能 `from src... import ...`。

在每个脚本顶部 `import _bootstrap` 即可（无需安装为 pip 包）。
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# 让 Windows 控制台（可能是 cp932/cp936 代码页）能打印中文而不抛 UnicodeEncodeError。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
