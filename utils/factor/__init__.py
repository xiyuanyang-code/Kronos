import os
import importlib
from pathlib import Path
from .base import FactorBase

factor_list = []
# 获取当前目录
current_dir = Path(__file__).parent
# 遍历当前目录下的所有策略文件
for file_path in current_dir.glob('*.py'):
    module_name = file_path.stem
    # 动态导入模块
    module = importlib.import_module(f'.{module_name}', package=__package__)
    # 查找继承自 FactorBase 的类
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and issubclass(attr, FactorBase) and attr != FactorBase:
            factor_list.append(attr)

# 生成策略映射
FACTOR_MAP = dict([(item.name.lower(), item) for item in factor_list])
