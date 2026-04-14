"""检查各服务的异常分布"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data_gaia.loaders import RunLoader
import pandas as pd

# 加载异常记录
loader = RunLoader('datasets/GAIA/MicroSS/run/run')
periods = loader.get_anomaly_periods()
df = pd.DataFrame(periods)

print(f"总异常时段数: {len(df)}")
print(f"\n各服务异常数量:")
print(df['service'].value_counts())

print(f"\nwebservice1 占比: {(df['service'] == 'webservice1').sum() / len(df) * 100:.2f}%")
print(f"webservice1 异常数: {(df['service'] == 'webservice1').sum()}")
