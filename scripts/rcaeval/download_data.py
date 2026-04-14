"""
下载 RCAEval 数据集

支持下载 RE1、RE2、RE3 的所有子数据集
"""
import sys
import argparse
from pathlib import Path

# 添加 RCAEval 到路径（RCAEval 在 TSFM_Anomaly_Detection 的同级目录）
project_dir = Path(__file__).parent.parent.parent  # TSFM_Anomaly_Detection
rcaeval_path = project_dir.parent / "RCAEval"      # code/RCAEval
sys.path.insert(0, str(rcaeval_path))

from RCAEval.utility import (
    download_re1ob_dataset,
    download_re1ss_dataset,
    download_re1tt_dataset,
    download_re2ob_dataset,
    download_re2ss_dataset,
    download_re2tt_dataset,
    download_re3ob_dataset,
    download_re3ss_dataset,
    download_re3tt_dataset,
)


def download_dataset(dataset_name: str, output_dir: str = None):
    """
    下载指定的 RCAEval 数据集
    
    Args:
        dataset_name: 数据集名称（re1-ob, re2-ob, re2-ss, re2-tt 等）
        output_dir: 输出目录（默认为 datasets/RCAEval）
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "datasets" / "RCAEval"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_name = dataset_name.lower()
    
    download_funcs = {
        're1-ob': download_re1ob_dataset,
        're1-ss': download_re1ss_dataset,
        're1-tt': download_re1tt_dataset,
        're2-ob': download_re2ob_dataset,
        're2-ss': download_re2ss_dataset,
        're2-tt': download_re2tt_dataset,
        're3-ob': download_re3ob_dataset,
        're3-ss': download_re3ss_dataset,
        're3-tt': download_re3tt_dataset,
    }
    
    if dataset_name not in download_funcs:
        print(f"❌ 未知的数据集: {dataset_name}")
        print(f"支持的数据集: {', '.join(download_funcs.keys())}")
        return False
    
    print(f"=" * 80)
    print(f"下载 {dataset_name.upper()} 数据集")
    print(f"输出目录: {output_dir}")
    print(f"=" * 80)
    
    try:
        download_funcs[dataset_name](local_path=str(output_dir))
        print(f"\n✅ {dataset_name.upper()} 数据集下载完成！")
        return True
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="下载 RCAEval 数据集")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="数据集名称（re1-ob, re2-ob, re2-ss, re2-tt 等）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（默认为 datasets/RCAEval）"
    )
    
    args = parser.parse_args()
    
    success = download_dataset(args.dataset, args.output_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
