"""
使用 ExperimentDataManager 进行分析的示例脚本

这个脚本展示了如何在分析流程中使用数据管理器来组织数据
"""
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from experiment_data_manager import ExperimentDataManager

# 假设你已经有了这些函数（从analyze_trajectory_curvature.py）
# from analyze_trajectory_curvature import (
#     load_model, load_latent_vectors_for_video,
#     compute_tphate_3d, compute_pca_3d, compute_curvature
# )


def analyze_embryo_with_manager(
    model_name: str,
    embryo_id: str,
    checkpoint_path: str,
    data_root: str,
    device: str = "cpu",
    method: str = "tphate"
):
    """
    使用数据管理器分析单个embryo
    
    Args:
        model_name: 模型版本名称，如 "v1_baseline"
        embryo_id: Embryo ID，如 "ZS435-5"
        checkpoint_path: 模型checkpoint路径
        data_root: 数据根目录
        device: 计算设备
        method: 降维方法 ("tphate" 或 "pca")
    """
    # 创建数据管理器
    manager = ExperimentDataManager(base_dir="experiments")
    
    # 确保模型版本和embryo目录存在
    try:
        manager.create_model_version(
            model_name,
            model_config={
                "checkpoint": checkpoint_path,
                "device": device,
                "method": method
            },
            description=f"Analysis with {method.upper()}"
        )
    except:
        pass  # 如果已存在则跳过
    
    manager.create_embryo_dir(model_name, embryo_id)
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {embryo_id} with model {model_name}")
    print(f"{'='*60}\n")
    
    # 1. 加载模型和latent vectors
    # model = load_model(checkpoint_path, device)
    # latents, frame_paths = load_latent_vectors_for_video(
    #     embryo_id, model, data_root, device
    # )
    
    # 示例：假设你已经有了latents
    # latents = ...  # [T, latent_dim]
    
    # 2. 保存原始latents
    # manager.save_data(model_name, embryo_id, "latents", latents)
    
    # 3. 计算3D embedding
    # if method == "tphate":
    #     trajectory = compute_tphate_3d(latents, n_components=3)
    #     manager.save_data(model_name, embryo_id, "tphate_3d", trajectory)
    # else:
    #     trajectory = compute_pca_3d(latents, n_components=3)
    #     manager.save_data(model_name, embryo_id, "pca_3d", trajectory)
    
    # 4. 计算curvature
    # curvatures = compute_curvature(trajectory)
    # manager.save_data(model_name, embryo_id, "curvature", curvatures)
    
    # 5. 保存其他metadata到npz
    # high_curvature_indices = np.where(curvatures > np.percentile(curvatures, 95))[0]
    # manager.save_metadata_npz(
    #     model_name, embryo_id,
    #     high_curvature_indices=high_curvature_indices,
    #     max_curvature=np.max(curvatures),
    #     max_curvature_index=np.argmax(curvatures),
    #     threshold=np.percentile(curvatures, 95)
    # )
    
    # 6. 创建并保存plots
    # fig = plot_trajectory_curvature(trajectory, curvatures, embryo_id, None, method.upper())
    # manager.save_plot(model_name, embryo_id, f"trajectory_{method}", fig)
    
    # 7. 生成汇总
    summary = manager.generate_summary(model_name)
    print(f"\n✓ Analysis complete!")
    print(f"  Model: {model_name}")
    print(f"  Embryos: {summary['num_embryos']}")
    
    return manager


def compare_models(
    model_names: list,
    embryo_id: str,
    manager: ExperimentDataManager
):
    """
    比较不同模型版本的结果
    
    Args:
        model_names: 要比较的模型版本列表
        embryo_id: Embryo ID
        manager: 数据管理器实例
    """
    print(f"\n{'='*60}")
    print(f"Comparing models for embryo {embryo_id}")
    print(f"{'='*60}\n")
    
    results = {}
    
    for model_name in model_names:
        try:
            # 加载数据
            curvatures = manager.load_data(model_name, embryo_id, "curvature")
            metadata = manager.load_metadata_npz(model_name, embryo_id)
            
            results[model_name] = {
                "max_curvature": np.max(curvatures),
                "mean_curvature": np.mean(curvatures),
                "high_curvature_count": len(metadata.get("high_curvature_indices", [])),
                "max_curvature_index": np.argmax(curvatures)
            }
            
            print(f"{model_name}:")
            print(f"  Max curvature: {results[model_name]['max_curvature']:.6f}")
            print(f"  Mean curvature: {results[model_name]['mean_curvature']:.6f}")
            print(f"  High curvature frames: {results[model_name]['high_curvature_count']}")
            print()
            
        except FileNotFoundError as e:
            print(f"⚠️  {model_name}: Data not found - {e}")
    
    return results


if __name__ == "__main__":
    # 示例使用
    manager = ExperimentDataManager(base_dir="experiments")
    
    # 列出所有模型版本
    print("Available model versions:")
    for model in manager.list_model_versions():
        print(f"  - {model}")
    
    # 列出特定模型的所有embryo
    model_name = "v1_baseline"
    if model_name in manager.list_model_versions():
        print(f"\nEmbryos in {model_name}:")
        for embryo in manager.list_embryos(model_name):
            print(f"  - {embryo}")
    
    # 比较模型
    # compare_models(["v1_baseline", "v2_no_smooth"], "ZS435-5", manager)

