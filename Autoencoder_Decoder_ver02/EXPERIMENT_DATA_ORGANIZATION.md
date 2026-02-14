# 📁 实验数据组织方案

## 🎯 设计目标

这个数据组织方案旨在支持：
- ✅ **多个模型版本**（ablation study）
- ✅ **多个embryo**
- ✅ **不同类型的数据**（numpy arrays, plots, metadata）
- ✅ **易于检索和比较**
- ✅ **版本控制和可追溯性**

## 📂 目录结构

```
experiments/
├── model_v1_baseline/
│   ├── model_config.json          # 模型配置和描述
│   ├── summary.json               # 所有embryo的汇总
│   ├── embryo_ZS435-5/
│   │   ├── metadata.json          # 该embryo的metadata
│   │   ├── data/
│   │   │   ├── latents.npy        # 原始latent vectors [T, latent_dim]
│   │   │   ├── tphate_3d.npy      # TPHATE 3D embedding [T, 3]
│   │   │   ├── pca_3d.npy         # PCA 3D embedding [T, 3]
│   │   │   ├── curvature.npy      # Curvature values [T]
│   │   │   └── metadata.npz       # 其他metadata (indices, thresholds等)
│   │   └── plots/
│   │       ├── trajectory_tphate.png
│   │       ├── trajectory_pca.png
│   │       ├── curvature_plot.png
│   │       └── high_curvature_frames.png
│   ├── embryo_RS363-7/
│   │   └── ...
│   └── embryo_XXX-XX/
│       └── ...
│
├── model_v2_no_temporal_smooth/
│   ├── model_config.json
│   ├── summary.json
│   └── embryo_ZS435-5/
│       └── ...
│
├── model_v3_different_loss/
│   └── ...
│
└── model_v4_ablation_xyz/
    └── ...
```

## 🚀 使用方法

### 1. 基本使用

```python
from experiment_data_manager import ExperimentDataManager

# 创建管理器
manager = ExperimentDataManager(base_dir="experiments")

# 创建模型版本
manager.create_model_version(
    "v1_baseline",
    model_config={
        "encoder_hidden_dim": 256,
        "decoder_hidden_dim": 128,
        "use_temporal_smooth": True
    },
    description="Baseline model with temporal smoothness"
)

# 创建embryo目录（会自动创建）
# 或者显式创建
manager.create_embryo_dir("v1_baseline", "ZS435-5")
```

### 2. 保存数据

```python
# 保存numpy数组
latents = np.array(...)  # [T, latent_dim]
manager.save_data("v1_baseline", "ZS435-5", "latents", latents)

trajectory = np.array(...)  # [T, 3]
manager.save_data("v1_baseline", "ZS435-5", "tphate_3d", trajectory)

curvatures = np.array(...)  # [T]
manager.save_data("v1_baseline", "ZS435-5", "curvature", curvatures)

# 保存多个数组到一个npz文件
manager.save_metadata_npz(
    "v1_baseline", "ZS435-5",
    high_curvature_indices=indices,
    max_curvature=max_val,
    threshold=threshold
)

# 保存matplotlib图像
fig = plt.figure(...)
manager.save_plot("v1_baseline", "ZS435-5", "trajectory_tphate", fig)
```

### 3. 加载数据

```python
# 加载numpy数组
latents = manager.load_data("v1_baseline", "ZS435-5", "latents")
trajectory = manager.load_data("v1_baseline", "ZS435-5", "tphate_3d")
curvatures = manager.load_data("v1_baseline", "ZS435-5", "curvature")

# 加载metadata.npz
metadata = manager.load_metadata_npz("v1_baseline", "ZS435-5")
high_indices = metadata["high_curvature_indices"]
max_curvature = metadata["max_curvature"]

# 获取embryo的metadata
embryo_meta = manager.get_embryo_metadata("v1_baseline", "ZS435-5")
print(embryo_meta["data_files"])  # 查看有哪些数据文件
print(embryo_meta["plots"])        # 查看有哪些plots
```

### 4. 查询和汇总

```python
# 列出所有模型版本
models = manager.list_model_versions()
print(models)  # ['v1_baseline', 'v2_no_smooth', ...]

# 列出特定模型的所有embryo
embryos = manager.list_embryos("v1_baseline")
print(embryos)  # ['RS363-7', 'ZS435-5', ...]

# 生成模型版本的汇总
summary = manager.generate_summary("v1_baseline")
print(summary)
# {
#   "model_name": "v1_baseline",
#   "num_embryos": 10,
#   "embryos": {
#     "ZS435-5": {
#       "data_files": ["latents.npy", "tphate_3d.npy", ...],
#       "num_plots": 4,
#       "created_at": "2024-01-01T12:00:00"
#     },
#     ...
#   }
# }
```

### 5. 在实际分析脚本中使用

参考 `scripts/analyze_with_data_manager.py` 查看完整示例。

基本流程：
1. 创建/获取模型版本
2. 创建embryo目录
3. 进行分析（计算latents, trajectory, curvature等）
4. 保存所有结果
5. 生成汇总

## 🔄 与现有代码集成

### 修改 `analyze_trajectory_curvature.py`

在保存结果的地方，替换为使用数据管理器：

```python
# 原来的代码
# np.savez(curvature_data_path, ...)
# plt.savefig(plot_path)

# 新的代码
from experiment_data_manager import ExperimentDataManager

manager = ExperimentDataManager()
manager.create_model_version(args.model_name, model_config)
manager.save_data(args.model_name, args.video_name, "latents", latents)
manager.save_data(args.model_name, args.video_name, f"{args.method}_3d", trajectory)
manager.save_data(args.model_name, args.video_name, "curvature", curvatures)
manager.save_plot(args.model_name, args.video_name, f"trajectory_{args.method}", fig)
```

## 📊 数据文件说明

### `latents.npy`
- **形状**: `[T, latent_dim]`
- **内容**: 原始latent vectors，从模型encoder输出
- **用途**: 进一步分析、降维、特征提取

### `tphate_3d.npy` / `pca_3d.npy`
- **形状**: `[T, 3]`
- **内容**: 3D embedding用于可视化
- **用途**: 轨迹可视化、curvature计算

### `curvature.npy`
- **形状**: `[T]`
- **内容**: 每个时间点的curvature值
- **用途**: 识别关键时间点、异常检测

### `metadata.npz`
- **内容**: 多个数组，如：
  - `high_curvature_indices`: 高curvature的帧索引
  - `max_curvature`: 最大curvature值
  - `max_curvature_index`: 最大curvature的帧索引
  - `threshold`: 使用的阈值

### `metadata.json`
- **内容**: 实验配置、文件列表、创建时间等
- **用途**: 追踪实验设置和数据文件

## 🎨 命名约定

### 模型版本命名
- `v1_baseline`: 基线模型
- `v2_no_temporal_smooth`: 移除temporal smoothness loss
- `v3_different_loss`: 不同的loss权重
- `v4_ablation_xyz`: 其他ablation实验

### Embryo ID
- 使用原始ID，如 `ZS435-5`, `RS363-7`
- 保持一致性，便于跨模型比较

### 数据文件命名
- `latents.npy`: 原始latents
- `tphate_3d.npy`: TPHATE 3D embedding
- `pca_3d.npy`: PCA 3D embedding
- `curvature.npy`: Curvature值
- `metadata.npz`: 其他metadata

### Plot命名
- `trajectory_tphate.png`: TPHATE轨迹图
- `trajectory_pca.png`: PCA轨迹图
- `curvature_plot.png`: Curvature曲线图
- `high_curvature_frames.png`: 高curvature帧的montage

## 🔍 比较不同模型版本

```python
# 比较多个模型版本的结果
def compare_models(model_names, embryo_id, manager):
    results = {}
    for model_name in model_names:
        curvatures = manager.load_data(model_name, embryo_id, "curvature")
        results[model_name] = {
            "max": np.max(curvatures),
            "mean": np.mean(curvatures)
        }
    return results

# 使用
results = compare_models(
    ["v1_baseline", "v2_no_smooth", "v3_different_loss"],
    "ZS435-5",
    manager
)
```

## 💡 优势

1. **清晰的层次结构**: 模型版本 → Embryo → 数据类型
2. **易于检索**: 通过API快速查找和加载数据
3. **版本控制友好**: 每个模型版本独立，便于git管理
4. **可扩展**: 容易添加新的数据类型
5. **元数据追踪**: 自动记录文件信息和实验配置
6. **便于比较**: 统一的结构便于跨模型比较

## 📝 注意事项

1. **模型版本命名**: 使用有意义的名称，便于识别
2. **数据一致性**: 确保同一embryo在不同模型版本中使用相同的数据
3. **存储空间**: 大量embryo和模型版本会占用较多空间，考虑定期清理
4. **备份**: 重要实验结果建议定期备份

## 🚀 未来扩展

可以考虑添加：
- 数据库支持（SQLite）用于更复杂的查询
- 自动生成对比报告
- 数据压缩选项
- 远程存储支持（S3, etc.）

