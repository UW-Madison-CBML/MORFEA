"""
Experiment Data Manager
用于管理多个模型版本和多个embryo的实验数据

目录结构：
experiments/
  model_v1_baseline/
    embryo_ZS435-5/
      data/
        latents.npy
        tphate_3d.npy            # TPHATE 3D embedding [T, 3]
        pca_3d.npy               # PCA 3D embedding [T, 3]
        curvature.npy            # Curvature values [T]
        metadata.npz
      plots/
        trajectory_tphate.png
        trajectory_pca.png
        curvature_plot.png
        high_curvature_frames.png
      metadata.json
    embryo_RS363-7/
      ...
    summary.json
  model_v2_no_temporal_smooth/
    ...
  model_v3_different_loss/
    ...
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import shutil


class ExperimentDataManager:
    
    def __init__(self, base_dir: str = "experiments"):
        """
        Args:
            base_dir: 实验数据的基础目录
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
    
    def create_model_version(
        self,
        model_name: str,
        model_config: Dict[str, Any],
        description: str = ""
    ) -> Path:
        """
        创建一个新的模型版本目录
        
        Args:
            model_name: 模型名称，如 "v1_baseline", "v2_no_smooth"
            model_config: 模型配置字典（会保存到metadata）
            description: 模型描述
        
        Returns:
            模型版本的目录路径
        """
        model_dir = self.base_dir / f"model_{model_name}"
        model_dir.mkdir(exist_ok=True)
        
        config = {
            "model_name": model_name,
            "description": description,
            "config": model_config,
            "created_at": datetime.now().isoformat(),
            "embryos": []
        }
        
        config_path = model_dir / "model_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Created model version: {model_name}")
        print(f"  Directory: {model_dir}")
        
        return model_dir
    
    def create_embryo_dir(
        self,
        model_name: str,
        embryo_id: str
    ) -> Path:
        """
        为特定模型版本创建embryo目录
        
        Args:
            model_name: 模型名称
            embryo_id: Embryo ID，如 "ZS435-5"
        
        Returns:
            Embryo目录路径
        """
        model_dir = self.base_dir / f"model_{model_name}"
        if not model_dir.exists():
            raise ValueError(f"Model version {model_name} does not exist. Create it first.")
        
        embryo_dir = model_dir / f"embryo_{embryo_id}"
        embryo_dir.mkdir(exist_ok=True)
        
        (embryo_dir / "data").mkdir(exist_ok=True)
        (embryo_dir / "plots").mkdir(exist_ok=True)
        
        metadata = {
            "embryo_id": embryo_id,
            "model_name": model_name,
            "created_at": datetime.now().isoformat(),
            "data_files": {},
            "plots": []
        }
        
        metadata_path = embryo_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self._add_embryo_to_model_config(model_name, embryo_id)
        
        return embryo_dir
    
    def save_data(
        self,
        model_name: str,
        embryo_id: str,
        data_name: str,
        data: np.ndarray,
        metadata: Optional[Dict] = None
    ) -> Path:
        """
        保存numpy数据
        
        Args:
            model_name: 模型名称
            embryo_id: Embryo ID
            data_name: 数据名称，如 "latents", "tphate_3d", "curvature"
            data: numpy数组
            metadata: 可选的额外metadata
        
        Returns:
            保存的文件路径
        """
        embryo_dir = self._get_embryo_dir(model_name, embryo_id)
        data_dir = embryo_dir / "data"
        
        file_path = data_dir / f"{data_name}.npy"
        np.save(file_path, data)
        
        self._update_embryo_metadata(
            model_name, embryo_id,
            data_files={data_name: {
                "file": f"{data_name}.npy",
                "shape": list(data.shape),
                "dtype": str(data.dtype),
                "metadata": metadata or {}
            }}
        )
        
        print(f"✓ Saved {data_name} to {file_path}")
        print(f"  Shape: {data.shape}, Dtype: {data.dtype}")
        
        return file_path
    
    def save_plot(
        self,
        model_name: str,
        embryo_id: str,
        plot_name: str,
        fig,
        dpi: int = 150
    ) -> Path:
        """
        保存matplotlib图像
        
        Args:
            model_name: 模型名称
            embryo_id: Embryo ID
            plot_name: 图像名称，如 "trajectory_tphate", "curvature_plot"
            fig: matplotlib figure对象
            dpi: 图像分辨率
        
        Returns:
            保存的文件路径
        """
        embryo_dir = self._get_embryo_dir(model_name, embryo_id)
        plots_dir = embryo_dir / "plots"
        
        file_path = plots_dir / f"{plot_name}.png"
        fig.savefig(file_path, dpi=dpi, bbox_inches='tight')
        
        self._update_embryo_metadata(
            model_name, embryo_id,
            plots=[plot_name]
        )
        
        print(f"✓ Saved plot {plot_name} to {file_path}")
        
        return file_path
    
    def save_metadata_npz(
        self,
        model_name: str,
        embryo_id: str,
        **arrays
    ) -> Path:
        """
        保存多个numpy数组到一个npz文件
        
        Args:
            model_name: 模型名称
            embryo_id: Embryo ID
            **arrays: 要保存的数组，如 curvature=curvature_array, indices=indices_array
        
        Returns:
            保存的文件路径
        """
        embryo_dir = self._get_embryo_dir(model_name, embryo_id)
        data_dir = embryo_dir / "data"
        
        file_path = data_dir / "metadata.npz"
        np.savez(file_path, **arrays)
        
        array_info = {name: {"shape": list(arr.shape), "dtype": str(arr.dtype)}
                     for name, arr in arrays.items()}
        
        self._update_embryo_metadata(
            model_name, embryo_id,
            data_files={"metadata.npz": array_info}
        )
        
        print(f"✓ Saved metadata.npz to {file_path}")
        print(f"  Contains: {list(arrays.keys())}")
        
        return file_path
    
    def load_data(
        self,
        model_name: str,
        embryo_id: str,
        data_name: str
    ) -> np.ndarray:
        """
        加载numpy数据
        
        Args:
            model_name: 模型名称
            embryo_id: Embryo ID
            data_name: 数据名称
        
        Returns:
            numpy数组
        """
        embryo_dir = self._get_embryo_dir(model_name, embryo_id)
        file_path = embryo_dir / "data" / f"{data_name}.npy"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        return np.load(file_path)
    
    def load_metadata_npz(
        self,
        model_name: str,
        embryo_id: str
    ) -> Dict[str, np.ndarray]:
        """
        加载metadata.npz文件
        
        Returns:
            包含所有数组的字典
        """
        embryo_dir = self._get_embryo_dir(model_name, embryo_id)
        file_path = embryo_dir / "data" / "metadata.npz"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {file_path}")
        
        data = np.load(file_path, allow_pickle=True)
        return {key: data[key] for key in data.files}
    
    def get_embryo_metadata(
        self,
        model_name: str,
        embryo_id: str
    ) -> Dict:
        embryo_dir = self._get_embryo_dir(model_name, embryo_id)
        metadata_path = embryo_dir / "metadata.json"
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_model_versions(self) -> List[str]:
        models = [d.name.replace("model_", "") for d in self.base_dir.glob("model_*")
                 if d.is_dir()]
        return sorted(models)
    
    def list_embryos(self, model_name: str) -> List[str]:
        model_dir = self.base_dir / f"model_{model_name}"
        if not model_dir.exists():
            return []
        
        embryos = [d.name.replace("embryo_", "") for d in model_dir.glob("embryo_*")
                  if d.is_dir()]
        return sorted(embryos)
    
    def generate_summary(self, model_name: str) -> Dict:
        """
        生成模型版本的汇总信息
        
        Returns:
            包含所有embryo信息的字典
        """
        model_dir = self.base_dir / f"model_{model_name}"
        if not model_dir.exists():
            raise ValueError(f"Model version {model_name} does not exist")
        
        embryos = self.list_embryos(model_name)
        summary = {
            "model_name": model_name,
            "num_embryos": len(embryos),
            "embryos": {}
        }
        
        for embryo_id in embryos:
            metadata = self.get_embryo_metadata(model_name, embryo_id)
            summary["embryos"][embryo_id] = {
                "data_files": list(metadata.get("data_files", {}).keys()),
                "num_plots": len(metadata.get("plots", [])),
                "created_at": metadata.get("created_at")
            }
        
        summary_path = model_dir / "summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return summary
    
    def _get_embryo_dir(self, model_name: str, embryo_id: str) -> Path:
        model_dir = self.base_dir / f"model_{model_name}"
        if not model_dir.exists():
            raise ValueError(f"Model version {model_name} does not exist")
        
        embryo_dir = model_dir / f"embryo_{embryo_id}"
        if not embryo_dir.exists():
            self.create_embryo_dir(model_name, embryo_id)
        
        return embryo_dir
    
    def _update_embryo_metadata(
        self,
        model_name: str,
        embryo_id: str,
        data_files: Optional[Dict] = None,
        plots: Optional[List] = None
    ):
        embryo_dir = self._get_embryo_dir(model_name, embryo_id)
        metadata_path = embryo_dir / "metadata.json"
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        if data_files:
            metadata.setdefault("data_files", {}).update(data_files)
        
        if plots:
            metadata.setdefault("plots", []).extend(plots)
            metadata["plots"] = list(set(metadata["plots"]))
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def _add_embryo_to_model_config(self, model_name: str, embryo_id: str):
        model_dir = self.base_dir / f"model_{model_name}"
        config_path = model_dir / "model_config.json"
        
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            if embryo_id not in config.get("embryos", []):
                config.setdefault("embryos", []).append(embryo_id)
                config["embryos"].sort()
                
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    manager = ExperimentDataManager(base_dir="experiments")
    
    model_config = {
        "encoder_hidden_dim": 256,
        "decoder_hidden_dim": 128,
        "use_temporal_smooth": True,
        "loss_weights": {"l1": 0.5, "ms_ssim": 0.5}
    }
    manager.create_model_version(
        "v1_baseline",
        model_config,
        description="Baseline model with temporal smoothness"
    )
    
    manager.create_embryo_dir("v1_baseline", "ZS435-5")
    
    # manager.save_data("v1_baseline", "ZS435-5", "latents", latents)
    
    # summary = manager.generate_summary("v1_baseline")
    # print(json.dumps(summary, indent=2))

