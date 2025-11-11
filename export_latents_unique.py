# export_latents_unique.py - 只處理不同的胚胎（不重複）
import numpy as np, torch, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset_ivf import IVFSequenceDataset
from model import Model
from sklearn.decomposition import PCA
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def export_and_plot_unique(checkpoint="model_weights.pth", n_unique_cells=50):
    ds = IVFSequenceDataset("index.csv", resize=500, norm="minmax01")
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    
    print(f"載入模型: {checkpoint}")
    model = Model()
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("✅ 模型載入成功")

    print(f"\n尋找 {n_unique_cells} 個不同的胚胎...")
    seen_cells = set()
    
    for vol, cell_id in loader:
        if cell_id[0] in seen_cells:
            continue
        
        print(f"\n處理胚胎 {len(seen_cells)+1}/{n_unique_cells}: {cell_id[0]}")
        seen_cells.add(cell_id[0])
        
        vol = vol.to(DEVICE)
        with torch.no_grad():
            recon, z_seq = model(vol)
        z = z_seq.squeeze(0).cpu().numpy()
        
        # 儲存特徵
        np.save(f"latents_unique/{cell_id[0]}_z.npy", z)
        print(f"  ✅ 儲存特徵")

        # 2D 投影（PCA）
        pca = PCA(n_components=2)
        z2 = pca.fit_transform(z)
        
        plt.figure(figsize=(8, 6))
        plt.plot(z2[:,0], z2[:,1], marker='o', color='blue', alpha=0.6, linewidth=2)
        
        # 添加箭頭
        for t in range(len(z2)-1):
            dx = z2[t+1,0] - z2[t,0]
            dy = z2[t+1,1] - z2[t,1]
            plt.arrow(z2[t,0], z2[t,1], dx, dy, 
                     head_width=0.02, head_length=0.03, 
                     fc='red', ec='red', alpha=0.5,
                     length_includes_head=True)
        
        # 標記起點和終點
        plt.scatter(z2[0,0], z2[0,1], c='green', s=200, marker='o', 
                   label='Start', zorder=5, edgecolors='black', linewidths=2)
        plt.scatter(z2[-1,0], z2[-1,1], c='red', s=200, marker='s', 
                   label='End', zorder=5, edgecolors='black', linewidths=2)
        
        plt.title(f"Latent Trajectory: {cell_id[0]}", fontsize=14, fontweight='bold')
        plt.xlabel('PC1', fontsize=12)
        plt.ylabel('PC2', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"latents_unique/{cell_id[0]}_traj.png", dpi=150)
        plt.close()
        print(f"  ✅ 儲存軌跡圖")

        # 速度曲線
        d = np.linalg.norm(z[1:]-z[:-1], axis=1)
        
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(d)), d, '-o', color='blue', linewidth=2, markersize=8)
        plt.fill_between(range(len(d)), d, alpha=0.3)
        plt.title(f"Development Speed: {cell_id[0]}", fontsize=14, fontweight='bold')
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Speed (||z(t+1)-z(t)||)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 添加平均速度線
        mean_speed = d.mean()
        plt.axhline(y=mean_speed, color='red', linestyle='--', 
                   label=f'Mean: {mean_speed:.4f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"latents_unique/{cell_id[0]}_speed.png", dpi=150)
        plt.close()
        print(f"  ✅ 儲存速度圖")
        
        if len(seen_cells) >= n_unique_cells:
            break
    
    print(f"\n🎉 完成！共處理 {len(seen_cells)} 個不同的胚胎")
    print(f"📁 結果儲存在: latents_unique/")
    print(f"   - 特徵文件: *_z.npy")
    print(f"   - 軌跡圖: *_traj.png")
    print(f"   - 速度圖: *_speed.png")

if __name__ == "__main__":
    Path("latents_unique").mkdir(exist_ok=True)
    export_and_plot_unique(checkpoint="ae_epoch17.pt", n_unique_cells=50)

