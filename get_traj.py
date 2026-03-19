import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from geometric_features import calculate_curvatures, get_path_sigs, get_acc, get_vel
import umap
import os
from sklearn.preprocessing import MinMaxScaler

def plot(traj, c, color_name, f_name, title):
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    im = ax.scatter(traj[0], traj[1], traj[2], c=c, cmap=None if len(c.shape) == 2 else 'viridis')
    ax.set_xlabel("R1")
    ax.set_ylabel("R2")
    ax.set_zlabel("R3")
    ax.set_title(title)
    if(len(c.shape) != 2): 
        fig.colorbar(im, ax=ax, label=color_name)
    fig.savefig(os.path.join("trajs", f"{f_name}.png"))
    plt.close(fig)

def main():
    num_points = 10
    points = np.random.rand(num_points,3)
    anchors = np.linspace(0,1,num_points)
    filled_points = np.linspace(0,1,1000)
    plot_x = make_interp_spline(anchors, points[:,0], k=3)(filled_points)
    plot_y = make_interp_spline(anchors, points[:,1], k=3)(filled_points)
    plot_z = make_interp_spline(anchors, points[:,2], k=3)(filled_points)
    trajectory = np.column_stack((plot_x, plot_y, plot_z))
    plot((plot_x,plot_y,plot_z), filled_points, "[a,b]", "time", "Trajectory")

    plot((plot_x,plot_y,plot_z), calculate_curvatures(trajectory, offset=80, how="triangle"), "Curve", "curve", "Curvature")
    
    sigs = get_path_sigs(trajectory, 6)
    sig_colors = umap.UMAP(n_neighbors=20, n_components=3, random_state=42).fit_transform(sigs)

    scaler = MinMaxScaler()
    normalized_colors = scaler.fit_transform(sig_colors)
    plot((plot_x,plot_y,plot_z), normalized_colors, "Sig", "path_sig", "Path Signatures")
    plot((normalized_colors[:,0],normalized_colors[:,1],normalized_colors[:,2]), filled_points, "Sig over Time", "path_sig_time", "Path Signatures")

    
    plot((plot_x,plot_y,plot_z), get_vel(trajectory), "Vel", "vel", "Velocity")
    plot((plot_x,plot_y,plot_z), get_acc(trajectory), "Acc", "acc", "Acceleration")
    
if __name__ == "__main__":
    main()
