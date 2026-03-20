import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from geometric_features import calculate_curvatures, get_path_sigs, get_acc, get_vel
import umap
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
def plot(traj, c, color_name, f_name, title, axis_label = "R", cmap="viridis"):
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    im = ax.scatter(traj[0], traj[1], traj[2], c=c, cmap=None if len(c.shape) == 2 else cmap)
    ax.set_xlabel(f"{axis_label}1")
    ax.set_ylabel(f"{axis_label}2")
    ax.set_zlabel(f"{axis_label}3")
    ax.set_title(title)
    if(len(c.shape) != 2): 
        fig.colorbar(im, ax=ax, label=color_name)
    fig.savefig(os.path.join("trajs", f"{f_name}.png"))
    plt.close(fig)

def main():
    num_points = 5
    points = np.random.rand(num_points,3)
    anchors = np.linspace(0,1,num_points)
    filled_points = np.linspace(0,1,1000)
    plot_x = make_interp_spline(anchors, points[:,0], k=3)(filled_points)
    plot_y = make_interp_spline(anchors, points[:,1], k=3)(filled_points)
    plot_z = make_interp_spline(anchors, points[:,2], k=3)(filled_points)
    trajectory = np.column_stack((plot_x, plot_y, plot_z))
    plot((plot_x,plot_y,plot_z), filled_points, "[a,b]", "time", "Trajectory")

    plot((plot_x,plot_y,plot_z), calculate_curvatures(trajectory, offset=80, how="triangle"), "Curve", "curve", "Curvature")
    
    sigs = get_path_sigs(trajectory, 7)
    sig_colors = umap.UMAP(n_neighbors=20, n_components=3, random_state=42).fit_transform(sigs)

    scaler = MinMaxScaler()
    normalized_colors = scaler.fit_transform(sig_colors)
    plot((plot_x,plot_y,plot_z), normalized_colors, "Sig", "path_sig", "Path Signatures")
    plot((normalized_colors[:,0],normalized_colors[:,1],normalized_colors[:,2]), filled_points, "Sig over Time", "path_sig_time", "Path Signatures", axis_label="UMAP ")

    kmeds_sigs = KMedoids(n_clusters=num_points, random_state=42).fit(sigs)
    plot((plot_x,plot_y,plot_z), kmeds_sigs.labels_ / num_points, "Sig Cluster Labels", "path_sig_clusters", "Path Signature Clusters", cmap="tab10")

    kmeds_traj = KMedoids(n_clusters=num_points, random_state=42).fit(trajectory)

    plot((plot_x,plot_y,plot_z), kmeds_traj.labels_ / num_points, "Path Cluster Labels", "path_clusters", "Path Clusters", cmap="tab10")
    plot((plot_x,plot_y,plot_z), get_vel(trajectory), "Vel", "vel", "Velocity")
    plot((plot_x,plot_y,plot_z), get_acc(trajectory), "Acc", "acc", "Acceleration")
    
if __name__ == "__main__":
    main()
