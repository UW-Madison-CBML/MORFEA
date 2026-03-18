import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from geometric_features import calculate_curvature
def plot(traj, c, color_name, f_name, title):
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    im = ax.scatter(traj[0], traj[1], traj[2], linestyle='-', marker='', c=c, cmap=None if len(c.shape) == 2 else 'viridis')
    ax.set_xlabel("R1")
    ax.set_ylabel("R2")
    ax.set_zlabel("R3")
    ax.set_title(title)
    
    fig.colorbar(im, ax=ax, label=color_name)
    fig.savefig(f"{f_name}.png")
    plt.close(fig)

def main():
    
    points = np.random.rand(10,3)
    anchors = np.linspace(0,1,10)
    filled_points = np.linspace(0,1,1000)
    plot_x = make_interp_spline(anchors, points[:,0], k=3)(filled_points)
    plot_y = make_interp_spline(anchors, points[:,1], k=3)(filled_points)
    plot_z = make_interp_spline(anchors, points[:,2], k=3)(filled_points)
    trajectory = np.column_stack((plot_x, plot_y, plot_z))
    plot((plot_x,plot_y,plot_z), filled_points, "[a,b]", "time", "Trajectory")

    plot((plot_x,plot_y,plot_z), calculate_curvature(trajectory), "[a,b]", "time", "Trajectory")
    
if __name__ == "__main__":
    main()
