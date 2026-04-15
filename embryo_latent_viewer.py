import pandas as pd
import pyvista as pv
import numpy as np

def visualize_embryo_embeddings(points, string_list):
    cloud = pv.PolyData(points)
    
    cloud.field_data['embryo_metadata'] = string_list

    def callback(picked_mesh):
        idx = cloud.find_closest_point(picked_mesh.points[0])
        
        print(f"--- Selection ---")
        print(f"Index: {idx}")
        print(f"Data:  {string_list[idx]}\n")

    plotter = pv.Plotter()
    plotter.add_mesh(
        cloud, 
        render_points_as_spheres=True, 
        point_size=12.0, 
        color='aquamarine'
    )

    plotter.enable_point_picking(
        callback=callback, 
        show_message="Press 'P' or Click a point to inspect"
    )

    plotter.show()

if __name__ == "__main__":
    T = 100
    sample_points = np.random.normal(size=(T, 3))
    sample_strings = [f"Embryo Frame {i}: Intensity=0.8{i}, Status=Normal" for i in range(T)]

    visualize_embryo_embeddings(sample_points, sample_strings)


