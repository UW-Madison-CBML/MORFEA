import pandas as pd

# run this with & python embryo_latent_viewer.py in VS code. Will take a sec to show up

# i am running this on windows not sure how it will work on other os's
# you will need to download latents and annotations to this folder, and eventually the cebra embeddings as well
# reqs: trame trame-vuetify trame-vtk "pyvista[jupyter]" trame 'pyvista[all]' panel:
# streamlit>=1.30.0
# pandas
# numpy
# pillow
# pyvista>=0.40.0
# stpyvista>=0.0.10
# trame
# trame-vuetify
# trame-vtk
# nest_asyncio
# scikit-learn
# umap-learn
import matplotlib.pyplot as plt
import asyncio
import nest_asyncio
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


nest_asyncio.apply()
import streamlit as st
import pyvista as pv
from stpyvista import stpyvista
import numpy as np
from PIL import Image
import streamlit.components.v1 as components
import umap
import os
import random 
from sklearn.decomposition import PCA
def get_annotations_col(group, group_name, group_len, annotations_dir):
    annotation_file = os.path.join(annotations_dir, f"{group_name}_phases.csv")
    df = pd.read_csv(annotation_file, names=['stage_id', 'stage_begin', 'stage_end'])

    new_column = []
    
    new_column += ["pre_phase"] * (df.iloc[0]["stage_begin"] - 1)
    col_len_seq = []
    for index, row in df.iterrows():
        new_column += [row["stage_id"]] * (row["stage_end"] - row["stage_begin"]+1)
        col_len_seq.append(len(new_column))

    


    new_column += ["post_phase"] * (group_len - len(new_column))
    new_column = new_column[:group_len]
    group = group.assign(phase=new_column)
    return group
#-------------------------------------------------------------------------
# do the dataframe stuff
@st.cache_data
def build_df():
    model_name = "control-2026-03-12"
    

    lat_df = pd.read_csv(os.path.abspath(f"latents/{model_name}.csv"), keep_default_na=False).rename(columns={"cell_id":"embryo_id"})
    lat_np = np.load(os.path.abspath(f"latents/{model_name}.npy"))
    if(len(lat_df) != lat_np.shape[0]):
        raise ValueError("keys and values sizes do not match")
    lat_columns = [f"z_{i}" for i in range(lat_np.shape[1])]
    values_df = pd.DataFrame(lat_np, columns=lat_columns, index=lat_df.index)
    df = pd.concat([lat_df, values_df], axis = 1)
    if(os.path.exists(os.path.join("cebra_latents", f"{model_name}.npy"))):
        
        cebra_np = np.load(os.path.join("cebra_latents", f"{model_name}.npy"))
        cebra_columns = [f"cebra_{i}" for i in range(cebra_np.shape[1])]
        cebra_df = pd.DataFrame(cebra_np, columns=cebra_columns, index=lat_df.index)
        df = pd.concat([df, cebra_df], axis = 1)
     
    embryos = random.sample(df["embryo_id"].unique().tolist(),5)
    df = df[df["embryo_id"].str.contains("|".join(embryos),regex=True)]

    trajectories = df[lat_columns].to_numpy()
    pca = PCA(n_components=3)
    pca_trajs = pca.fit_transform(trajectories)

    df = pd.concat([df, pd.DataFrame(pca_trajs, columns=["pca_0", "pca_1", "pca_2"], index=df.index)], axis=1)
    ump = umap.UMAP(n_components=3, n_neighbors=15)
    umap_trajs = ump.fit_transform(trajectories)

    df = pd.concat([df, pd.DataFrame(umap_trajs, columns=["umap_0", "umap_1", "umap_2"], index=df.index)], axis=1)
    df = df.groupby("embryo_id").apply(lambda group: get_annotations_col(group, group.name, len(group), "embryo_dataset_annotations"), include_groups=False).reset_index()
    return df
df = build_df()
df
#---------------------------------------------------------------------------
# now do streamlit


st.set_page_config(layout="wide")
st.title("Embryo Latent Space Explorer")


st.sidebar.header("Visualization Settings")
reduction_method = st.sidebar.selectbox(
    "Dimension Reduction Technique",
    ("PCA", "UMAP", "CEBRA") # also need cebra eventually
)
grade = st.sidebar.multiselect(
    "Grade",
    ("A", "B", "C", "NA") 
)
phase = st.sidebar.multiselect(
    "Phase",
    ('pre_phase', 'tPB2', 'tPNa', 'tPNf', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9+', 'tM','tSB','tB', 'tEB', 'tHB', 'post_phase') 
)
color_by = st.sidebar.selectbox(
    "Color By",
    ("Grade", "Phase", "Time") 
)
num_embryos = st.sidebar.number_input("Number of embryos", min_value=1, step=1)



embryos = random.sample(df["embryo_id"].unique().tolist(),int(num_embryos))
new_df = df[(df["TE"].str.contains("|".join(grade), regex=True)) & (df["phase"].str.contains("|".join(phase), regex=True)) & (df["embryo_id"].str.contains("|".join(embryos), regex=True))]

points = new_df[["pca_0", "pca_1", "pca_2"] if reduction_method == "PCA" else(["umap_0", "umap_1", "umap_2"] if reduction_method == "UMAP" else ["cebra_0", "cebra_1", "cebra_2"])].to_numpy()
time = new_df["time_step"].to_numpy() / new_df["time_step"].max()
grade = np.array([["NA","C","B","A"].index(g) for g in new_df["TE"].to_list()])
phase = np.array([['pre_phase', 'tPB2', 'tPNa', 'tPNf', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9+', 'tM','tSB','tB', 'tEB', 'tHB', 'post_phase'].index(g) for g in new_df["phase"].to_list()])

points
cloud = pv.PolyData(points)
if(color_by == "Phase"):
    cloud['Intensity'] = phase
elif(color_by == "Grade"):
    cloud['Intensity'] = grade
elif(color_by == "Time"):
    cloud['Intensity'] = time 
   



col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"3D Latent Space ({reduction_method})")
    pv.set_jupyter_backend('static') # Static is safer for initial debugging

    
    plotter = pv.Plotter(window_size=[600, 600])
    sphere_source = pv.Sphere(radius=0.1) 

  
    glyphed_cloud = cloud.glyph(geom=sphere_source, scale=False)
    

    if(color_by == "Phase"):
        plotter.add_mesh(
            glyphed_cloud, 
            scalars="Intensity", 
            cmap=plt.get_cmap("tab20c", 18), 
            smooth_shading=True,
            clim=[0, 17],
        )
    elif(color_by == "Grade"):
        plotter.add_mesh(
            glyphed_cloud, 
            scalars="Intensity", 
            cmap=["grey", "red", "yellow", "green"], 
            smooth_shading=True,
            clim=[0, 3],

        )
    elif(color_by == "Time"):
        plotter.add_mesh(
            glyphed_cloud, 
            scalars="Intensity", 
            cmap="magma", 
            smooth_shading=True,
            clim=[0, 1],
        )
    
    
    plotter.background_color = "black"
    
    html_io = plotter.export_html(None) 

   
    html_str = html_io.getvalue()

    
    components.html(html_str, height=500, scrolling=True)

with col2:
    st.subheader("Embryo Inspection")
    
    # Selection slider to simulate "picking" a point since stpyvista is static-interactive
    selected_idx = st.slider("Select Embryo Index", 0, len(points)-1, 0)
    
    st.write(f"**Frame:** {selected_idx}")
    #st.write(f"**Intensity:** {intensities[selected_idx]:.4f}")
    
    
    st.image("https://placeholder.com", caption=f"Embryo {selected_idx} View")

