import pandas as pd
err = False
df = pd.read_csv('latents.csv')
for (cell_id, time_step), grouped_df in df.groupby(['cell_id','time_step']):
    first_lat_row = grouped_df.iloc[0] # first two entries cell_id and index will always be the same
    for _, lat_row in grouped_df.iterrows():
        if(lat_row != first_lat_row):
            print("Error: ", str(cell_id), "-", str(time_step), " is not serving as a key!")
            err = True
if(not err):
    print("latents.csv is as expected")
    

