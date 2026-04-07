import pandas as pd


junctions = ['A0','A1','A2','A3','A4','B0','B1','B2','B3','B4','C0','C1','C2','C3','C4'
             ,'D0','D1','D2','D3','D4','E0','E1','E2','E3','E4']
flows = []
features = ['Density','Waiting Time','Time Loss','Speed','Travel Time']
for feature in features:
    final_df = pd.DataFrame()
    for junction in junctions:
        file_name = junction+".csv"
        dir = "../junction_data/2.75_18000_20/"+file_name
        file = pd.read_csv(dir)
        final_df[junction] = file[feature].values
        final_df.to_csv(f"../junction_data/2.75_18000_20/Data/{feature}.csv")
