import pandas as pd

junctions = ['A0','A1','A2','A3','A4','B0','B1','B2','B3','B4','C0','C1','C2','C3','C4'
             ,'D0','D1','D2','D3','D4','E0','E1','E2','E3','E4']
flows = []
final_df = pd.DataFrame()
for junction in junctions:
    file_name = junction+"_flow.csv"
    dir = "../junction_data/3.25_7200_40/"+file_name
    file = pd.read_csv(dir)
    final_df[junction] = file['Average Flow'].values[::2]

final_df.to_csv("../junction_data/3.25_7200_40/flows.csv")
