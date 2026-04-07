import pandas as pd

df = pd.read_csv('../outputs/queue_length.csv')
junctions = ['A0','A1','A2','A3','A4','B0','B1','B2','B3','B4','C0','C1','C2','C3','C4'
             ,'D0','D1','D2','D3','D4','E0','E1','E2','E3','E4']

cols = df.keys()
time_steps = df['Timestep']
junction_sum = {}
for junction in junctions:
    junction_sum[junction] = [0]*900
    for i in range(len(cols)):
        col = cols[i]
        if col[2:4] == junction:
            arr_1 = [float(x) for x in df[col]]
            arr_2 = junction_sum[junction]
            list_sum = [a + b for a, b in zip(arr_1, arr_2)]
            junction_sum[junction] = list_sum


df_final = pd.DataFrame({
        'Timestep': time_steps
    })

for junction in junction_sum.keys():
    df_final[junction] = junction_sum[junction]

df_final.to_csv(f'../junction_data/2.75_18000_20/queue_length_junctions.csv', index=False)