import matplotlib.pyplot as plt
import pandas as pd

df= pd.read_csv('summary_results_fold.csv')
df['model']= df['model'].replace({
    'Sage': 'GraphSAGE',
    'Attentive': 'AttentiveFP',

})
full_set= df.loc[df['features']=='atomic-number_chirality-type_hybridization_xyz', ['model','f1']]
without_coordinates= df.loc[df['features']=='atomic-number_chirality-type_hybridization', ['model','f1']]
without_hybridisation = df.loc[df['features']=='atomic-number_chirality-type_xyz', ['model','f1']]
without_atomic_number = df.loc[df['features']=='chirality-type_hybridization_xyz', ['model','f1']]
without_chirality_type = df.loc[df['features']=='atomic-number_hybridization_xyz', ['model','f1']]
atomic_chirality= df.loc[df['features']=='atomic-number_chirality-type', ['model','f1']]

ablations=['coordinates', 'hybridisation', 'atomic_number', 'chirality_type']
ablation_dfs = {
    'coordinates': without_coordinates,
    'chirality_type': without_chirality_type,
    'hybridisation': without_hybridisation,
    'atomic_number': without_atomic_number
}

for ablation in ablations:
    plt.figure(figsize=(7, 7), dpi=300)
    plt.plot(full_set['model'], full_set['f1'], marker='o', label='full set',markersize=12,)
    plt.plot(
        full_set['model'],
        ablation_dfs[ablation]['f1'],
        marker='^',
        markersize=12,
        label=f'without {ablation}'
    )

    plt.ylabel('F1 score', fontsize=14)
    plt.xticks(rotation=0, fontsize=14)
    plt.legend(fontsize=12)
    plt.show()



