import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd


def sam_vs_medsam_glottis(metric='dsc'):
    # Create a list to store the data
    data_list = []

    compare = ('1p0n',
              '1p1n',
              '1p3n',
              '2p3n')
    rows = ('SAMv2 1p0n',
            'MedSAM 1p0n',
            'SAMv2 1p1n',
            'MedSAM 1p1n',
            'SAMv2 1p3n',
            'MedSAM 1p3n',
            'SAMv2 2p3n',
            'MedSAM 2p3n')
    columns = ('Average DSC',
               'Median DSC',
               'Average IOU',
               'Median IOU',)



    # Iterate over the files and plot the data as a boxplot
    for folder in compare:
        data_sam = np.load(f'src/BAGLS/SAMv2/videos/results/{folder}/_t_results.npy', allow_pickle=True).item()
        data_medsam = np.load(f'src/BAGLS/SAMv2/videos/results/{folder}/ms2_results.npy', allow_pickle=True).item()

        data_list.append([
            round(data_sam['dsc_mean'],2),
            round(data_sam['dsc_median'],2),
            round(data_sam['iou_mean'],2),
            round(data_sam['iou_median'],2),
        ])
        data_list.append([
            round(data_medsam['dsc_mean'],2),
            round(data_medsam['dsc_median'],2),
            round(data_medsam['iou_mean'],2),
            round(data_medsam['iou_median'],2),
        ])

    # Create a pandas table to display the data
    df = pd.DataFrame(data_list, index=rows, columns=columns)

    print(df)
    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')  
    the_table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc='top', bbox=[0.1, 0.3, 1, 0.5])
    plt.title('SAMv2 vs MedSAM2 Glottis video segmentation')
    plt.show()

if __name__ == "__main__":
    sam_vs_medsam_glottis()