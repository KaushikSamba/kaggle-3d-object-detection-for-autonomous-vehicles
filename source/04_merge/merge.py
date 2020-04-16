from pathlib import Path

import pandas as pd
from tqdm import tqdm


def main():
"""
The idea in this file is to merge the predictions from each of the models in step 2 and 
step 3. Step 2 identifies large objects while Step 3 identifies small ones. Combining 
the detection results of these two detectors should result in an increased score. 
"""

    data1 = pd.read_csv('../02_ssd_large/_submission/submit_model_th0.5_gth0.3_ov0.1.csv')
    data1.index = data1['Id']

    data2 = pd.read_csv('../03_ssd_small/_submission/submit_model_th0.5_gth0.3_ov0.1.csv')
    data2.index = data2['Id']

    sample_submission = pd.read_csv('../../dataset/sample_submission.csv')

    subs = list()

    data1 = data1.fillna('')
    data2 = data2.fillna('')

    for _, row in tqdm(sample_submission.iterrows()):

        s1 = data1.loc[row['Id']]['PredictionString']
        s2 = data2.loc[row['Id']]['PredictionString']

        if s1 != '' and s2 != '':
            s_merged = s1 + ' ' + s2
        elif s1 == '':
            s_merged = s2
        elif s2 == '':
            s_merged = s1
        else:
            s_merged = ''

        subs.append(s_merged)

    dir_sub = Path('_submission')
    dir_sub.mkdir(exist_ok=True)

    sample_submission['PredictionString'] = subs
    sample_submission.to_csv(dir_sub / 'submit_merged.csv', index=False)


if __name__ == '__main__':
    main()
