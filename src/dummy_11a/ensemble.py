from common import *

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

## combile classification and segmentation csv
if 1:
    label_csv = ROOT_DIR + '/result/resnet34-cls-full-foldb0-0/submit/test/resnet34-cls-tta-0.50.csv'
    mask_csv = ROOT_DIR + '/result/resnet18-seg-full-softmax-foldb1-1-4balance/submit/test/resnet18-softmax-tta-0.50.csv'

    df_mask = pd.read_csv(mask_csv).fillna('')
    df_label = pd.read_csv(label_csv).fillna('')

    assert (np.all(df_mask['ImageId_ClassId'].values == df_label['ImageId_ClassId'].values))
    print((df_mask.loc[df_label['EncodedPixels'] == '', 'EncodedPixels'] != '').sum())  # 202
    df_mask.loc[df_label['EncodedPixels'] == '', 'EncodedPixels'] = ''

    csv_file = ROOT_DIR + '/result/merge-submission-11a.csv'
    df_mask.to_csv(csv_file, columns=['ImageId_ClassId', 'EncodedPixels'], index=False)
