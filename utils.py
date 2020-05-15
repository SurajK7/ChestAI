from fastai.vision import *
import pandas as pd
from sklearn.metrics import roc_auc_score
import glob

chexpert_targets = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
u_one_features = ['Atelectasis', 'Edema']
u_zero_features = ['Cardiomegaly', 'Consolidation', 'Pleural Effusion']

def add_patient_study_isvaild_columns(df, train_valid):
  df['train_valid'] = train_valid
  df['patient'] = df.Path.str.split('/',3,True)[2]
  df  ['study'] = df.Path.str.split('/',4,True)[3]
  return df

def create_dfs():
  full_train_df = pd.read_csv('./CheXpert-v1.0-small/train.csv')
  full_train_df = add_patient_study_isvaild_columns(full_train_df, False)
  full_valid_df = pd.read_csv('./CheXpert-v1.0-small/valid.csv')
  full_valid_df = add_patient_study_isvaild_columns(full_valid_df, True)
  return (full_train_df, full_valid_df)

def create_data(full_df, size, bs):
  tfms = get_transforms(do_flip=False, max_zoom=1.0, max_lighting=0.1)
  src = ImageList.from_df(full_df, '.', 'Path').split_from_df('train_valid').label_from_df('feature_string',label_delim=';')
  data = src.transform(tfms, size=size, padding_mode='zeros', resize_method=ResizeMethod.PAD).databunch(bs=bs).normalize(imagenet_stats)
  return data

def validation_eval(learn, full_valid_df):
    acts = full_valid_df.groupby(['patient','study'])[learn.data.classes].max().values
    valid_preds=learn.get_preds(ds_type=DatasetType.Valid)
    preds = valid_preds[0]
    preds_df = full_valid_df.copy()
    
    for i, c in enumerate(learn.data.classes):
        preds_df[c] = preds[:,i]

    preds = preds_df.groupby(['patient','study'])[learn.data.classes].mean().values
    auc_scores = {learn.data.classes[i]: roc_auc_score(acts[:,i],preds[:,i]) for i in range(len(chexpert_targets))}
    avg_auc = sum(list(auc_scores.values()))/len(auc_scores.values())

    return avg_auc

def print_val_score(learn, full_valid_df):
  for path in glob.glob("/content/models/*"):
    model = Path(path).stem
    print(model)
    learn.load(model)
    print(validation_eval(learn, full_valid_df))
