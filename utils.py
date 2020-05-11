from fastai.vision import *
from sklearn.metrics import roc_auc_score

chexpert_targets = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
u_one_features = ['Atelectasis', 'Edema']
u_zero_features = ['Cardiomegaly', 'Consolidation', 'Pleural Effusion']

def add_columns(df, train_valid):
  df['train_valid'] = train_valid
  df['patient'] = df.Path.str.split('/',3,True)[2]
  df  ['study'] = df.Path.str.split('/',4,True)[3]
  return df

def avg_auc_metric(input, targs):
    input=input.detach().cpu()
    targs=targs.detach().cpu().byte()
    auc_scores = []
    for i in range(targs.shape[1]):
      try:
        auc_scores.append(roc_auc_score(targs[:,i],input[:,i]))
      except ValueError:
        pass
    auc_scores = torch.tensor(auc_scores)
    return auc_scores.mean()

def validation_eval(learn, full_valid_df):
    acts = full_valid_df.groupby(['patient','study'])[learn.data.classes].max().values

    valid_preds=learn.get_preds(ds_type=DatasetType.Valid)
    preds = valid_preds[0]
    preds_df = full_valid_df.copy()
    
    print(len(full_valid_df), len(preds))
    for i, c in enumerate(learn.data.classes):
        preds_df[c] = preds[:,i]

    preds = preds_df.groupby(['patient','study'])[learn.data.classes].mean().values

    auc_scores = {learn.data.classes[i]: roc_auc_score(acts[:,i],preds[:,i]) for i in range(len(chexpert_targets))}

    #average results reported in the associated paper
    chexpert_auc_scores = {'Atelectasis':      0.858,
                           'Cardiomegaly':     0.854,
                           'Consolidation':    0.939,
                           'Edema':            0.941,
                           'Pleural Effusion': 0.936}

    max_feat_len = max(map(len, chexpert_targets))

    avg_chexpert_auc = sum(list(chexpert_auc_scores.values()))/len(chexpert_auc_scores.values())
    avg_auc          = sum(list(auc_scores.values()))/len(auc_scores.values())

    [print(f'{k: <{max_feat_len}}\t auc: {auc_scores[k]:.3}\t chexpert auc: {chexpert_auc_scores[k]:.3}\t difference:\
    {(chexpert_auc_scores[k]-auc_scores[k]):.3}') for k in chexpert_targets]

    print(f'\nAverage auc: {avg_auc:.3} \t CheXpert average auc {avg_chexpert_auc:.3}\t Difference {(avg_chexpert_auc-avg_auc):.3}')
