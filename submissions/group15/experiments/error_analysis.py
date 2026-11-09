"""Group 15 Q4 analysis. Run from submissions/group15/: python experiments/error_analysis.py

Default: analyze Model C. For the team's selected model, pass --model-name and
ID-keyed CSVs (id,p_ai) or team-order NPY arrays using --val-preds/--train-preds.
An NPY vector alone cannot prove ID alignment; companion *_ids.npy is preferred.
Never analyze validation predictions from a model trained on all 10,000 rows.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import warnings
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from model_c_style import load_csv, load_split, extract_style_features, SEED, TEAM_COMMON


def read_predictions(path, frame):
    path=Path(path)
    if path.suffix=='.npy':
        values=np.load(path,allow_pickle=False)
        if values.shape!=(len(frame),):
            raise ValueError(f'{path}: expected shape {(len(frame),)}, got {values.shape}')
        sidecar=path.with_name(path.stem+'_ids.npy')
        if sidecar.exists():
            ids=np.load(sidecar,allow_pickle=False)
            if ids.shape!=(len(frame),) or len(set(ids.tolist()))!=len(frame):
                raise ValueError(f'{sidecar}: invalid IDs')
            pred=pd.DataFrame({'id':ids,'p_ai':values})
        else:
            warnings.warn(f'{path}: no ID companion; assuming the README-specified row order. This is not an independent order verification.')
            pred=pd.DataFrame({'id':frame.id,'p_ai':values})
    else:
        pred=pd.read_csv(path,float_precision='round_trip')
    if not {'id','p_ai'}<=set(pred.columns) or pred.id.isna().any() or pred.id.duplicated().any():
        raise ValueError('Predictions require unique IDs and p_ai')
    if set(pred.id)!=set(frame.id):
        raise ValueError('Prediction IDs differ from the requested split; regenerate predictions using the TEAM validation file')
    result=frame.merge(pred,on='id',how='left',sort=False,validate='one_to_one')
    if 'y_true' in result and not np.array_equal(result.label,result.y_true):
        raise ValueError('Prediction labels disagree with train.csv')
    if not np.isfinite(result.p_ai).all() or not result.p_ai.between(0,1).all():
        raise ValueError('p_ai must contain finite P(AI) in [0,1], not raw SVM margins')
    return result


def table(frame):
    return frame.to_string(index=False,float_format=lambda x:f'{x:.6f}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default=TEAM_COMMON.DATA_DIR)
    p.add_argument('--val-ids',default=TEAM_COMMON.SPLIT_FILE)
    p.add_argument('--model-name',default='Model C: style Random Forest')
    p.add_argument('--val-preds',default='preds/model_c_val.npy')
    p.add_argument('--train-preds',default='preds/model_c_train.csv')
    p.add_argument('--threshold',type=float,default=.5)
    p.add_argument('--output-prefix',default='error_analysis')
    p.add_argument('--layout-stress',action='store_true',help='Model C only: evaluate the saved development model after collapsing blank lines')
    a=p.parse_args()
    if not 0<a.threshold<1:
        raise ValueError('Threshold must be strictly inside (0,1)')
    fit,val,origin=load_split(load_csv(a.data,'train.csv'),a.val_ids)
    metadata_path=Path('experiments/model_c_metrics.json')
    if Path(a.val_preds).resolve()==Path('preds/model_c_val.npy').resolve() and metadata_path.exists():
        metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
        current=hashlib.sha256(Path(a.val_ids).read_bytes()).hexdigest()
        if metadata['split']['val_id_sha256']!=current:
            raise ValueError('The validation ID file changed after Model C training; rerun model_c_style.py first')
    val=read_predictions(a.val_preds,val)
    train=read_predictions(a.train_preds,fit) if Path(a.train_preds).exists() else None
    if train is None:
        warnings.warn('No training predictions: Q4.1 train-validation gap will be explicitly unavailable')
    val['word_count']=val.text.str.split().str.len()
    val['prediction']=(val.p_ai>=a.threshold).astype(int)
    val['error_type']=np.select([(val.label==0)&(val.prediction==1),
                                (val.label==1)&(val.prediction==0)],['FP','FN'],default='correct')
    val['length_bin']=pd.cut(val.word_count,[0,150,250,np.inf],labels=['short <=150','medium 151-250','long >250'])
    groups=[]
    for name,g in val.groupby('length_bin',observed=True):
        groups.append({'length_bin':str(name),'count':len(g),'human_count':int((g.label==0).sum()),
                       'ai_count':int((g.label==1).sum()),
                       'val_auc':roc_auc_score(g.label,g.p_ai) if g.label.nunique()==2 else np.nan,
                       'FP':int((g.error_type=='FP').sum()),'FN':int((g.error_type=='FN').sum())})
    bins=pd.DataFrame(groups)
    selected=pd.concat([val[val.error_type=='FP'].nlargest(5,'p_ai'),
                        val[val.error_type=='FN'].nsmallest(5,'p_ai')]).copy()
    features=extract_style_features(selected.text)
    selected['newline_count']=selected.text.str.count('\n')
    selected['type_token_ratio']=features.type_token_ratio.to_numpy()
    selected['repeated_trigram_fraction']=features.repeated_trigram_fraction.to_numpy()
    selected['snippet']=selected.text.str.replace(r'\s+',' ',regex=True).str.slice(0,300)
    selected['failure_hypothesis']='Manual source-text review required; style statistics alone do not establish a cause.'
    output=Path('experiments');figures=Path('figures')
    output.mkdir(exist_ok=True);figures.mkdir(exist_ok=True)
    prefix=output/a.output_prefix
    bins.to_csv(str(prefix)+'_length_bins.csv',index=False)
    val.drop(columns='text').to_csv(str(prefix)+'_validation.csv',index=False)
    error_columns=['id','label','error_type','p_ai','word_count','newline_count','type_token_ratio',
                   'repeated_trigram_fraction','snippet','failure_hypothesis']
    selected[error_columns].to_csv(str(prefix)+'_examples.csv',index=False)
    cm=confusion_matrix(val.label,val.prediction,labels=[0,1])
    train_auc=float(roc_auc_score(train.label,train.p_ai)) if train is not None else None
    val_auc=float(roc_auc_score(val.label,val.p_ai))
    summary={'model':a.model_name,'seed':SEED,'split':origin,'threshold':a.threshold,
             'train_auc':train_auc,'val_auc':val_auc,
             'auc_gap':train_auc-val_auc if train_auc is not None else None,
             'TN':int(cm[0,0]),'FP':int(cm[0,1]),'FN':int(cm[1,0]),'TP':int(cm[1,1]),
             'length_bins':'whitespace word count: <=150; 151-250; >250',
             'case_selection':'five highest-P(AI) FP and five lowest-P(AI) FN; confident errors, not a random sample',
             'limitations':['Model-selection validation is not an independent final test.',
                            'No hidden domain/generator labels or test labels are used.']}
    if a.layout_stress:
        if Path(a.val_preds).resolve()!=Path('preds/model_c_val.npy').resolve():
            raise ValueError('--layout-stress is only for the saved Model C validation predictions')
        artifact=joblib.load('models/model_c_dev.joblib')
        if artifact.get('phase')!='development_8000':
            raise ValueError('Stress test must use the 8,000-row development model')
        current=artifact['model'].predict_proba(extract_style_features(val.text))[:,1]
        if not np.allclose(current,val.p_ai,rtol=0,atol=1e-12):
            raise ValueError('Saved Model C and supplied validation predictions differ')
        changed=val.text.map(lambda s:re.sub(r'\n(?:[ \t\r]*\n)+','\n',s))
        probs=artifact['model'].predict_proba(extract_style_features(changed))[:,1]
        flattened=val.text.map(lambda t:TEAM_COMMON.clean_text(t,mode='flatten'))
        flat_probs=artifact['model'].predict_proba(extract_style_features(flattened))[:,1]
        summary['layout_stress']={'operation':'collapse consecutive newlines; words unchanged',
                                  'changed_rows':int((changed!=val.text).sum()),
                                  'val_auc_after':float(roc_auc_score(val.label,probs)),
                                  'common_flatten_operation':'common.clean_text(mode="flatten") on validation text only',
                                  'common_flatten_val_auc':float(roc_auc_score(val.label,flat_probs)),
                                  'interpretation':'layout-sensitivity proxy, not hidden test performance'}
    (Path(str(prefix)+'.json')).write_text(json.dumps(summary,indent=2),encoding='utf-8')
    text=(f'{a.model_name}\nThreshold={a.threshold}; seed={SEED}\n'
          f'Split origin: {origin["origin"]}\n\nQ4.1\n'
          f'Train AUROC={train_auc}; Validation AUROC={val_auc:.6f}; gap={summary["auc_gap"]}\n'
          'Small random-holdout gaps do not prove robustness to unknown sources.\n\nQ4.2\n'+table(bins)+
          '\n\nQ4.3: confident errors; short snippets only\n'+table(selected[error_columns])+
          '\n\nQ4.4\nInspect original passages by the listed IDs before supplying failure hypotheses. '
          'Check whether short texts, recipes/lists, informal posts or biography-like passages recur. '
          'Distinguish observations from speculation. Format sensitivity can be tested by changing only line breaks '
          'while keeping words and the trained model fixed. Such a proxy test is not hidden-domain evaluation.\n'
          +'\nLayout stress result: '+json.dumps(summary.get('layout_stress','not requested'))+'\n')
    Path(str(prefix)+'.txt').write_text(text,encoding='utf-8')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fpr,tpr,_=roc_curve(val.label,val.p_ai)
    fig,ax=plt.subplots(figsize=(6,4))
    ax.plot(fpr,tpr,label=f'Validation AUROC = {val_auc:.5f}',color='#4477aa')
    ax.plot([0,1],[0,1],ls='--',color='gray',lw=1)
    ax.set(xlabel='False positive rate',ylabel='True positive rate',title=a.model_name)
    ax.legend(loc='lower right');fig.tight_layout();fig.savefig(figures/f'{a.output_prefix}_roc.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4))
    ax.bar(bins.length_bin,bins.val_auc,color=['#4477aa','#228833','#b89a20'])
    ax.set(ylim=(.5,1),ylabel='Validation AUROC',title='Validation by text length')
    for i,row in bins.iterrows():
        if np.isfinite(row.val_auc):
            ax.text(i,row.val_auc-.02,f'{row.val_auc:.3f}\nn={row["count"]}',ha='center',va='top',color='white')
    fig.tight_layout();fig.savefig(figures/f'{a.output_prefix}_length.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(4.5,4));ax.imshow(cm,cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Human','AI'],yticklabels=['Human','AI'],
           xlabel='Predicted label',ylabel='True label',title=f'Validation (threshold {a.threshold:g})')
    fig.tight_layout();fig.savefig(figures/f'{a.output_prefix}_confusion.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2));print(table(bins))
    if (selected.error_type=='FP').sum()<5 or (selected.error_type=='FN').sum()<5:
        print('Fewer than five errors of a category: report actual cases; never invent errors.')


if __name__=='__main__':
    main()
