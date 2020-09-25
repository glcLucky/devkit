- modify model_train_eval/training/mgcpredictx/mgcpredict.py

def mgcLoadNormData(self, **kwargs):

newadd:

df_raw_x_w = df_raw_x_w.reindex(index=dataset_ref['cell_date'].unique())

to keep **same order** with dataset_ref
