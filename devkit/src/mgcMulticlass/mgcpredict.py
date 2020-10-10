import numpy as np
import pandas as pd


def make_top_n_df(arr_prob, n=3, pred_labels=None):
    """
    make top n dataframe from prob
    @arr_prob <np.array>: nsamples*nclasses
    @n <int>: ONLY consider the largest n probs
    @pred_labels <list>: the label of classes
    """
    nsamples, nclasses = arr_prob.shape
    top_n_idx = np.argsort(arr_prob, axis=1)[:, ::-1][:, :n]
    if pred_labels is not None:
        pred_labels = [str(i) for i in pred_labels]
    else:
        pred_labels = list(range(nclasses))

    arr_labels = np.repeat(
        pred_labels, nsamples, axis=0).reshape(
            len(pred_labels), nsamples).T
    
    top_n_class = arr_labels[
        np.repeat(np.arange(nsamples), n),
        top_n_idx.ravel()].reshape(nsamples, n)

    top_n_prob = arr_prob[
        np.repeat(np.arange(nsamples), n),
        top_n_idx.ravel()].reshape(nsamples, n)

    class_cols = ["class{}".format(i+1) for i in range(n)]
    prob_cols = ["prob{}".format(i+1) for i in range(n)]
    cols_name =  class_cols + prob_cols
    df_pred_topn = pd.DataFrame(
        np.hstack((top_n_class, top_n_prob)), columns=cols_name)
    df_pred_topn[prob_cols] = df_pred_topn[prob_cols].astype('float32')

    return df_pred_topn
