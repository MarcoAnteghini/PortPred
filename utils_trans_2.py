#!/usr/bin/env python

import numpy as np
import pandas as pd
import pickle
import os
from Bio import SeqIO
from tqdm.notebook import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import confusion_matrix, f1_score, recall_score, roc_auc_score, fbeta_score, make_scorer, balanced_accuracy_score, accuracy_score, matthews_corrcoef
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

# General utils


def avg_and_std_list(lr):
    res = []
    for x in zip(*lr):
        res.append([sum(x)/len(x), np.std(x)])
    return res


def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn+fp)
    return specificity


# Load the embedded sequence. Each of the following functions return either a dict (proteinID:embeddding_array) or just the embedding.
def load_unirep_embedding(dire, dictio=False):
    emb_list, emb_ids = [], []
    for e in os.listdir(dire):
        # print(e[:-11])
        emb_list.append(np.load(dire+'/'+e))
        ids = e[:-11]
        emb_ids.append(str(ids))
    emb = np.asarray(emb_list)
    if dictio:
        dic = {}
        for i, j in zip(emb_ids, emb):
            dic[i] = j
        return dic, emb
    else:
        return emb


def load_protbert_embedding(dire, dictio=False):
    emb_list, emb_ids = [], []
    for e in os.listdir(dire):
        # print(e[:-11])
        emb_list.append(np.load(dire+'/'+e))
        ids = e[:-13]
        emb_ids.append(str(ids))
    emb = np.asarray(emb_list)
    if dictio:
        dic = {}
        for i, j in zip(emb_ids, emb):
            dic[i] = j
        return dic, emb
    else:
        return emb


def load_esmb1_embedding(dire, dictio=False):
    emb_list, emb_ids = [], []
    for e in os.listdir(dire):
        # print(e[:-11])
        emb_list.append(np.load(dire+'/'+e))
        ids = e[:-10]
        emb_ids.append(str(ids))
    emb = np.asarray(emb_list)
    if dictio:
        dic = {}
        for i, j in zip(emb_ids, emb):
            dic[i] = j
        return dic, emb
    else:
        return emb


def load_seqvec_embedding(dire, dictio=False):
    emb_list, emb_ids = [], []
    for e in os.listdir(dire):
        # print(e[:-11])
        emb_list.append(np.load(dire+'/'+e))
        ids = e[:-11]
        emb_ids.append(str(ids))
    emb = np.asarray(emb_list)
    if dictio:
        dic = {}
        for i, j in zip(emb_ids, emb):
            dic[i] = j
        return dic, emb
    else:
        return emb

# Input emb must be a string stating which embedding is desired among: 'esmb1', 'unirep', 'seqvec', 'protbert'


def load_embedding(emb, dire, dictio=False):
    if emb == 'unirep':
        if dictio == True:
            dic, emb = load_unirep_embedding(dire, dictio=True)
            return dic, emb
        else:
            emb = load_unirep_embedding(dire, dictio=True)
            return emb
    if emb == 'seqvec':
        if dictio == True:
            dic, emb = load_seqvec_embedding(dire, dictio=True)
            return dic, emb
        else:
            emb = load_seqvec_embedding(dire, dictio=True)
            return emb
    if emb == 'esmb1':
        if dictio == True:
            dic, emb = load_esmb1_embedding(dire, dictio=True)
            return dic, emb
        else:
            emb = load_esmb1_embedding(dire, dictio=True)
            return emb
    if emb == 'protbert':
        if dictio == True:
            dic, emb = load_protbert_embedding(dire, dictio=True)
            return dic, emb
        else:
            emb = load_protbert_embedding(dire, dictio=True)
            return emb


# FUNCTIONS USEFUL FOR CV AND GRIDSEARCH:
def create_model_param_grid(method, class_weight, random_state=42):
    if method == 'LR':
        model = LogisticRegression(random_state=random_state)
        param_grid = {'solver': ['liblinear', 'saga'],
                      'penalty': ['l1', 'l2'],
                      'C': np.logspace(-3, 9, 13),
                      'class_weight': class_weight}
    elif method == 'SVM':
        model = SVC(random_state=random_state)
        param_grid = {'C': np.logspace(-2, 10, 13),
                      'gamma': np.logspace(-9, 3, 13),
                      'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                      'class_weight': class_weight}
    elif method == 'RF':
        model = RandomForestClassifier(random_state=random_state)
        param_grid = {'n_estimators': [15, 25, 50, 75, 100, 200, 300],
                      'criterion': ['gini', 'entropy'],
                      'max_depth': [2, 5, 10, None],
                      'min_samples_split': [2, 4, 8, 10],
                      'max_features': ['sqrt', 'auto', 'log2'],
                      'class_weight': class_weight}
    elif method == 'MLP':
        model = MLPClassifier(random_state=random_state)
        param_grid = {'hidden_layer_sizes': [(200,), (100,), (50,), (200, 100, 6, 1)],
                      'activation': ['relu'],
                      'solver': ['lbfgs'],
                      'alpha': [1.0],
                      'learning_rate': ['constant']}
    return model, param_grid


def create_X_y(pos, neg):
    #pos=np.asarray([pos.values()[key] for key in pos.values()])
    #neg=np.asarray([neg.values()[key] for key in neg.values()])
    X = np.concatenate((pos, neg), axis=0)
    y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))), axis=0)
    return X, y


# iter a cv validation for a number of times,
# x is the dataset, y is the list of labels
# accepted methods: RF,MLP,SVM,RF
def CViter(x, y, method, result_filename='results_CV',
           randomstates=[10, 20, 30, 42, 50], folds=5, class_weight=[{0: 1, 1: 1}], refit='f1_macro'):
    results_10iteration = []
    for rs in tqdm(randomstates):
        max_score = 0
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=rs)
        M, p = create_model_param_grid(method, class_weight, random_state=rs)
        mp = GridSearchCV(M, param_grid=p, cv=cv, return_train_score=True,
                          scoring={'balanced_accuracy': make_scorer(balanced_accuracy_score),
                                   'f1_macro': make_scorer(f1_score, average='macro'), 'accuracy': make_scorer(accuracy_score), 'mcc': make_scorer(matthews_corrcoef),
                                   'roc_auc': make_scorer(roc_auc_score), 'recall': make_scorer(recall_score),
                                   'specificity': make_scorer(specificity)}, verbose=1, n_jobs=-1, refit=refit)
        mp.fit(x, y)

        results_full = pd.DataFrame(pd.DataFrame(mp.cv_results_))
        results_rank = results_full.sort_values(by=['rank_test_f1_macro'])
        res = results_rank.filter(items=['mean_test_accuracy', 'mean_test_balanced_accuracy',
                                         'mean_test_mcc', 'mean_test_roc_auc',
                                         'mean_test_f1_macro', 'mean_test_specificity', 'mean_test_recall'])[:1].values[0]
        results_10iteration.append(res)
        # print(res)

        if res[4] > max_score:
            max_score = res[4]
            best_model = mp
    final_res = avg_and_std_list(results_10iteration)
    new_dict = {}
    new_v = []
    for lista in final_res:

        # print(lista)

        new_v.append([round(lista[0], 4), round(lista[1], 4)])
        # print(k,round(n[0],4),round(n[1],4))
        # print(new_v)
        new_dict['score'] = new_v
    res_df = pd.DataFrame(new_dict)
    newcol = ['ACC', 'BACC', 'MCC', 'ROC_AUC', 'F1', 'specificity', 'recall']
    res_df.insert(loc=0, column='metrics', value=newcol)
    res_df.to_csv(result_filename+'_'+method+'.csv', sep='\t')
    filename = result_filename+method+'_model.sav'
    pickle.dump(best_model, open(filename, 'wb'))
    return res_df, best_model, res_df['score'][4][0]


# HRFE
def create_dataset(features, df):
    Ms = []
    for feature in features:
        M = get_feature(df, feature)
        Ms.append(M)
    return np.concatenate(Ms, axis=1)


def get_feature(df, feature):
    M = np.asarray(df[feature]).reshape(-1, 1)
    return M


def grid_search(method, model, param_grid, random_state, outer_cv, inner_cv, M, m, current_features):
    fold = 1  # 1 indexed because of results
    for outer_train, outer_test in (outer_cv.split(M, m)):
        clf = GridSearchCV(model, param_grid=param_grid, cv=inner_cv,
                           return_train_score=True, scoring='f1_macro')
        clf.fit(M[outer_train], m[outer_train])
        if fold == 1:
            results = pd.DataFrame(pd.DataFrame(clf.cv_results_)['params'])
        results['mean_scores' +
                str(fold)] = pd.DataFrame(clf.cv_results_)['mean_test_score'].values
        results['std_scores' +
                str(fold)] = pd.DataFrame(clf.cv_results_)['std_test_score'].values
        results.to_csv('results_'+'_'.join(str(current_features)
                                           )+'_'+method+'_inner_loop.tsv', sep="\t")
        fold += 1
    return results


def create_model(method, params, random_state=42):
    if method == 'LR':
        model = LogisticRegression(
            max_iter=200, random_state=random_state, **params)
    elif method == 'MLP':
        model = MLPClassifier(random_state=random_state,
                              max_iter=200, **params)
    elif method == 'SVM':
        model = SVC(random_state=random_state, **params)
    elif method == 'RF':
        model = RandomForestClassifier(random_state=random_state, **params)
    return model


def outer_cross_val(method, model, outer_cv, M, m):
    scores = cross_validate(model, X=M, y=m, cv=outer_cv,
                            scoring={'balanced_accuracy': make_scorer(balanced_accuracy_score),
                                     'f1_macro': make_scorer(f1_score, average='macro'),
                                     'accuracy': make_scorer(accuracy_score),
                                     'mcc': make_scorer(matthews_corrcoef),
                                     'roc_auc': make_scorer(roc_auc_score)})
    return scores


def save_scores(scores, current_features, method, f):
    score_df = pd.DataFrame(scores)[['test_f1_macro', 'test_balanced_accuracy', 'test_mcc',
                                     'test_accuracy', 'test_roc_auc']].T
    score_df['mean'] = score_df.mean(axis=1)
    score_df['std'] = score_df.std(axis=1)
    score_df.to_csv('results_'+'_'.join(str(current_features)) +
                    '_'+method+'_outer_loop.csv')
    f1_mean = round(np.mean(scores['test_f1_macro']), 3)
    f1_std = round(np.std(scores['test_f1_macro']), 3)
    bacc_mean = round(np.mean(scores['test_balanced_accuracy']), 3)
    bacc_std = round(np.std(scores['test_balanced_accuracy']), 3)
    mcc_mean = round(np.mean(scores['test_mcc']), 3)
    mcc_std = round(np.std(scores['test_mcc']), 3)
    acc_mean = round(np.mean(scores['test_accuracy']), 3)
    acc_std = round(np.std(scores['test_accuracy']), 3)
    auc_mean = round(np.mean(scores['test_roc_auc']), 3)
    auc_std = round(np.std(scores['test_roc_auc']), 3)
    f.write(', '.join(str(current_features))+'\n')
    f.write('{}±{}'.format(f1_inner_mean, f1_inner_std).rjust(15) +
            '{}±{}'.format(f1_mean, f1_std).rjust(15) +
            '{}±{}'.format(bacc_mean, bacc_std).rjust(15) +
            '{}±{}'.format(mcc_mean, mcc_std).rjust(15) +
            '{}±{}'.format(acc_mean, acc_std).rjust(15) +
            '{}±{}'.format(auc_mean, auc_std).rjust(15)+'\n')
    return f1_mean


def hybrid_embedding_full_comparison(x, y, methods=['LR'], random_state=[42], n_jobs=-1,
                                     folds=10, refit='f1_macro', class_weight=[{0: 1, 1: 1}], war='ignore',
                                     results_filename='RESULTS'):
    df = pd.DataFrame(x)
    class_weight = class_weight
    folds = folds
    refit = refit
    if war == 'ignore':
        warnings.filterwarnings("ignore")
        for method in tqdm(methods, desc="Methods"):

            results_iteration = []
            for rs in tqdm(random_states):
                #model, param_grid = create_model_param_grid(method, class_weight, random_state)
                features = list(df.columns)
                selected_features = []
                current_best = 0

                while True:
                    current_results = np.zeros(len(features))
                    for i, feature in tqdm(enumerate(features), total=len(features), desc="Feature selection "+method):
                        current_features = list(selected_features)+[feature]
                        D = create_dataset(current_features, df)
                        cv = StratifiedKFold(
                            n_splits=folds, shuffle=True, random_state=rs)
                        M, p = create_model_param_grid(
                            method, class_weight, random_state=rs)
                        mp = GridSearchCV(M, param_grid=p, cv=cv, return_train_score=True,
                                          scoring={'balanced_accuracy': make_scorer(balanced_accuracy_score),
                                                   'f1_macro': make_scorer(f1_score, average='macro'), 'accuracy': make_scorer(accuracy_score), 'mcc': make_scorer(matthews_corrcoef),
                                                   'roc_auc': make_scorer(roc_auc_score), 'recall': make_scorer(recall_score),
                                                   'specificity': make_scorer(specificity)}, verbose=1, n_jobs=n_jobs, refit=refit)

                        mp.verbose = False
                        mp.fit(D, y)

                        results_full = pd.DataFrame(
                            pd.DataFrame(mp.cv_results_))
                        results_rank = results_full.sort_values(
                            by=['rank_test_f1_macro'])
                        res = results_rank.filter(items=['mean_test_accuracy', 'mean_test_balanced_accuracy',
                                                         'mean_test_mcc', 'mean_test_roc_auc',
                                                         'mean_test_f1_macro', 'mean_test_specificity', 'mean_test_recall'])[:1].values[0]

                        best_params = mp.best_params_

                        current_results[i] = mp.best_score_

                    if current_results.max() > current_best:
                        selected_feature = np.asarray(list(features))[
                            current_results.argmax()]
                        selected_features.append(selected_feature)
                        features.remove(selected_feature)
                        current_best = current_results.max()
                        current_best_model = mp
                        final_res = res
                        # f.write()
                    else:
                        break
                        # f.write('\n')

                results_iteration.append(final_res)
            filename = result_filename+method+'_model.sav'
            pickle.dump(current_best_model, open(filename, 'wb'))

            all_res = avg_and_std_list(results_iteration)
            new_dict = {}
            new_v = []
            for lista in all_res:
                # print(lista)
                new_v.append([round(lista[0], 4), round(lista[1], 4)])
                # print(new_v)
            new_dict['score'] = new_v
            # print(new_dict)
            res_df = pd.DataFrame(new_dict)
            newcol = ['acc', 'bacc', 'mcc', 'roc_auc',
                      'f1_macro', 'specificity', 'recall']

            res_df.insert(loc=0, column='metrics', value=newcol)
            res_df.to_csv(results_filename+'_'+method+'.csv', sep='\t')
            print(method, all_res)


def hybrid_embedding_RFE(x, y, methods=['LR'], random_state=[42], n_jobs=-1,
                         folds=10, refit='f1_macro', class_weight=[{0: 1, 1: 1}], war='ignore',
                         results_filename='RESULTS'):
    class_weight = class_weight
    folds = folds
    refit = refit
    if war == 'ignore':
        warnings.filterwarnings("ignore")
        for method in tqdm(methods, desc="Methods"):
            current_best = 0
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
            M, p = create_model_param_grid(
                method, class_weight, random_state=42)
            mp = GridSearchCV(M, param_grid=p, cv=cv, return_train_score=True,
                              scoring={'balanced_accuracy': make_scorer(balanced_accuracy_score),
                                       'f1_macro': make_scorer(f1_score, average='macro'), 'accuracy': make_scorer(accuracy_score), 'mcc': make_scorer(matthews_corrcoef),
                                       'roc_auc': make_scorer(roc_auc_score), 'recall': make_scorer(recall_score),
                                       'specificity': make_scorer(specificity)}, verbose=1, n_jobs=n_jobs, refit=refit)

            mp.verbose = False
            mp.fit(x, y)

            #results_full = pd.DataFrame(pd.DataFrame(mp.cv_results_))
            # results_rank=results_full.sort_values(by=['rank_test_f1_macro'])
            # res=results_rank.filter(items=['mean_test_accuracy','mean_test_balanced_accuracy',\
            #                                  'mean_test_mcc','mean_test_roc_auc',\
            #                                  'mean_test_f1_macro','mean_test_specificity','mean_test_recall'])[:1].values[0]

            best_params = mp.best_params_
            filename_model_all = str(results_filename) + \
                '_'+str(method)+'_all_features_model.sav'
            print(filename_model_all)
            model = mp.best_estimator_
            pickle.dump(model, open(filename_model_all, 'wb'))

            results_iteration = []
            for rs in tqdm(random_states, desc="random states"):
                cv = StratifiedKFold(
                    n_splits=folds, shuffle=True, random_state=rs)
                if method == 'LR':
                    estimator = LogisticRegression(**best_params)
                elif method == 'SVM':
                    estimator = SVC(**best_params)
                elif method == 'RF':
                    estimator = RandomForestClassifier(**best_params)
                elif method == 'MLP':
                    estimator = MLPClassifier(**best_params)
                estimator.verbose = False
                estimator.fit(x, y)
                res = cross_validate(estimator=estimator, X=x, y=y, scoring={'balanced_accuracy': make_scorer(balanced_accuracy_score),
                                                                             'f1_macro': make_scorer(f1_score, average='macro'), 'accuracy': make_scorer(accuracy_score), 'mcc': make_scorer(matthews_corrcoef),
                                                                             'roc_auc': make_scorer(roc_auc_score), 'recall': make_scorer(recall_score),
                                                                             'specificity': make_scorer(specificity)})
                r = pd.DataFrame(res)
                results_full = r.mean().filter(items=['test_accuracy', 'test_balanced_accuracy',
                                                      'test_mcc', 'test_roc_auc',
                                                      'test_f1_macro', 'test_specificity', 'test_recall'])

                results_iteration.append(res_full)

            all_res = avg_and_std_list(results_iteration)
            new_dict = {}
            new_v = []
            for lista in all_res:
                new_v.append([round(lista[0], 4), round(lista[1], 4)])
            new_dict['score'] = new_v
            res_df = pd.DataFrame(new_dict)
            newcol = ['acc', 'bacc', 'mcc', 'roc_auc',
                      'f1_macro', 'specificity', 'recall']

            res_df.insert(loc=0, column='metrics', value=newcol)
            res_df.to_csv(results_filename+'_'+method+'.csv', sep='\t')
            print('all features results:'+'\n', method, res_df)

            list_of_n_of_selected_features, list_of_feature_ranks = [], []

            selector = RFECV(
                estimator, cv=folds, min_features_to_select=100, step=100, scoring='f1_macro')
            sel = selector.fit(x, y)
            n_of_selected_features = sel.n_features_
            list_of_n_of_selected_features.append(n_of_selected_features)
            list_of_feature_ranks.append(sel.ranking_)
            lofosf = np.asarray(list_of_n_of_selected_features)
            avg_n_of_selected_features = np.mean(lofosf)
            columns_list = []
            df_x = pd.DataFrame(x)
            for i, j in zip(list_of_feature_ranks[0], df_x.columns):
                if i == 1:
                    columns_list.append(j)
            columns4_emb = np.asarray(columns_list)
            new_emb = df_x[columns4_emb]
            print('New embedding shape:', new_emb.shape)
            results_iteration = []

            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
            M, p = create_model_param_grid(
                method, class_weight, random_state=42)
            mp = GridSearchCV(M, param_grid=p, cv=cv, return_train_score=True,
                              scoring={'balanced_accuracy': make_scorer(balanced_accuracy_score),
                                       'f1_macro': make_scorer(f1_score, average='macro'), 'accuracy': make_scorer(accuracy_score), 'mcc': make_scorer(matthews_corrcoef),
                                       'roc_auc': make_scorer(roc_auc_score), 'recall': make_scorer(recall_score),
                                       'specificity': make_scorer(specificity)}, verbose=1, n_jobs=n_jobs, refit=refit)

            mp.verbose = False
            mp.fit(new_emb, y)
            best_params = mp.best_params_
            print(results_filenemae)
            filename_model_hybrid = str(
                results_filename)+'_'+str(method)+'_hybrid_features_model.sav'
            print(filename_model_hybrid)
            model2 = mp.best_estimator
            pickle.dump(model2, open(filename_model_hybrid, 'wb'))

            for rs in tqdm(random_states, desc="random states"):
                cv = StratifiedKFold(
                    n_splits=folds, shuffle=True, random_state=rs)
                if method == 'LR':
                    estimator = LogisticRegression(**best_params)
                elif method == 'SVM':
                    estimator = SVC(**best_params)
                elif method == 'RF':
                    estimator = RandomForestClassifier(**best_params)
                elif method == 'MLP':
                    estimator = MLPClassifier(**best_params)
                estimator.verbose = False
                estimator.fit(new_emb, y)
                res = cross_validate(estimator=estimator, X=new_emb, y=y, scoring={'balanced_accuracy': make_scorer(balanced_accuracy_score),
                                                                                   'f1_macro': make_scorer(f1_score, average='macro'), 'accuracy': make_scorer(accuracy_score), 'mcc': make_scorer(matthews_corrcoef),
                                                                                   'roc_auc': make_scorer(roc_auc_score), 'recall': make_scorer(recall_score),
                                                                                   'specificity': make_scorer(specificity)})
                r = pd.DataFrame(res)
                results_full = r.mean().filter(items=['test_accuracy', 'test_balanced_accuracy',
                                                      'test_mcc', 'test_roc_auc',
                                                      'test_f1_macro', 'test_specificity', 'test_recall'])

                results_iteration.append(res_full)

            all_res = avg_and_std_list(results_iteration)
            new_dict = {}
            new_v = []
            for lista in all_res:
                new_v.append([round(lista[0], 4), round(lista[1], 4)])
            new_dict['score'] = new_v
            res_df = pd.DataFrame(new_dict)
            newcol = ['acc', 'bacc', 'mcc', 'roc_auc',
                      'f1_macro', 'specificity', 'recall']

            res_df.insert(loc=0, column='metrics', value=newcol)
            res_df.to_csv(results_filename+'RFE'+'_'+method+'.csv', sep='\t')
            print('selected_features:'+'\n', method, res_df)
