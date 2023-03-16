import numpy as np
import sys
import pandas as pd
import pickle
import os
from Bio import SeqIO
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import confusion_matrix,f1_score,recall_score,roc_auc_score,fbeta_score,make_scorer,balanced_accuracy_score,accuracy_score,matthews_corrcoef
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
import utils_trans_2
import warnings
import portpred as pp


def generate_pos_neg(dictio,pos_l,neg_l):
    pos_train,neg_train=[],[]
    for k,v in dictio.items():
        if k in pos_l:
            pos_train.append(v)

        elif k in neg_l:
            neg_train.append(v)
    return np.asarray(pos_train),np.asarray(neg_train)

def CViter(x, y, method, result_filename='results_CV',
           randomstates=[10,20,30,42,50,60,70,80,90,100], folds=10, class_weight=[{0: 1, 1: 1}], refit='f1_macro'):
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
        emb_list.append(np.load(dire+'/'+e,allow_pickle=True))
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

def testAndresults(file_model,x,true_y,resultfiletosave):
    model=pickle.load(open(file_model,'rb'))
    pred=model.predict(x)
    acc=accuracy_score(true_y,pred)
    spe=specificity(true_y,pred)
    sen=recall_score(true_y,pred)
    mcc=matthews_corrcoef(true_y,pred)
    roc_auc=roc_auc_score(true_y,pred)
    f1=f1_score(true_y,pred)
    print('sen:     ',round(sen,4)*100)
    print('spe:     ',round(spe,4)*100)
    print('acc:     ',round(acc,4)*100)
    print('mcc:     ',round(mcc,4)*100)
    print('roc_auc: ',round(roc_auc,4)*100)
    print('F1:      ',round(f1,4)*100)
    with open(resultfiletosave, 'w') as res:
        res.write('sen: '+'\t'+str(round(sen,4))+'\n'+\
                  'spe: '+'\t'+str(round(spe,4))+'\n'+\
                  'acc: '+'\t'+str(round(acc,4))+'\n'+\
                  'mcc: '+'\t'+str(round(mcc,4))+'\n'+\
                  'roc_auc: '+'\t'+str(round(roc_auc,4))+'\n'+\
                  'f1: '+'\t'+str(round(f1,4)))
    res=[sen,spe,acc,mcc,roc_auc,f1]
    return res

def checktestAndresults(file_model,x,true_y,resultfiletosave):
    model=pickle.load(open(file_model,'rb'))
    pred=model.predict(x)
    print(pred)
    acc=accuracy_score(true_y,pred)
    spe=specificity(true_y,pred)
    sen=recall_score(true_y,pred)
    mcc=matthews_corrcoef(true_y,pred)
    roc_auc=roc_auc_score(true_y,pred)
    f1=f1_score(true_y,pred)
    print('sen:     ',round(sen,4)*100)
    print('spe:     ',round(spe,4)*100)
    print('acc:     ',round(acc,4)*100)
    print('mcc:     ',round(mcc,4)*100)
    print('roc_auc: ',round(roc_auc,4)*100)
    print('F1:      ',round(f1,4)*100)
    with open(resultfiletosave, 'w') as res:
        res.write('sen: '+'\t'+str(round(sen,4))+'\n'+\
                  'spe: '+'\t'+str(round(spe,4))+'\n'+\
                  'acc: '+'\t'+str(round(acc,4))+'\n'+\
                  'mcc: '+'\t'+str(round(mcc,4))+'\n'+\
                  'roc_auc: '+'\t'+str(round(roc_auc,4))+'\n'+\
                  'f1: '+'\t'+str(round(f1,4)))
    res=[sen,spe,acc,mcc,roc_auc,f1]
    return res


def test_score_pred(model_file,pos_list,neg_list,col_list,percentage=True,r=2,hf=False):
    if hf:
        x,y=create_X_y(pos_list,neg_list)
        df=pd.DataFrame(x)
        new_x=df[col_list]
        model=pickle.load(open(model_file,'rb'))
        pred=model.predict(new_x)
        if percentage:
            print('SEN: '+str(round(recall_score(pred,y)*100,r))+'\n'+\
                  'SPE: '+str(round(specificity(pred,y)*100,r))+'\n'+\
                  'ACC: '+str(round(accuracy_score(pred,y)*100,r))+'\n'+\
                  'MCC: '+str(round(matthews_corrcoef(pred,y)*100,r))+'\n'+\
                  'ROC: '+str(round(roc_auc_score(pred,y)*100,r))+'\n'+\
                  'F1s: '+str(round(f1_score(pred,y)*100,r)))
        else:
            print('SEN: '+str(round(recall_score(pred,y),r))+'\n'+\
              'SPE: '+str(round(specificity(pred,y),r))+'\n'+\
              'ACC: '+str(round(accuracy_score(pred,y),r))+'\n'+\
              'MCC: '+str(round(matthews_corrcoef(pred,y),r))+'\n'+\
              'ROC: '+str(round(roc_auc_score(pred,y),r))+'\n'+\
              'F1s: '+str(round(f1_score(pred,y),r)))
    else:    
        model=pickle.load(open(model_file,'rb'))
        x,y=create_X_y(pos_list,neg_list)
        pred=model.predict(x)
        if percentage:
            print('SEN: '+str(round(recall_score(pred,y)*100,r))+'\n'+\
                  'SPE: '+str(round(specificity(pred,y)*100,r))+'\n'+\
                  'ACC: '+str(round(accuracy_score(pred,y)*100,r))+'\n'+\
                  'MCC: '+str(round(matthews_corrcoef(pred,y)*100,r))+'\n'+\
                  'ROC: '+str(round(roc_auc_score(pred,y)*100,r))+'\n'+\
                  'F1s: '+str(round(f1_score(pred,y)*100,r)))
        else:
            print('SEN: '+str(round(recall_score(pred,y),r))+'\n'+\
              'SPE: '+str(round(specificity(pred,y),r))+'\n'+\
              'ACC: '+str(round(accuracy_score(pred,y),r))+'\n'+\
              'MCC: '+str(round(matthews_corrcoef(pred,y),r))+'\n'+\
              'ROC: '+str(round(roc_auc_score(pred,y),r))+'\n'+\
              'F1s: '+str(round(f1_score(pred,y),r)))
            

def hybrid_embedding_RFE(x,y,methods=['LR'],random_state=[42],n_jobs=-1,\
                                     folds=10,refit='f1_macro',class_weight=[{0:1,1:1}],war='ignore',\
                                     results_filename='RESULTS'):
    random_states=[10,20,30,42,50,60,70,80,90,100]
    class_weight=class_weight
    folds=folds
    refit=refit
    if war=='ignore':
        warnings.filterwarnings("ignore")
        for method in tqdm(methods, desc="Methods"):
            current_best = 0
            cv = StratifiedKFold(n_splits=folds,shuffle=True,random_state=42)
            M,p=create_model_param_grid(method,class_weight,random_state=42)
            mp = GridSearchCV(M, param_grid=p, cv=cv, return_train_score=True,\
                                              scoring ={'balanced_accuracy':make_scorer(balanced_accuracy_score),\
                                'f1_macro':make_scorer(f1_score, average='macro'),'accuracy':make_scorer(accuracy_score),'mcc':make_scorer(matthews_corrcoef),\
                                'roc_auc':make_scorer(roc_auc_score),'recall':make_scorer(recall_score),\
                                                       'specificity':make_scorer(specificity)},verbose=1,n_jobs=n_jobs,refit=refit)

            mp.verbose = False
            mp.fit(x,y)
           
        
            best_params = mp.best_params_
            filename_model_all = str(results_filename)+'_'+str(method)+'_all_features_model.sav'
            #print(filename_model_all)
            model=mp.best_estimator_
            pickle.dump(model, open(filename_model_all, 'wb'))
            
            


            results_iteration=[]
            for rs in tqdm(random_states, desc="random states"):
                #print(rs)
                cv = StratifiedKFold(n_splits=folds,shuffle=True,random_state=rs)
                if method == 'LR':
                    estimator = LogisticRegression(**best_params, random_state=rs)
                    
                elif method == 'SVM':
                    estimator = SVC(**best_params, random_state=rs)
                    
                elif method == 'RF':
                    estimator = RandomForestClassifier(**best_params, random_state=rs)
                    
                elif method == 'MLP':
                    estimator = MLPClassifier(**best_params, random_state=rs)
                    
                estimator.fit(x,y)
                estimator.verbose = False
                res=cross_validate(estimator=estimator,cv=cv,X=x,y=y,scoring ={'balanced_accuracy':make_scorer(balanced_accuracy_score),\
                                'f1_macro':make_scorer(f1_score, average='macro'),'accuracy':make_scorer(accuracy_score),'mcc':make_scorer(matthews_corrcoef),\
                                'roc_auc':make_scorer(roc_auc_score),'recall':make_scorer(recall_score),\
                                                       'specificity':make_scorer(specificity)})
                r=pd.DataFrame(res)
                results_full=r.mean().filter(items=['test_accuracy','test_balanced_accuracy',\
                                              'test_mcc','test_roc_auc',\
                                              'test_f1_macro','test_specificity','test_recall'])
        



                            

                results_iteration.append(results_full)

            all_res=avg_and_std_list(results_iteration)
            new_dict={}
            new_v=[]
            for lista in all_res:
                new_v.append([round(lista[0],4),round(lista[1],4)])
            new_dict['score']=new_v
            res_df=pd.DataFrame(new_dict)
            newcol=['acc','bacc','mcc','roc_auc','f1_macro','specificity','recall']

            res_df.insert(loc=0, column='metrics', value=newcol)
            res_df.to_csv(results_filename+'_all_feat'+'_'+method+'.csv',sep='\t')
            print('all features results:'+'\n',method,res_df)


            
            list_of_n_of_selected_features,list_of_feature_ranks=[],[]
            
            print('Starting the RFE, and the hubtid embedding generation')
            ##RFECV
            cv = StratifiedKFold(n_splits=folds,shuffle=True,random_state=42)
            #M,p=create_model_param_grid('RF',class_weight,random_state=42)
            #mp = GridSearchCV(M, param_grid=p, cv=cv, return_train_score=True,\
            #                                  scoring ={'balanced_accuracy':make_scorer(balanced_accuracy_score),\
            #                    'f1_macro':make_scorer(f1_score, average='macro'),'accuracy':make_scorer(accuracy_score),'mcc':make_scorer(matthews_corrcoef),\
            #                    'roc_auc':make_scorer(roc_auc_score),'recall':make_scorer(recall_score),\
            #                                           'specificity':make_scorer(specificity)},verbose=1,n_jobs=n_jobs,refit=refit)

            #mp.verbose = False
            #mp.fit(x,y)
            #rf_best_params=mp.best_params_
            estimator=RandomForestClassifier(random_state=42)
            modelLR=np.load('binary_classification_LR_all_features_model.sav',allow_pickle=True)
            estimator2=modelLR
            selector = RFECV(estimator2,cv=cv, min_features_to_select=100, step=100, scoring='f1_macro')
            sel = selector.fit(x, y)
            n_of_selected_features=sel.n_features_
            list_of_n_of_selected_features.append(n_of_selected_features)
            list_of_feature_ranks.append(sel.ranking_)
            lofosf=np.asarray(list_of_n_of_selected_features)
            avg_n_of_selected_features=np.mean(lofosf)
            columns_list=[]
            df_x=pd.DataFrame(x)
            for i,j in zip(list_of_feature_ranks[0],df_x.columns):
                if i==1:
                    columns_list.append(j)
            columns4_emb=np.asarray(columns_list)
            with open('columns4_emb.txt' ,'w') as f:
                for e in columns4_emb:
                    f.write(str(e)+'\t')
            new_emb=df_x[columns4_emb]
            #new_emb.to_csv('selected_features_'+method)
            print('New embedding shape:',new_emb.shape)
            results_iteration =[]
           
            cv = StratifiedKFold(n_splits=folds,shuffle=True,random_state=42)
            M,p=create_model_param_grid(method,class_weight,random_state=42)
            mp = GridSearchCV(M, param_grid=p, cv=cv, return_train_score=True,\
                                              scoring ={'balanced_accuracy':make_scorer(balanced_accuracy_score),\
                                'f1_macro':make_scorer(f1_score, average='macro'),'accuracy':make_scorer(accuracy_score),'mcc':make_scorer(matthews_corrcoef),\
                                'roc_auc':make_scorer(roc_auc_score),'recall':make_scorer(recall_score),\
                                                       'specificity':make_scorer(specificity)},verbose=1,n_jobs=n_jobs,refit=refit)

            mp.verbose=False
            mp.fit(new_emb,y)
            best_params = mp.best_params_
            filename_model_hybrid = str(results_filename)+'_'+str(method)+'_hybrid_features_model.sav'
            print(filename_model_hybrid)
            model2=mp.best_estimator_
            pickle.dump(model2, open(filename_model_hybrid, 'wb'))
            

            for rs in tqdm(random_states, desc="random states"):
                cv = StratifiedKFold(n_splits=folds,shuffle=True,random_state=rs)
                if method == 'LR':
                    estimator = LogisticRegression(**best_params, random_state=rs)
                    estimator.fit(new_emb,y)
                elif method == 'SVM':
                    estimator = SVC(**best_params, random_state=rs)
                    estimator.fit(new_emb,y)
                elif method == 'RF':
                    estimator = RandomForestClassifier(**best_params, random_state=rs)
                    estimator.fit(new_emb,y)
                elif method == 'MLP':
                    estimator = MLPClassifier(**best_params, random_state=rs)
                    estimator.fit(new_emb,y)
                estimator.verbose = False
                res=cross_validate(estimator=estimator,X=new_emb,y=y,cv=cv, scoring ={'balanced_accuracy':make_scorer(balanced_accuracy_score),\
                                'f1_macro':make_scorer(f1_score, average='macro'),'accuracy':make_scorer(accuracy_score),'mcc':make_scorer(matthews_corrcoef),\
                                'roc_auc':make_scorer(roc_auc_score),'recall':make_scorer(recall_score),\
                                                       'specificity':make_scorer(specificity)})
                r=pd.DataFrame(res)
                results_full=r.mean().filter(items=['test_accuracy','test_balanced_accuracy',\
                                              'test_mcc','test_roc_auc',\
                                              'test_f1_macro','test_specificity','test_recall'])
        



                            

                results_iteration.append(results_full)
    
            all_res=avg_and_std_list(results_iteration)
            new_dict={}
            new_v=[]
            for lista in all_res:
                new_v.append([round(lista[0],4),round(lista[1],4)])
            new_dict['score']=new_v
            res_df=pd.DataFrame(new_dict)
            newcol=['acc','bacc','mcc','roc_auc','f1_macro','specificity','recall']

            res_df.insert(loc=0, column='metrics', value=newcol)
            res_df.to_csv(results_filename+'RFE'+'_'+method+'.csv',sep='\t')
            print('selected_features:'+'\n',method,res_df)

            
def load_data(fastafile):
    data={}
    for seq_record in SeqIO.parse(fastafile, "fasta"):
        data[seq_record.id]=seq_record.seq
    return data   

def concatenate_embeddings(protbert_dict,esm1b_dict,seqvec_dict,unirep_dict):
    full_dict={}
    full_emb=[]
    for k in esm1b_dict.keys():
        full_dict[k]=np.concatenate((esm1b_dict[k],protbert_dict[k],seqvec_dict[k],unirep_dict[k]))
    for k,v in full_dict.items():
        full_emb.append(v)
    full_emb=np.asarray(full_emb)
    return full_dict,full_emb

col_list_b=[0,1,3,5,6,10,11,13,16,21,25,36,37,38,40,42,43,45,50,56,57,66,67,70,79,81,85,86,89,91,92,97,100,103,105,114,115,126,132,133,134,141,146,149,152,155,157,158,163,
168,170,171,174,175,178,179,183,184,190,193,194,198,201,203,204,211,216,221,224,240,249,250,251,255,258,259,269,277,279,281,283,284,285,291,295,296,306,307,308,
312,313,314,315,322,326,328,333,334,335,337,342,343,344,349,351,357,358,359,361,362,363,369,370,379,380,381,382,384,386,387,391,392,396,397,403,404,408,409,433,
435,440,443,450,452,454,455,456,459,461,463,464,467,468,470,477,478,479,480,483,491,494,495,496,498,499,500,503,506,508,512,514,515,518,519,525,527,530,536,540,
544,548,556,559,565,566,567,568,571,573,575,581,584,585,587,589,590,598,600,605,609,610,617,619,621,623,624,626,632,633,634,636,640,642,643,649,654,662,667,669,
676,682,684,685,694,695,697,698,703,706,707,708,718,719,722,724,737,750,753,755,758,771,772,773,777,778,779,780,781,782,783,785,786,788,789,792,795,797,798,801,
802,807,810,815,826,833,834,840,842,843,848,856,860,862,864,866,879,883,888,889,892,893,899,900,901,903,906,909,910,916,918,928,932,937,938,940,941,946,948,951,
959,961,966,974,978,979,981,982,983,985,992,997,1003,1005,1006,1007,1008,1009,1012,1013,1015,1016,1020,1021,1022,1023,1026,1033,1036,1041,1042,1044,1046,1050,
1053,1055,1057,1063,1074,1075,1077,1078,1080,1082,1085,1086,1091,1093,1094,1100,1101,1103,1105,1106,1108,1110,1111,1112,1114,1117,1118,1120,1121,1125,1126,1127,
1131,1133,1139,1145,1147,1149,1152,1155,1156,1162,1163,1164,1166,1170,1173,1175,1189,1198,1199,1200,1207,1208,1211,1213,1216,1217,1223,1224,1227,1228,1230,1233,
1234,1236,1244,1247,1249,1250,1251,1255,1257,1258,1261,1264,1266,1270,1271,1272,1276,1278,1284,1342,1354,1375,1384,1436,1445,1455,1469,1480,1493,1522,1538,1541,
1545,1560,1585,1700,1710,1711,1760,1782,1790,1834,1839,1892,1919,1965,2004,2045,2077,2090,2106,2113,2162,2179,2312,2318,2326,2329,2332,2348,2352,2353,2356,2362,
2377,2378,2383,2394,2401,2402,2405,2410,2414,2427,2430,2435,2440,2446,2457,2466,2467,2474,2476,2493,2494,2501,2503,2512,2514,2516,2517,2521,2524,2534,2537,2541,
2542,2543,2545,2547,2549,2551,2552,2558,2563,2568,2571,2576,2579,2587,2596,2598,2599,2604,2605,2608,2610,2611,2624,2628,2630,2632,2633,2635,2637,2638,2639,2644,
2652,2653,2655,2656,2658,2659,2662,2666,2667,2669,2675,2677,2679,2680,2683,2686,2697,2699,2700,2702,2706,2711,2712,2713,2714,2715,2716,2718,2722,2738,2741,2748,
2754,2757,2764,2776,2777,2783,2786,2797,2799,2803,2806,2807,2809,2810,2818,2823,2830,2833,2838,2856,2868,2872,2883,2887,2891,2894,2903,2906,2909,2910,2915,2922,
2924,2928,2931,2933,2946,2947,2954,2958,2960,2961,2971,2973,2997,3010,3013,3015,3018,3029,3035,3039,3040,3042,3048,3051,3058,3059,3061,3064,3067,3070,3072,3073,
3081,3091,3095,3097,3098,3100,3107,3110,3113,3116,3133,3135,3137,3139,3144,3147,3149,3156,3160,3163,3166,3167,3168,3172,3178,3198,3208,3210,3212,3214,3218,3221,
3223,3233,3250,3252,3259,3260,3263,3266,3268,3270,3282,3291,3293,3295,3300,3301,3313,3319,3320,3322,3324,3327,3338,3409,3416,3421,3434,3442,3443,3462,3463,3476,
3503,3506,3514,3534,3544,3553,3557,3561,3590,3607,3619,3620,3634,3636,3641,3648,3660,3664,3667,3704,3710,3716,3754,3802,3810,3821,3825,3830,3838,3844,3862,3863,
3873,3900,3908,3915,3930,3940,3961,3968,3969,3971,3972,3976,3995,4008,4009,4017,4025,4039,4042,4065,4070,4079,4093,4097,4101,4102,4107,4119,4145,4150,4174,4197,
4198,4212,4225,4226,4234,4254,4257,4279,4288,4316,4327,4346,4364,4384,4397,4400,4412,4434,4437,4459,4465,4467,4482,4489,4493,4519,4520,4536,4564,4569,4573,4580,
4585,4607,4620,4632,4642,4661,4671,4684,4697,4703,4714,4723,4736,4744,4747,4759,4765,4767,4769,4776,4794,4797,4803,4813,4815,4824,4834,4837,4849,4869,4881,4917,
4918,4935,4936,4942,4962,4984,5002,5007,5011,5025,5038,5047,5054,5055,5090,5094,5095,5101,5131,5138,5143,5164,5173,5181,5191]

def predict_transporter(X,embedding='HFE',model='HFE',classification='all',output='PortPred_results.csv'):
    if model=='HFE':
        m = np.load('binary_classification_LR_hybrid_features_model.sav', allow_pickle=True)
        df=pd.DataFrame(X.values())
        df=df[col_list_b]
        pred=m.predict_proba(df)
        pred_col,pred_k=[],[]
        for p,k in zip(pred,X.keys()):
            pred_col.append(p[1])
            #print(pred_col)
            pred_k.append(k)
        df_results=pd.DataFrame(pred_k,columns=['protein_ID'])
        df_results.insert(len(df_results.columns),'probability',pred_col)
        df_results.loc[df_results['probability'] >= 0.5, 'transporter'] = 'True'
        df_results.loc[df_results['probability'] < 0.5, 'transporter'] = 'False'
        df_results.to_csv(output)
    return df_results
