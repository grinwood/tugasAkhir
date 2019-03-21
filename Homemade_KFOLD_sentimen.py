import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold as KFold
from sklearn.metrics import precision_score, recall_score, f1_score
import Homemadetfidf as lib
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as dekompos

def fiturpenting(namaFile,coef1,feature1,coef2,feature2,status):
    max_coef = 0
    min_coef = 0
    fn = ''
    fnn=''
    tempfn2 = []
    tempcoef2 = []
    tempfn1 = []
    tempcoef1 = []
    for i in range(len(feature2)):
        if feature2[i] not in tempfn2:
            for y in range(len(feature2)):
                if (feature2[i]==feature2[y] and coef2[y] >= coef2[i]):
                    max_coef = coef2[y]
                    fn = feature2[y]
            tempfn2.append(fn)
            tempcoef2.append(max_coef)
    for i in range(len(feature1)):
        if feature1[i] not in tempfn1:
            for y in range(len(feature1)):
                if (feature1[i]==feature1[y] and coef1[y] <= coef1[i]):
                    min_coef = coef1[y]
                    fnn = feature1[y]
            tempfn1.append(fnn)
            tempcoef1.append(min_coef)
    if (len(tempcoef1) > len(tempcoef2)):
        for i in range(len(tempcoef1)-len(tempcoef2)):
            tempcoef2.append(0)
            tempfn2.append('x')
    if (len(tempcoef1) < len(tempcoef2)):
        for i in range(len(tempcoef2)-len(tempcoef1)):
            tempcoef1.append(100)
            tempfn1.append('x')
    if (len(tempcoef1) == len(tempcoef2)):
        feat = {'coef_1': tempcoef1, 'feat_1': tempfn1, 'coef_2': tempcoef2, 'feat_2': tempfn2}
        dfx = pd.DataFrame(data=feat)
    if(status =='tfidf'):
        dfx.to_excel('' + namaFile + '_feature_tfidf.xlsx', index=False)
    if(status=='delta'):
        dfx.to_excel('' + namaFile + '_feature_deltatfidf.xlsx', index=False)
def plotClusters(x,y,a,b,t):
    # Takes in a set of datapoints x and y for two clusters,
    #  the hyperplane separating them in the form a'x -b = 0,
    #  and a slab half-width t
    d1_min = np.min([x[:,0],y[:,0]])
    d1_max = np.max([x[:,0],y[:,0]])
    # Line form: (-a[0] * x - b ) / a[1]
    d2_atD1min = (-a[0]*d1_min + b ) / a[1]
    d2_atD1max = (-a[0]*d1_max + b ) / a[1]

    sup_up_atD1min = (-a[0]*d1_min + b + t ) / a[1]
    sup_up_atD1max = (-a[0]*d1_max + b + t ) / a[1]
    sup_dn_atD1min = (-a[0]*d1_min + b - t ) / a[1]
    sup_dn_atD1max = (-a[0]*d1_max + b - t ) / a[1]

    # Plot the clusters!
    plt.scatter(x[:,0],x[:,1],color='blue')
    plt.scatter(y[:,0],y[:,1],color='red')
    plt.plot([d1_min,d1_max],[d2_atD1min[0,0],d2_atD1max[0,0]],color='black')
    plt.plot([d1_min,d1_max],[sup_up_atD1min[0,0],sup_up_atD1max[0,0]],'--',color='gray')
    plt.plot([d1_min,d1_max],[sup_dn_atD1min[0,0],sup_dn_atD1max[0,0]],'--',color='gray')
    plt.ylim([np.floor(np.min([x[:,1],y[:,1]])),np.ceil(np.max([x[:,1],y[:,1]]))])
def fold_tfidf(namaFile):
    if (os.path.isfile(''+namaFile+'_Performa_tfidf.xlsx')== True ):
        print('================')
        print('tf idf selesai di analisis sebelumnya')
        print('================')
    else:
        print('====================')
        print('Running TF-IDF')
        df = pd.read_excel(''+namaFile+'_PrePro.xlsx', sheet_name='Sheet1')
        reviewtext = df['review']
        sentimen_gold = df['sentimen_gold']
        kf = KFold(n_splits=10, random_state=42)
        acc = []
        prec = []
        rec = []
        f1 = []
        avgf1 = 0
        avgprec = 0
        avgrec = 0
        sumf1 = 0
        sumprec = 0
        sumrec = 0
        count = 1
        for train_index, test_index in kf.split(reviewtext,sentimen_gold):
            rec_coef_1 = []
            rec_fn_1 = []
            rec_coef_2 = []
            rec_fn_2 = []
            # print('train: ',train_index,' test : ', test_index)
            doc_train, doc_test = np.array(reviewtext)[train_index], np.array(reviewtext)[test_index]
            label_train, label_test = np.array(sentimen_gold)[train_index], np.array(sentimen_gold)[test_index]
            # training tf-idf
            doc_train_list = doc_train.tolist()
            label_train_list = label_train.tolist()
            doc_train_tfidf = lib.tfidf(doc_train,'train')
            #doc_train_tfidf.toarray()
            '''# PCA
            pca = dekompos(n_components=2)
            x = pca.fit_transform(doc_train_tfidf)
            y = label_train
            for i in range (len(x)):
                if (y[i]=='positif'):
                    plt.scatter(x[i,1],x[i,0],c='g')
                else:
                    plt.scatter(x[i,1],x[i,0],c='r')
            plt.show()
            # PCA'''
            # testing_tf-idf
            doc_test_list = doc_test.tolist()
            doc_test_tfidf = lib.tfidf(doc_test,'test')
            #doc_test_tfidf.toarray()

            #classifier
            clf = LinearSVC(C=0.1)
            clf.fit(doc_train_tfidf, label_train)
            pred = clf.predict(doc_test_tfidf)
            akurasi = accuracy_score(label_test, pred)
            acc.append(akurasi)
            precisi = precision_score(label_test, pred, average='binary',pos_label='positif')
            prec.append(precisi)
            recall = recall_score(label_test, pred, average='binary',pos_label='positif')
            rec.append(recall)
            fscore = f1_score(label_test, pred, average='binary',pos_label='positif')
            f1.append(fscore)
            sumf1 += fscore
            sumprec += precisi
            sumrec += recall
            print('akurasi : ',akurasi,' precisi : ',precisi,' recall : ',recall,' F1 Score : ',fscore)
            print('iterasi :',count)
            print('==========================')
            '''print('creating record..')
            d = {'gold': label_test,'review':doc_test, 'prediksi':pred}
            df = pd.DataFrame(data=d)
            df.to_excel(''+namaFile+'_tfidfrecord_fold_'+str(count)+'.xlsx', index=False)
            print('creating important feature.')
            n = 20
            feature_names = np.array(lib.getVocab())
            coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
            top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
            for (coef_1, fn_1), (coef_2, fn_2) in top:
                rec_coef_1.append(coef_1)
                rec_coef_2.append(coef_2)
                rec_fn_1.append(fn_1)
                rec_fn_2.append(fn_2)
            fiturpenting(namaFile,rec_coef_1,rec_fn_1,rec_coef_2,rec_fn_2,'tfidf')'''
            ######### ngambil nilai tfidf
            if (count == 10):
                dn = {'value': lib.gettfidf_val(), 'term': lib.getVocab(),'contained':lib.getcontained()}
                df = pd.DataFrame(data=dn)
                df.to_excel('' + namaFile + '_tfidf_value_all.xlsx', index=False)
            count += 1
        avgf1 = sumf1 / 10
        avgprec = sumprec / 10
        avgrec = sumrec / 10
        print('rata-rata akurasi : ',str(avgf1))
        d = {'Akurasi': acc , 'Precisi': prec, 'Recall': rec, 'F1 score': f1}
        df = pd.DataFrame(data=d)
        df.to_excel(''+namaFile+'_Performa_tfidf.xlsx', index=False)
    return avgf1, avgprec, avgrec

def fold_deltatfidf(namaFile):
    if (os.path.isfile(''+namaFile+'_Performa_deltatfidf.xlsx')== True ):
        print('================')
        print('delta tf idf selesai di analisis sebelumnya')
        print('================')
    else:
        print('====================')
        print('Running Delta TF-IDF')
        df = pd.read_excel(''+namaFile+'_PrePro.xlsx', sheet_name='Sheet1')
        reviewtext = df['review']
        sentimen_gold = df['sentimen_gold']
        kf = KFold(n_splits=10, random_state=42)
        acc = []
        prec = []
        rec = []
        f1 = []
        avgf1 = 0
        avgprec=0
        avgrec=0
        sumf1 = 0
        sumprec=0
        sumrec=0
        count = 1
        for train_index, test_index in kf.split(reviewtext,sentimen_gold):
            rec_coef_1 = []
            rec_fn_1 = []
            rec_coef_2 = []
            rec_fn_2 = []
            # print('train: ',train_index,' test : ', test_index)
            doc_train, doc_test = np.array(reviewtext)[train_index], np.array(reviewtext)[test_index]
            label_train, label_test = np.array(sentimen_gold)[train_index], np.array(sentimen_gold)[test_index]
            # training deltatf-idf
            doc_train_deltatfidf = lib.deltatfidf(doc_train,label_train,'train')
            '''## PCA
            pca = dekompos(n_components=2)
            x = pca.fit_transform(doc_train_deltatfidf)
            y = label_train
            for i in range(91):
                if (y[i] == 'positif'):
                    plt.scatter(x[i, 1], x[i, 0], c='g')
                else:
                    plt.scatter(x[i, 1], x[i, 0], c='r')
            plt.show()
            # PCA'''
            # testing_deltatf-idf
            doc_test_deltatfidf = lib.deltatfidf(doc_test,'','test')
            # classifier
            clf2 = LinearSVC(C=0.1)
            clf2.fit(doc_train_deltatfidf, label_train)
            '''print('creating important feature.')
            n = 20
            feature_names = np.array(lib.getvocab_delta())
            coefs_with_fns = sorted(zip(clf2.coef_[0], feature_names))
            top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
            for (coef_1, fn_1), (coef_2, fn_2) in top:
                rec_coef_1.append(coef_1)
                rec_coef_2.append(coef_2)
                rec_fn_1.append(fn_1)
                rec_fn_2.append(fn_2)
            #fiturpenting(namaFile, rec_coef_1, rec_fn_1, rec_coef_2, rec_fn_2, 'delta')
            feat = {'coef_1': rec_coef_1, 'feat_1': rec_fn_1, 'coef_2': rec_coef_2, 'feat_2': rec_fn_2}
            dfx = pd.DataFrame(data=feat)
            dfx.to_excel('' + namaFile + '_feature_deltatfidf.xlsx', index=False)'''
            pred = clf2.predict(doc_test_deltatfidf)
            akurasi = accuracy_score(label_test, pred)
            acc.append(akurasi)
            precisi = precision_score(label_test, pred, average='binary',pos_label='positif')
            prec.append(precisi)
            recall = recall_score(label_test, pred, average='binary',pos_label='positif')
            rec.append(recall)
            fscore = f1_score(label_test, pred, average='binary',pos_label='positif')
            f1.append(fscore)
            sumf1 += fscore
            sumprec += precisi
            sumrec += recall
            print('akurasi : ', akurasi, ' precisi : ', precisi, ' recall : ', recall, ' F1 Score : ', fscore)
            print('iterasi :', count)
            print('==========================')
            '''print('creating record..')
            d = {'gold': label_test, 'review': doc_test, 'prediksi': pred}
            df = pd.DataFrame(data=d)
            df.to_excel('' + namaFile + '_deltatfidfrecord_fold_' + str(count) + '.xlsx', index=False)
            
            d = {'doctrain': doc_train, 'label':label_train}
            df = pd.DataFrame(data=d)
            df.to_excel('' + namaFile + '_doctrain_deltatfidf.xlsx', index=False)'''
            ######### ngambil nilai delta
            if (count ==10):
                dn = {'value':lib.getIdfNeg(),'term':lib.getNeg_vocab()}
                df = pd.DataFrame(data=dn)
                df.to_excel('' + namaFile + '_deltatfidf_value_neg.xlsx', index=False)
                dp = {'value':lib.getIdfPos(),'term':lib.getPos_vocab()}
                dfp = pd.DataFrame(data=dp)
                dfp.to_excel('' + namaFile + '_deltatfidf_value_pos.xlsx', index=False)
                dv = {'vocab':lib.getvocab_delta()}
                dvv = pd.DataFrame(data=dv)
                dvv.to_excel('' + namaFile + '_deltatfidf_vocab.xlsx', index=False)
            count += 1
        avgf1 = sumf1/10
        avgprec = sumprec/10
        avgrec = sumrec/10
        print('rata-rata akurasi : ', str(avgf1))
        d = {'Akurasi': acc, 'Precisi': prec, 'Recall': rec, 'F1 score': f1}
        df = pd.DataFrame(data=d)
        df.to_excel('' + namaFile + '_Performa_deltatfidf.xlsx', index=False)
        vPos = np.array(lib.getPos_vocab())
        vNeg = np.array(lib.getNeg_vocab())
        jumlahPos = len(vPos)
        jumlahNeg = len(vNeg)
        print('jumlah pos ', jumlahPos)
        file = open(''+namaFile+'_jumlah_vocabdelta.txt', 'w')
        jp = 'jumlah pos : '+str(jumlahPos)
        jn = 'jumlah neg : '+str(jumlahNeg)
        file.write(jp)
        file.write(jn)
        file.close()
    return avgf1,avgprec,avgrec



