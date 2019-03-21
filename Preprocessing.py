import json
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import Homemadetfidf as lib
import re, string, unicodedata
from pprint import pprint
import csv
'''with open('E:\Document\Dokumen\kuliah\TA\dataset Delta TF-IDF\sorted_data\electronics\example.txt') as f:
    j = json.load(f)
pprint(j)
print('====')
print(j['sentimen'])
print(j['sentimen']['review'])
print(j['sentimen']['review'][0]['rating'])
print(j['sentimen']['review'][1]['rating'])
print(len(j))
print('======')
i=0
for i in range(len(j)+1):
    print(j['sentimen']['review'][i]['rating'])
    i= i+1'''
def old_jsonLoader(path,limit):
    if (os.path.isfile('output.xlsx')== True ):
        print('================')
        print('file output telah dibuat sebelumnya')
        print('================')
    else:
        target_path = path
        target_text = []
        target_overall = []
        target_label = []
        negatif = int(limit/2)
        positif = int(limit/2)
        print('opening file..')
        target_file = open(target_path, "r")
        print('success opening.. , loop Append..')
        i = 0

        # kalo mau tau jumlah item si json
        ##num = sum(1 for line in target_file)
        ###print('total item : '+str(num))
        for line in target_file:
            print('loading json')
            overall = json.loads(line)['rating']
            review = json.loads(line)['reviewText']
            if (review == '' or float(overall) == float(3)):
                continue
            else:
                if (float(overall) > float(3)):
                    if (positif!= 0):
                        target_label.append('positif')
                        positif = positif - 1
                    else:
                        continue
                else:
                    if(float(overall) < float(3)) :
                        if (negatif !=0):
                            target_label.append('negatif')
                            negatif = negatif - 1
                        else:
                            continue
                    else:
                        target_label.append('manual')
                print('overall : '+str(overall)+' review : ' + review)
                print('loaded, appending json')
                target_text.append(review)
                target_overall.append(overall)
                i += 1
                if (i == limit):
                    break
        d = {'overall': target_overall,'sentimen': target_label, 'review': target_text}
        df = pd.DataFrame(data=d)
        df.to_excel('output.xlsx', index=False)
        print('total item masuk : '+str(len(target_text)))
    return 1

def new_jsonLoader(posPath,negPath,limit,namaFile):
    if (os.path.isfile(''+namaFile+'.xlsx')== True ):
        print('================')
        print('file output telah dibuat sebelumnya')
        print('================')
    else:
        target_text = []
        target_overall = []
        target_label = []
        negatif = int(limit/2)
        positif = int(limit/2)
        print('opening file..')
        dataPos = []
        dataNeg = []
        with open(posPath,encoding='utf-8') as f:
            dataPos = json.loads(f.read())
        with open(negPath,encoding='utf-8') as f:
            dataNeg = json.loads(f.read())
        print('success opening.. , loop Append..')
        i = 1
        # kalo mau tau jumlah item si json
        ##num = sum(1 for line in target_file)
        ###print('total item : '+str(num))
        print(str(positif))
        while (i<=positif):
            overall = dataPos['sentimen']['review'][i]['rating']
            review = dataPos['sentimen']['review'][i]['review_text']
            target_overall.append(overall)
            target_label.append('positif')
            target_text.append(review)
            i += 1
        '''if (os.path.isfile('' + namaFile + '_reviewPositif.xlsx') == False):
            d = {'overall': target_overall, 'sentimen': target_label, 'review': target_text}
            df = pd.DataFrame(data=d)
            df.to_excel('' + namaFile + '_reviewPositif.xlsx', index=False)'''
        i = 1
        while (i<=negatif):
            overall = dataNeg['sentimen']['review'][i]['rating']
            review = dataNeg['sentimen']['review'][i]['review_text']
            target_overall.append(overall)
            target_label.append('negatif')
            target_text.append(review)
            i += 1
        '''if (os.path.isfile('' + namaFile + '_reviewNegatif.xlsx')== False ):
            d = {'overall': target_overall, 'sentimen': target_label, 'review': target_text}
            df = pd.DataFrame(data=d)
            df.to_excel('' + namaFile + '_reviewNegatif.xlsx', index=False)'''
        d = {'overall': target_overall,'sentimen': target_label, 'review': target_text}
        df = pd.DataFrame(data=d)
        df.to_excel(''+namaFile+'.xlsx', index=False)
        print('total item masuk : '+str(len(target_text)))
    return 1
def tokenize(word):
    new_words = nltk.word_tokenize(word)
    return new_words

def lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def remove_number(words):
    relevant_tokens = [token for token in words if not any(c.isdigit() for c in token)]
    #print(relevant_tokens)
    return relevant_tokens

def costum_stopwords(stop,words):
    new_stopwords = stop
    new_words = []
    for word in words:
        if word not in new_stopwords:
            new_words.append(word)
    return new_words
def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    stopword = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y']
    for word in words:
        if word not in stopword:
            new_words.append(word)
    return new_words

def low_frequency(words,min):
    new_words = []
    termfreq = 0
    for word in words:
        termfreq = words.count(word)
        if (int(termfreq) > int(min)):
            new_words.append(word)
    return new_words
def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = PorterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def runAllPreprocessing(namaFile,seleksifitur):
    if (os.path.isfile(''+namaFile+'_PrePro.xlsx')== True ):
        print('================')
        print('dataset telah di preprocess sebelumnya')
        print('================')
    else:
        df = pd.read_excel(''+namaFile+'.xlsx', sheet_name='Sheet1')
        reviewtext = df['review']
        sentimen_gold = df['sentimen']
        bersih = []
        i = 0
        for i in range(len(reviewtext)):
            words = tokenize(reviewtext[i])
            new_words = lowercase(words)
            new_words = remove_punctuation(new_words)
            new_words = remove_number(new_words)
            new_words = remove_stopwords(new_words)
            #new_words = stem_words(new_words)
            new_words = [w for w in new_words if len(w) > 2]
            bersih.append(new_words)

        from nltk.tokenize.moses import MosesDetokenizer
        detoken = MosesDetokenizer()
        words = []
        for i in range(len(bersih)):
            word = detoken.detokenize(bersih[i], return_str=True)
            words.append(word)
        if(seleksifitur=='on'):
            #seleksi berdasarkan frekuensi
            '''satustring = ",".join(map(str, words))
            token = tokenize(satustring)
            fredis = nltk.FreqDist(token)
            rarewords = list(filter(lambda x:x[1]<=3,fredis.items()))
            #newstopword = rarewords[:,0]'''
            newstopword = []
            #seleksi pakai delta tf-idf
            lib.deltatfidf(words,sentimen_gold,'train')
            newstopword = lib.getZero()
            '''for i in range(len(rarewords)):
                newstopword.append(rarewords[i][0])
            print(rarewords)'''
            #print(newstopword)
            d = {'zero term': newstopword}
            df = pd.DataFrame(data=d)
            df.to_excel('' + namaFile + '_zeroterm.xlsx', index=False)
            final = []
            for i in range(len(words)):
                word = tokenize(words[i])
                new_words = costum_stopwords(newstopword,word)
                final.append(new_words)
            cleanfinal = []
            for i in range(len(final)):
                word = detoken.detokenize(final[i], return_str=True)
                cleanfinal.append(word)
            d = {'review': cleanfinal,'sentimen_gold': sentimen_gold}
            df = pd.DataFrame(data=d)
            df.to_excel(''+namaFile+'_PrePro.xlsx', index=False)
        if (seleksifitur=='off'):
            d = {'review': words, 'sentimen_gold': sentimen_gold}
            df = pd.DataFrame(data=d)
            df.to_excel('' + namaFile + '_PrePro.xlsx', index=False)
        print('================')
        print('preprocessing sukses')
        print('================')

