
import math


tokenize = lambda doc: doc.lower().split(" ")

def term_frequency(term, tokenized_document):
    return tokenized_document.count(term)

vocab = []
def getVocab ():
    return vocab
tfidf_val = []
contain_doc = []
def getcontained ():
    return contain_doc
def gettfidf_val():
    return tfidf_val
def tfidf_train(tokenized_documents):
    if not vocab:
        print('')
    else:
        vocab.clear()
        tfidf_val.clear()
        contain_doc.clear()
    idf_values = {}
    all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
    for tkn in all_tokens_set:
        contains_token = map(lambda doc: tkn in doc, tokenized_documents)
        contained = sum(contains_token)
        idf_values[tkn] =  round(math.log10((len(tokenized_documents)+1)/(contained+1)),5)
        vocab.append(tkn)
        tfidf_val.append(idf_values[tkn])
        contain_doc.append(contained)
        if (tkn == 'mine'):
            print('term :',tkn,' idf : ',idf_values[tkn],'total doc',str(len(tokenized_documents)),'contained doc :',str(contained))
    return idf_values

def tfidf_test(tokenized_documents):
    idf_values = {}
    '''test_token_set = set([item for sublist in tokenized_documents for item in sublist])
    for tkn in test_token_set:
        contained = vocab.count(tkn)
        idf_values[tkn] = round(math.log10((len(tokenized_documents) + 1) / (contained + 1)), 3)'''
    for tkn in vocab:
        contains_token = map(lambda doc: tkn in doc, tokenized_documents)
        contained = sum(contains_token)
        idf_values[tkn] =  round(math.log10((len(tokenized_documents)+1)/(contained+1)),3)
    return idf_values

pos_doc = []
neg_doc = []

vocab_delta = []
vocab_pos = []
idf_neg = []
def getIdfNeg():
    return idf_neg
idf_pos = []
def getIdfPos():
    return  idf_pos
vocab_neg = []
zero_idf = []
def getZero():
    return zero_idf
def getvocab_delta():
    return vocab_delta
def getPos_vocab():
    return vocab_pos
def getNeg_vocab():
    return vocab_neg
def deltatfidf_train (tokenized_documents,labels):
    label = labels
    if not vocab_delta:
        print('')
    else:
        vocab_delta.clear()
        vocab_pos.clear()
        vocab_neg.clear()
        pos_doc.clear()
        neg_doc.clear()
        zero_idf.clear()
        idf_neg.clear()
        idf_pos.clear()
    idf_values = {}
    for i in range(len(tokenized_documents)):
        if (label[i]=='positif'):
            pos_doc.append(tokenized_documents[i])
        else:
            neg_doc.append(tokenized_documents[i])
    all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
    #vocab_pos.append(set([item for sublist in pos_doc for item in sublist]))
    print('doc positif : ',len(pos_doc))
    #vocab_neg.append(set([item for sublist in neg_doc for item in sublist]))
    print('doc negatif : ',len(neg_doc))
    for tkn in all_tokens_set:
        contains_token_pos = map(lambda doc: tkn in doc, pos_doc)
        contains_token_neg = map(lambda doc: tkn in doc, neg_doc)
        contained_pos = sum(contains_token_pos)
        contained_neg = sum(contains_token_neg)
        idf_values[tkn] = round(math.log10((contained_neg + 1) / (contained_pos + 1)), 3)
        #if (tkn == 'return'):
        #    print('term : ', tkn, ' delta idf :', idf_values[tkn],' jumlah di term positif : ',contained_pos, 'jumlah di negatif : ', contained_neg)
        #print('token : ',tkn,' contain_pos: ',contained_pos,' contain_neg : ',contained_neg)
        vocab_delta.append(tkn)
        if (idf_values[tkn]==0.0):
            #print('term : ', tkn, ' delta idf :', idf_values[tkn])
            zero_idf.append(tkn)
        if (idf_values[tkn]>0):
            idf_neg.append(idf_values[tkn])
            vocab_neg.append(tkn)
        if (idf_values[tkn]<0):
            idf_pos.append(idf_values[tkn])
            vocab_pos.append(tkn)
        '''if (contained_pos > 0) :
            vocab_pos.append(tkn)
        if (contained_neg > 0):
            vocab_neg.append(tkn)'''
    return idf_values

def detlatfidf_test(tokenized_documents):
    idf_values = {}
    test_token_set = set([item for sublist in tokenized_documents for item in sublist])
    test_token_list = list(test_token_set)
    for tkn in vocab_delta:
        if (test_token_list.count(tkn)>0):
            contains_token_pos = map(lambda doc: tkn in doc, pos_doc)
            contains_token_neg = map(lambda doc: tkn in doc, neg_doc)
            contained_pos = sum(contains_token_pos)
            contained_neg = sum(contains_token_neg)
        else:
            contained_neg = 0
            contained_pos = 0
        idf_values[tkn] = round(math.log10((contained_neg + 1) / (contained_pos + 1)), 3)
        #if (tkn == 'uncomfortable'):
         #   print('term : ', tkn, ' delta idf :', idf_values[tkn],' jumlah di term positif : ',contained_pos, 'jumlah di negatif : ', contained_neg)
        #print('term : ', tkn, ' idf :', idf_values[tkn])
    return idf_values


def tfidf(documents,status):
    tokenized_documents = [tokenize(d) for d in documents]
    if (status == 'train'):
        #print('masuk train')
        idf = tfidf_train(tokenized_documents)
    else:
        #print('masuk test')
        idf = tfidf_test(tokenized_documents)
    tfidf_documents = []
    for document in tokenized_documents:
        doc_tfidf = []
        for term in idf.keys():
            tf = term_frequency(term, document)
            doc_tfidf.append(tf * idf[term])
        tfidf_documents.append(doc_tfidf)
    return tfidf_documents

def deltatfidf(documents,labels,status):
    tmp = 0.0
    tokenized_documents = [tokenize(d) for d in documents]
    if (status == 'train'):
        idf = deltatfidf_train(tokenized_documents,labels)
    else:
        idf = detlatfidf_test(tokenized_documents)
    tfidf_documents = []
    for document in tokenized_documents:
        doc_tfidf = []
        for term in idf.keys():
            tf = term_frequency(term, document)
            doc_tfidf.append(tf * idf[term])
            tmp = (tf * idf[term])
            '''if (status == 'train'):
                if (tmp > 0):
                    idf_neg.append(tmp)
                    vocab_neg.append(term)
                if (tmp < 0):
                    idf_pos.append(tmp)
                    vocab_pos.append(term)'''
        tfidf_documents.append(doc_tfidf)

    return tfidf_documents

data = ['I love dog love', 'I hate knitting', 'I love your knitting skill', 'Mike hate you']
test = ['I hate you','I love you','you are bad']
labels = ['positif', 'negatif', 'positif', 'negatif']
label_test = ['neg','pos','neg']


'''nilai_tf = tfidf(data,'train')
test_tf = tfidf(test,'test')
clf = LinearSVC(random_state=0)
clf.fit(nilai_tf,labels)
pred = clf.predict(test_tf)
acc = accuracy_score(label_test,pred)
print(acc)'''

#deltatfidf(data,labels,'train')


