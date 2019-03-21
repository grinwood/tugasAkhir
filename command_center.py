import Preprocessing as prePro
import Homemade_KFOLD_sentimen as senti

posPath = 'E:\Document\Dokumen\kuliah\TA\dataset Delta TF-IDF\sorted_data\electronics\Rian_positive_json.txt'
negPath = 'E:\Document\Dokumen\kuliah\TA\dataset Delta TF-IDF\sorted_data\electronics\Rian_negative_json.txt'

'''jumData = 200
namaFile= '200Dataset'
prePro.new_jsonLoader(posPath,negPath,jumData,namaFile)
prePro.runAllPreprocessing(namaFile,'on')
senti.fold_tfidf(namaFile)
senti.fold_deltatfidf(namaFile)
jumData = 400
namaFile= '400Dataset'
prePro.new_jsonLoader(posPath,negPath,jumData,namaFile)
prePro.runAllPreprocessing(namaFile,'on')
senti.fold_tfidf(namaFile)
senti.fold_deltatfidf(namaFile)
jumData = 600
namaFile= '600Dataset'
prePro.new_jsonLoader(posPath,negPath,jumData,namaFile)
prePro.runAllPreprocessing(namaFile,'on')
senti.fold_tfidf(namaFile)
senti.fold_deltatfidf(namaFile)
jumData = 800
namaFile= '800Dataset'
prePro.new_jsonLoader(posPath,negPath,jumData,namaFile)
prePro.runAllPreprocessing(namaFile,'on')
senti.fold_tfidf(namaFile)
senti.fold_deltatfidf(namaFile)'''
jumData = 1800
namaFile= '1800Dataset'
prePro.new_jsonLoader(posPath,negPath,jumData,namaFile)
prePro.runAllPreprocessing(namaFile,'on')
senti.fold_tfidf(namaFile)
senti.fold_deltatfidf(namaFile)