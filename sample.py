import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import string
from sklearn import svm
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import metrics
from TurkishStemmer import TurkishStemmer
import numpy
import math
import numpy as np

class Sample():

    def preprocess(self, term):
        "get rid of punctiation"
        term = term.translate(string.punctuation)
        return term

    def stem_tokens(self, tokens, stemmer):
        stemmed=[stemmer.stem(item) for item in tokens]
        return stemmed

    def tokenize(self, text):
        stemmer = TurkishStemmer()
        tokens = nltk.word_tokenize(text)
        stems = self.stem_tokens(tokens, stemmer)
        return stems


    def run(self, path):

        print("Reading corpus names from path", path)
        fileNames = []
        classLabels = []
        for subdir, dirs, files in os.walk(path):
            for file in files:
                #print(file)
                if (file.endswith(".txt")):
                    file_path = subdir + os.path.sep + file
                    fileNames.append(file_path)  # output
                    classLabels.append(
                        subdir[len(path):])  # output - this needs to be converted to integers for classifier training!
        #convert class labels to 0..4
        classLabels = [w.replace('\\ekonomi', '0') for w in classLabels]
        classLabels = [w.replace('\\magazin', '1') for w in classLabels]
        classLabels = [w.replace('\\saglik', '2') for w in classLabels]
        classLabels = [w.replace('\\siyasi', '3') for w in classLabels]
        classLabels = [w.replace('\\spor', '4') for w in classLabels]
        #ytrain vector yazdirma/class labels string arrayini integer array e olarak kopyaladim
        y_train = [int(numeric_string) for numeric_string in classLabels]
        print(y_train)
        textVectorizer = TfidfVectorizer(min_df=2, preprocessor=self.preprocess, lowercase=False, use_idf=True, tokenizer=self.tokenize)
        #Xtrain matrix
        docTermMatrix = textVectorizer.fit_transform((open(f, encoding='iso-8859-9').read() for f in fileNames))



        #SMC Algorithm

        wordCountsInClass=numpy.zeros(shape=(5,14784))
        totalNumWordsInClass=numpy.zeros(shape=5)
        scores=numpy.zeros(shape=(5,14784))
        wordCounts=numpy.zeros(shape=14784)
        totalNumWords=0
        featureArray=textVectorizer.get_feature_names()
        print(featureArray.__len__())#14784
        print(docTermMatrix[0, 977])  # 14784
        for docIndex in range(0, 1150):
          for wordIndex in range(0, 14784):
            #kacinci idexte oldugunu anlamak icin yazdirma
            print(docIndex)
            print(wordIndex)
            wordCountsInClass[y_train[docIndex]][wordIndex] += docTermMatrix[docIndex, wordIndex]
            wordCounts[wordIndex] += docTermMatrix[docIndex, wordIndex]
            totalNumWordsInClass[y_train[docIndex]] += docTermMatrix[docIndex, wordIndex]
            totalNumWords += docTermMatrix[docIndex, wordIndex]

        #print("hello")
        for numberOfclass in range(0, 5):
            for numberOfWords in range(0, 14784):
                 if (wordCountsInClass[numberOfclass][numberOfWords] == 0):
                    # print("hello")
                    scores[numberOfclass][numberOfWords] = 0
                 else:
                     K = wordCounts[numberOfWords];
                     m=wordCountsInClass[numberOfclass][numberOfWords];
                     minus1_div_m = (-1 / m)
                     K_fak = math.lgamma(K + 1)
                     m_fak = math.lgamma(m+ 1)
                     K_minus_m_fak = math.lgamma(K - m + 1)
                     m_minus_1 = m - 1
                     N = totalNumWords / totalNumWordsInClass[numberOfclass]
                     secondPart = m_minus_1 * math.log(N)
                     firstPart = math.exp(K_fak - m_fak - K_minus_m_fak)
                     NFA = math.log(firstPart) - secondPart
                     scores[numberOfclass][numberOfWords] = minus1_div_m * NFA
                     print(scores[numberOfclass][numberOfWords])


ex = Sample()
ex.run("C:/Users/selin/PycharmProjects/SampleProject/1150haber/raw_texts")
