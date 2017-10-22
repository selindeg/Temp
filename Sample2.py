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



class NotSoElegantSample():


    @staticmethod
    def preprocess(term):  # term is one file in a one class
        exclude = set(string.punctuation)
        term = ''.join(ch for ch in term if ch not in exclude).lower()
        return term  # term.lower().translate( string.punctuation )

    @staticmethod
    def stemmerTrFps6(term):
        return term[:6]

    def tokenize(self,text):
        stemmer = self.stemmerTrFps6
        tokens = nltk.word_tokenize(text)
        stems = self.stem_tokens(tokens, stemmer)
        return stems

    def run(self, path):

        print("Reading corpus names from path", path)
        fileNames = []

        classLabels = []

        for subdir, dirs, files in os.walk(path):
            for file in files:

                file_path = subdir + os.path.sep + file

                print(file_path)
                fileNames.append(file_path) # output
                classLabels.append(subdir[len(path):]) # output - this needs to be converted to integers for classifier training!


        textVectorizer = TfidfVectorizer(min_df=2, preprocessor=self.preprocess, lowercase=False, use_idf=True,
                            tokenizer=self.tokenize)
        docTermMatrix = textVectorizer.fit_transform((open(f, encoding='utf-8').read() for f in fileNames))

        classifier = svm.SVC(kernel='linear', C=1.0, decision_function_shape=None)

        predictions = cross_validation.cross_val_predict(classifier, docTermMatrix, classLabels, cv=10)

        # score=cross_val_score(classifier, dataset, classlabels,cv=n,scoring='accuracy')  avg is same as metrics.accuracy_score

        accuracy = metrics.accuracy_score(classLabels, predictions)
        precision = precision_score(classLabels, predictions, average=None)
        recall = recall_score(classLabels, predictions, average=None)

        F1=metrics.f1_score(classLabels, predictions, average=None)
        F1_macro=metrics.f1_score(classLabels, predictions, average='macro')
        F1_micro=metrics.f1_score(classLabels, predictions, average='micro')

        confusionMatrix = confusion_matrix(classLabels, predictions)

        print("Accuracy :", accuracy)
        print("Precision:", precision)
        print("Recall   :", recall)
        print("F1       :" , F1)
        print("F1 macro :" , F1_macro)
        print("F1 micro :" , F1_micro)
        print("Confusion Matrix :\n " , confusionMatrix)


ex = NotSoElegantSample()
ex.run("C:/Users/selin/PycharmProjects/SampleProject/1150haber/raw_texts")
