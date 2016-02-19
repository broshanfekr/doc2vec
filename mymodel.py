import pandas as pd
import nltk
import numpy as np
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from gensim.models import doc2vec
from nltk.corpus import stopwords
import gensim
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup
import re

################## define variables-----------------------------------------------
num_features=100
num_Of_epoch = 10
sentences = []  # Initialize an empty list of sentences
mylabel = []    # labels for train sentences
#_________________________________________________________________________________
################## define functions-----------------------------------------------
def Review2Word(review, remove_stopwords=False):
#    review_text = BeautifulSoup(review).get_text()    #Remove HTML tags
    review_text = re.sub("[^a-zA-Z]"," ", review)   #Remove non-letters from words
    words = review_text.lower().split()     #Convert words to lower case and split them
    #remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return(words)   #Return a list of words

def Review2Doc(review, mytag, remove_stopwords=False):
    labeledSent = TaggedDocument(words = Review2Word(review, remove_stopwords), tags = [mytag])
    return labeledSent

# new labels to self.vocab
def add_new_labels(sentences, model):
    sentence_no = -1
    total_words = 0
    vocab = model.vocab
    model_sentence_n = len([l for l in vocab if l.startswith("SENT")])
    n_sentences = 0
    for sentence_no, sentence in enumerate(sentences):
        sentence_length = len(sentence.words)
        for label in sentence.labels:
            label_e = label.split("_")
            label_n = int(label_e[1]) + model_sentence_n
            label = "{0}_{1}".format(label_e[0], label_n)
            total_words += 1
            if label in vocab:
                vocab[label].count += sentence_length
            else:
                vocab[label] = gensim.models.word2vec.Vocab(
                    count=sentence_length)
                vocab[label].index = len(model.vocab) - 1
                vocab[label].code = [0]
                vocab[label].sample_probability = 1.
                model.index2word.append(label)
                n_sentences += 1
    return n_sentences
############################### End Functions ###################################################
# Read data from files
train = pd.read_csv("/home/bero/Desktop/dataset/kaggle/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("/home/bero/Desktop/dataset/kaggle/testData.tsv", header=0, delimiter="\t", quoting=3)
unlabeled_train = pd.read_csv("/home/bero/Desktop/dataset/kaggle/unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


print "Parsing sentences from training set"
Total_Train_Samples = len(train["review"])
for i in range(Total_Train_Samples):#train["review"]:
    row = train.iloc[i,:]
    review = row["review"]
    label = row["sentiment"]
    mylabel.append(label)
    sentences.append(Review2Doc(review = review, mytag = i))
'''
num_features = 100
min_word_count = 5
context = 8
num_workers = 4
print "Training model..."
model = Doc2Vec(sentences, workers=num_workers, size = num_features, min_count = min_word_count, window = context)
print "model Trained."
print model.most_similar('good')

for epoch in range(num_Of_epoch):
    model.train(sentences)


model.save('./imdb.d2v')
'''
model = Doc2Vec.load('./imdb.d2v')

mydocvec = model.docvecs
classifier = LogisticRegression()
classifier.fit(mydocvec, mylabel)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)

#########################################################################################################
#                   testing
#classifier.score(test_arrays, test_labels)

test_sentences_vec = []
test_label = []
print "Parsing sentences from testing set"
Total_Test_Samples = len(test["review"])
for i in range(Total_Test_Samples):#train["review"]:
    row = train.iloc[i,:]
    review = row["review"]
    label = row["sentiment"]
    test_label.append(label)
    mydocwords = Review2Doc(review = review, mytag = i)
    mydocwords = mydocwords[0]
    myvec = model.infer_vector(mydocwords, alpha=0.1, min_alpha=0.0001, steps=5)
    test_sentences_vec.append(myvec)

# Set model.train_words to False and model.train_labels to True
#model.train_words = False
#model.train_lbls = True
print classifier.score(test_sentences_vec, test_label)

'''
sentences=doc2vec.TaggedLineDocument("/home/bero/Desktop/dataset/kaggle/labeledTrainData.tsv")
model1 = doc2vec.Doc2Vec(sentences,size = 100, window = 300, min_count = 10, workers=4)
docvec1 = model1.docvecs
print docvec1[99]
'''
