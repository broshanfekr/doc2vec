from __future__ import unicode_literals
from hazm import Normalizer
from hazm import sent_tokenize, word_tokenize
from hazm import Stemmer, Lemmatizer
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from sklearn.linear_model import LogisticRegression
stemmer = Stemmer()
normalizer = Normalizer()


################## define variables-----------------------------------------------
num_features=100
num_Of_epoch = 10
sentences = []  # Initialize an empty list of sentences
mylabel = []    # labels for train sentences
#_________________________________________________________________________________
sentence_path = '/home/bero/Desktop/dataset/Persian Product Review Dataset/totaldata'
label_path = '/home/bero/Desktop/dataset/Persian Product Review Dataset/totallabel'

file_to_read = open(label_path, 'r')
labels = file_to_read.readlines()
mylabel = []
for line in labels:
    tmp = line.split('\n')
    mylabel.append(int(tmp[0]))
file_to_read.close()

file_to_read = open(sentence_path, 'r')
file_content = file_to_read.readlines()
file_to_read.close()

index  = 0
for line in file_content:
    tmp = line.split('\n')
    tmp = tmp[0]
    tmp = normalizer.normalize(tmp)
    #print(tmp)
    #print(sent_tokenize(tmp))
    word_tokenized = word_tokenize(tmp)
    #print(word_tokenized)
    labeledSent = TaggedDocument(words = word_tokenized, tags = [index])
    sentences.append(labeledSent)
    index += 1

num_features = 100
min_word_count = 5
context = 8
num_workers = 4
print("Training model...")
model = Doc2Vec(sentences, workers=num_workers, size = num_features, min_count = min_word_count, window = context)
print("model Trained.")

for epoch in range(num_Of_epoch):
    model.train(sentences)


mydocvec = model.docvecs
classifier = LogisticRegression()
classifier.fit(mydocvec, mylabel)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)


#########################################################################################################
#                   testing
sentence_path = '/home/bero/Desktop/dataset/Persian Product Review Dataset/testdata'
label_path = '/home/bero/Desktop/dataset/Persian Product Review Dataset/testlabel'

file_to_read = open(label_path, 'r')
labels = file_to_read.readlines()
test_label = []
for line in labels:
    tmp = line.split('\n')
    test_label.append(int(tmp[0]))
file_to_read.close()

file_to_read = open(sentence_path, 'r')
file_content = file_to_read.readlines()
file_to_read.close()

test_sentences_vec = []
index = 0
for line in file_content:
    tmp = line.split('\n')
    tmp = tmp[0]
    tmp = normalizer.normalize(tmp)
    #print(tmp)
    #print(sent_tokenize(tmp))
    word_tokenized = word_tokenize(tmp)
    #print(word_tokenized)
    myvec = model.infer_vector(word_tokenized, alpha=0.1, min_alpha=0.0001, steps=5)
    test_sentences_vec.append(myvec)
    index += 1

print(classifier.score(test_sentences_vec, test_label))
