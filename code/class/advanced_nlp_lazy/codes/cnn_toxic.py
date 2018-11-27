# https://deeplearningcourses.com/c/deep-learning-advanced-nlp
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer # string을 쪼개기 위해
from keras.preprocessing.sequence import pad_sequences # 모든 데이터에 대해 똑같은 길이를 만들어주기 위해
from keras.layers import Dense, Input, GlobalMaxPooling1D 
from keras.layers import Conv1D, MaxPooling1D, Embedding # 1-d operation
from keras.models import Model
from sklearn.metrics import roc_auc_score # useful for binary classification


# Download the data:
# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
# Download the word vectors:
# http://nlp.stanford.edu/data/glove.6B.zip


# some configuration
MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 20000 # 통계적으로 native english speaker들이 평균적으로 20,000개의 vocabulary를 사용한다고 함. (물론 데이터마다 다르겠지만, 평균적으로 이 수치를 많이 사용)
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 10



# load in pre-trained word vectors
print('Loading word vectors...')
word2vec = {}
with open(os.path.join('glove.6B/glove.6B.%sd.txt' % EMBEDDING_DIM), encoding='UTF8') as f:
    # is just a space-separated text file in the format:
    # word vec[0] vec[1] vec[2] ...
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        word2vec[word] = vec
print('Found %s word vectors.' % len(word2vec))



# prepare text samples and their labels
print('Loading in comments...')

train = pd.read_csv("toxic-comment/train.csv")
sentences = train["comment_text"].fillna("DUMMY_VALUE").values # NA에 dummay value를 할당
possible_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
targets = train[possible_labels].values

print("max sequence length:", max(len(s) for s in sentences))
print("min sequence length:", min(len(s) for s in sentences))
s = sorted(len(s) for s in sentences)
print("median sequence length:", s[len(s) // 2])




# convert the sentences (strings) into integers
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE) # 두 가지 일을 함: (1) 토큰으로 쪼갬 (2) 토큰을 숫자로 변경
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
# print("sequences:", sequences); exit()


# get word -> integer mapping
word2idx = tokenizer.word_index 
print('Found %s unique tokens.' % len(word2idx))


# pad sequences so that we get a N x T matrix
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', data.shape)



# prepare embedding matrix
print('Filling pre-trained embeddings...')
num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1) # word2idx의 0-index는 패딩이다. 따라서, 크기도 +1 해줘야 함.
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx.items():
    if i < MAX_VOCAB_SIZE:
        embedding_vector = word2vec.get(word) # get함수 장점: 없으면 예외처리를 발생하지 않고, 그냥 none만 출력.
        if embedding_vector is not None:
            # words not found in embedding index will be all zeros. # 위에서 디폴트로 zero로 초기화 했음.
            embedding_matrix[i] = embedding_vector



# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(
    num_words,
    EMBEDDING_DIM,
    weights=[embedding_matrix],
    input_length=MAX_SEQUENCE_LENGTH,
    trainable=False
)


print('Building model...')

# train a 1D convnet with global maxpooling
input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
x = embedding_layer(input_)
x = Conv1D(128, 3, activation='relu')(x)
x = MaxPooling1D(3)(x)
x = Conv1D(128, 3, activation='relu')(x)
x = MaxPooling1D(3)(x)
x = Conv1D(128, 3, activation='relu')(x)
x = GlobalMaxPooling1D()(x) # 데이터가 time-series이기 때문에 globalmaxpooling을 해준다: T x M -> M 으로.
x = Dense(128, activation='relu')(x)
output = Dense(len(possible_labels), activation='sigmoid')(x) # (주의!) multi-label 문제이므로 softmax가 아닌 sigmoid를 사용함.

model = Model(input_, output) # 심플하게 모델 object를 입력, 출력 이라는 argument로 만들 수 있음.
model.compile(
    loss='binary_crossentropy', # (주의!) categorical_entropy가 아닌 점을 유념하자.
    optimizer='rmsprop',
    metrics=['accuracy']
)

print('Training model...')
r = model.fit(
    data,
    targets,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT
)


# plot some data
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['acc'], label='acc')
plt.plot(r.history['val_acc'], label='val_acc')
plt.legend()
plt.show()

# plot the mean AUC over each label
p = model.predict(data)
aucs = []
for j in range(6): # 6개의 label을 모두 구함.
    auc = roc_auc_score(targets[:,j], p[:,j])
    aucs.append(auc)
print(np.mean(aucs))
