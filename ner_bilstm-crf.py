import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#Import dataset
data = pd.read_csv("ner_dataset.csv", encoding="latin1")

#Clean dataframe
data = data.rename(columns={"Sentence #": "Sentence"})
data.drop(['POS'], axis=1, inplace=True)
data["Tag"] = data["Tag"].str.upper()
data = data.fillna(method='ffill')

#List of non-O unique tags
print(data['Tag'].unique())

#Visualise distribution of unique non-O tags
plt.figure(figsize=(8, 6))
data.Tag[data.Tag != 'O']\
    .value_counts()\
    .plot\
    .barh();

# Retrive sentences and tags from dataset
class get_sentences(object):

    def __init__(self, data):
        self.sent = 1
        self.dataset = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["Word"].values.tolist(),
                                                     s["Tag"].values.tolist())]
        self.grouped = self.dataset.groupby("Sentence").apply(agg_func)
        self.sentences = [s for s in self.grouped]


getter = get_sentences(data)
sentences = getter.sentences

#Compare raw and tagged sentences
raw_sentence = [" ".join([s[0] for s in sent]) for sent in getter.sentences]
print("Raw sentence: {}".format(raw_sentence[27]))

print("Tagged sentence:" + "\n" + "\n".join(map(str, sentences[27])))

#Define attributes
MAX_LEN = max([len(s) for s in sentences])
DIM_EMBEDDINGS = 50

#Map sentences and NER tags
from future.utils import iteritems

words = list(set(data["Word"].values))
number_words = len(words)

word2idx = {w: i+1 for i, w in enumerate(words)}
idx2word = {i: w for w, i in iteritems(word2idx)}

tags = list(set(data["Tag"].values))
number_tags = len(tags)

tag2idx = {t: i+1 for i, t in enumerate(tags)}
idx2tag = {i: w for w, i in iteritems(tag2idx)}

#Pad sequence data
from tensorflow.keras.preprocessing.sequence import pad_sequences

X = [[word2idx[w[0]] for w in s] for s in sentences]
X = pad_sequences(maxlen=MAX_LEN, sequences=X, padding="post")
print(X)

#Pad sequence data
y = [[tag2idx[w[1]] for w in s] for s in sentences]
y = pad_sequences(maxlen=MAX_LEN, sequences=y, padding="post")

#One hot encode
from tensorflow.keras.utils import to_categorical
y = [to_categorical(i, num_classes=number_tags+1) for i in y]
print(y)

#Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Initialise model
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Dropout, TimeDistributed, Bidirectional
from keras_contrib.layers import CRF

#Input layer
input = Input(shape=(MAX_LEN,))

#Embedding layer
model = Embedding(input_dim=number_words+1,
                  input_length=MAX_LEN,
                  output_dim=DIM_EMBEDDINGS)(input)

#BiLSTM layer
model = Bidirectional(LSTM(units=DIM_EMBEDDINGS,
                           return_sequences=True,
                           dropout=0.5,
                           recurrent_dropout=0.5))(model)
model = LSTM(units=DIM_EMBEDDINGS*2,
             return_sequences=True,
             dropout=0.5,
             recurrent_dropout=0.5)(model)

#TimeDistributed layer
model = TimeDistributed(Dense(number_tags+1, activation="relu"))(model)

#CRF layer
crf = CRF(number_tags+1)
out = crf(model)

model = Model(input, out)

#Compile model
model.compile(optimizer="rmsprop",
              loss=crf.loss_function,
              metrics=[crf.accuracy, "accuracy"])
print(model.summary())

history = model.fit(X_train, np.array(y_train),
                    batch_size=32,
                    epochs=10,
                    validation_split=0.2,
                    verbose=1)

# Visualising model accuracy
plt.plot(history.history['crf_viterbi_accuracy'])
plt.plot(history.history['val_crf_viterbi_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'])
plt.show()

# Visualising model loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'])
plt.show()

#Evaluating the model
from sklearn.metrics import classification_report
y_pred = model.predict(np.array(X_test))
print(classification_report(np.argmax(y_test, 2).ravel(),
                            np.argmax(y_pred, axis=2).ravel(),
                            labels=list(idx2tag.keys()),
                            target_names=list(idx2tag.values())))

#Compare actual and predicted tags in a random sample
r = np.random.randint(0, X_test.shape[0])
y_rand = model.predict(np.array([X_test[r]]))
y_rand = np.argmax(y_rand, axis=-1)
y_true = np.argmax(np.array(y_test), axis=-1)[r]

print("{:15}{:5}\t{}".format("Word", "Actual", "Predicted"))
print("-"*35)

n = 0
for (w, t, pred) in zip(X_test[r], y_true, y_rand[0]):
    if n==20:
        break
    else:
        print("{:15}{}\t{}".format(words[w-1], tags[t], tags[pred]))
        n+=1

#Save model
import pickle
with open('word_to_index.pickle', 'wb') as f:
    pickle.dump(word2idx, f)

with open('tag_to_index.pickle', 'wb') as f:
    pickle.dump(tag2idx, f)