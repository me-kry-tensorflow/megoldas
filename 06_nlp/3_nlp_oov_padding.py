from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'I love dog',
    'I love cat',
]

test_data = [
    'I love my dog',
    'my dog love my cat',
    'my dog very love my beautiful cat',
]

# nem veszitunk elemet, mindig annyi elem lesz, amennyi a mintaban volt
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")  # maximalis szoszam, nem ismert uj elemek helyett '<OOV>' lesz
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index  # kerjuk a szotarat, szavak es azonositok

sequences = tokenizer.texts_to_sequences(sentences)
test_sequences = tokenizer.texts_to_sequences(test_data)

# a leghoszabb alapjan kiegesziti a rovideket alol 0-akkal
padded_test_data = pad_sequences(test_sequences)
print(word_index)
print('sequences ', sequences)
print('test_sequences ', test_sequences)
print('pad_sequences ', padded_test_data)

