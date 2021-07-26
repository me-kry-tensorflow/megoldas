from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    'I love dog',
    'I love cat',
]

test_data = [
    'I love my dog',
    'my dog love my little cat',
]

# nem veszitunk elemet, mindig annyi elem lesz, amennyi a mintaban volt
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")  # maximalis szoszam, nem ismert uj elemek helyett '<OOV>' lesz
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index  # kerjuk a szotarat, szavak es azonositok

sequences = tokenizer.texts_to_sequences(sentences)
# az '<OOV>' indexe (1) meg fog jelenni a test_sequences, mert a test_databan van ismeretlen elem.
# minden ismeretlen szo helyett az OOV indexe lesz.
# jelentesbol veszitunk, de a numerikus reprezentacio nem serul
test_sequences = tokenizer.texts_to_sequences(test_data)

print(word_index)
print('sequences ', sequences)
print('test_sequences ', test_sequences)

