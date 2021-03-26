from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    'I love dog',
    'I love cat',
]

test_data = [
    'I love my dog',
    'my dog love my cat',
]

tokenizer = Tokenizer(num_words=100)  # maximalis szoszam
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index  # kerjuk a szotarat, szavak es azonositok

sequences = tokenizer.texts_to_sequences(sentences)
test_sequences = tokenizer.texts_to_sequences(test_data)

print(word_index)
print('sequences ', sequences)
print('test_sequences ', test_sequences)

