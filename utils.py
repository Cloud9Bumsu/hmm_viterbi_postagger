import nltk
import random

from nltk.corpus import brown

def random_brown():
    return brown.tagged_sents()[random.randrange(0,57340)]

def add_random_noise(sentence, alpha = 0.1):

    noised_sentence = []

    for word in sentence:
        # alpha 확률로 word를 random word로 교체
        if random.random() < alpha:
            noise_sentence = random_brown()
            noise_word = noise_sentence[random.randrange(len(noise_sentence))]
            noised_sentence.append(noise_word[0])

        else:
            noised_sentence.append(word)
    
    #print("original_sentence :", sentence)
    #print("noised_sentence :", noised_sentence)

    return noised_sentence