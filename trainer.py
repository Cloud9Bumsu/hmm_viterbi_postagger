import nltk

from nltk.corpus import brown

def Trainer():
    brown_word_tags = []

    for brown_sent in brown.tagged_sents():
        brown_word_tags.append(('START', 'START'))

        for words, tag in brown_sent:
            brown_word_tags.extend([(tag[:2], words)])

        brown_word_tags.append(("END", "END"))

    # get the conditional frequency distribution for the brown word tags
    cfd_tag_words = nltk.ConditionalFreqDist(brown_word_tags)
    cpd_tag_words = nltk.ConditionalProbDist(cfd_tag_words, nltk.MLEProbDist)

    # Estimating P(ti | t{i-1}) from corpus data using Maximum Likelihood Estimation (MLE): 
    # P(ti | t{i-1}) = count(t{i-1}, ti) / count(t{i-1})
    brown_tags = []
    for tag, words in brown_word_tags:
        brown_tags.append(tag)

    #Make Conditional Frequency Distribution:
    # count(t{i-1} ti)
    cfd_tags = nltk.ConditionalFreqDist(nltk.bigrams(brown_tags))
    # make conditional probability distiribution, usimg mle:
    # P(ti | t{i-1})
    cpd_tags = nltk.ConditionalProbDist(cfd_tags, nltk.MLEProbDist)

    return cpd_tag_words, cpd_tags

def Log_prob(unigram, bigram):
    pass

def Distinguish(tags):
    return set(tags)


def find():
    import numpy as np
    import matplot.pyplot as plt
    n_samples, n_features = 200,50
    x = np.random.randn(n_samples, n_features)
