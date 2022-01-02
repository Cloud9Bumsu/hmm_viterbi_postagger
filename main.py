import trainer

def Viterbi(sentence):

    len_sentence = len(sentence)

    # Viterbi:
    # Viterbi gives a good way of computing all probabilities
    # as fast as possible.
    unigram_cpd, bigram_cpd = trainer.Trainer()
    distinct_unigram_tags = trainer.Distinguish(unigram_cpd)

    viterbi_tag = {}
    viterbi_backpointer = {}

    for tag in distinct_unigram_tags:
        if tag == "START":
            continue
        # Prob of ["START"] -> [initial state]
        viterbi_tag[tag] = bigram_cpd["START"].prob(tag) * unigram_cpd[tag].prob(sentence[0])
        viterbi_backpointer[tag] = "START"

    # for each step i in 1 .. len
    # store a dictionary
    # that maps each tag X
    # to the probability of the best tag sequence of length i that ends in X
    viterbi_main = []

    # for each step i in 1 .. len
    # store a dictionary
    # that maps each tag X
    # to the previous tag in the best tag sequence of length i that ends in X
    backpointer_main = []

    viterbi_main.append(viterbi_tag)
    backpointer_main.append(viterbi_backpointer)

    current_best = max(viterbi_tag.keys(), key=lambda tag: viterbi_tag[tag])

    print()
    print("Word","'" + sentence[0] + "'", "current best two-tag sequence:", viterbi_backpointer[current_best])

    for index in range(1, len_sentence):
        curr_viterbi = {}
        curr_backpointer = {}
        prev_viterbi = viterbi_main[-1]

        for brown_tag in distinct_unigram_tags:

            if brown_tag != "START":
                # if this tag is X and the current word is w, then
                # find the previous tag Y such that
                # the best tag sequence that ends in X
                # actually ends in Y X
                # that is, the Y that maximizes
                # prev_viterbi[ Y ] * P(X | Y) * P(w | X) 
                prev_best = max(prev_viterbi.keys(),
                                    key=lambda prevtag:\
                                        prev_viterbi[prevtag] * bigram_cpd[prevtag].prob(brown_tag) * unigram_cpd[brown_tag].prob(
                                        sentence[index]))

                curr_viterbi[brown_tag] = prev_viterbi[prev_best] * \
                                    bigram_cpd[prev_best].prob(brown_tag) * unigram_cpd[brown_tag].prob(sentence[index])
                curr_backpointer[brown_tag] = prev_best
        
        current_best = max(curr_viterbi.keys(), key=lambda tag: curr_viterbi[tag])
        print("Word", "'" + sentence[index] + "'", "current best two-tag sequence:", curr_backpointer[current_best], current_best)


        viterbi_main.append(curr_viterbi)
        backpointer_main.append(curr_backpointer)

    # now find the probability of each tag
    # to have "END" as the next tag,
    # and use that to find the overall best sequence
    prev_viterbi = viterbi_main[-1]
    prev_best = max(prev_viterbi.keys(), key=lambda prev_tag: \
                                    prev_viterbi[prev_tag])

    print()
    print()

    prob_tag_sequence = prev_viterbi[prev_best] * bigram_cpd[prev_best].prob("END")
    best_tag_sequence = ["END", prev_best]
    backpointer_main.reverse()

    # go backwards through the list of backpointers
    # in each case:
    # the following best tag is the one listed under
    # the backpointer for the current best tag
    current_best_tag = prev_best
    for backpointer in backpointer_main:
        print("current best tag:", current_best_tag, "backpointer[current_best_tag]:",backpointer[current_best_tag])
        best_tag_sequence.append(backpointer[current_best_tag])
        current_best_tag = backpointer[current_best_tag]


    best_tag_sequence.reverse()
    print()
    print("The sentence given is :")
    for word in sentence:
        print(word,"")
    
    print()
    print()
    print("The best tag sequence using HMM for the given")

    for best_tag in best_tag_sequence:
        print(best_tag, "",)
    
    print()
    print()
    print("The probability of the best tag sequence printed above is given by : ", prob_tag_sequence)


def main():
    sample_sentence = ["I","like","hamburger"]
    Viterbi(sample_sentence)

if __name__ == "__main__":
    main()