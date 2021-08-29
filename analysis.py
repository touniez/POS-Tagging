# Anthony Zheng

import math
import random
import numpy
from collections import *
import matplotlib.pyplot as plt


def read_pos_file(filename):
    """
    Parses an input tagged text file.
    Input:
    filename --- the file to parse
    Returns:
    The file represented as a list of tuples, where each tuple
    is of the form (word, POS-tag).
    A list of unique words found in the file.
    A list of unique POS tags found in the file.
    """
    file_representation = []
    unique_words = set()
    unique_tags = set()
    f = open(str(filename), "r")
    for line in f:
        if len(line) < 2 or len(line.split("/")) != 2:
            continue
        word = line.split("/")[0].replace(" ", "").replace("\t", "").strip()
        tag = line.split("/")[1].replace(" ", "").replace("\t", "").strip()
        file_representation.append( (word, tag) )
        unique_words.add(word)
        unique_tags.add(tag)
    f.close()
    return file_representation, unique_words, unique_tags

class HMM:
    """
    Simple class to represent a Hidden Markov Model.
    """
    def __init__(self, order, initial_distribution, emission_matrix, transition_matrix):
        self.order = order
        self.initial_distribution = initial_distribution
        self.emission_matrix = emission_matrix
        self.transition_matrix = transition_matrix

def bigram_viterbi(hmm, sentence):
    """
    Run the Viterbi algorithm to tag a sentence assuming a bigram HMM model.
    Inputs:
      hmm --- the HMM to use to predict the POS of the words in the sentence.
      sentence ---  a list of words.
    Returns:
      A list of tuples where each tuple contains a word in the
      sentence and its predicted corresponding POS.
    """
    # Initialization
    viterbi = defaultdict(lambda: defaultdict(int))
    backpointer = defaultdict(lambda: defaultdict(int))
    unique_tags = set(hmm.initial_distribution.keys()).union(set(hmm.transition_matrix.keys()))
    for tag in unique_tags:
        #print(hmm.emission_matrix[tag])
        if (hmm.initial_distribution[tag] != 0) and (hmm.emission_matrix[tag][sentence[0]] != 0):
            viterbi[tag][0] = math.log(hmm.initial_distribution[tag]) + math.log(hmm.emission_matrix[tag][sentence[0]])
        else:
            viterbi[tag][0] = -1 * float('inf')
    # Dynamic programming.
    for t in range(1, len(sentence)):
        backpointer["No_Path"][t] = "No_Path"
        for s in unique_tags:
            max_value = -1 * float('inf')
            max_state = None
            for s_prime in unique_tags:
                val1 = viterbi[s_prime][t-1]
                val2 = -1 * float('inf')
                if hmm.transition_matrix[s_prime][s] != 0:
                    val2 = math.log(hmm.transition_matrix[s_prime][s])
                    #print("val2 " + str(val2))
                curr_value = val1 + val2
                if curr_value > max_value:
                    max_value = curr_value
                    max_state = s_prime
            val3 = -1 * float('inf')
            if hmm.emission_matrix[s][sentence[t]] != 0:
                val3 = math.log(hmm.emission_matrix[s][sentence[t]])
            viterbi[s][t] = max_value + val3

            if max_state == None:
                backpointer[s][t] = "No_Path"
            else:
                backpointer[s][t] = max_state
    for ut in unique_tags:
        string = ""
        for i in range(0, len(sentence)):
            if (viterbi[ut][i] != float("-inf")):
                string += str(int(viterbi[ut][i])) + "\t"
            else:
                string += str(viterbi[ut][i]) + "\t"

    # Termination
    max_value = -1 * float('inf')
    last_state = None
    final_time = len(sentence) - 1
    for s_prime in unique_tags:
        if viterbi[s_prime][final_time] > max_value:
            max_value = viterbi[s_prime][final_time]
            last_state = s_prime
    if last_state == None:
        last_state = "No_Path"

    # Traceback
    tagged_sentence = []
    tagged_sentence.append((sentence[len(sentence)-1], last_state))
    for i in range(len(sentence)-2, -1, -1):
        next_tag = tagged_sentence[-1][1]
        curr_tag = backpointer[next_tag][i+1]
        tagged_sentence.append((sentence[i], curr_tag))
    tagged_sentence.reverse()
    return tagged_sentence


def compute_counts(training_data: list, order: int) -> tuple:
    """
    Processes tag and word counts from inputted training data
    Inputs:
        training_data --- a list of tuples of (word, tag) from the training corpus
        order --- the order of markov chain to process data for, can be 2nd or 3rd order
    Returns:
        tuple of number of tokens total, count of words for each tag, count for each tag, as well as
        counts for sequences of tags appearing in training_data, depeneding on the order inputted
    """
    tokens = len(training_data)
    ctw = defaultdict(lambda : defaultdict(int))
    ct = defaultdict(int)
    tagseq2 = defaultdict(lambda : defaultdict(int))
    for (word, tag) in training_data:
        #modify ctw and ct
        ct[tag] += 1
        ctw[tag][word] += 1
    for index in range(tokens-1):
        tagseq2[training_data[index][1]][training_data[index+1][1]] += 1
    if order == 3:
        tagseq3 = defaultdict(lambda : defaultdict(lambda : defaultdict(int)))
        for index in range(tokens-2):
            currentseq = (training_data[index][1], training_data[index+1][1], training_data[index+2][1])
            tagseq3[currentseq[0]][currentseq[1]][currentseq[2]] += 1
        return (tokens, ctw, ct, tagseq2, tagseq3)
    return (tokens, ctw, ct, tagseq2)

def compute_initial_distribution(training_data: list, order: int) -> dict:
    """
    Computes the initial distribution of tags for the inputted training corpus
    Inputs:
        training_data --- a list of (word, tag) of the training corpus
        order --- the order of markov chain to process data for, can be 2nd or 3rd order
    Returns:
        a default dictionary representing the probability that a certain sequence of tags will begin a sentence
    """
    tot_starts = 0
    pi_vals2 = defaultdict(int)
    pi_vals3 = defaultdict(lambda : defaultdict(int))
    if order == 2:
        for i in range(len(training_data)):
            if training_data[i-1][1] == '.' or training_data[i-1][1] == '!' or training_data[i-1][1] == '?':
                tot_starts += 1
                pi_vals2[training_data[i][1]] += 1
        for i in pi_vals2.keys():
            pi_vals2[i] /= tot_starts
        return pi_vals2
    if order == 3:
        for i in range(len(training_data)):
            if training_data[i-2][1] == '.' or training_data[i-2][1] == '!' or training_data[i-2][1] == '?':
                tot_starts += 1
                pi_vals3[training_data[i-1][1]][training_data[i][1]] += 1
        for i in pi_vals3.keys():
            for j in pi_vals3[i].keys():
                pi_vals3[i][j] /= tot_starts
        return pi_vals3
    return

def compute_emission_probabilities(unique_words: list, unique_tags: list, W: dict, C: dict) -> dict:
    """
    Calculates the emission probability of words for every tag
    Inputs:
        unique_words --- a sequence of unique words that appears in the training corpus
        unique_tags --- a sequence of unique tags that appears in the training corpus
        W --- a dictionary of how often a word appears for each tag in the training corpus
        C --- a dictionary of how often a tag appears in the training corpus
    Returns:
        dictionary of emisison probabilities of each word for each tag
    """
    emission = defaultdict(lambda : defaultdict(int))
    for tag in W:
        for word in W[tag]:
            emission[tag][word] = W[tag][word]/C[tag]
    return emission

def compute_lambdas(unique_tags: list, num_tokens: int, C1: dict, C2: dict, C3: dict, order: int) -> list:
    """
    Computes lambdas for a cross validation technique for transition probabilities
    Inputs:
        unique_words --- a sequence of unique words that appears in the training corpus
        unique_tags --- a sequence of unique tags that appears in the training corpus
        C1 --- the counts of each tag
        C2 --- the counts of each 2 tag sequence
        C3 --- the ocunts of each 3 tag sequence
        order --- the order of markov chain to process data for, can be 2nd or 3rd order
    Returns:
        a list of 3 lambdas for the smoothing equations
    """
    lamb0 = 0
    lamb1 = 0
    lamb2 = 0
    if order == 2:
        for t1 in unique_tags:
            for t2 in unique_tags:
                if t1 in C2.keys() and t2 in C2[t1].keys() and C2[t1][t2] > 0:
                    alpha0 = (C1[t2] - 1) / num_tokens
                    if C1[t1] - 1 == 0:
                        alpha1 = 0
                    else:
                        alpha1 = (C2[t1][t2] - 1) / (C1[t1] - 1)
                    if alpha0 >= alpha1:
                        lamb0 = lamb0 + C2[t1][t2]
                    else: # alpha1 > alpha0:
                        lamb1 = lamb1 + C2[t1][t2]
        lamb0 = lamb0 / (lamb0 + lamb1)
        lamb1 = lamb1 / (lamb0 + lamb1)
        return [lamb0, lamb1, lamb2]
    if order == 3:
        for t1 in unique_tags:
            for t2 in unique_tags:
                for t3 in unique_tags:
                    if C3[t1] and C3[t1][t2] and C3[t1][t2][t3] and C3[t1][t2][t3] > 0:
                        alpha0 = (C1[t3] - 1)/num_tokens
                        if C1[t2] - 1 == 0:
                            alpha1 = 0
                        else:
                            alpha1 = (C2[t2][t3] - 1)/(C1[t2] - 1)
                        if C2[t1][t2] - 1 == 0:
                            alpha2 = 0
                        else:
                            alpha2 = (C3[t1][t2][t3]-1)/(C2[t1][t2] - 1)
                        if alpha0 >= alpha1 and alpha0 >= alpha2:
                            lamb0 = lamb0 + C3[t1][t2][t3]
                        elif alpha1 >= alpha0 and alpha1 >= alpha2:
                            lamb1 = lamb1 + C3[t1][t2][t3]
                        else: #if alpha2 > alpha1 and alpha2 > alpha0:
                            lamb2 = lamb2 + C3[t1][t2][t3]
        lamb0 = lamb0/(lamb0 + lamb1 + lamb2)
        lamb1 = lamb1/(lamb0 + lamb1 + lamb2)
        lamb2 = lamb2/(lamb0 + lamb1 + lamb2)
        return [lamb0, lamb1, lamb2]

def build_hmm(training_data: list, unique_tags: list, unique_words: list, order: int, use_smoothing: bool):
    """
    Builds a HMM object based on data of the training corpus
    Inputs:
        training_data --- list of (word, tag) tuples of the training corpus
        unique_words --- a sequence of unique words that appears in the training corpus
        unique_tags --- a sequence of unique tags that appears in the training corpus
        order --- the order of markov chain to process data for, can be 2nd or 3rd order
        use_smoothing --- a boolean of whether or not to utilize the lambda equation
    Returns:
        a hidden markov model object
    """
    tokens, ctw, ct, ctt, cttt = compute_counts(training_data, 3)
    initial_distribution = compute_initial_distribution(training_data, order)
    emission_matrix = compute_emission_probabilities(unique_words, unique_tags, ctw, ct)
    if use_smoothing:
        lambdas = compute_lambdas(unique_tags, tokens, ct, ctt, cttt, order)
        if order == 2:
            transition_matrix = defaultdict(lambda : defaultdict(int))
            for t1 in ctt:
                for t2 in ctt[t1]:
                    transition_matrix[t1][t2] = lambdas[1] * ctt[t1][t2]/ct[t1] + lambdas[0] * ct[t2]/tokens
            return HMM(order, initial_distribution, emission_matrix, transition_matrix)
        else:
            transition_matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
            for t1 in cttt:
                for t2 in cttt[t1]:
                    for t3 in cttt[t1][t2]:
                        transition_matrix[t1][t2][t3] = (lambdas[2] * cttt[t1][t2][t3] / ctt[t1][t2]) + (lambdas[1] * ctt[t2][t3] / ct[t2]) + ((ct[t3] / tokens) * lambdas[0])
            return HMM(order, initial_distribution, emission_matrix, transition_matrix)
    else:
        if order == 2:
            transition_matrix = defaultdict(lambda : defaultdict(int))
            for t1 in ctt:
                for t2 in ctt[t1]:
                    transition_matrix[t1][t2] = ctt[t1][t2]/ct[t1]
            return HMM(order, initial_distribution, emission_matrix, transition_matrix)
        else:
            transition_matrix = defaultdict(lambda : defaultdict(lambda : defaultdict(int)))
            for t1 in cttt:
                for t2 in cttt[t1]:
                    for t3 in cttt[t1][t2]:
                        val1 = 0
                        if ctt[t1][t2] != 0:
                            val1 = cttt[t1][t2][t3]/ctt[t1][t2]
                        transition_matrix[t1][t2][t3] = val1
            return HMM(order, initial_distribution, emission_matrix, transition_matrix)
    return hmm

def update_hmm(hmm, sentence):
    """
    Assigns 0.00001 emission probability for each word that was not in the training data but seen in the testing data
    Inputs:
        hmm --- an instance of the HMM object
        sentence --- the testing data
    Returns:
         a modified hmm object with adjusted emission values for the testing data
    """
    for i in range(len(sentence)):
        for j in range(len(sentence[i])):
            for tag in hmm.emission_matrix:
                if sentence[i][j] not in hmm.emission_matrix[tag]:
                    for word in hmm.emission_matrix[tag]:
                        hmm.emission_matrix[tag][word] *= 0.99999
                    hmm.emission_matrix[tag][sentence[i][j]] = 0.00001
    return

def trigram_viterbi(hmm, sentence: list) -> list:
    """
    Run the Viterbi algorithm to tag a sentence assuming a trigram HMM model.
    Inputs:
        hmm --- the HMM to use to predict the POS of the words in the sentence.
        sentence ---  a list of words.
    Returns:
        A list of tuples where each tuple contains a word in the
        sentence and its predicted corresponding POS.
    """
    v = defaultdict(lambda: defaultdict(int))
    bp = defaultdict(lambda: defaultdict(int))
    emission = hmm.emission_matrix
    all_states = set(hmm.initial_distribution.keys()).union(set(hmm.transition_matrix.keys()))
    #print(all_states)
    initial_dist = hmm.initial_distribution
    transition_mat = hmm.transition_matrix
    for state1 in all_states:
        for state2 in all_states:
            if initial_dist[state1][state2] != 0:
                if (emission[state1][sentence[0]] != 0) and (emission[state2][sentence[1]] != 0):
                    v[(state1, state2)][0] = math.log(initial_dist[state1][state2]) + math.log(emission[state1][sentence[0]]) + math.log(emission[state2][sentence[1]])
                else:
                    v[(state1, state2)][0] = -1 * float('inf')
    for i in range(1, len(sentence)-1):
        bp["No_Path"][i] = "No_Path"
        for state1 in all_states:
            for state2 in all_states:
                maxstate = (None, -1 * float('inf'))
                for stateprime in all_states:
                    val1 = v[(stateprime, state1)][i-1]
                    val2 = -1 * float('inf')
                    if transition_mat[stateprime][state1][state2] != 0:
                        val2 = math.log(transition_mat[stateprime][state1][state2])
                    curr_value = val1 + val2
                    if curr_value > maxstate[1]:
                        maxstate = (stateprime, curr_value)
                val3 = -1 * float('inf')
                if hmm.emission_matrix[state1][sentence[i]] != 0 and emission[state2][sentence[i+1]] != 0:
                    val3 = math.log(emission[state1][sentence[i]]) + math.log(emission[state2][sentence[i+1]])
                v[(state1, state2)][i] = maxstate[1] + val3
                if maxstate[0] == None:
                    bp[(state1, state2)][i] = "No_Path"
                else:
                    bp[(state1, state2)][i] = maxstate[0]

    maxend = ('No_Path', 'No_Path', -1 * float('inf'))
    for state1 in all_states:
        for state2 in all_states:
            if v[(state1,state2)][len(sentence)-2] > maxend[2]:
                #print("here")
                maxend = (state1, state2, v[(state1,state2)][len(sentence)-2])
    #print(maxend)
    Z = [(sentence[-2],maxend[0]), (sentence[-1],maxend[1])]
    for i in range(len(sentence) - 3, -1, -1):
        Z.insert(0, (sentence[i], bp[(Z[0][1], Z[1][1])][i+1]))
    return Z

def test(trainfile, testfile, taggedtestfile, smoothing, order):
    """
    Runs the required tests for HMM accuracy trained at 1, 5, 10, 25, 50, 75, and 100 percent of the training data
    Inputs:
        trainfile --- the name of the training file
        testfile --- the name of untagged test file
        taggedtestfile --- the name of the tagged test file
        smoothing --- the boolean of whether or not to use smoothing
        order --- the desired order of the markov model to trian
    Returns:
        a list of accuracies of the trained HMM
    """
    training_corpus = read_pos_file(trainfile)
    training_data = training_corpus[0]
    unique_tags = training_corpus[2]
    unique_words = training_corpus[1]
    my_file = open(testfile, "r")
    content = my_file.read()
    sentences = content.split(" . ")
    paragraph = []
    for sentence in sentences:
        splitted = sentence.split(" ")
        if splitted[-1] == " ":
            splitted.pop()
            splitted.append(".")
        elif splitted[-1] != ".":
            splitted.append(".")
        paragraph.append(splitted)
    paragraph.pop()
    correctvals = read_pos_file(taggedtestfile)[0]
    hmm1percent = build_hmm(training_data[0: int(len(training_data) / 100)], unique_tags, unique_words, order, smoothing)
    hmm5percent = build_hmm(training_data[0: int(len(training_data) / 20)], unique_tags, unique_words, order, smoothing)
    hmm10percent = build_hmm(training_data[0: int(len(training_data) / 10)], unique_tags, unique_words, order, smoothing)
    hmm25percent = build_hmm(training_data[0: int(len(training_data) / 4)], unique_tags, unique_words, order, smoothing)
    hmm50percent = build_hmm(training_data[0: int(len(training_data) / 2)], unique_tags, unique_words, order, smoothing)
    hmm75percent = build_hmm(training_data[0: int(3 * len(training_data) / 4)], unique_tags, unique_words, order, smoothing)
    hmm100percent = build_hmm(training_data[0: int(len(training_data))], unique_tags, unique_words, order, smoothing)
    update_hmm(hmm1percent, paragraph)
    update_hmm(hmm5percent, paragraph)
    update_hmm(hmm10percent, paragraph)
    update_hmm(hmm25percent, paragraph)
    update_hmm(hmm50percent, paragraph)
    update_hmm(hmm75percent, paragraph)
    update_hmm(hmm100percent, paragraph)
    vit1percent = []
    vit5percent = []
    vit10percent = []
    vit25percent = []
    vit50percent = []
    vit75percent = []
    vit100percent = []
    if order == 2:
        for sentence in paragraph:
            bv1 = bigram_viterbi(hmm1percent, sentence)
            bv5 = bigram_viterbi(hmm5percent, sentence)
            bv10 = bigram_viterbi(hmm10percent, sentence)
            bv25 = bigram_viterbi(hmm25percent, sentence)
            bv50 = bigram_viterbi(hmm50percent, sentence)
            bv75 = bigram_viterbi(hmm75percent, sentence)
            bv100 = bigram_viterbi(hmm100percent, sentence)
            vit1percent.extend(bv1)
            vit5percent.extend(bv5)
            vit10percent.extend(bv10)
            vit25percent.extend(bv25)
            vit50percent.extend(bv50)
            vit75percent.extend(bv75)
            vit100percent.extend(bv100)
    if order == 3:
        for sentence in paragraph:
            tv1 = trigram_viterbi(hmm1percent, sentence)
            tv5 = trigram_viterbi(hmm5percent, sentence)
            tv10 = trigram_viterbi(hmm10percent, sentence)
            tv25 = trigram_viterbi(hmm25percent, sentence)
            tv50 = trigram_viterbi(hmm50percent, sentence)
            tv75 = trigram_viterbi(hmm75percent, sentence)
            tv100 = trigram_viterbi(hmm100percent, sentence)
            vit1percent.extend(tv1)
            vit5percent.extend(tv5)
            vit10percent.extend(tv10)
            vit25percent.extend(tv25)
            vit50percent.extend(tv50)
            vit75percent.extend(tv75)
            vit100percent.extend(tv100)
    vit1percentacc = 0
    vit5percentacc = 0
    vit10percentacc = 0
    vit25percentacc = 0
    vit50percentacc = 0
    vit75percentacc = 0
    vit100percentacc = 0
    print(vit100percent)
    for i in range(len(vit1percent)-1):
        if vit1percent[i] == correctvals[i]:
            vit1percentacc += 1
        if vit5percent[i] == correctvals[i]:
            vit5percentacc += 1
        if vit10percent[i] == correctvals[i]:
            vit10percentacc += 1
        if vit25percent[i] == correctvals[i]:
            vit25percentacc += 1
        if vit50percent[i] == correctvals[i]:
            vit50percentacc += 1
        if vit75percent[i] == correctvals[i]:
            vit75percentacc += 1
        if vit100percent[i] == correctvals[i]:
            vit100percentacc += 1
    print("1 percent accuracy: " + str(vit1percentacc/len(vit1percent)))
    print("5 percent accuracy: " + str(vit5percentacc/len(vit5percent)))
    print("10 percent accuracy: " + str(vit10percentacc/len(vit10percent)))
    print("25 percent accuracy: " + str(vit25percentacc/len(vit25percent)))
    print("50 percent accuracy: " + str(vit50percentacc/len(vit50percent)))
    print("75 percent accuracy: " + str(vit75percentacc/len(vit75percent)))
    print("100 percent accuracy: " + str(vit100percentacc/len(vit100percent)))
    return [vit1percentacc/len(vit1percent), vit5percentacc/len(vit5percent), vit10percentacc/len(vit10percent), vit25percentacc/len(vit25percent), vit50percentacc/len(vit50percent), vit75percentacc/len(vit75percent), vit100percentacc/len(vit100percent)]

nosmoothing2 = test("training.txt", "testdata_untagged.txt", "testdata_tagged.txt", False, 2)
nosmoothing3 = test("training.txt", "testdata_untagged.txt", "testdata_tagged.txt", False, 3)
smoothing2 = test("training.txt", "testdata_untagged.txt", "testdata_tagged.txt", True, 2)
smoothing3 = test("training.txt", "testdata_untagged.txt", "testdata_tagged.txt", True, 3)
percentages = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
plt.plot(percentages, nosmoothing2, label = 'bigram no smoothing')
plt.plot(percentages, nosmoothing3, label = 'trigram no smoothing')
plt.plot(percentages, smoothing2, label = 'bigram smoothing')
plt.plot(percentages, smoothing3, label = 'trigram smoothing')
plt.legend(loc='best')
plt.ylim(0,1)
plt.xlabel("Percentage Training Data")
plt.ylabel("Percentage Accuracy")
plt.show()


# testing the viterbi algorithms!
"""
training_corpus = read_pos_file("training.txt")
training_data = training_corpus[0]
unique_tags = training_corpus[2]
unique_words = training_corpus[1]
#print(training_data[0:int(len(training_data)/100000)])
hmm = build_hmm(training_data[0: int(len(training_data)/10)], unique_tags, unique_words, 2, False)
my_file = open("testdata_untagged.txt", "r")
#print(set(hmm1percent.initial_distribution.keys()).union(set(hmm1percent.transition_matrix.keys())))
content = my_file.read()
sentence = content.split(" ")
update_hmm(hmm, sentence)
#print(hmm.initial_distribution.values())
vit = bigram_viterbi(hmm, sentence)
vit1 = 0
correctvals = read_pos_file("testdata_tagged.txt")[0]
for i in range(len(vit) - 1):
    if vit[i] == correctvals[i]:
        vit1 += 1
print(vit1/len(vit))
"""




