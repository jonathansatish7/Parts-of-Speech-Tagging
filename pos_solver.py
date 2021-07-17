###################################
# CS B551 Spring 2021, Assignment #3
#
# Your names and user ids:bkmaturi-josatiru-sgaraga
#
# (Based on skeleton code by D. Crandall)
#


import random
import math
from collections import Counter, defaultdict


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#added respective dictionaries to store the counts and probabilities.
class Solver:
    def __init__(self):
        self.pos = ['adj', 'adv', 'adp', 'conj', 'det', 'noun', 'num', 'pron', 'prt', 'verb', 'x', '.']
        self.words_count = {}
        self.speech_count = {}
        self.word_speech = {}
        self.speech_prob = {}
        self.word_speech_prob = {}
        self.speech2_speech1 = {}
        self.s2_s1_prob = {}
        self.start_word_pos_prob = {}
        self.start = {}
        self.start_prob = {}
        self.end_s2_s1_count = {}
        self.end_s2_s1_prob = {}
        self.trans_prob_each_count = {}
        self.init_count_dict = {}

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):
        if model == "Simple":
            res = 0
            for i in range(len(sentence)):
                res += math.log(self.word_speech_prob.get((sentence[i], label[i]), 0.00000000001))
                res += math.log(self.speech_prob.get(label[i], 0.00000000001))
            return res
        elif model == "HMM":
            res = 0
            for i in range(len(sentence)):
                res += math.log(self.word_speech_prob.get((sentence[i], label[i]), 0.00000000001)) + math.log(
                    self.s2_s1_prob.get((label[i], label[i - 1]), 0.00000000001))
            return res
        elif model == "Complex":
            res = 0
            for i in range(len(sentence)):

                if len(sentence) == 1:
                    res += math.log(self.word_speech_prob.get((sentence[i], label[i]), 0.00000000000000001)) + math.log(
                        self.start_prob.get((label[i]), 0.00000000000000001))
                elif i == 0:
                    res += math.log(self.word_speech_prob.get((sentence[i], label[i]), 0.00000000000000001)) + math.log(
                        self.start_prob.get((label[i]), 0.00000000000000001)) + math.log(
                        self.s2_s1_prob.get((label[i], label[i - 1]), 0.00000000000000001))
                elif i == (len(sentence) - 1):
                    res += math.log(self.word_speech_prob.get((sentence[i], label[i]), 0.00000000000000001)) + math.log(
                        self.s2_s1_prob.get((label[i - 1], label[i]), 0.00000000000000001))
                else:
                    res += math.log(self.word_speech_prob.get((sentence[i], label[i]), 0.00000000000000001)) + math.log(
                        self.s2_s1_prob.get((label[i - 1], label[i]), 0.00000000000000001)) + math.log(
                        self.s2_s1_prob.get((label[i], label[i + 1]), 0.00000000000000001))
            return res
        else:
            print("Unknown algo!")

    # Do the training!
    #
    def train(self, data):
        for i in data:
            # start_speech = i[0][1]
            start_pos = i[1][0]
            # calculating the count of the number of times the sentence started with the same pos
            # Each element is a dictionary having every POS as its key and values as total number of occurence in the start position
            if start_pos in self.start:
                self.start[start_pos] += 1
            else:
                self.start[start_pos] = 1
            #s1 and s2 variables to calculate the  probablity of speech2 given speech1
            s1 = i[1]
            s2 = s1[1:]
            s_combi = zip(s2, s1)
            combi = zip(i[0], i[1])
            for i in combi:

                # in i we are getting a combination of the word with its pos
                if i[1] in self.speech_count:
                    # calculating the count of each pos in the training data
                    # Each element is a dictionary having every POS as its key and values as total number of occurences
                    self.speech_count[i[1]] += 1
                else:
                    self.speech_count[i[1]] = 1
                if i[0] in self.words_count:
                    # calculating the occurences of each word in the training data
                    # Each element is a dictionary having every word as its key and values as total number of occurences
                    self.words_count[i[0]] += 1
                else:
                    self.words_count[i[0]] = 1
                if i in self.word_speech:
                    # calculating the word_speech combination occurences in the data.
                    # Each element is a dictionary having word_speech combination tuple as its key and values as total number of occurences
                    self.word_speech[i] += 1
                else:
                    self.word_speech[i] = 1
            for i in s_combi:
                # getting all the combinations of speech2 given speech1 tuple for every sentence.(s2,s1)
                # Each element is a dictionary having speech2_speech1 combination tuple as its key and values as total number of occurences
                if i in self.speech2_speech1:
                    self.speech2_speech1[i] += 1
                else:
                    self.speech2_speech1[i] = 1
        # speech_total to calculate the sum of all the pos values
        speech_total = sum(self.speech_count.values())
        for x in self.speech_count:
            #calculating the probablity of each pos
            self.speech_prob[x] = self.speech_count[x] / speech_total
        # w_s_total = sum(self.word_speech.values())
        for (x, y) in self.word_speech:
            # calculating the probablity of all the word_speech combination
            self.word_speech_prob[(x, y)] = self.word_speech[(x, y)] / self.speech_count[y]
        for (s2, s1) in self.speech2_speech1:
            # calculating the probablity of speech2 given speech1
            self.s2_s1_prob[(s2, s1)] = self.speech2_speech1[(s2, s1)] / self.speech_count[s1]
        for i in self.start:
            # calculating the probablity of a particular pos in the start position
            self.start_prob[i] = self.start[i] / sum(self.start.values())

    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #uses the simple Bayes net where the parts of speech are independent of each other
    def simplified(self, sentence):

        out = []
        for i in sentence:
            # we randomly take the first word as noun to compare the probabilties further
            s = 'noun'
            if i not in self.words_count:
                out.append(s)
                continue
            max_till = 0
            for j in self.pos:
                # checking for the combination of the word and all the pos one after the other
                if (i, j) in self.word_speech:
                    prob_i_j = self.word_speech_prob[(i, j)] * self.speech_prob[j]
                    if prob_i_j > max_till:
                        max_till = prob_i_j
                        s = j

            out.append(s)
        # print('my print',out)
        return out

    def hmm_viterbi(self, sentence):
        v_table = [[] for i in range(12)]
        v_path = [[] for i in range(12)]

        n = 0
        for word in sentence:
            #print(word)
            if n == 0:
                #calculating the possible pos for the first word in a sentence
                for i in range(12):
                    prob_start = math.log(self.word_speech_prob.get((word, self.pos[i]), 0.0000000000001)) + \
                                 math.log(self.start_prob.get(self.pos[i], 0.0000000000001))
                    #print(self.start_prob)
                    v_table[i].append(prob_start)
                    #print(v_table)
            else:
                for i in range(12):
                    #calculating the emission probablity for each word with all the possible pos tags
                    emi_prob = math.log(self.word_speech_prob.get((word, self.pos[i]), 0.0000000000001))
                    #print(emi_prob)
                    prev_prob = []

                    for j in range(12):
                        # calculating the transition probablities taking the possible combinations of parts of speech
                        tran_prob = self.s2_s1_prob.get((self.pos[i], self.pos[j]), 0.0000000000001)
                        # a list to store the probabilities of the previous word related probabilities to choose the maximum from it
                        prev_prob.append(v_table[j][n - 1] + math.log(tran_prob))
                    #to store the maximum from the prev_prob list
                    max_prob = max(prev_prob)
                    # to store the index of the maximum probability from the prev_prob list
                    max_index = prev_prob.index(max_prob)
                    # v_path list to store the indexes to enable backtracking for finding the pos tags
                    v_path[i].append(max_index)
                    # update the v_table for the next words considering the emission and maximum of the transition probabilities
                    v_table[i].append(emi_prob + max_prob)
            n += 1
        # back tracking to get the sequence of parts of speech by considering the maximum probabilities
        #in all columns that is for each word
        lists = []
        for i in v_table:
            lists.append(i[len(i) - 1])
        # taking the maximum value from the list
        max_value = max(lists)
        indexes = lists.index(max_value)
        # sequence to store all the indexes of the maximum value
        seq = [indexes]
        count = len(v_path[0]) - 1
        while count >= 0:
            a = v_path[indexes][count]
            seq.append(a)
            indexes = a
            count -= 1
        seq.reverse()
        hmm_output = []
        for i in seq:
            hmm_output.append(self.pos[i])
        return hmm_output

    def complex_mcmc(self, sentence):
        count_values = {}
        #print(len(sentence))
        # randomly allocating the sentence to be all adv's
        list1 = ["noun"] * len(sentence)
        # creating a count dictionary to store the count of parts of speech when sampling
        for i in range(len(sentence)):
            count_values[i] = {}
            for j in self.pos:
                count_values[i][j] = 0
        for i in range(len(list1)):
            count_values[i][list1[i]] += 1
        for i in range(800):
            for j in range(len(list1)):
                prob = []
                if len(sentence) == 1:
                    for k in self.pos:
                        #calculating the probabilty if sentence contains only one word
                        prob.append(math.log(self.word_speech_prob.get((sentence[j], k), 0.00000000000000001)) +\
                                    math.log(self.start_prob.get((k), 0.00000000000000001)))
                elif j == 0:
                    for k in self.pos:
                        # starting from the first word calculating the probability by considering all the dependent observed
                        #and unobserved variables
                        prob.append(math.log(self.word_speech_prob.get((sentence[j], k), 0.00000000000000001))
                                 + math.log(self.start_prob.get((k), 0.00000000000000001))
                                 + math.log(self.s2_s1_prob.get((list1[j + 1], k), 0.00000000000000001)))
                elif j == len(list1) - 1:
                    for k in self.pos:
                        # probability of the last word
                        prob.append(math.log(self.word_speech_prob.get((sentence[j], k), 0.00000000000000001))
                                 + math.log(self.s2_s1_prob.get((list1[j - 1], list1[j]), 0.00000000000000001)))
                else:
                    for k in self.pos:
                        #calculating the probability of any word except the first and the last
                        prob.append(math.log(self.word_speech_prob.get((sentence[j], k), 0.00000000000000001))
                                 + math.log(self.s2_s1_prob.get((k, list1[j]), 0.00000000000000001))
                                 + math.log(self.s2_s1_prob.get((list1[j + 1], k), 0.00000000000000001)))
                orig_prob=[]
                #converting to probabilites by taking exponent
                for n in range(len(prob)):
                    orig_prob.append(math.exp(prob[n]))
                total=0
                # calculating the total probability
                for n in range(len(orig_prob)):
                    total+=orig_prob[n]
                norm_prob=[]
                # normalize the values by dividing the values by total
                for n in range(len(orig_prob)):
                    norm_prob.append(orig_prob[n]/total)
                # function to take some random value between 0 and 1
                rand = random.random()
                c = 0
                for k in range(len(norm_prob)):
                    c += norm_prob[k]
                    # if the random value is less than c we update the list1 of the pos to be pos[k]
                    if rand < c:
                        list1[j] = self.pos[k]
                        break
                #after 500 iterations we start updating the count_values of the pos tags.
                if i > 500:
                    for k in range(len(list1)):
                        count_values[k][list1[k]] += 1
        output_complex = []
        #by taking the values of maximum occuring pos tags we keep updating the pos tags in the output
        for i in range(len(list1)):
            max_till = 0
            for j in self.pos:
                if count_values[i][j] >= max_till:
                    max_till = count_values[i][j]
                    pos1 = j
            output_complex.append(pos1)
        # print(count_dict)
        return output_complex

        # return [ "noun" ] * len(sentence)

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself.
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        else:
            print("Unknown algo!")