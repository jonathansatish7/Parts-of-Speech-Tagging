# Parts-of-Speech-Tagging

The goal of the assignment was to tag the parts of speech to each word in all the sentences in the test data provided. There are 12 parts of speech namely 'adj', 'adv', 'adp', 'conj', 'det', 'noun', 'num', 'pron', 'prt', 'verb', 'x', '.'
The three approaches used for parts of speech tagging are :
1.	Simple
2.	Hidden Markov model / Viterbi algorithm
3.	Complex Markov chains Monte carlo method/ Gibbs Sampling
The first step towards assigning the parts of speech to the words in a sentence is to train the model from the data provided.

## Training:
 We process the given training data and learn the probabilities of the required information that we would be needing to implement the algorithms
We have taken different dictionaries to maintain the count of the following
1)	the occurrence of each word 
2)	the occurrence of 12 different parts of speech 
3)	the combination of word and the respective parts of speech 
4)	the combination of speech n-1 to speech n transitions 
5)	a particular parts of speech starting a sentence
 the probabilities that we calculated are :
1)	Emission probability : The probability of word given speech P(Word/Speech)
2)	Transition probability :The probability that a particular speech follows a given speech P(S2/S1)
3)	The probability that a speech comes as the first speech in the sentence 
4)	State probability :The probability of a particular parts of speech P(S)

## Algorithm Description:
* For the Simple Algorithm :
In this approach the parts of speech is considered to be independent of one another 
The formula for calculating the probability would be : 
P(S|W) = max. P(W|S)*P(S)/P(W)

First randomly we assign the parts of speech to the first word as Noun.
For each word in the sentence, we check for each speech considering all the 12 parts of speech, considering each word to be independent. We calculate the probability of the word given speech for all the different speeches. From the training the data, we have the probability of word given speech which we utilize here. 
Once selected a word, we take the probability for each speech given this word and multiply it with the probability of the speech. We consider the maximum of the probabilities calculated for each word given speech and take the speech which has the highest probability and append it to the out list.

* Viterbi algorithm :
The Viterbi algorithm is a dynamic programming algorithm for obtaining the maximum posteriori probability estimate of the most likely sequence of hidden states—called the Viterbi path—that results in a sequence of observed events. 
To find the best tag sequence for a given sentence (or a word sequence), we need to compare all tag sequences. we can only observe the words and we need to determine the Part of speech of that word. Here the words are observed variables and hidden variables are the parts of speeches.
The formula for calculating the probability of the most probable path ending in state k with observation I :
                                           
We considered two variables v_table to store the values of probabilities in for all possible parts of speech and for all the words in a matrix form. V_path to store the indexes of the maximum probabilities for each word(in a column) to get the sequence of tags
For the first word in the sentence it does not have any transition probabilities but two probabilities: The probability that the speech comes first in a sentence and the probability of word given speech. We calculate the probabilities for the first word and update it in the table 
For the second word onwards we use Emission probability, Transition probability and the previous Viterbi path probability . We compute the emission probability for each parts of speech and transition probability considering the each parts of speech with different combinations. For any given particular speech, we take the max of the transition probability added to the previous Viterbi path probability from a speech and multiply it with the emission.
We conitnue doing this procedure for all words, and keep filling the matrix used by us. We also maintain a variable v_path to store the indexes of maximum probabilities  for each word in the table.
We get the hmm_output by getting the respective pos from the sequence stored.

* Gibbs Sampling : 
We assign a random set of parts of speech to each word or the same parts of speech and then choose each word and assign it all the 12 parts of speech. For each word we compute the posterior given by the following formula:
	               P(Si|S1…Si-1,Si+1…Sn, W1…Wn) or P(Si|S-Si, W)  =
 marginalised over Si 
                            P (S1)P(W1|S1)P(S2|S1)P(S3|S1,S2)….P(Sn|Sn-1,Sn-2)	
Rearranging the above terms:
P(S1){P(S2|S1)P(S3|S2)…P(Sn|Sn-1)}{P(W2|S1) P(W3|S2)….P(Wn|Sn-1)}{P(W1|S1)…P(Wn|Sn)}

First we create a count dictionary to store the count of the parts of speech tags when sampling.
We take roughly some 800 samples and in each sample We randomly assign each parts of speech to all the words of a sentence and calculate the probabilities. We then marginalize the probabilities and a random function is created. According to the generated random values , comparing the random value to the probability we then rearrange the list containing the pos tags.
The first few samples are discarded and the from the rest the tag that occur the most for a word is assigned as the final tag. We are using a value of 500 after which the count dictionary is updated for each parts of speech
After 800 samples for each sentence the speech that as the maximum count is assigned to a particular word.

## Assumptions:
1. Fixing the probabilities of unknown words, unknown transitions and unknown emissions to a very small value:
we faced challenges for a word that is not present in the training data. For such words we set their probability to a very small value such as 0.0000001. We have used those values for calculating emission probabilities and transition probabilities in  Viterbi and MVMC.
2. Fixing the starting word to be the noun in simplified model
We randomly fixed the first parts of speech tag to be a noun in a sentence such that if a word in the test data is not found in the dictionary of the test data we fix that word to be a noun. We experimented with different parts of speech but with the first parts of speech as noun we were getting higher accuracy.
3.Fixing some random parts of speech in a list to start with gibbs sampling:
The list of parts of speech with the same length of the sentence is randomly assigned to be all adverbs. After getting the value from the random function and comparing with the normalized probabilities the list is updated accordingly.

## Challenges faced:
1.We faced challenges formulating the expressions for calculating the probabilities for gibbs sampling
2.We faced difficulty with increasing the accuracy and tried changing the number of samples for MCMC and and also changing the random pos tags assigned to the words.

