# SUTD Machine Learning Project
**A POS&NPC tagger for tweets with Hidden Markov Model**

Name: Yuan Bowei

Student ID: 1001916
## Directory Structure
**src** Source code, all following intructions are for this directory

**order2** A combination of MLE predictor, first order HMM and second order HMM ,plus some smoothing. The best performance parameter combination can give a 78% correction rate. 

**data** Training data, test data and output data

## Instruction
This package consists of four python modules:

1. **emission.py:**This module computes emission probability with given training text.
2. **transition.py:**This module computes transition probability with given training text.
3. **viterbi.py:** This module implements viterbi algorithm to predict the tag of given test text.
4. **toolbox.py:**This module is a box of tools, providing functionalities and optimization for three modules above, including preprocessor, evaluation of a prediction.

## How to run the code
for help:

	python run.py -h
	parameter list:
	-t Path of training file, required
	-i Path of testing file, required
	-o Path of output file, required
	--algorithm, range[0,1,2], the algorithm to use
		0: MLE predictor in emission.py
		1: use viterbi top 1 tagger
		2: use viterbi top N tagger, N can be 1 or 10
	-b, specifying number of sequences to find
		default value is 1
		for algorithm 0 and 1: it canonly be 1
		for algorithm 2: it can be 1 or 10
	-p, preprocess
		default value is True
		if it is True, word would be processed before
		computing probabilities
	
For example,running viterbi to get best 10 sequence without preprocessor:

	python run.py -t ../data/POS/ptrain 
					-i ../data/POS/dev.in 
					-o ../data/POS/dev.p3.out 
					--algorithm 2 
					-b 10
					-p False
##emission.py
This module contains the class *emission* which maintains the emission probabilty of text given in a file.
#### Algorithm Specification
This class computes emission probability of the given text with the Maximal Likelihood Estimator (MLE) introduced in Hidden Markov Model.

The general case formula is:
$$ e(word|tag) = \frac{count(word,tag)}{count(tag)+1}$$

As we can hardly learn all possible *"word"*s in the given text file, for a word that has never appeared in the corpus, we provide a special case formula to estimate its emission probability:
$$e(new|tag) = \frac{1}{count(tag)+1}$$

Internally, this class maintains two Python dictionaries, named *matrix* and *labels*, which have:

	matrix[word][tags] = count(word, tag)
	labels[tag] = count(tag)
#### import the module and initialization
	# import the module
	import emission
	# initialization
	e = emission.emission()
This step provides an "empty" emission object whose two internal dictionaries mentioned above are empty.
#### compute the parameters with a given corpus
	e.compute("training file name")
After this step the *matrix* and *labels* in emission classes are computed with the training file provided.
#### compute emission probability
	e.emit(word, tag, p=True)
	# attention: tag must appear in the given training corpus
	# or it would raise a runtime error
***word*** is the word to be emitted from ***tag***. Parameter *p* stands for "process", when *p* is set True, the object would process *word* (with the tools in toolbox) before computing the emission probability.

This function returns a floating number specifying the emission probability of the given ***word*** and ***tag***.
#### guess the most probable tag
	e.mostprob()
It happens in Hidden Markov Model that at some points in a sentence, all possible tags score 0. One enhancement is to tag this position as the most probable label among all possible labels, which could improve the result to some degree.

#### MLE predictor:
With the emission probability, one way to predict labels of a given corpus is to use Maximal Likelihood, that is, to find out the most probable tag for the word under emission probability.

	e.predict("input filename","output filename", p=True)
Then the object labels the input file with emission probability MLE and writes the labeled corpus into output file. 

When *p* is set True, the word would be processed with toolbox before computing emission probability.

## transition.py
This module provides a class similar to emission.py, which also maintains four internal dictionaries ***start***, ***stop***, ***matrix*** and ***states*** to compute the transition probability with the MLE in Hidden Markov Model, where:
$$T(y_{i-1}->y_i) = \frac{count(y_{i-1},y{i})}{count(y_{i-1})}$$
In particular, i=1 or i=n respectively denotes that this word is the first or last word of the sentence, under such case the transition probability is:
$$T(y_0->y_1)= \frac{count(START,y_1)}{count(START)}$$
$$T(y_n->y_{n+1}) = \frac{count(y_n,STOP)}{count(STOP)}$$

	matrix[front][current] = count(front, current)
	states[tag] = count(tag)
	start[tag] = count(START,tag)
	stop[tag] = count(tag,STOP)

### import and initialization
	
	import transition
	t = transition.transition()
Similarly to that in emission.py, this procedure gives an empty transition object with empty dictioinaries.

### compute the transition probability

	t.compute(training file name)
Also similarly to that in emission.py, this procedure computes the values in the four dictionaries mentioned above

### transit

	t.transit(front, current)
This procedure returns a single floating number denoting the transition probability of transitioning from ***front*** to ***current***, which respective denoting the front tag and current tag.

### startwith
	
	t.startwith(tag)
This method returns a single floating number, which is the value of $T(START,tag)$, denoting the probability of the sentence starts with ***tag***.

### stopwith
	t.stopwith(tag)
Similarly to ***startwith()***, it returns the probability that one sentence stops with ***tag***.

## viterbi.py
This part is the core of our viterbi algorithm, it has two classes and two methods.

#### use viterbi to find top1 tag sequence

	viterbi_best(e, t, inputfile, outputfile, p=True)
	e: computed emission object
	t: computed transition object
	inputfile: the path of input file
	outputfile: path of output file
	p: whether do preprocessing before computing probabilities
This function runs the dynamic programming to find the best tag sequence for given observation of word sequence. As the algorithm to compute top N tag sequence is more general, the procedure would be specified in next part.

#### use viterbi to find top N sequence

	viterbi_Nbest(e,t,input,output,best=10,p=True)
	e: computed emission object
	t: computed transition object
	input: path of input file
	output: path of output file
	best: number of best paths to find
	p: whether do preprocessing
This function is implemented with 2 classes:

* worditem,having following attributes:

		word: the previous tag of current position
		score: score of this previous tag at this position
		path: which path of the previous tag this score is from

* NBest:
  This class maintains a heap of worditem to rank previous tags with their respective scores. having following methods:
  		
  		NBest.NBest(best):
  		best value specifying the number of best paths we want
  
		NBest.add(word, prob, path)
		adding a new worditem object with this 3 values
		into the heap
		
		NBest.best()
		remove all paths after ranking behind the specified best
		value


#### algorithm specification

In viterbi algorithm, we have following dynamic transition function:

$Pi [ i , tag ] = Max(Pi [ i-1 , tag’ ] * Transition[ tag’ , tag] * Emission[ tag , word] )$

Base Case:

$Pi [ 0 , START ] = 1$

$Pi [ 0 , tag ] = 0 $ ( for tag is not START )

In order to get the top N sequences, we must save N-best POS tag sequences at every tag choice for every word.

We save the result in the queue in the following format.

$F[ i , tag ] = [ ( Pi1  , previous_tag1 ) …  ( Pi10  , previous_tag10 ) ] $
(If the number of possible choices are above 10. If not, just save all the possible choices in the heap)

The generating procedure is:

* enumerate all positions in a sequence  $O(N)$
* enumerate all possible tags for the word at this position $O(T)$
* enumerate all possible tags for previous position $O(T)$
* enumerate all n scores at the given previous tag and compute the score of current position with the dynamic transition function mentioned above, and push the object having related information into the heap$O(n)$
* for one certain current tag, if all previous tags are already enumerated, remove all previous tag paths ranking behind n.

The decoding procedure is:

* As we know the N best tags at a position and from which previous path of the previous tag it comes from, we can decode the sentence from the last to the beginning. $O(N)$

So all in all, the time complexity of the best N algorithm is $O(nNT^2)$ 

## toolbox.py
This module provides the tools that facilitates other modules, including preprocessor and evaluation methods:
#### Preprocessor
	toolbox.preprocess("input file", "output file")
This method preprocesses the input file and write the results into output file. It achieves this by using following two methods:

	#sentence processing
	pword, ptag = toolbox.processSentence(sentence)
This method processes one sentence into one processed word and one processed tag. Temporarily, no modifications are done to tag except extracting it from the sentence.
	
	#word processing
	pword = toolbox.processWord(word)
This method processes a word with the following rule:

  *  when one word contains only punctuations, for example ":)", ":(", remains the same
  *  the words begins with '@' are turned into USR to be currectly labeled as USR
  *  digits are turned into 'digit'
  *  words starting with '#' are turned into 'HT'
  *  for other words with not only punctuations like "hello!", remove the punctuations in them and modify into lower case. This generalization procedure could slightly improve the results.
  *  [deprecated]exception is URL, initially URL are retained, but in terms of the error rate there were not improvements, so deprecated:

		hasURL(word) #judge whether one sentence has URL


## Performance
#### POS with preprocessor
1. emission.py MLE, with preprocessor

		python run.py -t ../data/POS/ptrain 
		-i ../data/POS/dev.in 
		-o ../data/POS/dev.p4.out 
		--algorithm 0
	
Error rate: 0.305203938115

2. viterbi, best 1, with preprocessor
		python run.py -t ../data/POS/ptrain 
		-i ../data/POS/dev.in 
		-o ../data/POS/dev.p4.out 
		--algorithm 1
Error rate: 0.325246132208

3. viterbi, best 10, with preprocessor
 		
 		python run.py -t ../data/POS/ptrain 
		-i ../data/POS/dev.in 
		-o ../data/POS/dev.p4.out 
		--algorithm 2 
		-b 10
Error rate: 

		0.310829817159
		0.313291139241
		0.322784810127
		0.318213783404
		0.323839662447
		0.32876230661
		0.331575246132
		0.32876230661
		0.32841068917
		0.329465541491

#### NPC with preprocessor
1. emission.py MLE, with preprocessor

		python run.py -t ../data/NPC/ptrain 
		-i ../data/NPC/dev.in 
		-o ../data/NPC/dev.p2.out 
		--algorithm 0
		
Error rate:0.296279505546

2. viterbi, best 1, with preprocessor

		python run.py -t ../data/NPC/ptrain 
		-i ../data/NPC/dev.in 
		-o ../data/NPC/dev.p3.out 
		--algorithm 1

Error rate: 0.214767129084
3. viterbi, best 10 with preprocessor
		
		python run.py -t ../data/NPC/ptrain 
		-i ../data/NPC/dev.in 
		-o ../data/NPC/dev.p4.out 
		--algorithm 2 
		-b 10
Error rate: 

		0.214767129084
		0.235711911022
		0.240638317164
		0.254813068577
		0.256505576208
		0.264726327561
		0.268232236226
		0.275425393659
		0.266962855502
		0.257142857143
