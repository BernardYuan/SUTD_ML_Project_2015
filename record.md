# SUTD Machine Learning Project
**A POS&NPC tagger for tweets with Hidden Markov Model**

Name: Yuan Bowei

Student ID: 1001916
## Directory Structure
### Folders
**src** Source code, all following intructions are for this directory

**order2** A combination of MLE predictor, first order HMM and second order HMM ,plus some smoothing. The best performance parameter combination can give a 78% correction rate. 

**data** Training data, test data and output data

### Scripts
This package consists of four python modules:

1. **emission.py:** This module computes emission probability with given training text.
2. **transition.py:** This module computes transition probability with given training text.
3. **viterbi.py:** This module implements viterbi algorithm to predict the tag of given test text.
4. **toolbox.py:** This module is a box of tools, providing functionalities and optimization for three modules above, including preprocessor, evaluation of a prediction.

## How to run the code
You can always add `-h` argument to seek help.

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
	
For example, the following command runs viterbi algorithm to get best 10 sequence without preprocessing:

	python run.py -t ../data/POS/ptrain 
					-i ../data/POS/dev.in 
					-o ../data/POS/dev.p3.out 
					--algorithm 2 
					-b 10
					-p False
                    
## Technical Overview
### 1. `emission.py`
This module contains `class emission` which maintains the emission probabilty of text in a file.
#### Algorithm Specification
This class computes emission probability of the given text with Maximal Likelihood Estimator (MLE) introduced in Hidden Markov Model.

The general case formula is:

<a href="https://www.codecogs.com/eqnedit.php?latex=e(word|tag)&space;=&space;\frac{count(word,tag)}{count(tag)&plus;1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?e(word|tag)&space;=&space;\frac{count(word,tag)}{count(tag)&plus;1}" title="e(word|tag) = \frac{count(word,tag)}{count(tag)+1}" /></a>

We can hardly learn **all possible** words in the given text file. For a word that never appears in the corpus, we provide a special case formula to estimate its emission probability:

<a href="https://www.codecogs.com/eqnedit.php?latex=e(new|tag)&space;=&space;\frac{1}{count(tag)&plus;1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?e(new|tag)&space;=&space;\frac{1}{count(tag)&plus;1}" title="e(new|tag) = \frac{1}{count(tag)+1}" /></a>

#### Implementation
Internally, `class emission` maintains two Python `dictionaries`, named `matrix` and `labels`.
~~~~python
matrix[word][tags] = count(word, tag)
labels[tag] = count(tag)
~~~~    
#### (1) Import and Initialize the Module

~~~~python
# import the module
import emission
# initialization
e = emission.emission()
~~~~    
This step provides an empty `emission` object whose two internal dictionaries are empty.

#### (2) Compute the Parameters with Given Corpus
~~~~python	
e.compute("training file name")
~~~~
After this step the `matrix` and `labels` in `emission` object are computed with the training data.

#### (3) Compute Emission Probability
~~~~python
e.emit(word, tag, p=True)
# attention: tag must appear in the given training corpus
# or it would raise a runtime error
~~~~
***word*** is the word to be emitted from ***tag***. Parameter *p* stands for "process". When *p* is `True`, the object would process *word* (with the tools in toolbox) before computing the emission probability.

This function returns a floating number specifying the emission probability of the given ***word*** and ***tag***.

#### (4) Guess the Most Probable Tag

~~~~PYTHON
e.mostprob()
~~~~
It happens in Hidden Markov Model that at some points in a sentence, all possible tags score 0. One enhancement is to tag this position as the most probable label among all possible labels, which improves the result.

#### (5) MLE Predictor

With the emission probability, one way to predict labels of a given corpus is to use Maximal Likelihood, that is, to find out the most probable tag for the word under emission probability.
~~~~python
e.predict("input filename","output filename", p=True)
~~~~
Then the object labels the input file with emission probability MLE and writes the labeled corpus into output file. 

When *p* is set True, the word would be processed with toolbox before computing emission probability.

### 2. `transition.py`
This module contains `class Transition` which maintains four internal dictionaries ***start***, ***stop***, ***matrix*** and ***states*** to compute the transition probability with the MLE in Hidden Markov Model, where:
<a href="https://www.codecogs.com/eqnedit.php?latex=T(y_{i-1}->y_i)&space;=&space;\frac{count(y_{i-1},y{i})}{count(y_{i-1})}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T(y_{i-1}->y_i)&space;=&space;\frac{count(y_{i-1},y{i})}{count(y_{i-1})}" title="T(y_{i-1}->y_i) = \frac{count(y_{i-1},y{i})}{count(y_{i-1})}" /></a>
In particular, i=1 or i=n respectively denotes that this word is the first or the last word of the sentence. In this case, the transition probability is:
<a href="https://www.codecogs.com/eqnedit.php?latex=T(y_0->y_1)=&space;\frac{count(START,y_1)}{count(START)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T(y_0->y_1)=&space;\frac{count(START,y_1)}{count(START)}" title="T(y_0->y_1)= \frac{count(START,y_1)}{count(START)}" /></a>
<a href="https://www.codecogs.com/eqnedit.php?latex=T(y_n->y_{n&plus;1})&space;=&space;\frac{count(y_n,STOP)}{count(STOP)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T(y_n->y_{n&plus;1})&space;=&space;\frac{count(y_n,STOP)}{count(STOP)}" title="T(y_n->y_{n+1}) = \frac{count(y_n,STOP)}{count(STOP)}" /></a>.

~~~~python
matrix[front][current] = count(front, current)
states[tag] = count(tag)
start[tag] = count(START,tag)
stop[tag] = count(tag,STOP)
~~~~

#### (1) Import and Initialization
	
	import transition
	t = transition.transition()
This procedure gives an empty `transition` object with empty dictioinaries.

#### (2) Compute Transition Probability

	t.compute(training file name)
Similar to that in emission.py, this procedure computes the values in the four dictionaries.

#### (3) Transition

	t.transit(front, current)
This procedure returns a single floating number denoting the transition probability of transitioning from ***front*** to ***current***, corresponding to the front tag and current tag.

#### (4) `startwith`
	
	t.startwith(tag)
This method returns a single floating number, which is the value of <a href="https://www.codecogs.com/eqnedit.php?latex=T(START,tag)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T(START,tag)" title="T(START,tag)" /></a>, denoting the probability of the sentence starts with ***tag***.

#### (5) `stopwith`
	t.stopwith(tag)
Similar to ***startwith()***, this method returns the probability that one sentence stops with ***tag***.

### 3. `viterbi.py`
This part is the core of the viterbi algorithm, it has two classes and two methods.

#### Find Top-1 tag sequence
~~~~python
viterbi_best(e, t, inputfile, outputfile, p=True)
~~~~

`e`: computed emission object

`t`: computed transition object

`inputfile`: the path of input file

`outputfile`: path of output file

`p`: whether do preprocessing before computing probabilities

This function applies dynamic programming to find the best tag sequence for given observation of word sequence. As the algorithm to compute top N tag sequence is more general, the procedure would be specified in next part.

#### Find Top-N  Tag Sequence

~~~~python
viterbi_Nbest(e,t,input,output,best=10,p=True)
~~~~

`e`: computed emission object

`t`: computed transition object

`inputfile`: the path of input file

`outputfile`: path of output file

`p`: whether do preprocessing before computing probabilities

This function is implemented with the following 2 classes:

* `class worditem`:

		word: the previous tag of current position
		score: score of this previous tag at this position
		path: which path of the previous tag this score is from

* `class NBest`:
  
  This class maintains a heap of worditem(s) to rank previous tags with their respective scores.
  		
  		NBest.NBest(best):
  		best value specifying the number of best paths we want
  
		NBest.add(word, prob, path)
		adding a new worditem object with this 3 values
		into the heap
		
		NBest.best()
		remove all paths after ranking behind the specified best
		value


#### Algorithm Specification

In viterbi algorithm, we have following dynamic transition function:

<a href="https://www.codecogs.com/eqnedit.php?latex=Pi&space;[&space;i&space;,&space;tag&space;]&space;=&space;Max(Pi&space;[&space;i-1&space;,&space;tag’&space;]&space;*&space;Transition[&space;tag’&space;,&space;tag]&space;*&space;Emission[&space;tag&space;,&space;word]&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Pi&space;[&space;i&space;,&space;tag&space;]&space;=&space;Max(Pi&space;[&space;i-1&space;,&space;tag’&space;]&space;*&space;Transition[&space;tag’&space;,&space;tag]&space;*&space;Emission[&space;tag&space;,&space;word]&space;)" title="Pi [ i , tag ] = Max(Pi [ i-1 , tag’ ] * Transition[ tag’ , tag] * Emission[ tag , word] )" /></a>

Base Case:

<a href="https://www.codecogs.com/eqnedit.php?latex=Pi&space;[&space;0&space;,&space;START&space;]&space;=&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Pi&space;[&space;0&space;,&space;START&space;]&space;=&space;1" title="Pi [ 0 , START ] = 1" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=Pi&space;[&space;0&space;,&space;tag&space;]&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Pi&space;[&space;0&space;,&space;tag&space;]&space;=&space;0" title="Pi [ 0 , tag ] = 0" /></a> ( for tag that is not START )

In order to get the top-N sequences, we must save N-best POS tag sequences at each potential tag for each word.

The result is saved in a queue in the following format.

<a href="https://www.codecogs.com/eqnedit.php?latex=F[&space;i&space;,&space;tag&space;]&space;=&space;[&space;(&space;Pi_{1}&space;,&space;previous\_tag_1&space;)...&space;(&space;Pi_{10}&space;,&space;previous\_tag_{10}&space;)&space;]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F[&space;i&space;,&space;tag&space;]&space;=&space;[&space;(&space;Pi_{1}&space;,&space;previous\_tag_1&space;)...&space;(&space;Pi_{10}&space;,&space;previous\_tag_{10}&space;)&space;]" title="F[ i , tag ] = [ ( Pi_{1} , previous\_tag_1 )... ( Pi_{10} , previous\_tag_{10} ) ]" /></a>

(If the number of possible choices is above 10. If not, just save all the possible choices in the heap)

The generating procedure is:

* Enumerate all positions in a sequence. -> `O(N)`
* Enumerate all possible tags for the word at this position. -> `O(T)`
* Enumerate all possible tags for previous position. -> `O(T)`
* Enumerate all n scores at the given previous tag and compute the score of current position with the dynamic transition function mentioned above, and push the object having related information into the heap. -> `O(n)`
* For one certain current tag, if all previous tags are already enumerated, remove all previous tag paths ranking behind n.

The decoding procedure is:

* As we know the N best tags at a position and from which previous path of the previous tag it comes from, we can decode the sentence from the last to the beginning. -> `O(N)`

All in all, the time complexity of the best N algorithm is `O(nNT^2)`

### 4. `toolbox.py`
This module provides the tools that facilitates other modules, including preprocessor and evaluation methods:
#### Preprocessor
~~~~python
toolbox.preprocess("input file", "output file")
~~~~
This method preprocesses the input file and write the results into output file. It achieves this by using following two methods.
~~~~python
# sentence processing
pword, ptag = toolbox.processSentence(sentence)
~~~~
This method processes one sentence into one processed word and one processed tag. Temporarily, no modifications are done to tag except extracting it from the sentence.
~~~~python
# word processing
pword = toolbox.processWord(word)
~~~~
This method processes a word with the following rules:

  *  When one word contains only punctuations, for example `:)` or `:(`, it remains the same.
  *  The words begins with `@` are turned into USR to be currectly labeled as USR.
  *  Digits are turned into `digit`.
  *  Words starting with `#` are turned into `HT`.
  *  For other words with not only punctuations, like `hello!`, remove the punctuations and modify all alphabets into lower case. This generalization procedure could slightly improve the results.
  *  **[Deprecated]** Exception is URL. Initially URL are retained, but in terms of the error rate there were not improvements, so this rule is deprecated.

~~~~python
hasURL(word) #judge whether one sentence has URL
~~~~        


## Performance
#### POS with preprocessor
1. emission.py MLE, with preprocessor

		python run.py -t ../data/POS/ptrain 
		-i ../data/POS/dev.in 
		-o ../data/POS/dev.p4.out 
		--algorithm 0
	
`Error rate: 0.305203938115`

2. viterbi, best 1, with preprocessor
		python run.py -t ../data/POS/ptrain 
		-i ../data/POS/dev.in 
		-o ../data/POS/dev.p4.out 
		--algorithm 1
Error rate: 0.325246132208

3. viterbi, best 10, with preprocessor
~~~~	
python run.py -t ../data/POS/ptrain 
	-i ../data/POS/dev.in 
	-o ../data/POS/dev.p4.out 
	--algorithm 2 
	-b 10
~~~~
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

