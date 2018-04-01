# SUTD Machine Learning Project 2015
Machine Learning Project at SUTD, 2015

##instruction
This package consists of four python modules:

1. **emission.py:** This module computes emission probability with given training text.
2. **transition.py:** This module computes transition probability with given training text.
3. **viterbi.py:** This module implements viterbi algorithm to predict the tag of given test text
4. **toolbox.py:**This module is a box of tools, providing functionalities and optimization for three modules above, including preprocessor, evaluation of a prediction

##Result:
POS:

without preprocessor
POS,MLE: 0.399085794655
POS,DP: 0.409282700422
runtime: 183.357277
1 :POS, DP2: 0.385372714487
2 :POS, DP2: 0.386779184248
3 :POS, DP2: 0.388537271449
4 :POS, DP2: 0.389240506329
5 :POS, DP2: 0.393108298172
6 :POS, DP2: 0.39099859353
7 :POS, DP2: 0.398030942335
8 :POS, DP2: 0.392405063291
9 :POS, DP2: 0.396624472574
10 :POS, DP2: 0.395218002813
with preprocessor
POS, MLE: 0.373769338959
POS, DP: 0.354430379747
runtime: 308.317963
1 :POS, DP2: 0.339310829817
2 :POS, DP2: 0.341420534459
3 :POS, DP2: 0.347046413502
4 :POS, DP2: 0.346694796062
5 :POS, DP2: 0.348452883263
6 :POS, DP2: 0.350914205345
7 :POS, DP2: 0.351617440225
8 :POS, DP2: 0.348452883263
9 :POS, DP2: 0.349156118143
10 :POS, DP2: 0.353023909986

NPC:

without preprocessor
NPC,MLE: 0.299996977665
NPC,DP: 0.223622570798
runtime: 17.605402
1 :NPC, DP2: 0.223622570798
2 :NPC, DP2: 0.24311663191
3 :NPC, DP2: 0.244325565932
4 :NPC, DP2: 0.256173119352
5 :NPC, DP2: 0.258016743736
6 :NPC, DP2: 0.264726327561
7 :NPC, DP2: 0.263910297096
8 :NPC, DP2: 0.271224347931
9 :NPC, DP2: 0.267325535709
10 :NPC, DP2: 0.271428571429
with preprocessor
NPC, MLE: 0.296672409103
NPC, DP: 0.215673829601
runtime: 23.08525
1 :NPC, DP2: 0.215673829601
2 :NPC, DP2: 0.236739504942
3 :NPC, DP2: 0.240849880618
4 :NPC, DP2: 0.255780215795
5 :NPC, DP2: 0.25714026657
6 :NPC, DP2: 0.265361017922
7 :NPC, DP2: 0.268111342823
8 :NPC, DP2: 0.275425393659
9 :NPC, DP2: 0.267144195606
10 :NPC, DP2: 0.257142857143

##toolbox.py
This module provides the tools that facilitates other modules, including preprocessor and evaluation methods:
####Preprocessor
	toolbox.preprocess("input file", "output file")
This method preprocesses the input file and write the results into output file. It achieves this by using following two methods:

	#sentence processing
	pword, ptag = toolbox.processSentence(sentence)
This method processes one sentence into one processed word and one processed tag. Temporarily, no modifications are done to tag except extracting it from the sentence.
	
	#word processing
	pword = toolbox.processWord(word)
This method processes a word with the following rule:

  *  when one word contains only punctuations, for example ":)", ":(", remains the same
  *  for other words with not only punctuations like "hello!", remove the punctuations in them and modify into lower case. This generalization procedure could slightly improve the results.
  *  [deprecated]exception is URL, initially URL are retained, but in terms of the error rate there were not improvements, so deprecated:

		hasURL(word) #judge whether one sentence has URL
		
####Evaluation
This tool is to evaluate the prediction with the answer, and return the error rate:

	evaluate("test file","answer file",col=1, pr=False)
The *test file* contains our prediction, *answer file* contains the correct anser, *col* specifies which column of prediction is to be tested (in case there are multiple predictions like in Top-10 predictions), when *pr* is set True, the function would print in which line the prediction differs from the answer.

##emission.py
This module contains the class which maintains the emission probabilty of text given in a file.
#### Algorithm Specification
This class computes emission probability of the given text with the Maximal Likelihood Estimator introduced in Hidden Markov Model.

The general case formula is:
$$ e(word|tag) = \frac{count(word,tag)}{count(tag)+1}$$
As we can hardly enumerate all possible *"word"*s in the given text file, for a word that has never appeared in the corpus, we provide a special case formula to estimate it's emission probability:
$$e(new|tag) = \frac{1}{count(tag)+1}$$

An experiment, when we change the estimator for new words, the result is better than before:
$$e(new|tag) = \frac{count(tag)}{count(tag)+1}$$
**My thoughts, the score actually doesn't matter, but we should tend to give the new word a more probable tag, this trick applied to part 5**

without preprocessor
POS,MLE: 0.340717299578
POS,DP: 0.377285513361
runtime: 146.777487
1 :POS, DP2: 0.352320675105
2 :POS, DP2: 0.353375527426
3 :POS, DP2: 0.360407876231
4 :POS, DP2: 0.357946554149
5 :POS, DP2: 0.356891701828
6 :POS, DP2: 0.35970464135
7 :POS, DP2: 0.365682137834
8 :POS, DP2: 0.362869198312
9 :POS, DP2: 0.369549929677
10 :POS, DP2: 0.364978902954
with preprocessor
POS, MLE: 0.322784810127
POS, DP: 0.323136427567
runtime: 265.356141
1 :POS, DP2: 0.307665260197
2 :POS, DP2: 0.319971870605
3 :POS, DP2: 0.3129395218
4 :POS, DP2: 0.313291139241
5 :POS, DP2: 0.317862165963
6 :POS, DP2: 0.323136427567
7 :POS, DP2: 0.321729957806
8 :POS, DP2: 0.32770745429
9 :POS, DP2: 0.321026722925
10 :POS, DP2: 0.322081575246
without preprocessor
NPC,MLE: 0.242995738508
NPC,DP: 0.191072022244
runtime: 12.2817
1 :NPC, DP2: 0.191072022244
2 :NPC, DP2: 0.204732976698
3 :NPC, DP2: 0.21727566718
4 :NPC, DP2: 0.221930063167
5 :NPC, DP2: 0.23613503793
6 :NPC, DP2: 0.235500347569
7 :NPC, DP2: 0.240396530359
8 :NPC, DP2: 0.242935291806
9 :NPC, DP2: 0.244053555777
10 :NPC, DP2: 0.285714285714
with preprocessor
NPC, MLE: 0.269259830145
NPC, DP: 0.200652824372
runtime: 18.826656
1 :NPC, DP2: 0.200652824372
2 :NPC, DP2: 0.218091697646
3 :NPC, DP2: 0.226796022607
4 :NPC, DP2: 0.236195484631
5 :NPC, DP2: 0.245776286759
6 :NPC, DP2: 0.250551576148
7 :NPC, DP2: 0.255417535588
8 :NPC, DP2: 0.258167860489
9 :NPC, DP2: 0.257200713271
10 :NPC, DP2: 0.2

Internally, this class maintains two Python dictionaries, named *matrix* and *labels*, which has:

	matrix[word][tags] = count(word, tag)
	labels[tag] = count(tag)
####import the module and initialization
	# import the module
	import emission
	# initialization
	e = emission.emission()
This step provides an "empty" emission object.
####compute the parameters with a given corpus
	e.compute("file name")
After this step the *matrix* and *labels* in emission classes is computed. 
####compute emission probability
	e.emit(word, tag, p=True)
	# attention: tag must appear in the given training corpus
	# or it would raise a runtime error
*word* is the word we want to emit, *tag* is the tentative tag to emit, and *p* stands for "process", when *p* is set true, the object would process *word* before computing the emission probability.
####guess the most probable tag
	e.mostprob()
It occurs in Hidden Markov Model that at some points in a sentence, all possible tags score 0. When it occurs such scenario one enhacement is tag this position as the most probable label among all possible labels.

Another try: only change the word guess at 0 score point but don't change the original path, the result proves to be worse than before 
without preprocessor
runtime: 160.738827
1 :POS, DP2: 0.387834036568
2 :POS, DP2: 0.393108298172
3 :POS, DP2: 0.396272855134
4 :POS, DP2: 0.393459915612
5 :POS, DP2: 0.399789029536
6 :POS, DP2: 0.396976090014
7 :POS, DP2: 0.403305203938
8 :POS, DP2: 0.397679324895
9 :POS, DP2: 0.403656821378
10 :POS, DP2: 0.401195499297
with preprocessor
runtime: 274.522126
1 :POS, DP2: 0.340014064698
2 :POS, DP2: 0.347046413502
3 :POS, DP2: 0.352672292546
4 :POS, DP2: 0.352320675105
5 :POS, DP2: 0.353375527426
6 :POS, DP2: 0.356188466948
7 :POS, DP2: 0.356891701828
8 :POS, DP2: 0.353727144866
9 :POS, DP2: 0.356188466948
10 :POS, DP2: 0.360759493671
without preprocessor
runtime: 13.108931
1 :NPC, DP2: 0.224136367758
2 :NPC, DP2: 0.24311663191
3 :NPC, DP2: 0.244265119231
4 :NPC, DP2: 0.255659322392
5 :NPC, DP2: 0.257291383323
6 :NPC, DP2: 0.264333424003
7 :NPC, DP2: 0.262610693021
8 :NPC, DP2: 0.269682957053
9 :NPC, DP2: 0.26738598241
10 :NPC, DP2: 0.271428571429
with preprocessor
runtime: 20.377208
1 :NPC, DP2: 0.216036509807
2 :NPC, DP2: 0.236739504942
3 :NPC, DP2: 0.240819657267
4 :NPC, DP2: 0.255719769094
5 :NPC, DP2: 0.256566022909
6 :NPC, DP2: 0.264877444313
7 :NPC, DP2: 0.266358388491
8 :NPC, DP2: 0.274065342884
9 :NPC, DP2: 0.267476652462
10 :NPC, DP2: 0.285714285714

####MLE predictor:
With the emission probability, one method to tag a word in one sentence is to choose the tag which has the greatest emission probability to the word.

	e.predict("input filename","output filename", p=True)
Then the object labels the input file with emission probability and writes the labeled text into output file. 

When *p* is set True, the word to be tagged would be processed with toolbox before computing emission probability.

##transition.py
This module provides a class similar to emission.py, which also maintains two internal dictionaries ***matrix*** and ***states*** to compute the transition probability with the MLE in Hidden Markov Model, where:
$$T(y_{i-1}->y_i) = \frac{count(y_{i-1},y{i})}{count(y_{i-1})}$$
In particular, i=1 or i=n respectively denotes that this word is the first or last word of the sentence, under such case the transition probability is:
$$T(y_0->y_1)= \frac{count(START,y_1)}{count(START)}$$
$$T(y_n->y_{n+1}) = \frac{count(y_n,STOP)}{count(STOP)}$$

The ***matrix*** dictionary maintains the v
