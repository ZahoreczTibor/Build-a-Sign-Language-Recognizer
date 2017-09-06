import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
    
        Baysian Information Criterion: maximises the likelihood of data - penalising large-size models. (BIC = -2 * log L + p * log N)
        (where #L is likelihood of the model, # p is the quantity of free parameters, #p * log N is the "penalty term", #N is quantity of data points. ==> #p = num free params = transistion probs(n*n) + means(n*f) + covars(n*f) (*ref:https://discussions.udacity.com/t/understanding-better-model-selection/232987/3 - more about free parameters: https://ai-nd.slack.com/files/ylu/F4S90AJFR/number_of_parameters_in_bic.txt)
        
        
        References:
        - https://en.wikipedia.org/wiki/Hidden_Markov_model#Architecture
        - https://discussions.udacity.com/t/bayesian-information-criteria-equation/326887/2
        - https://discussions.udacity.com/t/verifing-bic-calculation/246165
        - https://github.com/hmmlearn/hmmlearn/blob/master/hmmlearn/hmm.py#L106
        - https://en.wikipedia.org/wiki/Bayesian_information_criterion
        - Bayesian Analysis with Python by by Osvaldo Martin (2016 Packt)

        """
        best_BIC = float('Inf')
        best_model = None 
       
        for num_components in range(self.min_n_components, self.max_n_components+1):

            try:        
                current_model = self.base_model(num_components)
                logL = current_model.score(self.X, self.lengths)

                BIC = -2 * logL + (num_components ** 2 + 2 * num_components * current_model.n_features - 1 ) * np.log(len(self.sequences))

                if BIC < best_BIC:

                    best_BIC = BIC

                    best_model = current_model
                    
            except:
               
                if self.verbose:

                    print("failure on {} with {} states, continuing".format(self.this_word, num_components))

                pass

        return best_model
        
class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    
    DIC: we should find the model that gives high likelihood to the original word and low likelihood to the other words.
    = log(P(original word)) - average(log(P(other words))) 
     
    '''
    def select(self):

        n_components_range = range(self.min_n_components, self.max_n_components + 1)
        M =len(n_components_range)

        try: 
            best_score = float("-inf") 
            best_model = None            
            for num_states in n_components_range:

                model = self.base_model(num_states) 
                LogL_P = model.score(self.X, self.lengths)
                scores = []

                for word in self.words:
                    if word != self.this_word:
                        scores.append(model.score(self.X, self.lengths)) 
                DIC_score = LogL_P -np.mean(scores)            

                if DIC_score > best_score:
                    best_score = DIC_score
                    best_model = model
                    
        except Exception as e:

            pass

        if best_model is None: 

            return self.base_model(self.n_constant)

        else:

            return best_model
        
class SelectorCV(ModelSelector):
    
    ''' select best model based on average log Likelihood of cross-validation folds
    
        CV technique breaking-down the training set into "folds", rotating which fold is "left out" of the training set.The fold that is "left out" is scored for validation.

        reference: https://www.r-bloggers.com/aic-bic-vs-crossvalidation/
        * Figure 2: Comparison of effectiveness of AIC, BIC and crossvalidation in selecting the most parsimonous model (black arrow) from the set of 7 polynomials that were fitted to the data
    '''
    def select(self):

        warnings.filterwarnings("ignore", category=DeprecationWarning)

        try:
            best_score = float("-Inf")
            best_model = None

            for n_components in range(self.min_n_components, self.max_n_components + 1):

                split_method = KFold(n_splits=min(3,len(self.sequences)))
                logL_arr = []
                hmm_model = None

                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)
                    hmm_model = self.base_model(n_components)

                    X_test, test_lengths = combine_sequences(cv_test_idx, self.sequences)
                    logL = hmm_model.score(X_test, test_lengths)
                    logL_arr.append(logL)

                avg_score = np.mean(logL_arr)
                
                if(avg_score > best_score):

                    best_score = avg_score
                    best_model = hmm_model

            return best_model

        except:

            return self.base_model(self.n_constant)