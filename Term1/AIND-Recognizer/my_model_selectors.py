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
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        lowest_bic = float('Inf')
        best_model = None
        for i in range(self.min_n_components, self.max_n_components+1):
            cur_model = self.base_model(i)
            if cur_model is None:
                continue
            # the log likelihood
            try:
                logL = cur_model.score(self.X, self.lengths)
            except:
                continue
            # the number of data points
            logN = np.log(len(self.X))

            # according to https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235/17
            # number of parameters is
            # transition props (num_states**2) + means (dims*states) + variances (dims*states) -1
            # p = n^2 + 2*d*n - 1
            dims = cur_model.n_features
            p = i ** 2 + 2 * dims * i - 1
            bic_score = -2.0 * logL + p * logN

            if bic_score < lowest_bic:
                lowest_bic = bic_score
                best_model = cur_model
        return best_model

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def dic_score(self, n):
        """
            score based on the estimation of the discriminative factor criterion
            see https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
        """
        model = self.base_model(n)
        scores = []
        # self.hwords.items() return the key of the dict (the word)
        # followed by the data required for scoring in model.score
        for word, (X, lengths) in self.hwords.items():
            # check to make sure we are only collecting the anti-evidence
            # (class models conflicting with the current class)
            if word != self.this_word:
                try:
                    scores.append(model.score(X, lengths))
                except:
                    continue

        try:
            score = model.score(self.X, self.lengths) - np.mean(scores)
        except:
            return float('-Inf'), model
        return score, model

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        highest_dic = float('-Inf')
        best_model = None
        for i in range(self.min_n_components, self.max_n_components + 1):
            dic_score, cur_model = self.dic_score(i)

            if dic_score > highest_dic:
                highest_dic = dic_score
                best_model = cur_model
        return best_model

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def cv_score(self, n):
        """
            score based on Kfold cross validation
            see https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
        """

        split_method = KFold()
        hmm_model = self.base_model(n)
        fold_scores = []
        # Check for more than 2 sequences as required by sensible kfold
        if len(self.sequences) > 2:
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)
                x_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)

                hmm_model = self.base_model(n)
                try:
                    score = hmm_model.score(x_test, lengths_test)
                    fold_scores.append(score)
                except:
                    continue
        else:
            try:
                hScore = hmm_model.score(self.X, self.lengths)
            except:
                # print("could not score SelectorCV: (word:{}, n:{})".format(self.this_word, n))
                return float('-Inf'), hmm_model
            return hScore, hmm_model

        # Find mean across all folds
        avg_score = np.mean(fold_scores)
        return avg_score, hmm_model

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        highest_dic = float('-Inf')
        best_model = None
        for i in range(self.min_n_components, self.max_n_components + 1):
            dic_score, cur_model = self.cv_score(i)

            if dic_score > highest_dic:
                highest_dic = dic_score
                best_model = cur_model
        return best_model
