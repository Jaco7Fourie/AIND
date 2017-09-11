import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    Xlengths = test_set.get_all_Xlengths()
    for X, lengths in Xlengths.values():
        likelihood_dict = {}
        top_score = float('-Inf')
        top_guess = ''
        for w, model in models.items():
            try:
                # for each word calculate the LogLValue
                score = model.score(X, lengths)
                # print('score = {} (w: {})'.format(score, w))
                likelihood_dict[w] = score

                if score > top_score:
                    top_guess = w
                    top_score = score
            except:
                # something went wrong
                likelihood_dict[w] = float('-Inf')

        probabilities.append(likelihood_dict)
        guesses.append(top_guess)

    return probabilities, guesses

