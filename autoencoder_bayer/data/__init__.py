## just copied
from .datasets import *


CORPUS_LIST = {
    'Cerf2007-FIFA': Cerf2007_FIFA,
    #'ETRA2019': ETRA2019,
    'MIT-LowRes': MIT_LowRes,
    #'GAZEBASE': GAZEBASE
}
#'EMVIC2014': EMVIC2014,

frequencies = {
    #1000: ['GAZEBASE'],
    250: ['MIT-LowRes'],
    1000: ['Cerf2007-FIFA']#, 'GAZEBASE'],
    #500: ['ETRA2019'],
    # the ones from MIT are 240Hz but oh well  
}
#1000: ['EMVIC2014', 'Cerf2007-FIFA'], 

def get_corpora(args, additional_corpus=None):
    corpora = []
    for f, c in frequencies.items():
        #if args.hz <= f:
        #    corpora.extend(c)
        corpora.extend(c)

    # corpora = list(CORPUS_LIST.keys())

    # used to add a corpus to evaluator to test for overfitting during
    # training time
    # if isinstance(additional_corpus, str) and additional_corpus not in corpora:
    #     corpora.append(additional_corpus)
    #     logging.info('[evaluator] Added an unseen data set: {}'.format(
    #         additional_corpus))
    return {c: CORPUS_LIST[c](args) for c in corpora}

