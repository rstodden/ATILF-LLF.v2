import logging
import os
import sys
from param import XPParams

from corpus import Corpus
from evaluation import Evaluation
from oracles import EmbeddingOracle
from param import FeatParams, XPParams, Paths
#if XPParams.usePMICalc:
#from parsers_pmi import Parser
#else:
from parsers import Parser

#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def identify(corpus_version):
    """ read corpus of language, read features of language and 
        identifyCorpus and createMWEFiles in different ways 
        for crossvalidation and without)
    """
    configFile = Paths.configFile
    constantConfigFolder = Paths.configsFolder
    FeatParams(os.path.join(constantConfigFolder, configFile))
    corpus = Corpus(configFile[:2], corpus_version)

    if XPParams.useCrossValidation:
        scores = [0] * 12
        testRange, trainRange = corpus.getRanges()
        for x in range(len(testRange)):
            logging.warn('Iteration no.' + str(x + 1))
            XPParams.currentIteration = x
            Paths.iterationPath = os.path.join(Paths.langResultFolder, str(x + 1))
            evalScores = identifyCorpus(corpus, x)
            for i in range(6):
                scores[i] += evalScores[i]
            createMWEFiles(corpus, configFile[:2], x)
        for i in range(len(scores)):
            scores[i] /= float(len(testRange))
        logging.warn(' F-Score: ' + str(scores[0]))
    else:
        identifyCorpus(corpus)
        createMWEFiles(corpus, configFile[:2])


def identifyCorpus(corpus, x=-1):
    """ update corpus with mwedictionaries (type, count, tokens), 
        train, predict and evaluate corpus
    """
    print(XPParams.use_extern_labels)
    if XPParams.use_extern_labels:
        Parser.parse(corpus, "") # -> prediction
        scores = Evaluation.evaluate(corpus) # -> evaluate
    else:
		corpus.update()
		clf = EmbeddingOracle.train(corpus, x) # -> training
		Parser.parse(corpus, clf) # -> prediction
		scores = Evaluation.evaluate(corpus) # -> evaluate
    return scores


def createMWEFiles(corpus, lang, x=-1):
    """ save corpus in CV or testSet directory. 
        Write goldfiles and results
        :param corpus: corpus object
        :param lang: language of current corpus
        :param x: number of crossvalidation
    """
    folder = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "Results/MWEFiles")
    if XPParams.useCrossValidation:
        folder += '/CV/' + lang
    else:
        folder += '/testSet/'
    if Paths.descriptionPath:
        folder += Paths.descriptionPath + "/"
        if not os.path.exists(folder):
            os.makedirs(folder)
    if not os.path.exists(folder):
        os.makedirs(folder)
    if not XPParams.predictOnly:
        if not os.path.exists(folder+  str(XPParams.numberModuloReduction)+ '/' +"CNN-SVM/"):
            os.makedirs(folder+  str(XPParams.numberModuloReduction)+ '/' +"CNN-SVM/")
        if not os.path.exists(folder+  str(XPParams.numberModuloReduction)+ '/' +"CNN/"):
            os.makedirs(folder+  str(XPParams.numberModuloReduction)+ '/' +"CNN/")
        if not os.path.exists(folder+  str(XPParams.numberModuloReduction)+ '/' +"CNN-SVM/"+lang+'/'):
            os.makedirs(folder+  str(XPParams.numberModuloReduction)+ '/' +"CNN-SVM/"+lang+'/')
        if not os.path.exists(folder+  str(XPParams.numberModuloReduction)+ '/' +"CNN/"+lang+'/'):
            os.makedirs(folder+  str(XPParams.numberModuloReduction)+ '/' +"CNN/"+lang+'/')
        if x == -1:
            x = ''
        if XPParams.useCNNandSVM:
            # mwePath = os.path.join(folder + "CNN-SVM/" + lang + '/', lang + str(x) + '.txt')
            mwePath = os.path.join(folder +  str(XPParams.numberModuloReduction)+ '/' + "CNN-SVM/" + lang + '/', "test.system.cupt")
            goldenPath = os.path.join(folder + str(XPParams.numberModuloReduction)+ '/' + "CNN-SVM/" + lang + '/', lang + str(x) + '.gold.txt')
        elif XPParams.useCNNOnly:
            # mwePath = os.path.join(folder + "CNN/" + lang + '/', lang + str(x) + '.txt')
            mwePath = os.path.join(folder + str(XPParams.numberModuloReduction)+ "/CNN/" + lang + '/', "test.system.cupt")
            goldenPath = os.path.join(folder + str(XPParams.numberModuloReduction)+ "/CNN/" + lang + '/', lang + str(x) + '.gold.txt')
        else:
            mwePath = os.path.join(folder + lang + '/', "test.system.cupt")
            goldenPath = os.path.join(folder + lang + '/', str(x) + '.gold.txt')

        with open(mwePath, 'w') as f:
            logging.warn('MWE file is being written to {0}'.format(mwePath))
            f.write(str(corpus))
        with open(goldenPath, 'w') as f:
            logging.warn(' Golden MWE file is being written to {0}'.format(goldenPath))
            f.write(corpus.getGoldenMWEFile())
    else:
        if not os.path.exists(folder + str(XPParams.numberModuloReduction) + '/' + str(XPParams.nrRun) + "/CNN-SVM/"):
            os.makedirs(folder + str(XPParams.numberModuloReduction) + '/' + str(XPParams.nrRun) + "/CNN-SVM/")
        if not os.path.exists(folder + str(XPParams.numberModuloReduction) + '/' + str(XPParams.nrRun) + "/CNN/"):
            os.makedirs(folder + str(XPParams.numberModuloReduction) + '/' + str(XPParams.nrRun) + "/CNN/")
        if not os.path.exists(folder + str(XPParams.numberModuloReduction) + '/' + str(XPParams.nrRun) + "/CNN-SVM/" + lang + '/'):
            os.makedirs(folder + str(XPParams.numberModuloReduction) + '/' + str(XPParams.nrRun) + "/CNN-SVM/" + lang + '/')
        if not os.path.exists(folder + str(XPParams.numberModuloReduction) + '/' + str(XPParams.nrRun) + "/CNN/" + lang + '/'):
            os.makedirs(folder + str(XPParams.numberModuloReduction) + '/' + str(XPParams.nrRun) + "/CNN/" + lang + '/')
        if x == -1:
            x = ''
        if XPParams.useCNNandSVM:
            # mwePath = os.path.join(folder + "CNN-SVM/" + lang + '/', lang + str(x) + '.txt')
            mwePath = os.path.join(folder + str(XPParams.numberModuloReduction) + '/' + str(XPParams.nrRun) + "/CNN-SVM/" + lang + '/',
                                   "test.system.cupt")
            #goldenPath = os.path.join(folder + str(XPParams.numberModuloReduction) + '/' + str(XPParams.nrRun) + "/CNN-SVM/" + lang + '/',
            #                          lang + str(x) + '.gold.txt')
        elif XPParams.useCNNOnly:
            # mwePath = os.path.join(folder + "CNN/" + lang + '/', lang + str(x) + '.txt')
            mwePath = os.path.join(folder + str(XPParams.numberModuloReduction) + '/' + str(XPParams.nrRun) + "/CNN/" + lang + '/',
                                   "test.system.cupt")
            #goldenPath = os.path.join(folder + str(XPParams.numberModuloReduction) + '/' + str(XPParams.nrRun) + "/CNN/" + lang + '/',
            #                          lang + str(x) + '.gold.txt')
        else:
            mwePath = os.path.join(folder + lang + '/', "test.system.cupt")

        #if not os.path.exists(mwePath[:-4]):
        #    os.makedirs(mwePath[:-4])
        with open( mwePath, 'w') as f:
            logging.warn('MWE file is being written to {0}'.format(mwePath))
            f.write(str(corpus))


#reload(sys)
#sys.setdefaultencoding('utf8')
#logging.basicConfig(level=logging.WARNING)

# run main program with settings of param.py
#Paths.train_file_tsv = 'train.parsemetsv'
#Paths.test_file_tsv = 'test.parsemetsv'
#Paths.configFile = 'FA.json'
#identify()

import argparse
from param import XPParams

parser = argparse.ArgumentParser(description="""
        train input file and predict testfile.""")
parser.add_argument("--svm", type=int, default=1,
        help="""use svm""")
parser.add_argument("--cnn", type=int, default=1,
        help="""use cnn""")
parser.add_argument("--epoches", type=int, default=10,
        help="""number of epoches of cnn""")
parser.add_argument("--number_modulo", type=int, default=500,
        help="""number of new dimensions""")
parser.add_argument("--nrrun", type=str, default='',
        help="""number of run""")
parser.add_argument("--predictonly", type=int, default=0,
        help="""only prediction without training and model building""")
parser.add_argument("--corpora_version", type=str, default="1.1",
        help="""The version of the corpus and parseme task""")
parser.add_argument("--descriptionPath", type=str, default="",
        help="""description of the current test (modulo, batch generator, new features, cv, changed svm parameters?)""")
parser.add_argument("--useBatchGenerator", type=int, default=0,
        help="""use batchgenerator instead of modulo?""")
# ppmi;trainset;multichannel;kernelsizes;dim_hidden;filters;batchsize;embedding_dims;dropout;compile_metric;pooling;
parser.add_argument("--kernelsizes", type=str, default="4,6,8",
       help="""use these kernelsizes""")
parser.add_argument("--dimensions_hiddenlayer", type=int, default=50,
       help="""size of dimensions of hiddenlayer""")
parser.add_argument("--filters", type=int, default=32,
       help="""size of filters""")
parser.add_argument("--batchsize", type=int, default=32,
       help="""size of batch""")
parser.add_argument("--dimensions_embedding", type=int, default=32,
       help="""size of dimensions of embedding layer""")
parser.add_argument("--dropout", type=float, default=0.025,
       help="""size of dropout-rate""")
parser.add_argument("--compile_metric", type=str, default="fbeta_score",
       help="""name of compile metric""")
parser.add_argument("--pooling", type=str, default="maxpooling",
       help="""name of pooling layer""")
parser.add_argument("--ppmi", type=int, default=0,
        help="""use weighted features""")
        
parser.add_argument("--use_extern_labels", type=int, default=0,
        help="""use extern features""")
parser.add_argument("--label_file", type=str, default="", 
        help="file name of extern labels")
# parser.add_argument("--usePercentile", type=int, default=0,
#        help="""use feature selection with percentile""")
#parser.add_argument("--useMI", type=int, default=0,
#        help="""use feature selection mi""")
#parser.add_argument("--useCHI2", type=int, default=0,
#        help="""use feature selection chi2""")
#parser.add_argument("--numPerc", type=int, default=1,
#        help="""number of percentile for feature selection""")
parser.add_argument("--allFilters", type=int, default=0,
        help="""enable all filters?""")

parser.add_argument("config_file", type=str,
        help="""The config file""")
parser.add_argument("training_file", type=str,
        help="""The training file""")
parser.add_argument("test_file", type=str,
        help="""The test file""")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.cnn == 1 and args.svm == 0 and args.predictonly == 1:
        XPParams.useCNNOnly = True
        XPParams.useCNNandSVM = False
        XPParams.buildModel = False #False
        XPParams.predictOnly = True
    elif args.cnn == 1 and args.svm == 0 and args.predictonly == 0:
        XPParams.useCNNOnly = True
        XPParams.useCNNandSVM = False
        XPParams.buildModel = True #False
    elif args.cnn == 1 and args.svm == 1 and args.predictonly == 0:
        XPParams.useCNNandSVM = True
        XPParams.useCNNOnly = False
        XPParams.buildModel = False # True
        XPParams.buildSVM = True # False
    elif args.cnn == 1 and args.svm == 1 and args.predictonly == 1:
        XPParams.useCNNandSVM = True
        XPParams.useCNNOnly = False
        XPParams.buildModel = False # True
        XPParams.buildSVM = False # False
        XPParams.predictOnly = True
    else:
        raise ValueError("CNN or SVM needed. SVM only not allowed.")
    if args.descriptionPath != '':
        Paths.descriptionPath = args.descriptionPath
    else:
        Paths.descriptionPath = None
    if args.useBatchGenerator == 1:
        XPParams.useBatchGenerator = True
        XPParams.useModuloReduction = False
    else:
        XPParams.useBatchGenearator = False
        XPParams.useModuloReduction = True
    if args.allFilters == 1:
        XPParams.useAllFilters = True
    else:
        XPParams.useAllFilters = False
    #if args.usePercentile == 1:
    #    XPParams.use_feature_selection_percentile = True
    #    XPParams.numPerc = args.numPerc
    #if args.useMI == 1:
    #    XPParams.use_feature_selection_mi = True
    #elif args.useCHI2 == 1:
    #    XPParams.use_feature_selection_chi = True
    #else:
    #    XPParams.useModuloReduction = True
    XPParams.num_epochs = args.epoches
    XPParams.numberModuloReduction = args.number_modulo
    # if XPParams.numberModuloReduction == 500:
    #     XPParams.dropout = 0.025
    # if XPParams.numberModuloReduction == 1000:
    #     XPParams.dropout = 0.025
    # elif XPParams.numberModuloReduction == 2000:
    #     XPParams.dropout = 0.05
    # elif XPParams.numberModuloReduction == 5000:
    #     XPParams.dropout = 0.075
    # elif XPParams.numberModuloReduction == 10000:
    #     XPParams.dropout = 0.1

    XPParams.dropout = args.dropout
    XPParams.kernelsize = args.kernelsizes.split(',')
    XPParams.dim_hidden = args.dimensions_hiddenlayer
    XPParams.filters = args.filters
    XPParams.batchSize = args.batchsize
    XPParams.dim_embedd = args.dimensions_embedding
    XPParams.compile_metric = args.compile_metric
    XPParams.pooling = args.pooling
    XPParams.usePMICalc = args.ppmi
    XPParams.use_extern_labels = args.use_extern_labels
    Paths.extern_labels = args.label_file

    XPParams.nrRun = args.nrrun
    Paths.train_file = args.training_file
    Paths.test_file = args.test_file

    Paths.configFile = args.config_file
    XPParams.corpora_version = float(args.corpora_version)
    projectPath = os.path.dirname(__file__)[:-len(os.path.basename(os.path.dirname(__file__)))]
    if XPParams.corpora_version == 1.1:
        Paths.corporaPath = os.path.join(projectPath, "sharedtask_11/")
    #print(args.test_file, args.training_file, args.config_file, XPParams.num_epochs, XPParams.numberModuloReduction, XPParams.useCNNOnly, XPParams.useCNN)
    identify(XPParams.corpora_version)
