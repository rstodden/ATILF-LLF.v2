import collections
import operator
import os
import logging

from param import FeatParams, XPParams, Paths, PrintParams


class Corpus:
    """
        a class used to encapsulate all the information of the corpus
    """

    mweTokenDic, mweDictionary, mwtDictionary, mwtDictionaryWithSent = {}, {}, {}, {}
    counter_words = collections.Counter()
    counter_cooc = collections.Counter()
    total_number_words = 0
    context_window = XPParams.context_window

    def __init__(self, langName, corpus_version):
        """
            an initializer of the corpus, responsible of creating a structure of objects encapsulating all the information
            of the corpus, its sentences, tokens and MWEs.

            This function iterate over the lines of corpus document to create the precedent ontology
        """
        self.testSentNum, self.testMweNum = 0, 0
        self.corpus_version = corpus_version
        self.header = ''


        # logging.warn(langName)
        print("language:", langName)
        path = os.path.join(Paths.corporaPath, langName)
        # initialize all attributes of the object
        # mweFiles : traindata
        # testMweFile : testdata
        self.langName = langName
        if corpus_version == 1.0:
            self.sentNum, self.tokenNum, self.mweNum, self.intereavingNum, self.emeddedNum, self.singleWordExp, \
            self.continousExp, self.trainingSents, self.testingSents, self.trainDataSet, mweFile, testMweFile = \
                0, 0, 0, 0, 0, 0, 0, [], [], [], os.path.join(path, Paths.train_file_tsv), os.path.join(path, Paths.test_file_tsv)
            conlluFile, testConllu = self.getTrainAndTestConlluPath(path)
            testFile = os.path.join(path, Paths.test_file_tsv)

            if conlluFile is not None and testConllu is not None:
                self.trainDataSet = Corpus.readConlluFile(conlluFile)
                self.mweNum = Corpus.readMweFile(mweFile, self.trainDataSet)
                self.sentNum = len(self.trainDataSet)

                for sent in self.trainDataSet:
                    self.tokenNum += len(sent.tokens)
                    self.emeddedNum += sent.recognizeEmbedded()
                    self.intereavingNum += sent.recognizeInterleavingVMWEs()
                    x, y = sent.recognizeContinouosandSingleVMWEs()
                    self.singleWordExp += x
                    self.continousExp += y
                if XPParams.realExper:
                    self.testDataSet = Corpus.readConlluFile(testConllu)
                    Corpus.readMweFile(testMweFile, self.testDataSet)
            else:
                # if there are no train.conllu.autoPOS.autoDep and train.conllu.autoPOS.autoPOS files
                self.trainDataSet, self.sentNum, self.mweNum = Corpus.readSentences(mweFile)
                self.testDataSet, self.testSentNum, self.testMweNum = Corpus.readSentences(testFile, forTest=True)
                for sent in self.trainDataSet:
                    self.tokenNum += len(sent.tokens)
                    self.emeddedNum += sent.recognizeEmbedded()
                    self.intereavingNum += sent.recognizeInterleavingVMWEs()
                    x, y = sent.recognizeContinouosandSingleVMWEs()
                    self.singleWordExp += x
                    self.continousExp += y
            self.getTrainAndTestSents()
            if XPParams.useCrossValidation:
                self.testRange, self.trainRange = self.getRanges()
            else:
                self.testRange, self.trainRange = None, None
        elif corpus_version == 1.1:
            self.sentNum, self.tokenNum, self.mweNum, self.intereavingNum, self.emeddedNum, self.singleWordExp, \
            self.continousExp, self.trainingSents, self.testingSents, self.trainDataSet, mweFile, testMweFile = \
                0, 0, 0, 0, 0, 0, 0, [], [], [], os.path.join(path, Paths.train_file), os.path.join(path, Paths.test_file)
            file_train, file_test = self.getTrainAndTestConlluPath(path)
            print(self.langName, file_train, file_test, self.corpus_version, "epoches", XPParams.num_epochs, "features", XPParams.numberModuloReduction, "CNN and SVM", XPParams.useCNNandSVM, "only CNN", XPParams.useCNNOnly)
            self.trainDataSet, self.mweNum = self.readConlluAndMweFile(file_train, train=True)
            self.sentNum = len(self.trainDataSet)

            for sent in self.trainDataSet:
                self.tokenNum += len(sent.tokens)
                self.emeddedNum += sent.recognizeEmbedded()
                self.intereavingNum += sent.recognizeInterleavingVMWEs()
                x, y = sent.recognizeContinouosandSingleVMWEs()
                self.singleWordExp += x
                self.continousExp += y
            if XPParams.realExper:
                self.testDataSet = self.readConlluAndMweFile(file_test, train=False)[0]
                # Corpus.readMweFile(testMweFile, self.testDataSet)
            self.getTrainAndTestSents()
            if XPParams.useCrossValidation:
                self.testRange, self.trainRange = self.getRanges()
            else:
                self.testRange, self.trainRange = None, None
        else:
            raise ValueError("wrong corpora version")

    def update(self):
        """ update corpus with mwedictionary (number of occurrences of mwe), 
            mwetokendictionary (tokens in mwe) and 
            mwtdictionary (type of mwe)
        """
        if XPParams.useCrossValidation:
            self.divideSents()
            self.initializeSents()
        if XPParams.print_debug:
            print("trainsents", len(self.trainingSents))
        Corpus.mweDictionary, Corpus.mweTokenDic, Corpus.mwtDictionary = Corpus.getMWEDic(self.trainingSents)

    def getTrainAndTestConlluPath(self, path):
        """ return path of train and test conllu data. 
            Path depends on parameters like AutoGeneratePOS and AutoGenerateDEP
        """
        conlluFile, testConllu = None, None
        version = self.corpus_version
        if version == 1.0:
            if XPParams.useAutoGeneratedPOS and XPParams.useAutoGeneratedDEP and os.path.isfile(
                    os.path.join(path, 'train.conllu.autoPOS.autoDep')):
                # only True for HU
                conlluFile = os.path.join(path, 'train.conllu.autoPOS.autoDep')
                if os.path.isfile(os.path.join(path, 'test.conllu.autoPOS.autoDep')):
                    testConllu = os.path.join(path, 'test.conllu.autoPOS.autoDep')
            elif XPParams.useAutoGeneratedPOS and os.path.isfile(os.path.join(path, 'train.conllu.autoPOS')):
                conlluFile = os.path.join(path, 'train.conllu.autoPOS')
                if os.path.isfile(os.path.join(path, 'test.conllu.autoPOS')):
                    testConllu = os.path.join(path, 'test.conllu.autoPOS')
            if os.path.isfile(os.path.join(path, Paths.train_file_conllu)):
                conlluFile = os.path.join(path, Paths.train_file_conllu)
                if os.path.isfile(os.path.join(path, Paths.test_file_conllu)):
                    testConllu = os.path.join(path, Paths.test_file_conllu)
        elif version == 1.1:
            if os.path.isfile(os.path.join(path, Paths.train_file)):
                conlluFile = os.path.join(path, Paths.train_file)
                if os.path.isfile(os.path.join(path, Paths.test_file)):
                    testConllu = os.path.join(path, Paths.test_file)
        else:
            raise ValueError("wrong corpora version")
        return conlluFile, testConllu

    def divideCorpus(self, foldIdx, foldNum=5):
        testFoldSize = int(len(self.trainDataSet) * 0.2)
        self.testingSents = self.trainDataSet[foldIdx * testFoldSize: (foldIdx + 1) * testFoldSize]
        if foldIdx == 0:
            self.trainingSents = self.trainDataSet[(foldIdx + 1) * testFoldSize:]
        elif foldIdx == foldNum - 1:
            self.trainingSents = self.trainDataSet[:foldIdx * testFoldSize]
        else:
            self.trainingSents = self.trainDataSet[0:foldIdx * testFoldSize] + self.trainDataSet[
                                                                               (foldIdx + 1) * testFoldSize:]

    @staticmethod
    def readConlluFile(conlluFile):
        """ read the mwe corpus file (version 1.0)
            read file linewise. If line is a sent beginning initialize a new sent-object.
            If line is a part of sentence split line and identify elements (also mwes)
            and save in token object. Save tokens in sentence-object.
            At end of sentence store sentence object in sentences list.
            return sentences list
        """
        sentences = []
        with open(conlluFile) as corpusFile:
            # Read the corpus file
            lines = corpusFile.readlines()
            sent = None
            senIdx = 0
            sentId = ''

            lineNum = 0
            missedUnTag = 0
            missedExTag = 0

            for line in lines:
                if len(line) > 0 and line.endswith('\n'):
                    line = line[:-1]
                if line.startswith('# sentid:'):
                    sentId = line.split('# sentid:')[1].strip()
                elif line.startswith('# sentence-text:'):
                    continue
                elif line.startswith('1\t'):
                    if sentId.strip() != '':
                        sent = Sentence(senIdx, sentid=sentId)
                    else:
                        sent = Sentence(senIdx)
                    senIdx += 1
                    sentences.append(sent)

                if not line.startswith('#'):
                    lineParts = line.split('\t')

                    if len(lineParts) != 10 or '-' in lineParts[0]:
                        continue


                    lineNum += 1
                    if lineParts[3] == '_':
                        missedUnTag += 1
                    if lineParts[4] == '_':
                        missedExTag += 1

                    morpho = ''
                    if lineParts[5] != '_':
                        morpho = lineParts[5].split('|')
                    if lineParts[6] != '_':
                        token = Token(lineParts[0], lineParts[1], lemma=lineParts[2],
                                      universalPosTag=lineParts[3], morphologicalInfo=morpho,
                                      dependencyParent=int(lineParts[6]),
                                      dependencyLabel=lineParts[7])
                    else:
                        token = Token(lineParts[0], lineParts[1], lemma=lineParts[2],
                                      universalPosTag=lineParts[3], morphologicalInfo=morpho,
                                      dependencyLabel=lineParts[7])
                    if FeatParams.useUPOS and lineParts[3] != '_':
                        token.universalPosTag = lineParts[3]
                    else:
                        if lineParts[4] != '_':
                            token.posTag = lineParts[4]
                        else:
                            token.posTag = lineParts[3]
                    # Associate the token with the sentence
                    sent.tokens.append(token)
                    sent.text += token.text + ' '
            return sentences


    def readConlluAndMweFile(self, corpus_file, train=True, print_debug=XPParams.print_debug):
        """ read the cupt corpus file with mwe and conllu (version 1.1)
            read file linewise. If line is a sent beginning initialize a new sent-object.
            If line is a part of sentence split line and identify elements (also mwes)
            and save in token object. Save tokens in sentence-object.
            At end of sentence store sentence object in sentences list.
            return sentences list
        """
        sentences = []
        with open(corpus_file) as corpusFile:
            # Read the corpus file
            print("filename", corpus_file)
            lines = corpusFile.readlines()
        sent = None
        senIdx = 0
        sentId = ''
        sent_text = ''
        new_sent = False
        childDict = dict()

        lineNum = 0
        missedUnTag = 0
        missedExTag = 0

        #token_before_no_space = False

        mweNum = 0

        for line in lines:
            if print_debug:
                print("line", line)
            if len(line) > 0 and line.endswith('\n'):
                line = line[:-1]
            if line.startswith('# global.columns = '):
                self.header = line.split('# global.columns = ')[1].strip()
                nr_id, nr_form, nr_lemma, nr_upos, nr_xpos, nr_feats, nr_head, nr_deprel, nr_deps, nr_misc, nr_mwe = Corpus.column_order(self.header)

            elif line.startswith('# source_sent_id = '):
                sentId = line.split('# source_sent_id = ')[1].strip()
                new_sent = True
                if print_debug:
                    print("sentid", sentId)
            elif line.startswith('# text = '):
                sent_text = line.split('# text = ')[1].strip()
                continue

            elif (line.startswith('1\t') or line.startswith('1-')) and new_sent == True:
                # first part of sentence, initialize new sentence
                if sentId.strip() != '':
                    sent = Sentence(senIdx, sentid=sentId, senttext=sent_text)
                    if print_debug:
                        print("sentence object", sent)
                else:
                    sent = Sentence(senIdx)
                for head in childDict.keys():
                    # add number of childs and childs to token
                    #@todo
                    pass
                childDict = dict()
                senIdx += 1
                sentences.append(sent)

            if not line.startswith('#'):
                new_sent = False
                lineParts = line.split('\t')
                if print_debug:
                    print("lineparts", lineParts)
                # id 0 | token 1 | lemma 2 | utag 3 | postag 4 | morpho 5 | depline 6 | deplabel 7 | dependencies 8 | spaceafter attribute 9 | mwelabel 10

                if len(lineParts) != 11 or '-' in lineParts[0]:
                    continue

                lineNum += 1

                # convert float id in english corpus
                if "." in lineParts[nr_id]:
                    lineParts[nr_id] = lineParts[nr_id].split('.')[0]
                    print("wrong type of id", lineParts[nr_id])
                if "." in lineParts[nr_head]:
                    lineParts[nr_head] = lineParts[nr_head].split('.')[0]
                    print("wrong type of id", lineParts[nr_id])

                if lineParts[nr_upos] == '_':
                    missedUnTag += 1
                if lineParts[nr_xpos] == '_':
                    missedExTag += 1

                morpho = ''
                if lineParts[nr_feats] != '_':
                    morpho = lineParts[nr_feats].split('|')
                # generate token object with dependency parent if exist
                if lineParts[nr_head] != '_' and lineParts[nr_head].isdigit():
                    token = Token(lineParts[nr_id],
                                  lineParts[nr_form],
                                  morphologicalInfo=morpho,
                                  dependencyParent=int(lineParts[nr_head]))
                    if lineParts[nr_head] in childDict.keys():
                        childDict[lineParts[nr_head]]["childs"].append(lineParts[nr_lemma])
                        childDict[lineParts[nr_head]]["numChilds"] += 1
                    else:
                        childDict[lineParts[nr_head]] = {"childs": [lineParts[nr_lemma]], "numChilds": 1}
                else:
                    token = Token(lineParts[nr_id],
                                  lineParts[nr_form],
                                  morphologicalInfo=morpho)
                # if value uneqal to '_' and contains information set it otherwise use default of ''
                if FeatParams.usePOS and FeatParams.useUPOS and lineParts[nr_upos] != '_' and lineParts[nr_xpos] != '_' and lineParts[nr_xpos] == lineParts[nr_upos]:
                    token.posTag = lineParts[nr_xpos]
                elif FeatParams.useUPOS and lineParts[nr_upos] != '_':
                    token.universalPosTag = lineParts[nr_upos]
                elif FeatParams.usePOS and lineParts[nr_xpos] != '_':
                    token.posTag = lineParts[nr_xpos]
                if lineParts[nr_lemma] != '_':
                    token.lemma = lineParts[nr_lemma]
                if lineParts[nr_deprel] != '_':
                    token.dependencyLabel = lineParts[nr_deprel]
                #if space_before == False:
                #    token.spaceBefore = True
                #if lineParts[nr_misc] != 'SpaceAfter=No':
                    #token.spaceAfter = None
                    space_before = None
                #else:
                    #token.spaceAfter = True
                    #space_before = False

                Corpus.counter_words[token.getLemma()] += 1
                if len(sent.tokens) >= XPParams.context_window:
                    for prev_word in sent.tokens[-XPParams.context_window:]:
                        Corpus.counter_cooc[(prev_word.getLemma(), token.getLemma())] += 1
                else:
                    for prev_word in sent.tokens:
                        Corpus.counter_cooc[(prev_word.getLemma(), token.getLemma())] += 1

                # append vMWEs to sent
                if lineParts[nr_mwe] != '*' and lineParts[nr_mwe] != '_':
                    vMWEids = lineParts[nr_mwe].split(';')
                    if print_debug:
                        print("vmweids", vMWEids)
                    for vMWEid in vMWEids:
                        id = int(vMWEid.split(':')[0])
                        # New MWE captured
                        if id not in sent.getWMWEIds():
                            if len(vMWEid.split(':')) > 1:
                                type = str(vMWEid.split(':')[1])
                                vMWE = VMWE(id, token, type)
                                if print_debug:
                                    print("vmwe", id, token, type)
                            else:
                                vMWE = VMWE(id, token)
                                if print_debug:
                                    print("vmwe", id, token, type)
                            mweNum += 1
                            sent.vMWEs.append(vMWE)
                        # Another token of an under-processing MWE
                        else:
                            vMWE = sent.getVMWE(id)
                            if vMWE is not None:
                                vMWE.addToken(token)
                        # associate the token with the MWE
                        token.setParent(vMWE)

                # Associate the token with the (last) sentence
                sent.tokens.append(token)
                sent.text += token.text + ' '
        Corpus.total_number_words = sum(Corpus.counter_words.values())
        return sentences, mweNum

    @staticmethod
    def readMweFile(mweFile, sentences):
        """ read the mwe corpus file (version 1.0)"""
        mweNum = 0
        with open(mweFile) as corpusFile:

            # Read the corpus file
            lines = corpusFile.readlines()
            noSentToAssign = False
            sentIdx = 0
            for line in lines:
                if line == '\n' or line.startswith('# sentence-text:') or (
                            line.startswith('# sentid:') and noSentToAssign):
                    continue
                if len(line) > 0 and line.endswith('\n'):
                    line = line[:-1]
                if line.startswith('1\t'):
                    sent = sentences[sentIdx]
                    sentIdx += 1
                lineParts = line.split('\t')
                if '-' in lineParts[0]:
                    continue
                if lineParts is not None and len(lineParts) == 4 and lineParts[3] != '_':
                    # append vMWE to sent object
                    token = sent.tokens[int(lineParts[0]) - 1]
                    vMWEids = lineParts[3].split(';')
                    for vMWEid in vMWEids:
                        id = int(vMWEid.split(':')[0])
                        # New MWE captured
                        if id not in sent.getWMWEIds():
                            if len(vMWEid.split(':')) > 1:
                                type = str(vMWEid.split(':')[1])
                                vMWE = VMWE(id, token, type)
                            else:
                                vMWE = VMWE(id, token)
                            mweNum += 1
                            sent.vMWEs.append(vMWE)
                        # Another token of an under-processing MWE
                        else:
                            vMWE = sent.getVMWE(id)
                            if vMWE is not None:
                                vMWE.addToken(token)
                        # associate the token with the MWE
                        token.setParent(vMWE)
        return mweNum

    @staticmethod
    def readSentences(mweFile, forTest=False):
        """ read sentence of train and test dataset. 
            return sentence-objects, number of sentences and number of mwes
        """
        sentences = []
        sentNum, mweNum = 0, 0
        with open(mweFile) as corpusFile:
            # Read the corpus file
            lines = corpusFile.readlines()
            sent = None
            senIdx = 1
            for line in lines:
                if len(line) > 0 and line.endswith('\n'):
                    line = line[:-1]
                if line.startswith('1\t'):
                    # sentId = line.split('# sentid:')[1]
                    if sent is not None:
                        # Represent the sentence as a sequece of tokens and POS tags
                        sent.setTextandPOS()
                        # if not forTest:
                        sent.recognizeEmbedded()
                        sent.recognizeInterleavingVMWEs()

                    sent = Sentence(senIdx)
                    senIdx += 1
                    sentences.append(sent)

                elif line.startswith('# sentence-text:'):
                    if len(line.split(':')) > 1:
                        sent.text = line.split('# sentence-text:')[1]

                lineParts = line.split('\t')

                # Empty line or lines of the form: "8-9    can't    _    _"
                if len(lineParts) != 4 or '-' in lineParts[0]:
                    continue
                token = Token(lineParts[0], lineParts[1])
                # Trait the MWE
                # if not forTest and lineParts[3] != '_':
                if lineParts[3] != '_':
                    vMWEids = lineParts[3].split(';')
                    for vMWEid in vMWEids:
                        id = int(vMWEid.split(':')[0])
                        # New MWE captured
                        if id not in sent.getWMWEIds():
                            type = str(vMWEid.split(':')[1])
                            vMWE = VMWE(id, token, type)
                            mweNum += 1
                            sent.vMWEs.append(vMWE)
                        # Another token of an under-processing MWE
                        else:
                            vMWE = sent.getVMWE(id)
                            if vMWE is not None:
                                vMWE.addToken(token)
                        # associate the token with the MWE
                        token.setParent(vMWE)
                # Associate the token with the sentence
                sent.tokens.append(token)
            sentNum = len(sentences)
            return sentences, sentNum, mweNum

    @staticmethod
    def getMWEDic(sents):
        """ generate MWE Dictionary with lemma of mwe (as key) and 
            1) type and lemma of singleword mwe = mwtDictionary
            2) count of mwe = mweDictionary
            3) number of tokens = mweTokenDictionary (always = 1?)
            4) mwtDictionaryWithSent
        """
        mweDictionary, mweTokenDictionary, mwtDictionary = {}, {}, {}
        for sent in sents:
            for mwe in sent.vMWEs:
                lemmaString = mwe.getLemmaString()
                if len(mwe.tokens) == 1:
                    if lemmaString not in mwtDictionary:
                        mwtDictionary[lemmaString] = mwe.type
                    if lemmaString not in Corpus.mwtDictionaryWithSent:
                        Corpus.mwtDictionaryWithSent[lemmaString] = [sent]
                    elif lemmaString in Corpus.mwtDictionaryWithSent and Corpus.mwtDictionaryWithSent[
                        lemmaString] is not None:
                        Corpus.mwtDictionaryWithSent[lemmaString] = Corpus.mwtDictionaryWithSent[lemmaString].append(
                            sent)
                if lemmaString in mweDictionary:
                    mweDictionary[lemmaString] += 1
                    for token in mwe.tokens:
                        if token.lemma.strip() != '':
                            mweTokenDictionary[token.lemma] = 1
                        else:
                            mweTokenDictionary[token.text] = 1
                else:
                    mweDictionary[lemmaString] = 1
                    for token in mwe.tokens:
                        if token.lemma.strip() != '':
                            mweTokenDictionary[token.lemma] = 1
                        else:
                            mweTokenDictionary[token.text] = 1
        if FeatParams.usePreciseDictionary:
            for key1 in mweDictionary.keys():
                for key2 in mweDictionary.keys():
                    if key1 != key2:
                        if key1 in key2:
                            mweDictionary.pop(key1, None)
                        elif key2 in key1:
                            mweDictionary.pop(key2, None)
        if XPParams.print_debug:
            print("mweDictionary", len(mweDictionary), "mweTokenDictionary", len(mweTokenDictionary), "mwtDictionary", len(mwtDictionary))
        return mweDictionary, mweTokenDictionary, mwtDictionary

    def initializeSents(self, training=True):
        # Erasing each effect of the previous iteration
        sents = self.trainingSents
        if not training:
            sents = self.testingSents

        for sent in sents:
            sent.identifiedVMWEs = []
            sent.initialTransition = None
            sent.featuresInfo = []
            sent.blackMergeNum = 0
            for mwe in sent.vMWEs:
                mwe.isInTrainingCorpus = 0

    def getTrainAndTestSents(self):

        if XPParams.realExper:
            logging.warn('training sent: train data set')
            logging.warn('test sent: test data set')
            self.trainingSents = self.trainDataSet
            self.testingSents = self.testDataSet

        if XPParams.useCrossValidation:
            logging.warn('Cross validation on train data set')
            self.trainDataSet = self.trainDataSet
            self.testDataSet = []
            return [], []

        if len(self.trainingSents) <= 0:
            logging.warn('test sent: development data set of train data set')
            idx = 0
            self.trainingSents, self.testingSents = [], []
            for sent in self.trainDataSet:
                if idx % 5 == 0:
                    self.testingSents.append(sent)
                else:
                    self.trainingSents.append(sent)
                idx += 1

        return self.trainingSents, self.testingSents

    def getRanges(self):
        """ splits data set in 1/5 test set and 4/5 train set. Iterate over whole corpus and each 1/5 get test set
        """
        sents = self.trainDataSet
        testNum = int(len(sents) * 0.2)
        testRanges = [[0, testNum], [testNum, 2 * testNum], [2 * testNum, 3 * testNum], [3 * testNum, 4 * testNum],
                      [4 * testNum, len(sents)]]

        trainRanges = [[testNum, len(sents)], [0, testNum, 2 * testNum, len(sents)],
                       [0, 2 * testNum, 3 * testNum, len(sents)], [0, 3 * testNum, 4 * testNum, len(sents)],
                       [0, 4 * testNum]]

        return testRanges, trainRanges

    def divideSents(self):
        x = XPParams.currentIteration
        if self.testRange is None or self.trainRange is None:
            return
        self.testingSents = self.trainDataSet[self.testRange[x][0]: self.testRange[x][1]]
        if len(self.trainRange[x]) == 2:
            self.trainingSents = self.trainDataSet[self.trainRange[x][0]: self.trainRange[x][1]]
        else:
            self.trainingSents = self.trainDataSet[self.trainRange[x][0]: self.trainRange[x][1]] + \
                                 self.trainDataSet[self.trainRange[x][2]: self.trainRange[x][3]]

    @staticmethod
    def getNewIdentifiedMWE(testingSents):

        idenMWEs = 0
        newIdenMWEs = 0
        semiNewIdenMWEs = 0
        for sent in testingSents:
            for mwe in sent.vMWEs:
                if mwe.getLemmaString() not in Corpus.mweDictionary.keys():
                    idenMWEs += 1
            for mwe in sent.identifiedVMWEs:
                if mwe.getLemmaString() not in Corpus.mweDictionary.keys():
                    for vmw1 in sent.vMWEs:
                        if vmw1.getLemmaString() == mwe.getLemmaString():
                            newIdenMWEs += 1
                            break
                elif mwe.getLemmaString() in Corpus.mweDictionary.keys() and \
                                Corpus.mweDictionary[mwe.getLemmaString()] < 5:
                    semiNewIdenMWEs += 1

        return float(newIdenMWEs) / idenMWEs, float(semiNewIdenMWEs) / idenMWEs

    def toMWEFile(self):
        res = ''
        for sent in self.testingSents:
            if sent.identifiedVMWEs:
                pass
            idx = 1
            tokenLemmas = []
            for lemma in sent.tokens:
                tokenLemmas.append(lemma.getLemma().strip())
            for lemma in tokenLemmas:
                tokenLbl = ''
                for mwe in sent.identifiedVMWEs:
                    mweLemmas = []
                    for l in mwe.tokens:
                        mweLemmas.append(l.getLemma().strip())
                    if lemma in mweLemmas:
                        tokenIdx = mweLemmas.index(lemma)
                        # if tokenIdx == 0:
                        if tokenLbl:
                            tokenLbl += ';' + str(mwe.id)
                        else:
                            tokenLbl += str(mwe.id)
                if not tokenLbl:
                    tokenLbl = '_'
                res += '{0}\t{1}\t{2}\t{3}\n'.format(idx, lemma, '_', tokenLbl)
                idx += 1
            res += '\n'
        return res

    def __iter__(self):
        for sent in self.trainingSents:
            yield sent

    def getGoldenMWEFile(self):
        res = ''
        if self.corpus_version == 1.1:
            res += '# global.columns = ' + self.header + '\n'
        for sent in self.testingSents:
            if self.corpus_version == 1.1:
                res += '# source_sent_id = '+sent.sentid+'\n'
                res += '# text = '+sent.senttext+'\n'
            idx = 0
            for token in sent.tokens:
                tokenLbl = ''
                if token.parentMWEs:
                    for parent in token.parentMWEs:
                        if tokenLbl:
                            tokenLbl += ';' + str(parent.id)
                        else:
                            tokenLbl += str(parent.id)
                        if token.getLemma() == parent.tokens[0].getLemma():
                            tokenLbl += ':' + parent.type
                if tokenLbl == '':
                    if self.corpus_version == 1.1:
                        tokenLbl = '*'
                    else:
                        tokenLbl = '_'
                if self.corpus_version == 1.0:
                    res += '{0}\t{1}\t{2}\t{3}\n'.format(idx+1, token.text.strip(), '_', tokenLbl)
                elif self.corpus_version == 1.1:
                    if not sent.tokens[idx].dependencyParent:
                        dep_parent = '_'
                    else:
                        dep_parent = sent.tokens[idx].dependencyParent
                    if sent.tokens[idx].lemma == '' and sent.tokens[idx].form == '':
                        line_elements = '\t'.join(['_', '_', '_', '_', '_', '_', '_', '_', '_'])
                    elif sent.tokens[idx].lemma == '':
                        line_elements = '\t'.join([str(idx + 1), sent.tokens[idx].form.strip(),
                                                   '_', '_', '_', '_', '_', '_', '_', '_'])
                    elif sent.tokens[idx].form == '':
                        line_elements = '\t'.join([str(idx + 1), '_',
                                                   sent.tokens[idx].lemma.strip(), '_', '_', '_', '_', '_', '_', '_'])
                    else:
                        line_elements = '\t'.join([str(idx + 1), sent.tokens[idx].form.strip(),
                                                   sent.tokens[idx].lemma.strip(), '_', '_', '_', '_', '_', '_', '_'])

                    res += line_elements + '\t' + str(tokenLbl) + '\n'
                idx += 1
            res += '\n'
        return res

    def __str__(self):
        """" used to print/write the corpus to the resulting file
        """
        res = ''
        if self.corpus_version == 1.1:
            res += '# global.columns = ' + self.header + '\n'
        for sent in self.testingSents:
            if self.corpus_version == 1.1:
                res += '# source_sent_id = '+sent.sentid+'\n'
                res += '# text = '+sent.senttext+'\n'
                labels = ['*'] * len(sent.tokens)
            else:
                labels = ['_'] * len(sent.tokens)
            if XPParams.print_debug:
                print("labels", labels[:250])
                print("tokenlist", sent.tokens[:250])
            for mwe in sent.identifiedVMWEs:
                for token in mwe.tokens:
                    #print(str(mwe.id), mwe.type, "id", "type")
                    if self.corpus_version == 1.0:
                        if labels[token.position - 1] == '_':
                            labels[token.position - 1] = str(mwe.id)
                        else:
                            labels[token.position - 1] += ';' + str(mwe.id)
                        if mwe.tokens[0] == token:
                            labels[token.position - 1] += ':' + mwe.type
                    else:
                        if labels[token.position - 1] == '*':
                            labels[token.position - 1] = str(mwe.id)
                        else:
                            labels[token.position - 1] += ';' + str(mwe.id)
                        if mwe.tokens[0] == token:
                            labels[token.position - 1] += ':' + mwe.type
            #print("labels", labels)
            if self.corpus_version == 1.0:
                for i in range(len(sent.tokens)):
                    res += '{0}\t{1}\t{2}\t{3}\n'.format(i + 1, sent.tokens[i].form.strip(), '_', labels[i])
                res += '\n'
            elif self.corpus_version == 1.1:
                for i in range(len(sent.tokens)):
                    if not sent.tokens[i].dependencyParent:
                        dep_parent = '_'
                    else:
                        dep_parent = sent.tokens[i].dependencyParent
                    if sent.tokens[i].lemma == '' and sent.tokens[i].form == '':
                        if sent.tokens[i].text != '':
                            form = sent.tokens[i].text.strip()
                        else:
                            form = 'missing'
                        line_elements = '\t'.join(
                            [str(i + 1), form, '_',
                             '_', '_', '_', '_', '_', '_', '_'])
                    elif sent.tokens[i].lemma == '':
                        line_elements = '\t'.join(
                            [str(i + 1), sent.tokens[i].form.strip(), '_',
                             '_', '_', '_', '_', '_', '_', '_'])
                    elif sent.tokens[i].form == '':
                        if sent.tokens[i].text != '':
                            form = sent.tokens[i].text.strip()
                        else:
                            form = "missing"
                        line_elements = '\t'.join(
                            [str(i + 1), form, sent.tokens[i].lemma.strip(),
                             '_', '_', '_', '_', '_', '_', '_'])
                    else:
                        line_elements = '\t'.join([str(i+1), sent.tokens[i].form.strip(), sent.tokens[i].lemma.strip(),
                                                   '_', '_', '_', '_', '_', '_', '_'])
                    res += line_elements+'\t'+str(labels[i])+'\n'
                res += '\n'
        return res
        #     idx = 1
        #     tokenLemmas, tokenText = [], []
        #     for token in sent.tokens:
        #         tokenLemmas.append(token.getLemma().strip())
        #         tokenText.append(token.text.strip())
        #     lemmaIdx = 0
        #     for lemma in tokenLemmas:
        #         tokenLbl = ''
        #         for mwe in sent.identifiedVMWEs:
        #             mweLemmas = []
        #             for l in mwe.tokens:
        #                 mweLemmas.append(l.getLemma().strip())
        #             if lemma in mweLemmas:
        #                 # tokenIdx = mweLemmas.index(lemma)
        #                 # if tokenIdx == 0:
        #                 if tokenLbl:
        #                     tokenLbl += ';' + str(mwe.id)
        #
        #                 else:
        #                     tokenLbl += str(mwe.id)
        #                 if lemma == mwe.tokens[0].getLemma():
        #                     tokenLbl += ':OTH'  # TODO
        #         if not tokenLbl:
        #             tokenLbl = '_'
        #         res += '{0}\t{1}\t{2}\t{3}\n'.format(idx, tokenText[lemmaIdx], '_', tokenLbl)
        #         idx += 1
        #         lemmaIdx += 1
        #     res += '\n'
        # return res

    def getMWEDictionary(self):
        mweDictionary = {}
        for sent in self:
            for mwe in sent.vMWEs:
                lemmaString = mwe.getLemmaString()
                if lemmaString in mweDictionary:
                    mweDictionary[lemmaString] += 1
                else:
                    mweDictionary[lemmaString] = 1
        return mweDictionary

    def getMWEDictionaryWithWindows(self):
        mweDictionary = {}
        for sent in self:
            for mwe in sent.vMWEs:
                windows = ''
                for i in range(len(mwe.tokens)):
                    if i > 0:
                        distance = str(sent.tokens.index(mwe.tokens[i]) - sent.tokens.index(mwe.tokens[i - 1]))
                        if windows:
                            windows += ';' + distance
                        else:
                            windows = distance
                lemmaString = mwe.getLemmaString()
                if lemmaString in mweDictionary and mweDictionary[lemmaString] != windows:
                    oldWindow = mweDictionary[lemmaString]
                    oldWindowDistances = oldWindow.split(';')
                    newWindowDistances = windows.split(';')
                    newWindows = ''
                    if len(oldWindowDistances) == len(newWindowDistances):
                        for i in range(len(oldWindowDistances)):
                            if oldWindowDistances[i] > newWindowDistances[i]:
                                newWindows += oldWindowDistances[i] + (';' if i < (len(oldWindowDistances) - 1) else '')
                            else:
                                newWindows += newWindowDistances[i] + (';' if i < (len(newWindowDistances) - 1) else '')
                    else:
                        raise ValueError("Problem with windowsize")
                    mweDictionary[lemmaString] = newWindows
                else:
                    mweDictionary[lemmaString] = windows
        return mweDictionary

    @staticmethod
    def column_order(header):
        """ get order of columns to make sure that the index contains the right values.
        :param header: ID FORM LEMMA UPOS (universal POS-Tag) XPOS (POS-Tag) FEATS (morphological features) HEAD (id of parent) DEPREL (dependency label) DEPS () MISC (space after attribute) PARSEME:MWE
        :return: position of columns
        """
        column_order = header.split(" ")
        nr_id = column_order.index("ID")
        nr_form = column_order.index("FORM")
        nr_lemma = column_order.index("LEMMA")
        nr_upos = column_order.index("UPOS")
        nr_xpos = column_order.index("XPOS")
        nr_feats = column_order.index("FEATS")
        nr_head = column_order.index("HEAD")
        nr_deprel = column_order.index("DEPREL")
        nr_deps = column_order.index("DEPS")
        nr_misc = column_order.index("MISC")
        nr_mwe = column_order.index("PARSEME:MWE")
        return nr_id, nr_form, nr_lemma, nr_upos, nr_xpos, nr_feats, nr_head, nr_deprel, nr_deps, nr_misc, nr_mwe



class Sentence:
    """
       a class used to encapsulate all the information of a sentence
    """

    def __init__(self, id, sentid='', senttext=''):

        self.sentid = sentid
        self.senttext = senttext
        self.id = id
        self.tokens = []
        self.vMWEs = []
        self.identifiedVMWEs = []
        self.text = ''
        self.initialTransition = None
        self.featuresInfo = []
        self.containsEmbedding = False
        self.containsInterleaving = False
        self.containsDistributedEmbedding = False
        self.withRandomSelection = False
        self.blackMergeNum, self.interleavingNum, self.embeddedNum, self.distributedEmbeddingNum = 0, 0, 0, 0

    def getWMWEs(self):
        return self.vMWEs

    def getWMWEIds(self):
        result = []
        for vMWE in self.vMWEs:
            result.append(vMWE.getId())
        return result

    def getVMWE(self, id):

        for vMWE in self.vMWEs:
            if vMWE.getId() == int(id):
                return vMWE
        return None

    def setTextandPOS(self):

        tokensTextList = []
        for token in self.tokens:
            self.text += token.text + ' '
            tokensTextList.append(token.text)
        self.text = self.text.strip()

    def recognizeEmbedded(self, recognizeIdentified=False):
        if recognizeIdentified:
            vmws = self.identifiedVMWEs
        else:
            vmws = self.vMWEs

        if len(vmws) <= 1:
            return 0
        result = 0
        # [x1; x2; x3
        for vMwe1 in vmws:
            if vMwe1.isEmbedded:
                continue
            for vMwe2 in vmws:
                if vMwe1 is not vMwe2 and len(vMwe1.tokens) < len(vMwe2.tokens):
                    if vMwe1.getString() in vMwe2.getString():
                        vMwe1.isEmbedded = True
                        if not recognizeIdentified:
                            self.embeddedNum += 1
                            self.containsEmbedding = True
                        result += 1
                    else:
                        isEmbedded = True
                        vMwe2Lemma = vMwe2.getLemmaString()
                        for token in vMwe1.tokens:
                            if token.getLemma() not in vMwe2Lemma:
                                isEmbedded = False
                                break
                        if isEmbedded:
                            vMwe1.isDistributedEmbedding = True
                            vMwe1.isEmbedded = True
                            if not recognizeIdentified:
                                self.containsDistributedEmbedding = True
                                self.embeddedNum += 1
                                self.distributedEmbeddingNum += 1
                                self.containsEmbedding = True
                            result += 1
        if not recognizeIdentified:
            self.getDirectParents()
        return result

    def recognizeContinouosandSingleVMWEs(self):
        singleWordExp, continousExp = 0, 0
        for mwe in self.vMWEs:
            if len(mwe.tokens) == 1:
                mwe.isSingleWordExp = True
                mwe.isContinousExp = True
                singleWordExp += 1
                continousExp += 1
            else:
                if self.isContinousMwe(mwe):
                    continousExp += 1
        return singleWordExp, continousExp

    def isContinousMwe(self, mwe):
        idxs = []
        for token in mwe.tokens:
            idxs.append(self.tokens.index(token))
        range = xrange(min(idxs), max(idxs))
        mwe.isContinousExp = True
        for i in range:
            if i not in idxs:
                mwe.isContinousExp = False
        return mwe.isContinousExp

    def recognizeInterleavingVMWEs(self):
        if len(self.vMWEs) <= 1:
            return 0
        result = 0
        for vmwe in self.vMWEs:
            if vmwe.isEmbedded or vmwe.isInterleaving:
                continue
            for token in vmwe.tokens:
                if len(token.parentMWEs) > 1:
                    for parent in token.parentMWEs:
                        if parent is not vmwe:
                            if parent.isEmbedded or parent.isInterleaving:
                                continue
                            if len(parent.tokens) <= len(vmwe.tokens):
                                parent.isInterleaving = True
                            else:
                                vmwe.isInterleaving = True
                            self.containsInterleaving = True
                            self.interleavingNum += 1
                            result += 1
        return result

    def getCorpusText(self, gold=True):
        if gold:
            mwes = self.vMWEs
        else:
            mwes = self.identifiedVMWEs
        lines = ''
        idx = 1
        for token in self.tokens:
            line = str(idx) + '\t' + token.text + '\t_\t'
            idx += 1
            for mwe in mwes:
                if token in mwe.tokens:
                    if line.endswith('\t'):
                        line += str(mwe.id)
                    else:
                        line += ';' + str(mwe.id)
                    if token == mwe.tokens[0]:
                        line += ':' + str(mwe.type)
            if line.endswith('\t'):
                line += '_'
            lines += line + '\n'
        return lines

    def getCorpusTextWithPlus(self):
        goldMwes = self.vMWEs
        predMwes = self.identifiedVMWEs
        lines = ''
        idx = 1
        for token in self.tokens:
            line = str(idx) + '\t' + token.text + '\t_\t'
            idx += 1
            for mwe in goldMwes:
                if token in mwe.tokens:
                    if line.endswith('\t'):
                        line += '+'
                        break

            if line.endswith('\t'):
                line += '_\t'
            else:
                line += '\t'
            for mwe in predMwes:
                if token in mwe.tokens:
                    if line.endswith('\t'):
                        line += '+'
                        break
            if line.endswith('\t'):
                line += '_'
            lines += line + '\n'
        return lines

    def isPrintable(self):
        if len(self.vMWEs) > 2:
            return True
        # if not PrintParams.printSentsWithEmbeddedMWEs:
        #     return
        # for mwe in self.vMWEs:
        #     if mwe.isEmbedded:
        #         return True
        return False



        # if PrintParams.printSentsWithEmbeddedMWEs and len(self.vMWEs) > 2:
        #     for mwe in self.vMWEs:
        #         if mwe.isEmbedded:
        #             return True
        # return False

    def getDirectParents(self):

        for token in self.tokens:
            token.getDirectParent()

    @staticmethod
    def getTokens(elemlist):
        """return all Token objects from given elemlist (sentence)"""
        if isinstance(elemlist, Token):
            return [elemlist]
        if isinstance(elemlist, collections.Iterable):
            result = []
            for elem in elemlist:
                if isinstance(elem, Token):
                    result.append(elem)
                elif isinstance(elem, list):
                    result.extend(Sentence.getTokens(elem))
            return result
        return [elemlist]

    @staticmethod
    def getTokenLemmas(tokens):
        """return all Tokens objects which are combined with their lemmas from given elemlist (sentence)"""
        text = ''
        tokens = Sentence.getTokens(tokens)
        for token in tokens:
            if token.lemma != '' and FeatParams.useLemma:
                text += token.lemma + ' '
            else:
                text += token.text + ' '
        return text.strip()

    def printSummary(self):
        vMWEText = ''
        for vMWE in self.vMWEs:
            vMWEText += str(vMWE) + '\n'
        if len(self.identifiedVMWEs) > 0:
            identifiedMWE = '### Identified MWEs: \n'
            for mwe in self.identifiedVMWEs:
                identifiedMWE += str(mwe) + '\n'
        else:
            identifiedMWE = ''

        return '## Sentence No. ' + str(self.id) + ' - ' + self.sentid + '\n' + self.text + \
               '\n### Existing MWEs: \n' + vMWEText + identifiedMWE

    def __str__(self):

        vMWEText = ''
        for vMWE in self.vMWEs:
            vMWEText += str(vMWE) + '\n\n'
        if len(self.identifiedVMWEs) > 0:
            identifiedMWE = '### Identified MWEs: \n'
            for mwe in self.identifiedVMWEs:
                identifiedMWE += str(mwe) + '\n\n'
        else:
            identifiedMWE = ''
        # featuresInfo = ''

        result = ''
        transition = self.initialTransition
        idx = 0
        tab = '&nbsp;'
        while True:
            if transition is not None:
                if transition.type is not None:
                    type = transition.type.name
                else:
                    type = tab * 8
                configuration = str(transition.configuration)
                if type.startswith('MERGE') or type.startswith('WHITE'):
                    type = '**' + type + '**' + tab * 3
                if len(type) == 'SHIFT':
                    type = type + tab * 3
                result += '\n\n' + str(
                    transition.id) + '- ' + type + tab * 3 + '>' + tab * 3 + configuration + '\n\n'
                if transition.next is None:
                    break
                transition = transition.next
                if PrintParams.printFeaturesOfSent and len(self.featuresInfo) == 2 and len(self.featuresInfo[1]) > 0:
                    sortedDic = sorted(self.featuresInfo[1][idx].items(), key=operator.itemgetter(0))
                    for item in sortedDic:
                        result += str(item[0]) + ': ' + str(item[1]) + ', '
                idx += 1
            else:  # result += str(self.featuresInfo[1][idx]) + '\n\n'
                break
        text = ''
        for token in self.tokens:
            if token.parentMWEs is not None and len(token.parentMWEs) > 0:
                text += '**' + token.text + '**' + ' '
            else:
                text += token.text + ' '

        return '## Sentence No. ' + str(self.id) + ' - ' + self.sentid + '\n' + text + \
               '\n### Existing MWEs: \n' + vMWEText + identifiedMWE  # + 'black Merge Num : ' + str(self.blackMergeNum) + ' Interleaving Num: ' + str(self.interleavingNum) \
        # + '\n' + result #+ str(self.initialTransition) + '\n### Features: \n' + featuresInfo

    def __iter__(self):
        for vmwe in self.vMWEs:
            yield vmwe


class VMWE:
    """
        A class used to encapsulate the information of a verbal multi-word expression
    """

    def __init__(self, id, token=None, type='', isEmbedded=False, isInterleaving=False, isDistributedEmbedding=False,
                 isInTrainingCorpus=0):
        self.id = int(id)
        self.isInTrainingCorpus = isInTrainingCorpus
        self.tokens = []
        self.isSingleWordExp = False
        self.isContinousExp = False
        if token is not None:
            self.tokens.append(token)
        self.type = type
        self.isEmbedded = isEmbedded
        self.isDistributedEmbedding = isDistributedEmbedding
        self.isInterleaving = isInterleaving
        self.isVerbal = True
        self.directParent = None

    def getTokenIdx(self, targetToken, sent):
        for token in self.tokens:
            if token.getLemma() == targetToken.getLemma() and \
                            sent.tokens.index(targetToken) == sent.tokens.index(token):
                return self.tokens.index(token)

    def getId(self):
        return self.id

    def addToken(self, token):
        self.tokens.append(token)

    @staticmethod
    def getVMWENumber(tokens):
        result = 0
        for token in tokens:
            if isinstance(token, VMWE):
                result += 1
        return result

    @staticmethod
    def haveSameParents(tokens):
        # Do they have all a parent?
        for token in tokens:
            if not token.parentMWEs:
                return None
        # Get all parents of tokens
        parents = set()
        for token in tokens:
            for parent in token.parentMWEs:
                parents.add(parent)
        if len(parents) == 1:
            return list(parents)

        selectedParents = list(parents)
        for parent in parents:
            for token in tokens:
                if parent not in token.parentMWEs:
                    if parent in selectedParents:
                        selectedParents.remove(parent)

        for parent in list(selectedParents):
            if parent.isInterleaving or parent.isDistributedEmbedding:
                selectedParents.remove(parent)
        return selectedParents

        # @staticmethod
        # def haveSameDirectParents(s0, s1):
        #     if isinstance(s0, Token) and isinstance(s1, Token):
        #         return s0.directParent == s1.directParent
        # if isinstance(s0, list):

    def inVMWE(self, targetToken, sent):
        for token in self.tokens:
            if token.getLemma() == targetToken.getLemma() \
                    and sent.tokens.index(token) == sent.tokens.index(targetToken):
                return True
        return False

    @staticmethod
    def getParents(tokens, type=None):
        if len(tokens) == 1:
            if tokens[0].parentMWEs:
                for vmwe in tokens[0].parentMWEs:
                    if len(vmwe.tokens) == 1:  # and vmwe.type.lower() != type:
                        if type is not None:
                            if vmwe.type.lower() == type.lower():
                                return [vmwe]
                            else:
                                return None
                        else:
                            return [vmwe]

        # Do they have all a parent?
        for token in tokens:
            if len(token.parentMWEs) == 0:
                return None

        # Get all parents of tokens
        parents = set()
        for token in tokens:
            for parent in token.parentMWEs:
                parents.add(parent)
        selectedParents = list(parents)
        for parent in parents:
            if len(parent.tokens) != len(tokens):
                if parent in selectedParents:
                    selectedParents.remove(parent)
                continue
            for token in tokens:
                if parent not in token.parentMWEs:
                    if parent in selectedParents:
                        selectedParents.remove(parent)
        for parent in list(selectedParents):
            if parent.isInterleaving or parent.isDistributedEmbedding:
                selectedParents.remove(parent)
        if type is not None:
            for parent in list(selectedParents):
                if parent.type.lower() != type:
                    selectedParents.remove(parent)
        return selectedParents

    def __str__(self):
        tokensStr = ''
        for token in self.tokens:
            tokensStr += token.text + ' '
        tokensStr = tokensStr.strip()
        isInterleaving = ''
        if self.isInterleaving:
            isInterleaving = ', Interleaving '
        isEmbedded = ''
        if self.isEmbedded:
            if self.isDistributedEmbedding:
                isEmbedded = ', DistributedEmbedding '
            else:
                isEmbedded = ', Embedded '
                # isContinousExp =''
                # if self.isContinousExp:
                # isContinousExp = 'Continous'
        type = ''
        if self.type != '':
            type = '(' + self.type
            if self.isInTrainingCorpus != 0:
                type += ', ' + str(self.isInTrainingCorpus) + ')'
            else:
                type += ')'
        return str(self.id) + '- ' + '**' + tokensStr + '** ' + type + isEmbedded + isInterleaving

    def __iter__(self):
        for t in self.tokens:
            yield t

    def getString(self):
        result = ''
        for token in self.tokens:
            result += token.text + ' '
        return result[:-1].lower()

    def getLemmaString(self):
        result = ''
        for token in self.tokens:
            if token.lemma.strip() != '' and FeatParams.useLemma:
                result += token.lemma + ' '
            else:
                result += token.text + ' '
        return result[:-1].lower()

    def In(self, vmwes):

        for vmwe in vmwes:
            if vmwe.getString() == self.getString():
                return True

        return False

    def __eq__(self, other):
        if not isinstance(other, VMWE):
            raise TypeError()
        if self.getLemmaString() == other.getLemmaString():
            return True
        return False

    def __hash__(self):
        return hash(self.getLemmaString())

    def __contains__(self, vmwe):
        if not isinstance(vmwe, VMWE):
            raise TypeError()
        if vmwe is self or vmwe.getLemmaString() == self.getLemmaString():
            return False
        if vmwe.getLemmaString() in self.getLemmaString():
            return True
        for token in vmwe.tokens:
            if token.getLemma() not in self.getLemmaString():
                return False
        return True


class Token:
    """
        a class used to encapsulate all the information of a sentence tokens
    """

    def __init__(self, position, txt, lemma='', posTag='', universalPosTag='', morphologicalInfo=[],
                 dependencyParent=-1, dependencyLabel='', spaceAfter=''):
        self.position = int(position)
        self.form = txt
        self.text = txt.lower()
        self.lemma = lemma
        self.universalPosTag = universalPosTag
        self.posTag = posTag
        self.morphologicalInfo = morphologicalInfo
        self.dependencyParent = dependencyParent
        self.dependencyLabel = dependencyLabel
        self.parentMWEs = []
        self.directParent = None
        self.spaceAfter = spaceAfter
        self.childTokens = []
        self.numChilds = 0

    def setParent(self, vMWE):
        self.parentMWEs.append(vMWE)

    def getLemma(self):
        """return lemma (string) from Token object"""
        if self.lemma != '' and FeatParams.useLemma:
            return self.lemma.strip()
        return self.text.strip()

    def getDirectParent(self):
        """return and set parent MWEs if exist"""
        self.directParent = None
        if self.parentMWEs is not None and len(self.parentMWEs) > 0:
            if len(self.parentMWEs) == 1:
                if not self.parentMWEs[0].isInterleaving:
                    self.directParent = self.parentMWEs[0]
            else:
                parents = sorted(self.parentMWEs,
                                 key=lambda VMWE: (VMWE.isInterleaving, VMWE.isEmbedded, len(VMWE.tokens)),
                                 reverse=True)
                for parent in parents:
                    if not parent.isInterleaving:
                        self.directParent = parent
                        break
        return self.directParent

    def In(self, vmwe):
        for token in vmwe.tokens:
            if token.text.lower() == self.text.lower() and token.position == self.position:
                return True
        return False

    def isMWT(self):
        """ if parentMWEs is not empty and vmw contains only one token return vmw
            single multiwordelement found!
        """
        if self.parentMWEs:
            for vmw in self.parentMWEs:
                if len(vmw.tokens) == 1:
                    return vmw
        return None

    def __str__(self):
        parentTxt = ''
        if len(self.parentMWEs) != 0:
            for parent in self.parentMWEs:
                parentTxt += str(parent) + '\n'

        return str(self.position) + ' : ' + self.text + ' : ' + self.posTag + '\n' + 'parent VMWEs\n' + parentTxt
