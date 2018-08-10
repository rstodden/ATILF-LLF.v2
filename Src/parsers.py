from features import Extractor
from param import XPParams, Paths
from transTypes import TransitionType
from transitions import EmbeddingTransition, Transition
from oracles import EmbeddingOracle
from scipy.sparse import csr_matrix
import numpy
from numpy import argmax
from keras.models import load_model, Model
from modulo_reduction import ModuloReduction
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2, mutual_info_classif
from collections import Counter
import os


class Parser:
	@staticmethod
	def parse(corpus, clf):
		"""
        Iterate on sentences of the test set and predict transitions for it.
        The predicted transition will be applied and and the integer of it will be appended to labels (list) andd to saved in the featuresInfo of  the sentence object
        :param corpus: corpus object with sentence objects of testing set
        :param clf: classifier model
        """
		count_transitions = Counter()
		all_labels = list()
		if XPParams.useCrossValidation:
			corpus.initializeSents(training=False)
		label_file_content = list()
		folder = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "Results/")
		if Paths.descriptionPath:
			folder += Paths.descriptionPath + "/"
			if not os.path.exists(folder):
				os.makedirs(folder)
		with open(EmbeddingOracle.get_path(folder + 'features/' + str(XPParams.numberModuloReduction) + '/',
										   corpus.langName, 0, "test_modulo_features.csv"), "w+") as f:
			f.write("")
		if XPParams.use_extern_labels:
			i = 0
			with open(os.path.join(Paths.projectPath, Paths.extern_labels)) as f:  # FA_test_labels
				label_file_content = f.readlines()
				label_file_content = [int(l.split(" ")[0]) for l in label_file_content]
			for sent in corpus.testingSents:
				sent.initialTransition = EmbeddingTransition(None, isInitial=True, sent=sent)
				# new emdiddingtransition which inherit everything from Transition -> full buffer, empty stack
				transition = sent.initialTransition
				labels = []
				while not transition.configuration.isTerminalConf() and i < len(label_file_content):
					transTypeValue = label_file_content[i]
					transType = TransitionType.getType(transTypeValue)
					# transType: enumeration object with transitiontype
					legalTansDic = transition.getLegalTransDic()
					if transType in legalTansDic:
						newTransition = legalTansDic[transType]
					elif len(legalTansDic):
						newTransition = Transition.initialize(legalTansDic.keys()[0], sent)
					else:
						print("no trans")
						newTransition = None
					if newTransition:
						# newTransition: Transition object
						newTransition.apply(transition, sent, parse=True)
						labels.append(newTransition.type.value)
						count_transitions[newTransition.type.value] += 1
						# labels array/list contains the predicted labels (integer) of the sentence
						transition = newTransition
					i += 1
				sent.featuresInfo = [labels,
									 []]  # second argument is needed to have the same structure as in training (features.py -> Extractor.extract)
		else:
			for sent in corpus.testingSents:
				sent.initialTransition = EmbeddingTransition(None, isInitial=True, sent=sent)
				# new emdiddingtransition which inherit everything from Transition -> full buffer, empty stack
				transition = sent.initialTransition
				labels = []
				while not transition.configuration.isTerminalConf():
					# while configuration is not None -> buffer and stack unequal to 0 (config.py)
					# feats used here BQ
					# clf[0] SVM classifier(clf) and clf[1] DictVectorizer(vec) without values, clf[2] cnn (if used)
					newTransition = Parser.getNextTransition(transition, sent, clf, corpus, folder)
					if newTransition:
						# newTransition: Transition object
						newTransition.apply(transition, sent, parse=True)
						labels.append(newTransition.type.value)
						count_transitions[newTransition.type.value] += 1
						# labels array/list contains the predicted labels (integer) of the sentence
						transition = newTransition
				sent.featuresInfo = [labels,
									 []]  # second argument is needed to have the same structure as in training (features.py -> Extractor.extract)
				all_labels.extend(labels)
			with open(EmbeddingOracle.get_path(folder + 'labels/' + str(XPParams.numberModuloReduction) + '/',
											   corpus.langName, 0, "test_labels.txt"), "w+") as file_labels:
				file_labels.write('\n'.join(str(x) for x in all_labels))
		# print(count_transitions)

	@staticmethod
	def getNextTransition(transition, sent, clf, corpus, folder):
		"""
        extract features of the sentence of the trainingset and predict a transition for it
        :param transition:
        :param sent: sentence object of test corpus (corpus.py)
        :param clf[0]: trained model in oracles.py (sklearn.multiclass OutputCodeClassifier),
                clf[1]: DictVectorizer (vec) without values,
                clf[2] cnn model
        :return: transition object
        """
		if not XPParams.use_extern_labels:
			classifier = clf[0]
			dictVectorizer = clf[1]
			if XPParams.use_feature_selection_chi:
				feat_selection = clf[2]
			elif XPParams.use_feature_selection_mi:
				feat_selection = clf[2]
			else:
				feat_selection = ''
		legalTansDic = transition.getLegalTransDic()
		# legalTansDic: dictionary which contains transition names as keys (<TransitionType.SHIFT: 0>) and the transition object as value (transitions.Shift)
		if len(legalTansDic) == 1:
			return Transition.initialize(legalTansDic.keys()[0], sent)
		# ----- Extract Features ---- #
		featDic = Extractor.getFeatures(transition, sent)
		if not isinstance(featDic, list):
			featDic = [featDic]

		X = dictVectorizer.transform(featDic)  # transforms valueDict to sparse matrix

		# ---- PREDICTION ------- #

		if XPParams.use_feature_selection_chi:
			# mi = SelectKBest(mutual_info_classif)
			# mi = SelectPercentile()
			# print("pred", X.shape)
			X = feat_selection.transform(X)
			# chi = SelectKBest()
		if XPParams.use_feature_selection_mi:
			X = feat_selection.transform(X)
			# modulo reduction and prediction
		if XPParams.useModuloReduction:
			n_new_features = XPParams.numberModuloReduction
			if XPParams.twoD_feats:
				X = ModuloReduction.reduce_matrix_with_modulo(X, n_new_features, twoD_feats=True,
															  new_y_number=XPParams.new_y_number)
			else:
				X = ModuloReduction.reduce_matrix_with_modulo(X, n_new_features, calc_ppmi=XPParams.usePMICalc)
			# print(type(NewX), NewX)
			# transTypeValue = classifier.predict(X)[0]
		if XPParams.useCNNandSVM:
			# layer_model = Model(inputs=cnn.input, outputs=cnn.get_layer(XPParams.CNNlayerName).output)
			if not XPParams.useMultiChannel:
				X = X.toarray()
				X = X.reshape((1, 1, X.shape[1], 1))
			layer_model = clf[3]
			cnn_prediction = layer_model.predict(X)

			clf_prediction = classifier.predict(cnn_prediction)[0]
			# print(clf_prediction)
			# transTypeValue = argmax(clf_prediction) # numpy function which convert onehotvector back to integer
			transTypeValue = clf_prediction

		elif XPParams.useCNNOnly:
			# layer_model = Model(inputs=cnn.input, outputs=cnn.get_layer(XPParams.CNNlayerName).output)
			if not XPParams.twoD_feats:
				X = X.toarray()
				with open(EmbeddingOracle.get_path(folder + 'features/' + str(XPParams.numberModuloReduction) + '/',
												   corpus.langName, 0, "test_modulo_features.csv"), "a+") as f:
					numpy.savetxt(f, X, delimiter="\t", fmt='%i')
				if not XPParams.useMultiChannel:
					X = X.reshape((1, 1, X.shape[1], 1))
			else:
				X = X.reshape((1, 1, X.shape[1], X.shape[2]))
			cnn_prediction = classifier.predict(X)  # vector with one element per class
			# print("cnn",cnn_prediction)
			# clf_prediction = classifier.predict(cnn_prediction)[0]
			# print(clf_prediction)
			# transTypeValue = argmax(clf_prediction) # numpy function which convert onehotvector back to integer
			transTypeValue = numpy.argmax(cnn_prediction)
			# print(transTypeValue, "transtypevalue")

		else:
			transTypeValue = classifier.predict(X.toarray())[0]
			# transTypeValue: type integer, value of predicted transition
		# ----------------------- #

		transType = TransitionType.getType(transTypeValue)
		# transType: enumeration object with transitiontype
		if transType in legalTansDic:
			# legalTansDic[transType]: transition
			return legalTansDic[transType]
		if len(legalTansDic):
			# if only one possible transition in dict then initialize and return it
			return Transition.initialize(legalTansDic.keys()[0], sent)
		# raise an error if transtype is not in legaltransdict and legaltransdict contains one element
		raise ValueError("No transition found.")

