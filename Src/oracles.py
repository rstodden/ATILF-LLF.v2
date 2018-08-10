from __future__ import print_function
from __future__ import division
import os

# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import datetime
import logging
import gc

import numpy
from numpy import genfromtxt
import scipy.sparse
from scipy.sparse import csr_matrix
import pickle

from param import FeatParams, XPParams, Paths
from keras.layers import Convolution2D
from sklearn.feature_extraction import DictVectorizer
from sklearn.multiclass import OutputCodeClassifier, OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.externals import joblib

from features import Extractor
from transitions import Reduce, BlackMerge, EmbeddingTransition, MergeAsMWT
from transitions import Shift

from modulo_reduction import ModuloReduction

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling2D, Flatten, Conv2D, MaxPooling1D, Convolution1D, \
	Concatenate, AveragePooling1D
from keras.models import load_model, Model

from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2, mutual_info_classif

import keras
from keras import backend as K
from keras.utils.vis_utils import plot_model, model_to_dot

import threading


class EmbeddingOracle:
	@staticmethod
	def train(corpus, x):
		"""
        :param corpus: corpus object with sentece objects ot the training test
        :param x: use for crross validation
        :return: SVM classifier (clf) and DictVectorizer (vec) without values
        """
		time = datetime.datetime.now()
		logging.info('Static Embedding Oracle')
		folder = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "Results/")
		if Paths.descriptionPath:
			folder += Paths.descriptionPath + "/"
			if not os.path.exists(folder):
				os.makedirs(folder)
		# create directory for models
		if not os.path.exists(folder + "model/"):
			os.makedirs(folder + "model/")
		if not os.path.exists(folder + "model/" + str(XPParams.numberModuloReduction) + '/'):
			os.makedirs(folder + "model/" + str(XPParams.numberModuloReduction) + '/')
		print(XPParams.buildModel, XPParams.buildSVM)
		if XPParams.buildModel:
			Y, X_dic = EmbeddingOracle.parseCorpus(corpus.trainingSents, EmbeddingOracle)
			# X_dic dictionary with symbolic features names and feature labels
			vec = DictVectorizer()
			X = vec.fit_transform(X_dic)
			# type of X is sparse matrix can be transformed with toarray to arrays.
			print("X finished", X.shape[0], X.shape[1])

			# ------- save features to file
			# EmbeddingOracle.save_labels_and_features(Y, X, folder, corpus.langName, x)
			# save labels
			with open(EmbeddingOracle.get_path(folder + 'labels/' + str(XPParams.numberModuloReduction) + '/',
											   corpus.langName, x, "train_labels.txt"), "w+") as file_labels:
				file_labels.write('\n'.join(str(x) for x in Y))
			with open(EmbeddingOracle.get_path(folder + 'features/' + str(XPParams.numberModuloReduction) + '/',
											   corpus.langName, x, "train_vec"), "wb") as vec_file:
				pickle.dump(vec, vec_file)
			print("train", X.shape)
			# ----- use absolute value and modulo
			if XPParams.useModuloReduction:
				n_new_features = XPParams.numberModuloReduction
				if XPParams.twoD_feats:
					X = ModuloReduction.reduce_matrix_with_modulo(X, n_new_features, folder, training=True,
																  twoD_feats=True, new_y_number=XPParams.new_y_number)
					# numpy.save(EmbeddingOracle.get_path(folder + 'features/' + str(XPParams.numberModuloReduction) + '/',
					#                             corpus.langName, x, "modulo_features"), X)
					# scipy.sparse.save_npz(
					#    EmbeddingOracle.get_path(folder + 'features/' + str(XPParams.numberModuloReduction) + '/',
					#                             corpus.langName, x, "modulo_features"), X)
				else:
					X = ModuloReduction.reduce_matrix_with_modulo(X, n_new_features, folder, training=True,
																  calc_ppmi=XPParams.usePMICalc)
					scipy.sparse.save_npz(
						EmbeddingOracle.get_path(folder + 'features/' + str(XPParams.numberModuloReduction) + '/',
												 corpus.langName, x, "train_modulo_features"), X)
					numpy.savetxt(
						EmbeddingOracle.get_path(folder + 'features/' + str(XPParams.numberModuloReduction) + '/',
												 corpus.langName, x, "train_modulo_features") + '.csv',
						X.toarray(), delimiter="\t", fmt='%i')
					print(X.count_nonzero())
				print("New X (sparse matrix) finished")
				# --- train CNN ------------
			if XPParams.use_feature_selection_chi:
				# mi = SelectKBest(mutual_info_classif)
				# mi = SelectPercentile()
				if XPParams.use_feature_selection_percentile:
					n_feats = X.shape[1]
					# percentile = int(round(((n_feats / 1000)/n_feats)*100))
					# print("percentile", ((n_feats / 1000)/n_feats)*100)
					percentile = XPParams.numPerc
					feat_selection = SelectPercentile(chi2, percentile)
				elif XPParams.use_feature_selection_kbest:
					feat_selection = SelectKBest(chi2)
				else:
					feat_selection = ''
				X = feat_selection.fit_transform(X, Y)
				scipy.sparse.save_npz(
					EmbeddingOracle.get_path(folder + 'features/' + str(XPParams.numberModuloReduction) + '/',
											 corpus.langName, x, "chi2_features"), X)
				print("chi square", X.shape)
				# chi = SelectKBest()
			elif XPParams.use_feature_selection_mi:
				if XPParams.use_feature_selection_percentile:
					n_feats = X.shape[1]
					# percentile = int(round(((n_feats / 1000)/n_feats)*100))
					# print("percentile", percentile)
					percentile = XPParams.numPerc
					feat_selection = SelectPercentile(chi2, percentile)
				elif XPParams.use_feature_selection_kbest:
					feat_selection = SelectKBest(chi2)
				else:
					feat_selection = ''
				X = feat_selection.fit_transform(X, Y)
				scipy.sparse.save_npz(
					EmbeddingOracle.get_path(folder + 'features/' + str(XPParams.numberModuloReduction) + '/',
											 corpus.langName, x, "mi_features"), X)
				print("mi", X.shape)
			else:
				feat_selection = ''
			if XPParams.useCNNandSVM:
				# use cnn and svm
				# ---------- initialize SVM ----- #
				# clf = OutputCodeClassifier(LinearSVC(random_state=0), code_size=2, random_state=0)
				clf = SVC()
				number_of_classes = XPParams.numberOfClasses
				print(set(Y))
				Y_one_hot = keras.utils.to_categorical(Y, num_classes=number_of_classes)

				if XPParams.useBatchGenerator:
					cnn = EmbeddingOracle.train_CNN_batch(Y_one_hot, X)
					cnn.save(folder + "model/" + str(
						XPParams.numberModuloReduction) + "/cnn_model_" + corpus.langName + ".h5")
				else:
					if XPParams.useMultiChannel:
						cnn = EmbeddingOracle.train_CNN_multichannel(Y_one_hot, X)
					else:
						cnn = EmbeddingOracle.train_CNN(Y_one_hot, X)
					cnn.save(folder + "model/" + str(
						XPParams.numberModuloReduction) + "/cnn_model_" + corpus.langName + ".h5")

				# -------- get output of an intermediate layer ---------------
				# https://keras.io/getting-started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer
				layer_model = Model(inputs=cnn.input, outputs=cnn.get_layer(XPParams.CNNlayerName).output)
				# layer_model type keras.layers.convolutional.Conv1D without methods
				print("shape of CNN layer", X.shape)

				features = X.toarray()
				features = features.reshape((X.shape[0], 1, X.shape[1], 1))
				# ((number_of_samples_X, 1, number_of_features, 1))
				cnn_features = layer_model.predict(features)
				# print(cnn_features.shape) #(None, 26569, 250)
				# 2D layer is needed: reshape conv_layer (produces memoryerror while predicting) or use hidden_layer
				clf.fit(cnn_features, Y)  # Y_one_hot hasnt suitable shape
				# joblib.dump(clf, folder + "model/"+str(XPParams.numberModuloReduction)+"/svm_svc_model_" + corpus.langName + ".pkl")
				with open(folder + "model/" + str(
						XPParams.numberModuloReduction) + "/svm_svc_model_" + corpus.langName + ".pkl",
						  "wb") as clf_file:
					pickle.dump(clf, clf_file)
				print(cnn.summary())
				logging.info('Training Time: ' + str(int((datetime.datetime.now() - time).seconds / 60.)))

				return clf, vec, feat_selection, layer_model

			elif XPParams.useCNNOnly:
				# use only cnn
				number_of_classes = XPParams.numberOfClasses
				Y_one_hot = keras.utils.to_categorical(Y, num_classes=number_of_classes)
				print("trained labels", set(Y))
				if XPParams.useBatchGenerator:
					cnn = EmbeddingOracle.train_CNN_batch(Y_one_hot, X)
					cnn.save(folder + "model/" + str(
						XPParams.numberModuloReduction) + "/cnn_model_" + corpus.langName + ".h5")
				else:
					if XPParams.useMultiChannel:
						cnn = EmbeddingOracle.train_CNN_multichannel(Y_one_hot, X)
					else:
						cnn = EmbeddingOracle.train_CNN(Y_one_hot, X)
					cnn.save(folder + "model/" + str(
						XPParams.numberModuloReduction) + "/cnn_model_" + corpus.langName + ".h5")
				print(cnn.summary())
				logging.info('Training Time: ' + str(int((datetime.datetime.now() - time).seconds / 60.)))
				return cnn, vec, feat_selection
			else:
				# use only svm
				# clf = OutputCodeClassifier(LinearSVC(random_state=0), code_size=2, random_state=0)
				clf = SVC()
				clf.fit(X.toarray(), Y)
				# joblib.dump(clf, 'model/svm_model_'+corpus.langName+'.pkl')
				with open(folder + 'model/svm_model_' + corpus.langName + '.pkl', "wb") as clf_file:
					pickle.dump(clf, clf_file)
				logging.info('Training Time: ' + str(int((datetime.datetime.now() - time).seconds / 60.)))
				return clf, vec, feat_selection

		elif not XPParams.buildModel and not XPParams.buildSVM:
			# Y, X_dic = EmbeddingOracle.parseCorpus(corpus.trainingSents, EmbeddingOracle)
			# X_dic dictionary with symbolic features names and feature labels
			# vec = DictVectorizer()
			# X = vec.fit_transform(X_dic)
			# with open(EmbeddingOracle.get_path(folder + 'labels/' + str(XPParams.numberModuloReduction) + '/',
			#                                   corpus.langName, x, "labels"), "w+") as file_labels:
			#    Y = file_labels.readlines()
			#
			# Y = scipy.sparse.load_npz(EmbeddingOracle.get_path(folder + 'labels/' + str(XPParams.numberModuloReduction) + '/',
			#                                   corpus.langName, x, "labels"))
			with open(EmbeddingOracle.get_path(folder + 'features/' + str(XPParams.numberModuloReduction) + '/',
											   corpus.langName, x, "vec"), "rb") as vec_file:
				vec = pickle.load(vec_file)
			if XPParams.use_feature_selection_chi:
				with open(EmbeddingOracle.get_path(folder + 'features/' + str(XPParams.numberModuloReduction) + '/',
												   corpus.langName, x, "chi2_features"), "rb") as feat_file:
					feat_selection = pickle.load(feat_file)
			elif XPParams.use_feature_selection_chi:
				with open(EmbeddingOracle.get_path(folder + 'features/' + str(XPParams.numberModuloReduction) + '/',
												   corpus.langName, x, "mi_features"), "rb") as feat_file:
					feat_selection = pickle.load(feat_file)
			else:
				feat_selection = ''
			if XPParams.useCNNandSVM:
				cnn = load_model(
					folder + "model/" + str(XPParams.numberModuloReduction) + "/cnn_model_" + corpus.langName + ".h5",
					custom_objects={'fbeta_score': fbeta_score})
				layer_model = Model(inputs=cnn.input, outputs=cnn.get_layer(XPParams.CNNlayerName).output)
				# clf = joblib.load(folder + "model/"+str(XPParams.numberModuloReduction)+"/svm_svc_model_" + corpus.langName + ".pkl")
				with open(folder + "model/" + str(
						XPParams.numberModuloReduction) + "/svm_svc_model_" + corpus.langName + ".pkl") as clf_file:
					clf = pickle.load(clf_file)
				print("loaded files for cnn and svm prediction")
				return clf, vec, feat_selection, layer_model
			elif XPParams.useCNNOnly:
				cnn = load_model(
					folder + "model/" + str(XPParams.numberModuloReduction) + "/cnn_model_" + corpus.langName + ".h5",
					custom_objects={'fbeta_score': fbeta_score})
				print("loaded files for cnn prediction")
				return cnn, vec, feat_selection
			else:
				# clf = joblib.load("model/"+str(XPParams.numberModuloReduction)+"/svm_model_"+corpus.langName+'.pkl')
				with open("model/" + str(
						XPParams.numberModuloReduction) + "/svm_model_" + corpus.langName + '.pkl') as clf_file:
					clf = pickle.load(clf_file)
				print("loaded files for svm prediction")
				return clf, vec, feat_selection

		elif not XPParams.buildModel and XPParams.buildSVM:
			# Y, X_dic = EmbeddingOracle.parseCorpus(corpus.trainingSents, EmbeddingOracle)
			# X_dic dictionary with symbolic features names and feature labels
			# vec = DictVectorizer()
			# X = vec.fit_transform(X_dic)
			# with open(EmbeddingOracle.get_path(folder + 'labels/' + str(XPParams.numberModuloReduction) + '/',
			#                                   corpus.langName, x, "labels"), "w+") as file_labels:
			#    Y = file_labels.readlines()
			#
			# Y = scipy.sparse.load_npz(EmbeddingOracle.get_path(folder + 'labels/' + str(XPParams.numberModuloReduction) + '/',
			#                                   corpus.langName, x, "labels"))
			with open(EmbeddingOracle.get_path(folder + 'features/' + str(XPParams.numberModuloReduction) + '/',
											   corpus.langName, x, "vec"), "rb") as vec_file:
				vec = pickle.load(vec_file)
			if XPParams.useModuloReduction:
				X = scipy.sparse.load_npz(
					EmbeddingOracle.get_path(folder + 'features/' + str(XPParams.numberModuloReduction) + '/',
											 corpus.langName, x, "modulo_features.npz"))
			if XPParams.use_feature_selection_chi:
				with open(EmbeddingOracle.get_path(folder + 'features/' + str(XPParams.numberModuloReduction) + '/',
												   corpus.langName, x, "chi2_features"), "rb") as feat_file:
					feat_selection = pickle.load(feat_file)
			elif XPParams.use_feature_selection_chi:
				with open(EmbeddingOracle.get_path(folder + 'features/' + str(XPParams.numberModuloReduction) + '/',
												   corpus.langName, x, "mi_features"), "rb") as feat_file:
					feat_selection = pickle.load(feat_file)
			else:
				feat_selection = ''
			if XPParams.useCNNandSVM:
				# Y, X_dic = EmbeddingOracle.parseCorpus(corpus.trainingSents, EmbeddingOracle)
				with open(EmbeddingOracle.get_path(folder + 'labels/' + str(XPParams.numberModuloReduction) + '/',
												   corpus.langName, x, "train_labels.txt")) as f:
					c = f.readlines()
					Y = list()
					for line in c:
						line = int(line.strip())
						Y.append(line)

				cnn = load_model(
					folder + "model/" + str(XPParams.numberModuloReduction) + "/cnn_model_" + corpus.langName + ".h5",
					custom_objects={'fbeta_score': fbeta_score})
				layer_model = Model(inputs=cnn.input, outputs=cnn.get_layer(XPParams.CNNlayerName).output)
				features = X.toarray()
				features = features.reshape((X.shape[0], 1, X.shape[1], 1))
				# ((number_of_samples_X, 1, number_of_features, 1))
				cnn_features = layer_model.predict(features)
				print("cnn features finished")
				clf = SVC()
				clf.fit(cnn_features, Y)  # Y_one_hot hasnt suitable shape
				print("svm training finished")
				# joblib.dump(clf, folder + "model/" + str(
				#    XPParams.numberModuloReduction) + "/svm_svc_model_" + corpus.langName + ".pkl")
				with open(folder + "model/" + str(
						XPParams.numberModuloReduction) + "/svm_svc_model_" + corpus.langName + ".pkl",
						  "wb") as clf_file:
					pickle.dump(clf, clf_file)
				print("svm saved")
				print(cnn.summary())
				logging.info('Training Time: ' + str(int((datetime.datetime.now() - time).seconds / 60.)))

				print("loaded files for cnn and svm prediction")
				return clf, vec, feat_selection, layer_model
			elif XPParams.useCNNOnly:
				cnn = load_model(
					folder + "model/" + str(XPParams.numberModuloReduction) + "/cnn_model_" + corpus.langName + ".h5",
					custom_objects={'fbeta_score': fbeta_score})
				print("loaded files for cnn prediction")
				return cnn, vec, feat_selection
			else:
				with open("model/" + str(
						XPParams.numberModuloReduction) + "/svm_model_" + corpus.langName + '.pkl') as clf_file:
					clf = pickle.load(clf_file)
				# clf = joblib.load("model/"+str(XPParams.numberModuloReduction)+"/svm_model_"+corpus.langName+'.pkl')
				print("loaded files for svm prediction")
				return clf, vec, feat_selection

	@staticmethod
	def parseCorpus(sents, cls):
		""" iterate on each sentence and parse them. Get label and feature
            back which will be returned.
        """
		labels, features = [], []
		for sent in sents:
			# Parse the sentence
			trainingInfo = cls.parseSentence(sent, cls)
			if trainingInfo is not None:
				labels.extend(trainingInfo[0])
				features.extend(trainingInfo[1])

		return labels, features

	@staticmethod
	def parseSentence(sent, cls):
		""" get integer label and dict feature from each sentence
            and return it
            :param sent: sentence object
            :param cls: EmbeddingOracle class
        """
		sent.initialTransition = EmbeddingTransition(isInitial=True, sent=sent)
		# new embeddingtransition which inherit everything from Transition -> full buffer, empty stack, ...
		transition = sent.initialTransition
		while not transition.isTerminal():
			# while buffer and stack != 0 get transition
			# similar to transition.configuration.isTerminalConf n parsers.py
			transition = cls.getNextTransition(transition, sent)
		labels, features = Extractor.extract(sent)
		return labels, features

	@staticmethod
	def getNextTransition(parent, sent):
		""" get next transition, test if merge to multiwordtoken is possible,
            then if can blackmerge is possible, then if vmwe is completed,
            otherwise return shift
            :param parent: transition before
            :param sent: sentence object (corpus)
        """
		# Check if parent and current can be merged to multiwordtoken
		newTransition = MergeAsMWT.check(parent)
		if newTransition is not None:
			return newTransition

		# Check if parent and current can be merged to mwe
		newTransition = BlackMerge.check(parent)
		if newTransition is not None:
			return newTransition

		# Check for VMWE complete
		newTransition = Reduce.check(parent)
		if newTransition is not None:
			return newTransition

		# Apply the default transition: SHIFT
		shift = Shift(sent=sent)
		shift.apply(parent, sent)
		return shift

	@staticmethod
	def save_labels_and_features(labels, features, folder, lang, x):
		""" save labels and features in tab-separated files.
        """
		# folder = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "Results/")
		folder_labels = folder + 'labels/' + str(XPParams.numberModuloReduction) + '/'
		folder_features = folder + 'features/' + str(XPParams.numberModuloReduction) + '/'
		if not os.path.exists(folder_labels):
			os.makedirs(folder_labels)
		if not os.path.exists(folder_features):
			os.makedirs(folder_features)
		if XPParams.useCrossValidation:
			path_labels = os.path.join(folder_labels, lang + str(x) + 'labels.txt')
			path_features = os.path.join(folder_features, lang + str(x) + 'features.txt')
		else:
			path_labels = os.path.join(folder_labels, lang + '.labels.txt')
			path_features = os.path.join(folder_features, lang + '.features.txt')

		# with open(path_features, "a+") as f:
		#    for i in features.toarray():
		#		print(len(i))
		#        f.write('\t'.join(str(x) for x in i)+'\n')
		# print(features.toarray())
		# numpy.savetxt(path_features, features.toarray(), delimiter="\t", fmt="%f")
		scipy.sparse.save_npz(path_features, features)

		with open(path_labels, "w+") as file_labels:
			file_labels.write('\n'.join(str(x) for x in labels))
		return 0

	@staticmethod
	def train_CNN(label_vectors, features):
		number_of_classes = XPParams.numberOfClasses
		number_of_samples_X = features.shape[0]  # number of samples
		number_of_features = features.shape[1]  # number of features
		# print("I must be 500 " , number_of_features, features.shape[2])
		# print("This must be a one hot vector " , label_vectors[0])
		if not XPParams.twoD_feats:
			features = features.toarray()
			features = features.reshape((number_of_samples_X, 1, number_of_features, 1))
		else:
			number_third_dim = features.shape[2]
			features = features.reshape((number_of_samples_X, 1, number_of_features, number_third_dim))

		cnn = Sequential()
		nb_filter = 128  # Number of convolution filters to use.
		nb_row = 8  # Number of rows in the convolution kernel.
		nb_col = 1  # Number of columns in the convolution kernel.
		# cnn.add(Convolution2D(nb_filter, nb_row, nb_col,
		#                      border_mode="same",
		#                      #border_mode="valid",
		#                      activation="relu",
		#                      input_shape=(1, number_of_features, 1)))
		if not XPParams.twoD_feats:
			cnn.add(Conv2D(nb_filter, (nb_row, nb_col),
						   padding="same",
						   # border_mode="valid",
						   activation="relu",
						   input_shape=(1, number_of_features, 1)))
		else:
			cnn.add(Conv2D(nb_filter, (nb_row, nb_col),
						   padding="same",
						   # border_mode="valid",
						   activation="relu",
						   input_shape=(1, number_of_features, number_third_dim)))
		# cnn.add(Conv2D(nb_filter, (3, nb_col),
		#               padding="same",
		#               # border_mode="valid",
		#               activation="relu",
		#               input_shape=(1, number_of_features, 1)))
		# cnn.add(Conv2D(nb_filter, (8, 1, padding="same", activation="relu"))
		cnn.add(MaxPooling2D(pool_size=(1, 2)))  # pick the clearest/most distinct element of the vector

		# cnn.add(Convolution2D(128, 3, 1, border_mode="same", activation="relu"))
		# cnn.add(Convolution2D(128, 3, 1, border_mode="same", activation="relu"))
		# cnn.add(Conv2D(nb_filter, (nb_row, nb_col), padding="same", activation="relu"))
		# cnn.add(MaxPooling2D(pool_size=(1, 2)))

		cnn.add(Flatten())
		cnn.add(Dense(512, activation="relu", name="hidden_layer"))
		# cnn.add(MaxPooling2D(pool_size=(1,2)))
		cnn.add(Dropout(XPParams.dropout))  # delete neurons to avoid overfitting 0.075 # 0.025
		# cnn.add(Dropout(0.2))
		cnn.add(Dense(number_of_classes, activation="softmax"))

		# cnn.compile(loss="categorical_crossentropy", optimizer="adam",  metrics=['accuracy'])
		cnn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=[fbeta_score])
		# plot_model(cnn, to_file='model_plot_no_names_no_shape.png', show_shapes=False, show_layer_names=False)
		# model_to_dot(cnn).create(prog='dot', format='svg')
		# tbCallBack = keras.callbacks.TensorBoard(log_dir='Graph', histogram_freq=0, write_graph=True,
		#                                         write_images=True)
		print('Train...')
		# cnn.fit(features, label_vectors, epochs=XPParams.num_epochs, callbacks=[tbCallBack])
		cnn.fit(features, label_vectors, epochs=XPParams.num_epochs)

		return cnn

	@staticmethod
	def train_CNN_multichannel(label_vectors, features):
		number_of_classes = XPParams.numberOfClasses
		number_of_samples_X = features.shape[0]  # number of samples
		number_of_features = features.shape[1]  # number of features
		# print("I must be 500 " , number_of_features, features.shape[2])
		# print("This must be a one hot vector " , label_vectors[0])
		if not XPParams.twoD_feats:
			features = features.toarray()
			# features = features.reshape((number_of_samples_X, 1, number_of_features, 1))
		# else:
		# number_third_dim = features.shape[2]
		# features = features.reshape((number_of_samples_X, 1, number_of_features, number_third_dim))

		dropout_rate = XPParams.dropout
		# compile_metric = "fbeta_score"
		embedding_dim = XPParams.dim_embedd
		kernel_sizes = XPParams.kernelsize
		num_filters = XPParams.filters  # 10
		hidden_dims = XPParams.dim_hidden
		input_shape = (number_of_features,)
		model_input = Input(shape=input_shape)
		print(embedding_dim, input_shape, num_filters, hidden_dims, input_shape, dropout_rate, kernel_sizes)
		z = Embedding(number_of_samples_X, embedding_dim, input_length=input_shape, name="embedding")(model_input)
		# z = Dropout(0.025)(z)

		conv_blocks = []
		for k_size in kernel_sizes:
			# old keras layer, change to Conv1D
			conv = Convolution1D(filters=num_filters,
								 kernel_size=int(k_size),
								 padding="valid",
								 activation="relu",
								 strides=1)(z)
			if XPParams.pooling == "avg_pool":
				conv = AveragePooling1D(pool_size=2)(conv)
			else:
				conv = MaxPooling1D(pool_size=2)(conv)
			conv = Flatten()(conv)
			conv_blocks.append(conv)
		z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

		z = Dropout(dropout_rate)(z)
		z = Dense(hidden_dims, activation="relu", name="hidden_layer")(z)
		model_output = Dense(number_of_classes, activation="softmax")(z)

		model = Model(model_input, model_output)
		model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=[fbeta_score])

		# Train the model
		model.fit(features, label_vectors, batch_size=XPParams.batchSize, epochs=XPParams.num_epochs, verbose=1)

		return model

	@staticmethod
	def train_CNN_batch1(labels, features):
		max_features = features.shape[0]  # number of samples
		maxlen = features.shape[1]  # number of features
		# features = features.toarray()
		batch_size = XPParams.batchSize
		# batch size = the number of training examples in one forward/backward pass. The higher the batch size, the more memory space you'll need.
		embedding_dims = 100  # 300
		filters = 128  # 250
		kernel_size = 5
		epochs = 10
		# one epoch = one forward pass and one backward pass of all the training examples
		print(maxlen, " X ", max_features)
		print('Pad sequences (samples x time)')
		# x_train = sequence.pad_sequences(features, maxlen=maxlen)
		# Pads sequences to the same length
		print('x_train shape:', features.shape)

		print('Build model...')
		model = Sequential()

		# we start off with an efficient embedding layer which maps
		# our vocab indices into embedding_dims dimensions
		model.add(Embedding(max_features, embedding_dims, input_length=maxlen, name="input_layer"))

		# we add a Convolution1D, which will learn filters
		# word group filters of size filter_length:
		model.add(Conv1D(filters,
						 kernel_size,
						 padding='valid',
						 activation='relu',
						 strides=1,
						 name="conv_layer"))
		# we use max pooling:
		model.add(MaxPooling1D(name="pooling_layer"))

		model.add(Flatten())
		model.add(Dense(filters, name="hidden_layer_1", activation="relu"))
		model.add(Dropout(0.075))

		# We add a vanilla hidden layer:
		model.add(Dense(XPParams.numberOfClasses, name="hidden_layer"))

		model.compile(loss='sparse_categorical_crossentropy',
					  optimizer='adam',
					  metrics=[fbeta_score])
		# model.fit(x_train, labels,
		#          batch_size=batch_size,
		#          epochs=epochs)
		model.fit_generator(generator=batch_generator(features, labels, batch_size), nb_epoch=epochs,
							steps_per_epoch=features.shape[0])
		# model.fit_generator(EmbeddingOracle.samples(features, labels, batch_size), features.shape[0], epochs, verbose=1)
		# model.fit_generator(generator=sparse_generator(features, labels, batch_size, True),
		#                    samples_per_epoch = features.shape[0], steps_per_epoch=features.shape[0]/batch_size, nb_epoch = epochs, verbose=1,)
		return model

	@staticmethod
	def train_CNN_batch(labels, features):
		# RO num featues 2192525
		max_features = features.shape[0]  # number of samples
		maxlen = features.shape[1]  # number of features
		# features = features.toarray()
		batch_size = 32
		# batch size = the number of training examples in one forward/backward pass. The higher the batch size, the more memory space you'll need.
		epochs = XPParams.num_epochs
		# one epoch = one forward pass and one backward pass of all the training examples

		number_of_classes = XPParams.numberOfClasses
		number_of_samples_X = features.shape[0]  # number of samples
		number_of_features = features.shape[1]  # number of features

		print(maxlen, " X ", max_features)
		print('Pad sequences (samples x time)')
		# x_train = sequence.pad_sequences(features, maxlen=maxlen)
		# Pads sequences to the same length
		print('x_train shape:', features.shape)

		print('Build model...')
		cnn = Sequential()
		nb_filter = 8  # Number of convolution filters to use.
		nb_row = 8  # Number of rows in the convolution kernel.
		nb_col = 1  # Number of columns in the convolution kernel.
		cnn.add(Conv2D(nb_filter, (nb_row, nb_col),
					   padding="same",
					   # border_mode="valid",
					   activation="relu",
					   input_shape=(1, number_of_features, 1)))
		# cnn.add(Conv2D(128, 3, 1, border_mode="same", activation="relu"))
		cnn.add(MaxPooling2D(pool_size=(1, 2)))  # pick the clearest/most distinct element of the vector

		# cnn.add(Convolution2D(128, 3, 1, border_mode="same", activation="relu"))
		# cnn.add(Convolution2D(128, 3, 1, border_mode="same", activation="relu"))
		# cnn.add(Conv2D(nb_filter, (nb_row, nb_col), padding="same", activation="relu"))
		# cnn.add(MaxPooling2D(pool_size=(1, 2)))

		cnn.add(Flatten())
		cnn.add(Dense(nb_filter, activation="relu", name="hidden_layer"))
		# cnn.add(MaxPooling2D(pool_size=(1,2)))
		cnn.add(Dropout(0.2))  # delete neurons to avoid overfitting 0.075 # 0.025
		# cnn.add(Dense(nb_filter, activation="relu", name="hidden_layer_2"))
		# cnn.add(Dropout(0.2))
		# cnn.add(Dense(nb_filter, activation="relu", name="hidden_layer_3"))
		# cnn.add(Dropout(0.2))
		cnn.add(Dense(number_of_classes, activation="softmax"))

		# cnn.compile(loss="categorical_crossentropy", optimizer="adam",  metrics=['accuracy'])
		cnn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=[fbeta_score])

		# model.fit(x_train, labels,
		#          batch_size=batch_size,
		#          epochs=epochs)
		# data_genereator(batch_size, features, labels)
		# cnn.fit_generator(generator=data_genereator(batch_size, features, labels), nb_epoch=epochs,
		#                    samples_per_epoch=features.shape[0])
		# model.fit_generator(EmbeddingOracle.samples(features, labels, batch_size), features.shape[0], epochs, verbose=1)
		"""counter = 0
        while batch_size * counter <= features.shape[0]:
            features_batch = features[(batch_size * counter):min(batch_size * (counter + 1), features.shape[0]), :]
            labels_batch = labels[(batch_size * counter):min(batch_size * (counter + 1), labels.shape[0]), :]
            counter += 1
            features_batch = features_batch.toarray()
            features_batch = features_batch.reshape((features_batch.shape[0], 1, features_batch.shape[1], 1))
            cnn.train_on_batch(features_batch, labels_batch)
            print(counter*batch_size)"""

		cnn.fit_generator(generator=batch_generator(features, labels, batch_size), epochs=1,
						  steps_per_epoch=features.shape[0] / batch_size, use_multiprocessing=True, shuffle=False,
						  max_queue_size=10)
		# [14.497.408,128]

		# cnn.fit_generator(generator=generator(features, lookback=128, delay=144, min_index=0, max_index=200000, shuffle=False, step=6),
		#                    samples_per_epoch = features.shape[0], steps_per_epoch=features.shape[0]/batch_size, nb_epoch = epochs, verbose=1,)
		return cnn

	@staticmethod
	def samples(x_source, y_source, size):
		""" convert sparse matrix to array
        https://github.com/keras-team/keras/issues/4984#issuecomment-304965374"""
		while True:
			for i in range(0, x_source.shape[0], size):
				j = i + size

				if j > x_source.shape[0]:
					j = x_source.shape[0]

				yield x_source[i:j].toarray(), y_source[i:j]

	@staticmethod
	def save_features(X, output_path):
		# convert sparse matrix X line-by-line to array and save to file
		"""with open(output_path, "w") as f:
            num_samples = X.shape[0]
            num_features = X.shape[1]
            for index, vector in enumerate(X):
                #new_vector = ["0"]*num_features
                new_vector = numpy.zeros(num_features)
                for i, value in zip(vector.indices, vector.data):
                    new_vector[i] = str(value)
                #f.write("\t".join(new_vector) + "\n")
                f.write("\t".join(str(x) for x in new_vector))"""
		with open(output_path, "w") as f:
			num_samples = X.shape[0]
			for row_index in range(num_samples):
				numpy.savetxt(f, X.getrow(row_index).toarray(), delimiter="\t", fmt="%i")
		return 0

	@staticmethod
	def convert_matrix_to_array(X, file_path):
		num_samples = X.shape[0]
		tmp = list()
		for row_index in range(num_samples):
			tmp.append(X.getrow(row_index).toarray())
			if row_index == 0:
				output = tmp
			if not row_index % 100000:
				print(row_index, "of", num_samples)
				output.extend(tmp)
		numpy.save(file_path, output)
		print(len(output), num_samples)
		return output

	"""def convert_matrix_to_array(X, file_path):
        num_samples = X.shape[0]
        num_features = X.shape[1]
        output = numpy.array([[]])
        for index, vector in enumerate(X):
            # new_vector = ["0"]*num_features
            new_vector = numpy.zeros(num_features)
            for i, value in zip(vector.indices, vector.data):
                new_vector[i] = str(value)
            output = numpy.concatenate((output, new_vector), axis=0)
            if not index%10000:
                print(index, "of", num_samples)
        return output"""

	@staticmethod
	def get_path(folder, lang, x, ending):
		"""get path and filename for saving files"""
		if not os.path.exists(folder) and not os.path.exists(folder):
			os.makedirs(folder)
		if XPParams.useCrossValidation:
			path = os.path.join(folder, lang + '_' + str(x) + ending)
		else:
			path = os.path.join(folder, lang + '_' + ending)
			# print(path)
		return path


def precision(y_true, y_pred):
	'''Calculates the precision, a metric for multi-label classification of
    how many selected items are relevant.
    '''
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return precision


def recall(y_true, y_pred):
	'''Calculates the recall, a metric for multi-label classification of
    how many relevant items are selected.
    '''
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	recall = true_positives / (possible_positives + K.epsilon())
	return recall


def fbeta_score(y_true, y_pred, beta=1):
	'''Calculates the F score, the weighted harmonic mean of precision and recall.

    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.

    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.

    Source: https://raw.githubusercontent.com/GeekLiB/keras/master/keras/metrics.py
    https://faroit.github.io/keras-docs/1.2.2/metrics/#fbeta_score

     # According to @fchollet, he explained in #5794 that it was intentionally removed in version 2.0 because
     it performs only approximation by batchwise evaluation.
    '''
	if beta < 0:
		raise ValueError('The lowest choosable beta is zero (only precision).')

	# If there are no true positives, fix the F score at 0 like sklearn.
	if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
		return 0

	p = precision(y_true, y_pred)
	r = recall(y_true, y_pred)
	bb = beta ** 2
	fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
	return fbeta_score


#
# Generator (https://github.com/pplonski/keras-sparse-check)
#
class threadsafe_iter:
	"""Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    https://keunwoochoi.wordpress.com/2017/08/24/tip-fit_generator-in-keras-how-to-parallelise-correctly/
    """

	def __init__(self, it):
		self.it = it
		self.lock = threading.Lock()

	def __iter__(self):
		return self

	def next(self):
		with self.lock:
			return self.it.next()


def threadsafe_generator(f):
	"""A decorator that takes a generator function and makes it thread-safe.
    """

	def g(*a, **kw):
		return threadsafe_iter(f(*a, **kw))

	return g


@threadsafe_generator
def batch_generator(features, labels, batchsize):
	counter = 0
	while True:
		features_batch = features[(batchsize * counter):min(batchsize * (counter + 1), features.shape[0]), :]
		labels_batch = labels[(batchsize * counter):min(batchsize * (counter + 1), labels.shape[0]), :]
		counter += 1
		features_batch = features_batch.toarray()
		features_batch = features_batch.reshape((features_batch.shape[0], 1, features_batch.shape[1], 1))
		# print(counter, features_batch.shape)
		yield features_batch, labels_batch
		if batchsize * counter >= features.shape[0]:
			counter = 0
