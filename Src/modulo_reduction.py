from __future__ import division
from scipy.sparse import csr_matrix, save_npz
import numpy
import math
from collections import Counter
from param import XPParams


class ModuloReduction():
    @staticmethod
    def get_expected_cooc(total, count_feature, count_vector):
        """ return expected value
        """
        if total == 0 or count_feature == 0 or count_vector == 0:
            return 0
        return (count_vector*count_feature)/(total*total)

    @staticmethod
    def ppmi(count_cooc, expected_cooc):
        """ return pointwise mutual information. 
            If count of observed or expected co-ooccurrences is 0 return 0.
        """
        if count_cooc == 0 or expected_cooc == 0:
            ppmi = 0
        else:
            ppmi = max(0, math.log(count_cooc/expected_cooc))
        return ppmi

    @staticmethod
    def reduce_matrix_with_modulo(X, n_new_features, folder=None, training=False, calc_ppmi=False, new_y_number=None, twoD_feats=False):
        data = []  # contains all values
        rows = []  # contains all rownumbers of the data
        cols = []  # contains all colnumbers of the data
        #count_feature_dict = Counter()
        #count_vector_dict = Counter()
        count_feature_vec = numpy.zeros(n_new_features)
        #count_vector_vec = []
        if twoD_feats and new_y_number != None:
            #output_matrix = numpy.zeros((X.shape[0], 1))
            output_matrix = numpy.empty((X.shape[0], n_new_features, new_y_number))
            for row_nr, vector in enumerate(X):
                new_data = numpy.zeros((n_new_features, new_y_number))
                for col_nr, value in zip(vector.indices, vector.data):
                    if value != 0:
                        new_dimension_1 = abs(col_nr % n_new_features)
                        new_dimension_2 = abs((col_nr * 7) % n_new_features)
                        new_dimension_3 = abs((col_nr * 17) % n_new_features)
                        new_dimension_4 = abs((col_nr * 37) % n_new_features)
                        new_dimension_5 = abs((col_nr * 47) % n_new_features)

                        new_y = abs(col_nr % new_y_number)
                        new_data[new_dimension_1, new_y] += value
                        new_data[new_dimension_2, new_y] += value
                        new_data[new_dimension_3, new_y] += value
                        new_data[new_dimension_4, new_y] += value
                        new_data[new_dimension_5, new_y] += value
                        #print(len(new_data[numpy.nonzero(new_data)]))
                        if n_new_features > 500:
                            new_dimension_6 = abs((col_nr * 67) % n_new_features)
                            new_dimension_7 = abs((col_nr * 97) % n_new_features)
                            new_dimension_8 = abs((col_nr * 107) % n_new_features)
                            new_dimension_9 = abs((col_nr * 127) % n_new_features)
                            new_dimension_10 = abs((col_nr * 137) % n_new_features)
                            new_data[new_dimension_6, new_y] += value
                            new_data[new_dimension_7, new_y] += value
                            new_data[new_dimension_8, new_y] += value
                            new_data[new_dimension_9, new_y] += value
                            new_data[new_dimension_10, new_y] += value

                output_matrix[row_nr] = new_data
            #print(output_matrix)
            #output_matrix = csr_matrix(output_matrix)

        else:
            for row_nr, vector in enumerate(X):
                new_data = numpy.zeros(n_new_features)
                for col_nr, value in zip(vector.indices, vector.data):
                    #print(value)
                    if value != 0:
                        new_dimension_1 = abs(col_nr % n_new_features)
                        new_dimension_2 = abs((col_nr*7) % n_new_features)
                        new_dimension_3 = abs((col_nr * 17) % n_new_features)
                        new_dimension_4 = abs((col_nr * 37) % n_new_features)
                        new_dimension_5 = abs((col_nr * 47) % n_new_features)

                        new_data[new_dimension_1] += value
                        new_data[new_dimension_2] += value
                        new_data[new_dimension_3] += value
                        new_data[new_dimension_4] += value
                        new_data[new_dimension_5] += value

                        if n_new_features > 500:
                            new_dimension_6 = abs(col_nr*67 % n_new_features)
                            new_dimension_7 = abs((col_nr*97) % n_new_features)
                            new_dimension_8 = abs((col_nr * 107) % n_new_features)
                            new_dimension_9 = abs((col_nr * 127) % n_new_features)
                            new_dimension_10 = abs((col_nr * 137) % n_new_features)

                            new_data[new_dimension_6] += value
                            new_data[new_dimension_7] += value
                            new_data[new_dimension_8] += value
                            new_data[new_dimension_9] += value
                            new_data[new_dimension_10] += value


                for nr_value, value in enumerate(new_data):
                    #print(value)
                    # add values to new_matrix if not zero
                    if value != 0:
                        data.append(value)  # add new value if not 0
                        rows.append(row_nr)  # add row number of current vector
                        cols.append(nr_value)  # add index of column/feature
                    count_feature_vec[nr_value] += value # number 2
                #count_vector_vec[row_nr] = sum(count_feature_vec)
            output_matrix = csr_matrix((data, (rows, cols)), shape=(X.shape[0], n_new_features))
            #print("mod", output_matrix.toarray())

        if calc_ppmi:
            #ppmi
            if training:
                dim =n_new_features
                output_matrix = output_matrix.toarray()
                vectorOfSumAllVectors = numpy.zeros(dim)
                sumAllComponents = 0
                vecWeighted = numpy.zeros(shape=(output_matrix.shape[0], output_matrix.shape[1]))
                for vector in output_matrix:
                    for i in range(dim):
                        vectorOfSumAllVectors[i] += vector[i]

                for i in range(dim):
                    sumAllComponents += vectorOfSumAllVectors[i]
                #print("sumAllComponents", sumAllComponents)
                for n, vector in enumerate(output_matrix):
                    #if training == False:
                        #print(vector)
                    sumThisRow = sum(vector)
                    for i in range(dim):
                        pmi = 0
                        if vector[i] != 0:
                            pmi = max(0,math.log(vector[i]*sumAllComponents)-math.log(sumThisRow*vectorOfSumAllVectors[i]))
                        vecWeighted[n,i] = pmi
                    #if not training:
                        #print("vectorOfSumAllVectors", vectorOfSumAllVectors[i])
                        #print("sumThisRow", sumThisRow, "vector", vector[i])
                #print("ppmi", vecWeighted)
                XPParams.vectorOfSumAllVectors = vectorOfSumAllVectors
                XPParams.sumAllComponents = sumAllComponents
            else:
                dim = n_new_features
                output_matrix = output_matrix.toarray()
                vectorOfSumAllVectors = XPParams.vectorOfSumAllVectors
                sumAllComponents = XPParams.sumAllComponents
                vecWeighted = numpy.zeros(shape=(output_matrix.shape[0], output_matrix.shape[1]))

                for n, vector in enumerate(output_matrix):
                    # if training == False:
                    # print(vector)
                    sumThisRow = sum(vector)
                    for i in range(dim):
                        pmi = 0
                        if vector[i] != 0:
                            pmi = max(0, math.log(vector[i] * sumAllComponents) - math.log(
                                sumThisRow * vectorOfSumAllVectors[i]))
                            #print(pmi)
                        vecWeighted[n, i] = pmi
                    # if not training:
                    # print("vectorOfSumAllVectors", vectorOfSumAllVectors[i])
                    # print("sumThisRow", sumThisRow, "vector", vector[i])
                #print("ppmi", vecWeighted)

            return csr_matrix(vecWeighted)
        else:
            return output_matrix
            
            
            
        #     data_list_pmi = []
        #     rows_list_pmi = []
        #     cols_list_pmi = []
        #     """for row_nr, vector in enumerate(output_matrix):
        #         count_vector = sum(vector.data)
        #         for col_nr, value in zip(vector.indices, vector.data):
        #             expected_cooc = get_expected_cooc(sum(output_matrix.data), count_feature_dict[col_nr], count_vector_dict[row_nr])
        #             ppmi_value = ppmi(value, expected_cooc)
        #             data_list_pmi.append(ppmi_value)  # add new value if not 0
        #             rows_list_pmi.append(row_nr)  # add row number of current vector
        #             cols_list_pmi.append(col_nr)  # add index of column/feature"""
        #
        #     total = sum(output_matrix.data) # number 1
        #     output_matrix = output_matrix.toarray()
        #     for row_nr, vector in enumerate(output_matrix):
        #         count_vector = sum(vector)
        #
        #         for col_nr, value in enumerate(vector):
        #             pmi=0
        #             if value != 0:
        #                 pmi = max(0,math.log(value * total)-math.log(count_vector * count_feature_vec[col_nr]))
        #
        #         #    print pmi
        #
        #             #expected_cooc = get_expected_cooc(sum(output_matrix.data), count_feature_dict[col_nr], count_vector_dict[row_nr])
        #             #ppmi_value = ppmi(value, expected_cooc)
        #             data_list_pmi.append(pmi)  # add new value if not 0
        #             rows_list_pmi.append(row_nr)  # add row number of current vector
        #             cols_list_pmi.append(col_nr)  # add index of column/feature"""
        #         #print('vector', vector, row_nr, count_vector)
        #
        #     output_matrix = csr_matrix((data_list_pmi, (rows_list_pmi, cols_list_pmi)), shape=(X.shape[0], n_new_features))
        # return output_matrix