import joblib
from arrowflow.arrowflow import SortFlowHybridNetwork, DataGraph, SortFlow_MoE
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime

import numpy as np
import os
import time
import warnings

warnings.filterwarnings("ignore")
import multiprocessing
from time import sleep
import pandas as pd
from arrowflow.config import sortnet_config
from arrowflow.classical_classifiers import run_classical_classifiers
from arrowflow.mlp_classifier import run_mlp_classifier

if sortnet_config.profile_flag:
    from pyinstrument import Profiler
    profiler = Profiler()
    profiler.start()


def initialize_sortnet_and_train(sim_iter, input_sortnet, number_of_classes, return_dict):
    seed_no = (os.getpid() * int(time.time()*1000)) % 123456
    # print("Seed no: ", seed_no)
    np.random.seed(seed_no)
    adj_list_input, file_name, data, sortnet_config = input_sortnet

    if sortnet_config.moe_no_of_networks > 1:
        sortflow_moe = SortFlow_MoE('sf', adj_list_input, number_of_classes, file_name, sortnet_config)
        errors_train = sortflow_moe.train(data, sortnet_config)
    else:
        sortflow_net = SortFlowHybridNetwork('sf', adj_list_input, number_of_classes, file_name, sortnet_config)
        errors_train = sortflow_net.train(data, sortnet_config)

    return_dict[sim_iter] = errors_train

    return errors_train


def run():
    data_graph = DataGraph('cn_data')

    if sortnet_config.data_type == 'artificial':
        data_train_net, data_test_net, adj_list_input, number_of_classes, max_input_size = \
            data_graph.generate_sequence_data(distort_prob=sortnet_config.distort_prob,
                                              dim_expand_ratio=sortnet_config.dim_expand_ratio)
        # preprocess dataset, split into training and test part
        X_train, y_train = data_graph.transform_2_numpy_dataset(data_train_net, max_input_size)
        X_test, y_test = data_graph.transform_2_numpy_dataset(data_test_net, max_input_size)
        input_dim = len(adj_list_input)
        vocabulary_size = len(adj_list_input)
    elif sortnet_config.data_type == 'sorting':
        size_array = sortnet_config.sorting_dataset_array_size
        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'docs',
                                                 'sorting_dataset_' + str(int(size_array)) + '.pkl'))
        dataset_sorting = joblib.load(file_path)
        data_train_net = dataset_sorting['data_train']
        data_test_net = dataset_sorting['data_test']
        adj_list_input = dataset_sorting['adj_list_input']
        number_of_classes = dataset_sorting['number_of_classes']
        max_input_size = len(adj_list_input)

        X_train, y_train = data_graph.transform_2_numpy_sorting_dataset(data_train_net, max_input_size)
        X_test, y_test = data_graph.transform_2_numpy_sorting_dataset(data_test_net, max_input_size)
    elif sortnet_config.data_type == 'sliding8':
        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'docs', 'sliding8.pkl'))
        dataset_sliding_8 = joblib.load(file_path)
        data_train_net = dataset_sliding_8['data_train']
        train_data_size = sortnet_config.sliding8_no_of_data
        data_train_net = data_train_net[:train_data_size]
        data_test_net = dataset_sliding_8['data_test']
        test_data_size = int(sortnet_config.sliding8_no_of_data / 10)
        data_test_net = data_test_net[:test_data_size]
        adj_list_input = dataset_sliding_8['adj_list_input']
        number_of_classes = dataset_sliding_8['number_of_classes']
        max_input_size = len(adj_list_input)

        X_train, y_train = data_graph.transform_2_numpy_dataset(data_train_net, max_input_size)
        X_test, y_test = data_graph.transform_2_numpy_dataset(data_test_net, max_input_size)
        input_dim = len(adj_list_input)
        vocabulary_size = len(adj_list_input)
        # counts, bins = np.histogram(y_test)
        # plt.stairs(counts, bins)
        # plt.savefig(path_files+'sliding_distance.png', dpi=100)
    else:
        if sortnet_config.data_type == 'imdb':
            vocabulary_size = sortnet_config.vocabulary_size
            vector_size = sortnet_config.vector_size
            number_of_data_selected = sortnet_config.number_of_data_selected

            # This is imdb vectors that are bag of words
            df = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'docs', 'IMDB_Dataset.csv')))
            # Add a new column 'Category' for binary classification
            df["Category"] = df.sentiment.apply(lambda x: 1 if x == "positive" else 0)

            # Split the dataset into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(df["review"], df['Category'], test_size=0.2,
                                                                random_state=42)

            cv = CountVectorizer(max_features=vocabulary_size)
            cv_fit = cv.fit_transform(X_train)
            X_train = cv_fit.toarray()
            X_test = cv.transform(X_test).toarray()
            word_list = cv.get_feature_names_out()
            code_list = np.arange(0, len(word_list))
            count_list = X_train.sum(axis=0) / X_train.sum()
            frequency_dict_names = dict(zip(word_list, count_list))
            frequency_dict_codes = dict(zip(code_list, count_list))
            sortnet_config.frequency_dict = frequency_dict_codes

            matrix = []
            for vector in X_train:
                vector = np.nonzero(vector)[0]
                # vector = np.argsort(-vector, axis=0)
                vec_dense = np.pad(vector[:vector_size],
                                   (0, vector_size - len(vector[:vector_size])),'constant')
                matrix.append(vec_dense)
            X_train = np.asarray(matrix)

            matrix = []
            for vector in X_test:
                vector = np.nonzero(vector)[0]
                # vector = np.argsort(-vector, axis=0)
                vec_dense = np.pad(vector[:vector_size],
                                   (0, vector_size - len(vector[:vector_size])), 'constant')
                matrix.append(vec_dense)
            X_test = np.array(matrix)

            y_train = y_train.to_numpy()
            y_test = y_test.to_numpy()

            X_train = X_train[:number_of_data_selected]
            y_train = y_train[:number_of_data_selected]
            X_test = X_test[:int(number_of_data_selected/4)]
            y_test = y_test[:int(number_of_data_selected/4)]

            if sortnet_config.evaluation_algorithms_flag[3]:
                X_all = np.concatenate((X_train, X_test))
                y_all = np.concatenate((y_train, y_test))
                data_train__, adj_list_input = data_graph.convert_sequence_data_sortnet(X_all, y_all)
                data_train_net, adj_list_input_ = data_graph.convert_sequence_data_sortnet(X_train, y_train)
                data_test_net, adj_list_input__ = data_graph.convert_sequence_data_sortnet(X_test, y_test)
            else:
                data_train_net, data_test_net, adj_list_input = None, None, None

            input_dim = X_train.max() + 1
            number_of_classes = 2
        else:
            if sortnet_config.data_type == 'moons':
                ds = make_moons(n_samples=1000, noise=0.3, random_state=0)
                input_dim = 2
                number_of_classes = 2
            elif sortnet_config.data_type == 'circles':
                ds = make_circles(n_samples=1000, noise=0.2, factor=0.5, random_state=1)
                input_dim = 2
                number_of_classes = 2
            elif sortnet_config.data_type == 'linear':
                X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2,
                                           random_state=1, n_clusters_per_class=1)
                rng = np.random.RandomState(2)
                X += 2 * rng.uniform(size=X.shape)
                ds = (X, y)
                input_dim = 2
                number_of_classes = 2

            X, y = ds
            max_input_size = input_dim
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=.3, random_state=42)

            data_train_net, adj_list_input, W_random = \
                data_graph.project_random_space(X_train,y_train,None, sortnet_config.poly_expansion,
                                                sortnet_config.expand_diff, sortnet_config.pol_deg,
                                                sortnet_config.no_of_embedding_dim)
            data_test_net, adj_list_input_, W_random = \
                data_graph.project_random_space(X_test,y_test,W_random, sortnet_config.poly_expansion,
                                                sortnet_config.expand_diff, sortnet_config.pol_deg,
                                                sortnet_config.no_of_embedding_dim)

    sortnet_config.no_of_filters.append(number_of_classes)
    file_name = '{}_{}_{}_{}'.format(sortnet_config.data_type, sortnet_config.train_type,
                                     str(sortnet_config.no_of_filters[0]), str(sortnet_config.no_of_filters[1]))

    sortnet_config.data_initial = data_train_net
    data = [data_train_net, data_test_net]
    input_sortnet = [adj_list_input, file_name, data, sortnet_config]

    # LSTM training
    if sortnet_config.evaluation_algorithms_flag[0]:
        pass

    # Dense NN training
    if sortnet_config.evaluation_algorithms_flag[1]:
        run_mlp_classifier(X_train, y_train, X_test, y_test, sortnet_config.vocabulary_size)

    # Classifier training
    if sortnet_config.evaluation_algorithms_flag[2]:
        run_classical_classifiers(X_train, y_train, X_test, y_test)

    # SortNet training
    if sortnet_config.evaluation_algorithms_flag[3]:

        best_validation_error = []
        test_error = []

        if sortnet_config.multiprocessing and sortnet_config.no_of_sims > 1:
            manager = multiprocessing.Manager()

            tsim_init = datetime.now()
            jobs = []

            return_dict = manager.dict()
            for sim_iter in range(sortnet_config.no_of_sims):
                sleep(0.5*np.random.random())
                p = multiprocessing.Process(target=initialize_sortnet_and_train, args=(sim_iter, input_sortnet,
                                                                                       number_of_classes, return_dict))
                jobs.append(p)
                p.start()

            for proc in jobs:
                proc.join()

            result_list = return_dict.values()

            for iter_no, iter_result in enumerate(result_list):
                best_validation_error.append(iter_result[2])
                test_error.append(iter_result[3])
                print("Simulation iteration {}. Best val error: {:.1f}  Test Error: {:.1f}".format(iter_no,100 *
                                                                                                   iter_result[2],
                                                                                                   100 * iter_result[3]))
            print("Simulation per iter in {} sec".format((datetime.now() - tsim_init) / sortnet_config.no_of_sims))

        else:
            return_dict = {}
            for sim_iter in range(sortnet_config.no_of_sims):
                tsim_init = datetime.now()

                errors_train = initialize_sortnet_and_train(sim_iter, input_sortnet, number_of_classes, return_dict)

                best_validation_error.append(errors_train[2])
                test_error.append(errors_train[3])
                print("Simulation iteration {} in {} sec. Best val error: {:.1f}  Test Error: {:.1f}".format(sim_iter,
                                                                                         datetime.now() - tsim_init,
                                                                                         100*errors_train[2],100*errors_train[3]))
        bve = np.asarray(best_validation_error)
        te = np.asarray(test_error)
        if sortnet_config.problem == 'classification':
            bve = 100 * bve
            te = 100 * te
        print("Simulation stats Best Validation Error:   error mean:{:.1f},  error min: {:.1f},  error max: {:.1f},  error st.err: {:.1f}".
              format(np.mean(bve), np.min(bve), np.max(bve), np.std(bve)/np.sqrt(sortnet_config.no_of_sims)))
        print("Simulation stats Test Error:   error mean:{:.1f},  error min: {:.1f},  error max: {:.1f},  error st.err: {:.1f}".
              format(np.mean(te), np.min(te), np.max(te), np.std(te)/np.sqrt(sortnet_config.no_of_sims)))

        if sortnet_config.profile_flag:
            profiler.stop()
            profiler.print()

        print('Test Done!')


if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn', force=True)
    run()
