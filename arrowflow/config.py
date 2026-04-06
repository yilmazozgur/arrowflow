from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor


class SortNetConfig:
    data_type = 'sorting'    # 'artificial' or 'moons' or 'circles' or 'imdb', or 'sliding8', 'sorting'
    problem = 'sorting'  # 'regression' or 'classification', or 'sorting'

    s_neigh = None
    t_neigh = None
    atomic_f = None

    # LSTM and Dense NN variables
    no_of_neurons_list = [64]  #[32, 64, 128, 256]
    no_of_epochs = 30
    batch_size_mlp = 32
    no_of_iters_mlp = 32

    # IMDB dataset variables
    vocabulary_size = 1000
    vector_size = 64
    number_of_data_selected = 1000

    # Sliding8 dataset variables
    sliding8_no_of_data = 10000

    sorting_dataset_array_size = 8

    no_of_embedding_dim = 16  # 16
    poly_expansion = False
    expand_diff = False

    distort_prob = 0  # 0, 0.2
    dim_expand_ratio = 1  # 1, 2

    # Classifier variables
    names_classifiers = [#"RankSVM",
             "Nearest Neighbors-1",
             "Nearest Neighbors-2",
             "Nearest Neighbors-3",
             "Nearest Neighbors-4",
             "Nearest Neighbors-5",
             "Nearest Neighbors-6",
             "Linear SVM",
             "RBF SVM",
             "Gaussian Process",
             "Decision Tree",
             "Random Forest",]
    names_regressors = ["Nearest Neighbors-3",
                        "Linear SVR",
                        "RBF SVR",
                        "Decision Tree",
                        "Random Forest",]
    classifiers = [
        #RankSVM(),
        KNeighborsClassifier(1),
        KNeighborsClassifier(2),
        KNeighborsClassifier(3),
        KNeighborsClassifier(4),
        KNeighborsClassifier(5),
        KNeighborsClassifier(6),
        SVC(kernel="linear", C=0.025, random_state=42),
        SVC(gamma=2, C=1, random_state=42),
        GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
        DecisionTreeClassifier(max_depth=5, random_state=42),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=42)
    ]
    regressors = [
        KNeighborsRegressor(3),
        SVR(kernel='linear', C=0.025),
        SVR(gamma=2, C=1),
        DecisionTreeRegressor(max_depth=5, random_state=42),
        RandomForestRegressor(max_depth=5, n_estimators=10, max_features=1, random_state=42)
    ]
    linear_models = False
    optimize_rf_flag = True
    optimize_rbf_flag = False
    optimize_mlp_flag = False

    # Which algorithms will be tested in the experiment. LSTM, Dense, ML models, SortNet
    evaluation_algorithms_flag = [False, False, False, True]

    # SortNet variables
    network_type = "feedforward"  # or "feedforward" or "recurrent". Data structure changes with this selection. "hybrid" if there is tensor interface
    train_type = 'supervised'
    regression_method = 'quantize'  # quantize, distance_encode, shift_encode. Only for sort layer based regressin at the last layer.
    eval_period = 1
    val_data_ratio = 0.1
    no_of_filters = [256, 64, 256]
    layer_types = ['sort', 'sort', 'sort', 'sort']
    metric_flag = [None, None, None, None, None]
    embedding_layer_flag = False
    filter_rfs = [None, None, None, None, None]  # First is the rf for the input dimension
    last_layer_update = True  #False
    distance_computation_metric = 'l1' #l1
    dist_nonlinearity = 1.2  #0.8
    min_motion_zeroed = 0  #0
    apply_nonlinearity_distance = False  #False
    nonlinear_activation = False  #False
    average_method_motion = 'mean'  #mean
    average_method_position = 'mean' #mean
    change_probability_when_decision_correct = 0.01  # 0.01     0.2, 0.125
    tensor_gradient_threshold = 0.0001  # 0.0001
    learning_rate = 0.1  # 0.1
    lr_tensor = 0.01  # 0.01
    multistep_lr = True  #True
    motion_normalization_mult = 0.125  # 0.125
    ratio_data_backprop = 0.5  # 0.5
    missing_data_cost_weight = 0.5  #0.5
    missing_data_loc_mult = 0.5 #0.5
    no_of_iters = 200
    multiprocessing = True
    device = 'cpu'  # 'cpu' or 'cuda'
    no_of_sims = 21
    moe_no_of_networks = 1  # 7
    moe_data_ratio = 1.1
    batch_size = 32  # 32
    frequency_based_filter = False
    frequency_dict = None
    initial_filter_with_data = False
    data_initial = None  #None
    backprop_signal_replication_flag = False  # False. Only used for sorting problem. If True, last layer node sends backprop signal individually, without averaging.
    verbose = 2
    evaluate_train_data = False
    profile_flag = False

    if layer_types[0] == 'sort':
        pol_deg = 3  # 0 for Tensor first layer and 3 for sort first layer
    else:
        pol_deg = 0

sortnet_config = SortNetConfig()

