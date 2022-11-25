import numpy as np
import tensorflow as tf
import deepchem as dc


# Run before every test for reproducibility
def seed_all():
    np.random.seed(123)
    tf.random.set_seed(123)


def MultitaskRegressor():
    # MultitaskRegressor¶
    seed_all()

    # Load dataset with default 'scaffold' splitting
    tasks, datasets, transformers = dc.molnet.load_delaney(reload=False)
    train_dataset, valid_dataset, test_dataset = datasets
    #
    # # We want to know the pearson R squared score, averaged across tasks
    avg_pearson_r2 = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)

    # We'll train a multitask regressor (fully connected network)
    model = dc.models.MultitaskRegressor(len(tasks), n_features=1024, layer_sizes=[500])
    model.fit(train_dataset)

    # We now evaluate our fitted model on our training and validation sets
    train_scores = model.evaluate(train_dataset, [avg_pearson_r2], transformers)
    print(train_scores)
    valid_scores = model.evaluate(valid_dataset, [avg_pearson_r2], transformers)
    print(valid_scores)


def GraphConvModel():
    # GraphConvModel
    seed_all()
    tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='GraphConv', reload=False)  # 用图卷积进行分子表征学习
    train_dataset, valid_dataset, test_dataset = datasets

    avg_rms = dc.metrics.Metric(dc.metrics.rms_score, np.mean)#后面一个参数是为了多任务学习，对每个任务的评估指标进行平均得到一个值

    model = dc.models.GraphConvModel(len(tasks), mode='regression', dropout=0.5)
    model.fit(train_dataset, nb_epoch=30)

    train_scores = model.evaluate(train_dataset, [avg_rms], transformers)
    print(train_scores)
    valid_scores = model.evaluate(valid_dataset, [avg_rms], transformers)
    print(valid_scores)


def MPNNModel():
    # MPNNModel
    seed_all()
    tasks, datasets, transformers = dc.molnet.load_delaney(featurizer=dc.feat.MolGraphConvFeaturizer(use_edges=True), reload=False)
    train_dataset, valid_dataset, test_dataset = datasets

    avg_rms = dc.metrics.Metric(dc.metrics.rms_score, np.mean)

    model = dc.models.torch_models.MPNNModel(len(tasks), mode='regression', dropout=0.5)
    model.fit(train_dataset, nb_epoch=30)

    train_scores = model.evaluate(train_dataset, [avg_rms], transformers)
    print(train_scores)
    valid_scores = model.evaluate(valid_dataset, [avg_rms], transformers)
    print(valid_scores)


MPNNModel()

