import numpy as np
import tensorflow as tf
import deepchem as dc

# Run before every test for reproducibility
def seed_all():
    np.random.seed(123)
    tf.random.set_seed(123)


seed_all()
# Load ChEMBL 5thresh dataset with random splitting
chembl_tasks, datasets, transformers = dc.molnet.load_chembl(
    shard_size=2000, featurizer="ECFP", set="5thresh", split="random")
train_dataset, valid_dataset, test_dataset = datasets
print(len(chembl_tasks))
print(f'Compound train/valid/test split: {len(train_dataset)}/{len(valid_dataset)}/{len(test_dataset)}')

# We want to know the RMS, averaged across tasks
avg_rms = dc.metrics.Metric(dc.metrics.rms_score, np.mean)
# Create our model
n_layers = 3
model = dc.models.MultitaskRegressor(
    len(chembl_tasks),
    n_features=1024,
    layer_sizes=[1000] * n_layers,
    dropouts=[.25] * n_layers,
    weight_init_stddevs=[.02] * n_layers,
    bias_init_consts=[1.] * n_layers,
    learning_rate=.0003,
    weight_decay_penalty=.0001,
    batch_size=100)
model.fit(train_dataset, nb_epoch=5)

# We now evaluate our fitted model on our training and validation sets
train_scores = model.evaluate(train_dataset, [avg_rms], transformers)
print(train_scores['mean-rms_score'])

valid_scores = model.evaluate(valid_dataset, [avg_rms], transformers)
print(valid_scores['mean-rms_score'])



# Load ChEMBL dataset
chembl_tasks, datasets, transformers = dc.molnet.load_chembl(
    shard_size=2000, featurizer="GraphConv", set="5thresh", split="random")
train_dataset, valid_dataset, test_dataset = datasets

# RMS, averaged across tasks
avg_rms = dc.metrics.Metric(dc.metrics.rms_score, np.mean)
model = dc.models.GraphConvModel(len(chembl_tasks), batch_size=128, mode='regression')
# Fit trained model
model.fit(train_dataset, nb_epoch=5)

# We now evaluate our fitted model on our training and validation sets
train_scores = model.evaluate(train_dataset, [avg_rms], transformers)
print(train_scores['mean-rms_score'])

valid_scores = model.evaluate(valid_dataset, [avg_rms], transformers)
print(valid_scores['mean-rms_score'])
