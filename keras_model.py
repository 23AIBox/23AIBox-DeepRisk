import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from tensorflow.keras import layers, metrics, losses, optimizers, callbacks, regularizers
from tensorflow import keras
from tensorflow.compat import v1
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import scale
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
config = v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = v1.Session(config=config)
v1.keras.backend.set_session(sess)

from layers import PartialDense, PartialConnected1D
from data import Dataset


def get_model(snp_num, gene_num, mask, gene_ids):

    ipt = layers.Input(shape=(snp_num,))

    x = layers.Lambda(
        lambda x: tf.stack([x * tf.cast(x != 3, 'float32'), (2 - x) * tf.cast(x != 3, 'float32')], axis=-1)
    )(ipt)
    x = PartialConnected1D(mask, units=1)(x)

    lstm = layers.Bidirectional(
        v1.keras.layers.LSTM(
            units=4,
            return_sequences=True,
            name='lstm',
            kernel_regularizer=regularizers.l2(1e-3),
            recurrent_regularizer=regularizers.l2(1e-3)
        )
    )

    xs = list()
    for ids in gene_ids:
        xi = layers.Concatenate(axis=-2)([x[:, i:i+1] for i in ids])
        xs.append(lstm(xi))
    x = layers.Concatenate(axis=-2)(xs)
    

    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)

    opt = layers.Dense(
        units=1,
        activation='sigmoid',
        name='dense',
        kernel_regularizer=regularizers.l2(1e-3)
    )(x)

    model = keras.Model(inputs=[ipt], outputs=[opt])

    model.compile(
        optimizer=tfa.optimizers.AdamW(
            weight_decay=1e-4,
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            amsgrad=False
        ),
        loss=losses.binary_crossentropy,
        metrics=['acc', metrics.AUC()]
    )

    return model


def train_fold(model: keras.Model, class_weight, x_train, y_train, x_val, y_val, x_test, y_test):
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=512,
        epochs=20,
        verbose=1,
        validation_data=(x_val, y_val),
        callbacks=[
            callbacks.ModelCheckpoint(
                filepath='./checkpoint/weights.h5',
                save_weights_only=True
            )
        ],
        class_weight=class_weight,
        shuffle=True
    )
    model.load_weights(filepath='./checkpoint/weights.h5')
    pred = model.predict(x=x_test, batch_size=512)
    return pred


def compute_class_weight(y):
    pos_weight = len(y) / 2.0 / np.sum(y)
    neg_weight = len(y) / 2.0 / (len(y) - np.sum(y))
    return {0: neg_weight, 1: pos_weight}


if __name__ == "__main__":

    import sys

    disease, p_v, r2 = sys.argv[1:4]
    suffix = '%s_%s_%s_' % (disease, p_v, r2)

    diseases = {'AD': "AD.IGAP", 'T2D': "T2D.DIAGRAM", 'IBD': "IBD.IIBDGC.IBD", 'BC': "BC.BCAC.all",
                'CAD': "CAD.C4D.fixed", 'RA': "RA.Okada"}
    disease = diseases[disease]
    dataset = Dataset(disease)

    p_v = float(p_v)
    r2 = float(r2)
    
    snps = dataset.filter(p_value=p_v, r2=r2)

    chroms, posits = dataset.GWAS.loc[snps, 'Chromosome'], dataset.GWAS.loc[snps, 'Position']

    genotype, phenotype, _ = dataset.extract(snps)
    x, y = genotype.values.T, phenotype.values
    print(len(y), y.sum())

    mask_gene, _ = dataset.association(snps, return_matrix=False)

    mask_gene = sorted(mask_gene, key=lambda i: (chroms[i[0]], min([posits[j] for j in i])))

    mask, gene_ids = list(), {i: list() for i in set(chroms)}
    for i, j in enumerate(mask_gene):
        mask.extend([(k, i) for k in j])
        gene_ids[chroms[j[0]]].append(i)
        
    y = np.expand_dims(y, 1)

    snp_num = x.shape[1]
    gene_num = len(mask_gene)

    kfold = StratifiedKFold(n_splits=10, random_state=222, shuffle=True)

    test_auc, preds = list(), np.zeros((x.shape[0],))
    for fold, (train_index, test_index) in enumerate(kfold.split(x, y)):
        print("-" * 10 + "No. :", fold + 1)

        y_train, y_test = y[train_index], y[test_index]
        x_train, x_test = x[train_index], x[test_index]

        class_weight = compute_class_weight(y_train)

        model = get_model(snp_num, gene_num, mask, gene_ids.values())
        # if fold == 0:
        #     model.summary()

        pred = train_fold(model, class_weight, x_train, y_train, x_test, y_test, x_test, y_test)
        preds[test_index] = pred[:, 0]
        test_auc.append(roc_auc_score(y_test, pred))

        # del model

        keras.backend.clear_session()
        v1.reset_default_graph()

    print("-" * 10 + ' ' + disease + ' ' + suffix + ' ' + str(p_v) + ' ' + str(r2) + ' ' + "-" * 10)
    print(snp_num, gene_num)
    print("10 fold auc :", test_auc)
    print("10 fold mean auc :", np.mean(test_auc))

    result = pd.DataFrame(index=phenotype.index)
    result['label'] = phenotype
    result['pred'] = preds
    result.to_csv('our_result/%sprobs.csv' % (suffix))
