import argparse, os
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.utils import set_random_seed

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, AdamW



from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import scikitplot as skplt

from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint, WandbCallback

import params

# defaults
default_config = {
    "framework": "keras",
    "kernel_size": 32, # Kernels
    "filter_size": 7, # n of filters
    "lr": 2e-3,
    "weight_decay": 0.9,
    "log_preds": True,
    "embedding_size": 128,
    "max_tokens": 128,
    "batch_size": 128,
    "cnn_layers": 1,
    "epochs": 8,
    "dropout": 0,
    "num_classes": 5,
    "pretrained": False,
    "seed": 42
}

def parse_args():
    "Overriding default argments"
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--embedding_size', type=int, default=default_config['embedding_size'], help='Embedding size')
    argparser.add_argument('--batch_size', type=int, default=default_config['batch_size'], help='batch size')
    argparser.add_argument('--max_tokens', type=int, default=default_config['max_tokens'], help='Vocabulary size')    
    argparser.add_argument('--kernel_size', type=int, default=default_config['kernel_size'], help='Kernel  size')
    argparser.add_argument('--filter_size', type=int, default=default_config['filter_size'], help='Filter size')
    argparser.add_argument('--epochs', type=int, default=default_config['epochs'], help='number of training epochs')
    argparser.add_argument('--lr', type=float, default=default_config['lr'], help='learning rate')
    argparser.add_argument('--weight_decay', type=float, default=default_config['weight_decay'], help='Weight decay')
    argparser.add_argument('--dropout', type=float, default=default_config['dropout'], help='Dropout')
    argparser.add_argument('--seed', type=int, default=default_config['seed'], help='random seed')
    #argparser.add_argument('--log_preds', type=t_or_f, default=default_config['log_preds'], help='log model predictions')
    argparser.add_argument('--log_preds', default= True, action=argparse.BooleanOptionalAction)
    args = argparser.parse_args()
    #vars(default_config).update(vars(args))
    default_config.update(vars(args))
    return

def download_data(run):
    # Define the artifact to use
    processed_data_at = run.use_artifact(f'{params.PROCESSED_DATA_AT}:latest', type='split_data')
    # Download the data in the artifact
    processed_dataset_dir = Path(processed_data_at.download())
    return processed_dataset_dir

def get_target_dict(df):
    # Extract the unique category values
    target_classes = df['product'].unique()
    # Create a dictionary
    target_values = dict(zip(target_classes, range(5)))

    return target_values, target_classes

def get_datasets(processed_dataset_dir):
    # Read the data from the artifact as a csv file
    df = pd.read_csv(processed_dataset_dir / params.SPLIT_FILENAME)
    # GEt target values
    target_values, target_classes = get_target_dict(df)
    # Replace values in column product with integer values using the dict
    df.replace({'product': target_values}, inplace=True)
    # Create the train dataset
    X_train= df[df['stage']=='train']['narrative'].values
    y_train= df[df['stage']=='train']['product'].values
    # Create the validation dataset
    X_val= df[df['stage']=='validation']['narrative'].values
    y_val= df[df['stage']=='validation']['product'].values
    # Create the validation dataset
    X_test= df[df['stage']=='test']['narrative'].values
    y_test= df[df['stage']=='test']['product'].values

    return X_train, y_train, X_val, y_val, X_test, y_test, target_values, target_classes

def tokenize_text(X_train,X_val,X_test,max_tokens, tokenizer):
    # Adjust the tokenizer
    tokenizer.fit_on_texts(np.concatenate([X_train,X_val]))

    ## Vectorizing data to keep max_tokens words per sample.
    X_train_vect = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_tokens, padding="post", truncating="post", value=0.)
    X_val_vect  = pad_sequences(tokenizer.texts_to_sequences(X_val), maxlen=max_tokens, padding="post", truncating="post", value=0.)
    X_test_vect  = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_tokens, padding="post", truncating="post", value=0.)

    return X_train_vect, X_val_vect, X_test_vect

def create_model(tokenizer, embed_len, max_tokens, n_kernel, n_filter, target_values):
    # Create the input layer
    inputs = Input(shape=(max_tokens, ))
    # Embedding layer
    embeddings_layer = Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=embed_len,  input_length=max_tokens)
    # Create a Conv1D layer
    conv = Conv1D(n_kernel, n_filter, padding="same") ## Channels last
    # Create the final Dense layer
    dense = Dense(len(target_values), activation="softmax")
    # Build the model
    x = embeddings_layer(inputs)
    x = conv(x)
    x = tf.reduce_max(x, axis=1)
    output = dense(x)

    model = Model(inputs=inputs, outputs=output)

    return model

def evaluate_model(model, X_val, X_val_vect, y_val, target_classes,log_preds):
    val_preds = model.predict(X_val_vect)
    # Create the classification report
    report = classification_report(y_val, np.argmax(val_preds, axis=1), target_names=target_classes, output_dict=True)
    # Create a df with the report
    df_report = pd.DataFrame(report).transpose()
    df_report.reset_index(inplace=True)
    df_report = df_report.rename(columns = {'index':'category'})
    # Draw the confussion matrix for the test dataset
    skplt.metrics.plot_confusion_matrix([target_classes[i] for i in y_val], [target_classes[i] for i in np.argmax(val_preds, axis=1)],
                                        normalize=True,
                                        title="Confusion Matrix",
                                        cmap="Blues",
                                        hide_zeros=True,
                                        figsize=(5,5)
                                        )
    plt.xticks(rotation=90)
    plt.savefig('confussion_matrix.png')

    if log_preds:
        print("Create the predictions table")
        df_val= pd.concat([pd.Series(X_val), pd.Series(y_val), pd.DataFrame(val_preds), pd.Series(np.argmax(val_preds, axis=1))], axis=1)
        df_val.columns = ['complaint','product','prob0','prob1','prob2','prob3','prob4','prediction']

    return df_report, df_val, 'confussion_matrix.png'



def train(config):
    # Set the seed for reproducibility
    set_random_seed(default_config['seed'])

    # Init the wandb session in our project
    run = wandb.init(project=params.WANDB_PROJECT, entity=None, job_type="hyperparameters", config=config)
    # good practice to inject params using sweeps
    config = wandb.config

    # Download the data in the artifact
    processed_dataset_dir = download_data(run)
    # Split the datasets
    X_train, y_train, X_val, y_val, X_test, y_test, target_values, target_classes = get_datasets(processed_dataset_dir)
    # Create the tokenizer
    tokenizer = Tokenizer()
    # Tokenize the texts
    X_train_vect, X_val_vect, X_test_vect = tokenize_text(X_train,X_val,X_test, config.max_tokens, tokenizer)
    # Define the model
    model = create_model(tokenizer, config.embedding_size, config.max_tokens, config.kernel_size, config.filter_size, target_values)
    # Create the optimizer
    adam_optimizer = AdamW(learning_rate=config.lr, weight_decay=config.weight_decay)
    # Compile the model
    model.compile(adam_optimizer, "sparse_categorical_crossentropy", metrics=["accuracy"])
    # Create the callbacks
    #callbacks = [WandbCallback(monitor="val_accuracy", mode="max", log_preds=False, log_model=True),
    #             WandbModelCheckpoint(filepath="model-{epoch:02d}", monitor="val_accuracy", save_best_only=True, mode="max"),]

    callbacks= [EarlyStopping(monitor='val_accuracy', patience = 2, restore_best_weights = False),
            WandbMetricsLogger(),
            WandbModelCheckpoint(filepath="model", monitor="val_accuracy", save_best_only=True, mode="max"),]

    # Fit the model
    history = model.fit(X_train_vect, y_train, batch_size=config.batch_size, epochs=config.epochs, validation_data=(X_val_vect, y_val),
                    callbacks=[callbacks])    # Restore the best model

    #model.load_weights("model")
    model = tf.keras.models.load_model("model")

    # Evaluate the model
    df_report, df_val, conf_matrix = evaluate_model(model, X_val, X_val_vect, y_val, target_classes, config.log_preds)    
    # Save artifacts in W&B
    #Create the artifact
    tuning_data_at = wandb.Artifact(params.HPTUNING_DATA_AT, type="tuning")
    # Add the classifation report
    tuning_data_at.add_file(conf_matrix)
    # Save the classification report
    report_table = wandb.Table(dataframe=df_report)
    tuning_data_at.add(report_table, "classification_report")
    # Save evaluation data
    if config.log_preds:
        evaluation_table = wandb.Table(dataframe=df_val)
        tuning_data_at.add(evaluation_table, "evaluation_table")

    # Log artifacts
    run.log_artifact(tuning_data_at)
    # Close W&B session
    wandb.finish()

if __name__ == '__main__':
    # Parse the arguments
    parse_args()
    # Train the model
    train(default_config)