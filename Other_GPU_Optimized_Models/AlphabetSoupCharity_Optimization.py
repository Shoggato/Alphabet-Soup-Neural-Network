# Import our dependencies
from email.mime import application
import sklearn as skl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
import numpy as np

#  Import and read the charity_data.csv.
application_df = pd.read_csv("https://static.bc-edx.com/data/dl-1-2/m21/lms/starter/charity_data.csv")
  
# Drop the non-beneficial ID columns, 'EIN' and 'NAME'.
application_df = application_df.drop(columns=['EIN', 'NAME', 'STATUS', 'SPECIAL_CONSIDERATIONS', 'USE_CASE', 'ORGANIZATION'])

application_df.head()

# Identify the range of values
ask_amt_min = application_df['ASK_AMT'].min()
ask_amt_max = application_df['ASK_AMT'].max()
#print(ask_amt_max, ask_amt_min)

#Going to attempt to normalize my ASK_AMT data so to minimize the extreme values and attempt to minimize the effects of the majority value 5000
application_df['ASK_AMT'] = np.log1p(application_df['ASK_AMT'])

# Choose a cutoff value and create a list of income amount to be replaced
cutoff = 1000
IncomeAmt_to_replace = []
for i in application_df['INCOME_AMT'].value_counts().index:
    if application_df['INCOME_AMT'].value_counts()[i] < cutoff:
        IncomeAmt_to_replace.append(i)

# Replace in dataframe
for i in IncomeAmt_to_replace:
    application_df['INCOME_AMT'] = application_df['INCOME_AMT'].replace(i, "Other")

#print(application_df['INCOME_AMT'].value_counts())

# Choose a cutoff value and create a list of application types to be replaced
cutoff = 1000
Applications_to_replace = []
for i in application_df['APPLICATION_TYPE'].value_counts().index:
    if application_df['APPLICATION_TYPE'].value_counts()[i] < cutoff:
        Applications_to_replace.append(i)

# Replace in dataframe
for i in Applications_to_replace:
    application_df['APPLICATION_TYPE'] = application_df['APPLICATION_TYPE'].replace(i, 'Other')
    
#print(application_df['APPLICATION_TYPE'].value_counts())

# Choose a cutoff value and create a list of classifications to be replaced
cutoff = 500
classifications_to_replace = []
for i in application_df['CLASSIFICATION'].value_counts().index:
    if application_df['CLASSIFICATION'].value_counts()[i] < cutoff:
        classifications_to_replace.append(i)

# Replace in dataframe
for i in classifications_to_replace:
    application_df['CLASSIFICATION'] = application_df['CLASSIFICATION'].replace(i, "Other")

#print(application_df['CLASSIFICATION'].value_counts(), application_df.head())

# Convert categorical data to numeric with `pd.get_dummies` I decided to leave this here, and try One_Hot encoding
#application_df = pd.get_dummies(application_df, columns=['APPLICATION_TYPE', 'AFFILIATION', 'CLASSIFICATION', 'INCOME_AMT'], dtype=np.float32)
#print(application_df)

#set up OneHotEncoder
enc = OneHotEncoder()

#fit categorical columns to encoder then take that np array and convert to a pandas dataframe
EncOneHot = enc.fit_transform(application_df[['APPLICATION_TYPE', 'AFFILIATION', 'CLASSIFICATION', 'INCOME_AMT']]).toarray()
EncOneHot_df = pd.DataFrame(EncOneHot)
print(EncOneHot_df)

#grab remaining columns
remaining_col = application_df[['ASK_AMT', 'IS_SUCCESSFUL']]
remaining_col = remaining_col.rename(columns={
    'ASK_AMT': 23,
    'IS_SUCCESSFUL': 24
})
print(remaining_col)

#concat the columns together into preprocess_df ready for spliting
preprocess_df = pd.concat([EncOneHot_df, remaining_col], axis=1)
print(f"This is my preprocessed dataframe: {preprocess_df}")

# Split our preprocessed data into our features and target arrays
y = preprocess_df[24]
X = preprocess_df.drop(columns=[24])

# Split the preprocessed data into a training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    random_state=1,
                                                    stratify=y)

# Create a StandardScaler instances
scaler = StandardScaler()

# Fit the StandardScaler
X_scaler = scaler.fit(X_train)

# Scale the data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

# Create a method that creates a new Sequential model with hyperparameter options
def create_model(hp):
    
    input_dim = len(X.columns)
    model = tf.keras.models.Sequential()

    # Allow kerastuner to decide which activation function to use in hidden layers
    activation = hp.Choice('activation',['relu'])
    
    # Allow kerastuner to decide number of neurons in first layer
    model.add(tf.keras.layers.Dense(units=hp.Int('units',
        min_value=1,
        max_value=512,
        step=32), 
        activation=activation,
        input_dim=input_dim
        ))

    # Allow kerastuner to decide number of hidden layers and neurons in hidden layers
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(tf.keras.layers.Dense(units=hp.Int('units_' + str(i),
            min_value=1,
            max_value=512,
            step=32),
            activation=activation,
            kernel_regularizer=tf.keras.regularizers.l1(l=hp.Choice('l1_weight', [0.025]))
            ))
    
    #this is my drop out layer
    model.add(tf.keras.layers.Dropout(hp.Choice('dropout', [0.07])))

    #this is my final layer output
    model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

    # Compile the model
    model.compile(loss="binary_crossentropy",
                     optimizer='Adam',
                     metrics=["accuracy"])

    return model

# set up hyperband tuner
tuner = kt.Hyperband(
    create_model,
    objective="val_accuracy",
    max_epochs=100,
    hyperband_iterations=5,
    project_name='optimization_AlphabetSoupCharity_kt_gpu')

# Setting up an early_stop that will have a patience of 10 epochs
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# Run the kerastuner search for best hyperparameters
tuner.search(X_train_scaled, y_train,epochs=300, validation_split=0.2, callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

#Print tuner results
tuner.results_summary()

# Build the model with the optimal hyperparameters and train it on the data for 300 epochs with early stopping so model doesn't overfit
model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train_scaled, y_train, epochs=300, validation_split=0.2, callbacks=[stop_early])

#grabs val_accuracy
val_acc_per_epoch = history.history['val_accuracy']

#prints the best iteration of model
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

eval_result = model.evaluate(X_test_scaled, y_test)
print("[test loss, test accuracy]:", eval_result)

# Export our model to HDF5 file
model.save('models/AlphabetSoupCharityOptimized_gpu.HDF5')