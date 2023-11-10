# Import our dependencies
from email.mime import application
import sklearn as skl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import tensorflow as tf
from tensorflow.keras import regularizers
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

# Convert categorical data to numeric with `pd.get_dummies` I decided to leave this here
application_df = pd.get_dummies(application_df, columns=['APPLICATION_TYPE', 'AFFILIATION', 'CLASSIFICATION', 'INCOME_AMT'], dtype=np.float32)
print(application_df)

# Split our preprocessed data into our features and target arrays
y = application_df['IS_SUCCESSFUL']
X = application_df.drop(columns=['IS_SUCCESSFUL'])

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
def create_model():
    #This model was found by running Keras Tuner 0.025181
    input_features= len(X.columns)
    model = tf.keras.Sequential([
    tf.keras.layers.Dense(224, activation='relu', input_dim=24),
    tf.keras.layers.Dense(97, activation='relu'),
    tf.keras.layers.Dropout(0.06),
    tf.keras.layers.Dense(129, activation='relu'),
    tf.keras.layers.Dropout(0.06),
    tf.keras.layers.Dense(417, activation='relu'),
    tf.keras.layers.Dropout(0.06),
    tf.keras.layers.Dense(193, activation='relu'),
    tf.keras.layers.Dense(193, activation='relu')
    ])
   

    #this is my final layer output
    model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

    # Compile the model
    model.compile(loss="binary_crossentropy",
                     optimizer='Adam',
                     metrics=["accuracy"])

    return model

#create the model
nn_model = create_model()

# Setting up an early_stop that will have a patience of 10 epochs
stop_early = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)

#train the model
fit_model_nn = nn_model.fit(X_train_scaled, y_train, epochs=300, callbacks=[stop_early])

# Evaluate the model using the test data
model_loss, model_accuracy = nn_model.evaluate(X_test_scaled,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

# Export our model to HDF5 file
nn_model.save('models/AlphabetSoupCharity_PCA_FinalModel.HDF5')

