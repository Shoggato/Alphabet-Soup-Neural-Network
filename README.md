# deep-learning-challenge

Background
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

EIN and NAME—Identification columns
APPLICATION_TYPE—Alphabet Soup application type
AFFILIATION—Affiliated sector of industry
CLASSIFICATION—Government organization classification
USE_CASE—Use case for funding
ORGANIZATION—Organization type
STATUS—Active status
INCOME_AMT—Income classification
SPECIAL_CONSIDERATIONS—Special considerations for application
ASK_AMT—Funding amount requested
IS_SUCCESSFUL—Was the money used effectively

I have multiple files in my Repository.  Two of them I wrote in Visual Studio and ran in an Ubuntu Virtual Machine so I could utilize Tensor Flow GPU acceleration to speed up my models.

I put my Neural Network Model Report into my ReadMe for this project please see below.

Alphabet Soup Neural Network Model Report:

How many neurons, layers, and activation functions did you select for your neural network model, and why?
Were you able to achieve the target model performance?
What steps did you take in your attempts to increase model performance?
Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.

<h1><loud>Alphabet Soup Neural Network Model Report</loud><h1>

<h2>Overview of the Analysis:</h2><br>

The purpose of this analysis was to train a neural network that could successfully confirm if a applicant would be successful if funded by Alphabet Soup >75% of the time.

Results:

<h2>Data Preprocessing</h2><br>

There was only one variable that was utilized as a target for my model that was "IS_SUCCESSFUL".  The reason for this was that there was only two options, either the funded project succeeded '1', or it failed '0'. <br>

The variables that were my features for my model are listed below.<br>

<ul>
    <li>APPLICATION_TYPE </li>
    <li>AFFILIATION </li>
    <li>CLASSIFICATION </li>
    <li>USE_CASE </li>
    <li>ORGANIZATION </li>
    <li>INCOME_AMT </li>
    <li>ASK_AMT </li>
</ul>

I had removed four variables from my data since I decided that they were either too vague, or didn't have enough variability to warrant them being used.
<ul>
    <li>EIN</li>
    <li>NAME</li>
    <li>STATUS</li>
    <li>SPECIAL_CONSIDERATIONS</li>

</ul>

<h2>Compiling, Training, and Evaluating the Model</h2>

I decided rather than attempt to optimize the model by hand. I went with KerasTuner to automate the process.  I decided my activation function should be relu while my output should be sigmoid.  The reason for this is that relu chooses values between 0 and 1, while sigmoid output needs to be categorical, ie. either successful or failed.  Since I have a large amount of features that I was utilizing for my model.  I figured four layers would be good enough, I also chose my neuron values to be between 32, and 256 for both my starting layer and my hidden layers neurons as well.  I also decided to use L1 regularization to help keep my model from overfitting as well for each of my layers.  For my compiler I went with a loss function of binary_crossentropy since my problem was a binary classification (successful or failed).  My optimizer I went with Adam since it seemed to be a generally overall good optimizer for this data.  One additional thing I added for my model was an EarlyStopping metric.  This would set my entire model to monitor value loss.  If after three epochs (iterations) if my val_loss doesn't improve it will stop there.  This will stop my model form overfitting.  I tried multiple approaches for trying to decrease the complexity of my features for my model.  The first and easiest route was eliminating more columns of data that I felt would only confuse my model.  When that didn't change my accuracy or decrease my loss, I decided that I didn't really fully understand the data that I was setting the model up on.  Since I didn't have an expert on this data that I could go too.  I decided to use some unsupervised learning to help me find the most useful features.  T-SNE (t-distributed stochastic neighbor embedding) and PCA (principal component analysis) are both dimensionality reduction techniques that I utilized to try and reduce the dimensionality of the data and find the most useful features.  However, this did not improve my model.  When that didn't seem to affect my model (actually it decreased my accuracy to 0.5, and my loss to 1.0 which I beleive is equivalent to the model making random guesses).  From running multiple different settings with keras tuner I finally got a model that didn't seem to overfit (my accuracy during training was less than or equivalent to the accuracy of my testing sets).

I wasn't able to achieve the target model performance of >75%.  However, I feel pretty good about the model I have been able to create. It has a loss of around 0.5 and an accuracy of 72%.  

Summary:
I have trained a neural network model that can predict whether an applicant will be successful if funded by Alphabet Soup with an accuracy of 72%. This model is not perfect, but it can be a useful tool for Alphabet Soup to help select the applicants with the best chance of success.

Recommendation for a different model

Firstly, I think getting a expert who has knows this information from Alphabet would greatly increase the accuracy of the model.  The problem I kept running into, I wasn't sure which columns were important, or weren't important (EIN and NAME excluded).  Even after trying decreasing my dimensionality my model then started to randomly guess answers.  This tells me that most if not all of these columns are important.

Finally another possibility would be to use a different machine learning algorithm altogether, such as a support vector machine (SVM) or a random forest classifier. These algorithms are often more robust to overfitting and can be more effective on small datasets.
