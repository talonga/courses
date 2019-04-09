Machine Learning A-Z : Hands-on Python and R In Data Science

Link : https://www.udemy.com/machinelearning/

Second course I took on Machine Learning - this is a fairly long course that covers many of the common machine learning techniques.

Pros :

* Covers almost all types of Machine Learning
* Python and R code samples for all types, that really shows you how easy most libraries make ML implementation nowadays!
* Instructor explains the intuition behind each type of Machine Learning, such that you understand conceptually how it works. Easier to remember too
* Every chapter has the same consistent style, easy to understand and follow

Cons :

* Some of the intuition explanations are too brief or not well explained (e.g. Multi-arm bandit problem ...)
* The coding sections assume near-zero Python knowledge hence a bit too basic and draggy

<h2>Part 1 - Data Preprocessing</h2>

* Provides several coding templates to pre-process data
* sklearn.cross_validation.train_test_split to split your dataset to test and train
* sklearn.impute.SimpleImputer to fill in blanks in dataset based on strategy (default = mean)
* sklearn.preprocessing.LabelEncoder to encode data as categories
* sklearn.preprocessing.StandardScaler for feature scaling

<h2>Part 2 - Regression</h2>

Supervised Learning - Predicting numerical data
* Simple Linear Regression using sklearn.linear_model.LinearRegression
* Multiple Linear Regression using the same library but to take note of Dummy Variable trap
* Polynominal Regression using sklearn.preprocessing.PolynomialFeatures to generate additional features from input
* Support Vector Regression using sklearn.svm.SVR
  * not commonly used
* Decision Tree Regression using sklearn.tree.DecisionTreeRegressor
  * also not commonly used
* Random Forest using sklearn.ensemble.RandomForestRegressor - improves upon Decision Trees using ensemble learning
  * fancy way of saying running a bunch of times and averaging
* Goes through pros and cons of each technique, and using regularization to prevent over fitting

<h2>Part 3 - Classification</h2>

Supervised Learning - Predicting category or class
* Logistic Regression using sklearn.linear_model.LogisticRegression, using sklearn.metrics.confusion_matrix to check accuracy of predictions
* K-Nearest Neightbours (KNN) using sklearn.neighbors.KNeighborsClassifier
  * classifies based on the class of the K nearest neighbours of new data
* Support Vector Machines using sklearn.svm.SVC
  * computes support vector based on maximum distance seperating the classes
* Kernel SVM using the same library but using kernel='rbf' (radial basis function) for non-linear classifications
* Naive Bayes using sklearn.naive_bayes.GaussianNB
  * based on computation of probability of new data appearing within given dataset
* Decision Tree Classifier using sklearn.tree.DecisionTreeClassifier
* Random Forest using sklearn.ensemble.RandomForestClassifier
* Goes through pros and cons of each technique, and using regularization to prevent over fitting

<h2>Part 4 - Clustering</h2>

Unsupervised Learning - Deriving clusters or segmentations in data
* K-Means Clustering using sklearn.cluster.KMeans and using elbow method to find optimum clusters
  * initialises random cluster centroids, recursively assigns nearest points to centroid until data eventually stabilizes - cost defined as distance between clusters
* Hierarchy Clustering using sklearn.cluster.AgglomerativeClustering. Interesting technique!
  * recursively groups closest points / centroids together, using the scipy.cluster.hierarchy.dendrogram to visualize and determine the optimum number of clusters

<h2>Part 5 - Association Rule Learning</h2>

Unsupervised Learning - Finding items that occur commonly with other items
* Apriori Algorithm - simple implementation
  * Defines a support threshold, computes 1-element, 2-element, 3-element etc. tuples of items with occurences >= support threshold. Searches in Breadth-First Search manner
* ECLAT (Equivalence Class Clustering and bottom up Lattice Traversal)
  * Similar in concept but Depth-First Search
  
<h2>Part 6 - Reinforcement Learning</h2>
  
Supervised Learning - algorithms that continually learn and adjust based on actual events
* Upper Confidence Bound - simple implementation
  * For each potential option, first pick at random and determine outcome (e.g. reward or no). Based on reward, confidence bound of that option is adjusted. Eventually the optimum option will converge.
* Thomson Sampling - simple implementation
  * For each potential option, first pick at random and overtime, build an assumed distribution over the option and the reward. At every step, sample at random and pick the option that maximises the reward.
  
<h2>Part 7 - Natural Language Processing</h2>

General approaches to using text for classification or clustering
* Removing stopwords using ntlk.corpus.stopwords
  * Stopwords are words that have no 'meaning' and can be removed during pre-processing
* Stemming using ntlk.stem.porter
  * Stemming means to derive the root of a word, and sometimes truncates the trailing vowels as well
* Bag of words model using sklearn.feature_extraction.text.CountVectorizer
  * Simply converts input data to dictionary counts, as input to neural network or other classification algorithms
  
<h2>Part 8 - Deep Learning</h2>

Multi-layer neural networks
* Build regular Artificial Neural Networks using keras.models.Sequential and keras.layers.Dense
* Convulutional Neural Networks using multiple layers to interpret and perform classification on images
  * Convulation Layer (keras.layers.Convulation2D) - Layer of feature detectors which extracts features from an image, which downsizes the data to a feature map representing closeness to the feature
  * ReLu Layer - Rectifier (Linear Unit) to introduce non linearity
  * Max Pooling (keras.layers.MaxPooling2D) - Also known as downsampling, identify a size e.g. 3x3 pixels, to take the maximum value within into a single pixel. Preserves features while greatly reducing image size. Many pooling options exist (average, etc.) but max is proven to be the best.
  * Flattening (keras.layers.Flatten) - Aligns the images from 2d matrix of pixels to 1d array of numbers
  * Full Connection - Create as many layers of neural networks that are fully connected from flattened layer to output layer
  * Using softmax (improves interpretability of output probability) together with cross entropy cost function
  * Using keras.preprocessing.image.ImageDataGenerator to load and automatically pre-process images into the network
  
<h2>Part 9 - Dimensionality Reduction</h2>

Reducing dimensions of input data
* Principal Component Analysis (PCA) using sklearn.discriminant_analysis.PCA
 * Unsupervised algorithm to extract number of principal components that optimally describe cross-covariance of data
* Linear Discriminant Analysis (LDA) using sklearn.discriminant_analysis.LinearDiscriminantAnalysis
 * Supervised algorithm to extract number of principal components that maximum class separation
* Kernal PCA using sklearn.discriminant_analysis.KernelPCA
 * Unsupervised algorithm that uses kernel function to project input to higher dimensions, before running PCA for principal component extraction

<h2>Part 10 - Model Selection & Boosting</h2>

Methods to select appropriate model
* K-Fold Cross Validation using sklearn.model_selection.cross_val_score
 * Splits train and test set automatically across k number of sets to aggregate the accuracy score
* Grid Search using sklearn.model_selection.GridSearchCV
 * Provides a list of training parameters which GridSearch will iterate - useful in deciding between linear or non-linear options

Boosting
* XGBoost using xgboost.XGBClassifier - very popular implementation of gradient boosted decision trees with (1) high performance (2) fast execution speed (3) keep interpreration of model, no scaling required
 * Boosting is an algorithm by which a model is built, then a second model is added that attempts to correct the errors from the first model. Models are added until no further improvement can be made, or maximum no. of models are reached.
