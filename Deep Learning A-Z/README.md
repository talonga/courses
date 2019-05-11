Deep Learning A-Z : Hands On Python & R

Link : https://www.udemy.com/deeplearning/learn/v4/overview

3rd course on Udemy by the same instructor, first 2 sections are overlapped with Machine Learning A-Z course.

<h2>Part 1 - ANN</h2>

* Using keras.wrappers.scikit_learn.KerasClassifier as a scikit-learn wrapper, to run k-fold cross validation using keras models
* Using keras.layers.Dropout for dropout regularization, to solve overfitting
  * Dropout is a technique which randomly 'drops' several neurons in a particular ANN layer, which makes the network less reliant on specific weights

<h2>Part 2 - Convulutional Neural Networks</h2>

* Same course material as ML A-Z Course.
* Saving weights in HD5 form, and model definition using to_yaml()
* Classifying a single image by using keras.preprocessing.image to import image into numerical array

<h2>Part 3 - Recurrent Neural Networks</h2>

Concept of using a neural networks connected to itself via time steps. Good for regression of time series as RNN will retain contextual knowledge of past timesteps to predict values of next timestep

* Using sklearn.preprocessing.MinMaxScaler, normalization feature scaling better for RNN especially if there is sigmoid function
* Using keras.layers.LSTM for LSTM layers

<h2>Part 4 - Self Organizing Maps</h2>

Unsupervised deep learnig used for feature detection. Using input data, weights are assigned in fully connected neural network. By checking MSE of weights to each input data, for each unit, find BMU (best matching unit) and updates weights to move it closer to input data. Update units within radius as well, to be closer to input data.

* using minisom.MiniSom to implement Self Organizing Maps
* using pylab.pcolor to plot the SOM grid. Able to visually detect outliers by seeing grid cells that are unlike surrounding cells.

<h2Part 5 - Boltzmann Machines>/h2>
