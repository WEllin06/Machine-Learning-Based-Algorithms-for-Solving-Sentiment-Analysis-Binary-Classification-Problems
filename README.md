I will now proceed to provide a brief overview of the principal elements of the project.

The project is divided into two distinct phases:
1.The initial phase is predicated upon a linear hypothesis space and a multitude of activation functions, including the Softmax function. The model solves the weight w based on mean square loss, cross-entropy loss, and hinge loss, and it is combined with multiple optimisation algorithms, including stochastic and batch gradient descent methods. Finally, the optimal classification model for solving this sentiment analysis binary classification in linear space is Lasso regression based on gradient descent. 

2.In the second stage, a new weight function, r(x;a), was introduced, based on the nonlinear assumption space of the Gaussian kernel function. The final classification function remains within the linear assumption space. The parameter a in the weight function r(x;a) can be solved by the L-BFGS-B optimisation algorithm. Finally, the combination of the mean square loss, logistic loss, and hinge loss, and the subsequent solution of the weight parameter θ at the L-BFGS-B optimisation algorithm demonstrates that the IWL model is more accurate than the unweighted LSC algorithm on the test set. The SC (Important Weight Least Squares Classification) algorithm, which incorporates the weight function r(x;a), has been demonstrated to be more accurate than the unweighted LSC algorithm (Least Squares Classification). This enhanced accuracy is subjected to a comprehensive analysis.
