## package ml - Machine Learning Libraries
###import "github.com/malaschitz/go_ml"

[![GoDoc](https://godoc.org/github.com/malaschitz/go_ml?status.png)](https://godoc.org/github.com/alonsovidales/go_ml)


Package ml provides some implementations of usefull machine learning algorithms for data mining and data analysis.

The implemented algorithms are:

	- Linear Regression
	- Logistic Regression
	- Neural Networks
	- Collaborative Filtering
	- Gaussian Multivariate Distribution for anomaly detection systems

Is implemented too the fmincg function in order to calculate the optimal theta configuration to reduce the cost value for all the implemented solutions.

Author: Alonso Vidales <alonso.vidales@tras2.es>

Changes: 

	- Improved Normalize function (now is based on standard deviation)
	- Improved MapFeatures function
	- InitializeTheta is renamed to Initalize function. This function has option for Normalize data.

Author: Richard Malaschitz <malaschitz@gmail.com>	

Use of this source code is governed by a BSD-style. These programs and documents are distributed without any warranty, express or implied. All use of these programs is entirely at the user's own risk.

For further information about this libraries, please visit the online documentation on: <http://godoc.org/github.com/malaschitz/go_ml>
