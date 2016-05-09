// Package ml provides some implementations of usefull machine learning
// algorithms for data mining and data analysis.
//
// The implemented algorithms are:
//    - Linear Regression
//    - Logistic Regression
//    - Neural Networks
//    - Collaborative Filtering
//    - Gaussian Multivariate Distribution for anomaly detection systems
//
// Is implemented too the fmincg function in order to calculate the optimal
// theta configuration to reduce the cost value for all the implemented solutions.
//
// Author: Alonso Vidales <alonso.vidales@tras2.es>
//
// Use of this source code is governed by a BSD-style.
// These programs and documents are distributed without any warranty, express or
// implied. All use of these programs is entirely at the user's own risk.
//
package ml

// General purpose machine learning functions

import (
	"math"
)

// Normalize Returns all the values of the given matrix normalized, the formula
// applied to all the elements is: (Xn - Avg) / (max - min) If all the elements
// in the slice have the same values, or the slice is empty, the slice can't be
// normalized, then returns false in the valid parameter
func Normalize(values []float64) (norm []float64, mean float64, stddev float64) {
	mean, stddev = stdDevF(values)
	for _, val := range values {
		norm = append(norm, (val-mean)/stddev)
	}
	return
}

func stdDevF(numbers []float64) (float64, float64) {
	total := 0.0
	for _, number := range numbers {
		total += number
	}
	mean := total / float64(len(numbers))
	total = 0.0
	for _, number := range numbers {
		total += math.Pow(number-mean, 2)
	}
	variance := total / float64(len(numbers)-1)
	return mean, math.Sqrt(variance)
}

// MapFeatures Retrrns the x matrix with all the elements at the power of
// x, x-1, x-2, ... 1 and adds at the being of each row a 1 in order to be used
// as bias value.
// For example for a given matrix like:
//    2 3 5
//	  10 11 12
// Prepared at the power of 2:
//    1 2 3 5 4 6 10 9 15 25
//	  1 10 11 12 100 110 120 121 132 144
// Prepared at the power of 3:
//    1 2 3 5 4 6 10 9 15 25 8 12 20 18 30 50 27 45 75 125
//	  1 10 11 12 100 110 120 121 132 144 1000 1100 1200 1210 1320 1440 1331 1452 1584 1728

func MapFeatures(x [][]float64, degree int) (newX [][]float64) {
	for _, values := range x {
		x := []float64{1.0}
		for i := 0; i < degree; i++ {
			result := make([]float64, 0)
			combinations(0, make([]int, i+1), values, &result)
			x = append(x, result...)
		}
		newX = append(newX, x)
	}

	return
}

func combinations(pos int, p []int, a []float64, result *[]float64) {
	start := 0
	if pos > 0 {
		start = p[pos-1]
	}
	for i := start; i < len(a); i++ {
		p[pos] = i
		if pos == len(p)-1 { //end of game
			v := 1.0
			for j := 0; j < len(p); j++ {
				v *= a[p[j]]
			}
			*result = append(*result, v)
		} else {
			combinations(pos+1, p, a, result)
		}
	}
}

// multElems returns the result of multiply all the elements contained on the
// slice
func multElems(elems []float64) (resilt float64) {
	resilt = 1
	for _, elem := range elems {
		resilt *= elem
	}

	return
}

// Auxiliar functions to work with matrix elements

// neg returns the negation of the given float
func neg(n float64) float64 {
	return -n
}

// sigmoid calculates the sigmoid funcion for logistic regression
func sigmoid(z float64) float64 {
	return 1 / (1 + math.Pow(math.E, neg(z)))
}

// oneMinus returns one minus the given float
func oneMinus(x float64) float64 {
	return 1 - x
}

// powTwo returns the number at the power of two
func powTwo(x float64) float64 {
	return x * x
}
