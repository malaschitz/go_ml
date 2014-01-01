/*
Package ml, implements a set of machine learning algorithm for linear regrassion

*/
package ml

import (
	"math"
)

func neg(n float64) float64 {
	return -n
}

func Normalize(values []float64) (norm []float64, valid bool) {
	avg := 0.0
	max := math.Inf(-1)
	min := math.Inf(1)
	math.Inf(1)
	for _, val := range values {
		avg += val
		if val < min {
			min = val
		}
		if val > max {
			max = val
		}
	}
	avg /= float64(len(values))

	if max == min {
		valid = false
		return
	}

	valid = true
	for _, val := range values {
		norm = append(norm, (val - avg) / (max - min))
	}

	return
}

func multElems(elems []float64) (resilt float64) {
	resilt = 1
	for _, elem := range elems {
		resilt *= elem
	}

	return
}

func combinations(iterable []float64, r int) (results []float64) {
	pool := iterable
	n := len(pool)

	if r > n {
		return
	}

	indices := make([]int, r)
	for i := range indices {
		indices[i] = i
	}

	result := make([]float64, r)
	for i, el := range indices {
		result[i] = pool[el]
	}

	results = append(results, multElems(result))
	for {
		i := r - 1
		for ; i >= 0 && indices[i] == i+n-r; i -= 1 {
		}

		if i < 0 {
			return
		}

		indices[i] += 1
		for j := i + 1; j < r; j += 1 {
			indices[j] = indices[j-1] + 1
		}

		for ; i < len(indices); i += 1 {
			result[i] = pool[indices[i]]
		}
		results = append(results, multElems(result))
	}

	return
}

// This method calculates all the possible combinations of the features and
// returns them with the specified degree, for example, for a data.X with x1, x2
// and degree 2 will convert data.X to 1, x1, x2, x1 * x2, x1 ** 2, x2 ** 2,
// (x1 * x2) ** 2
// Use this method with care in order to calculate the model who fits better with
// the problem
func MapFeatures(x [][]float64, degree int) (ret [][]float64) {
	ret = make([][]float64, len(x))
	elems := len(x[1])
	for i := 0; i < len(x); i++ {
		aux := make([]float64, len(x[i]))
		copy(aux, x[i])
		ret[i] = make([]float64, len(x[i]))
		copy(ret[i], x[i])

		for l := 2; l <= elems; l++ {
			x[i] = append(x[i], combinations(aux, l)...)
		}
	}

	ret = PrepareX(x, degree)

	return
}

func PrepareX(x [][]float64, degree int) (newX [][]float64) {
	for _, values := range x {
		result := []float64{1}

		for _, value := range values {
			for calcDeg := 1; calcDeg <= degree; calcDeg++ {
				result = append(result, math.Pow(value, float64(calcDeg)))
			}
		}

		newX = append(newX, result)
	}

	return
}

func sigmoid(z float64) float64 {
	return 1 / (1 + math.Pow(math.E, neg(z)))
}

func oneMinus(x float64) float64 {
	return 1 - x
}

func powTwo(x float64) float64 {
	return math.Pow(x, 2)
}