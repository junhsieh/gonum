// Copyright ©2017 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package spatial // import "gonum.org/v1/gonum/stat/spatial"

import (
	"math"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

// TODO(kortschak): Make use of banded matrices when they exist in mat.

// GetisOrd performs Local Getis-Ord G*i statistic calculation.
type GetisOrd struct {
	data     *mat.Vector
	locality *mat.Dense

	mean, s float64
}

// NewGetisOrd returns a GetisOrd based on the provided data and locality matrix.
// NewGetisOrd will panic if locality is not a square matrix with dimensions the
// same as the length of data.
func NewGetisOrd(data, weights []float64, locality mat.Matrix) GetisOrd {
	if weights != nil {
		panic("spatial: weighted data not yet implemented")
	}
	r, c := locality.Dims()
	if r != len(data) || c != len(data) {
		panic("spatial: data length mismatch")
	}

	var g GetisOrd
	d := make([]float64, len(data))
	copy(d, data)
	g.data = mat.NewVector(len(data), d)

	g.locality = mat.DenseCopyOf(locality)
	g.mean = stat.Mean(data, nil)
	var ss float64
	for _, v := range data {
		ss += v * v
	}
	g.s = ss/float64(len(data)) - g.mean*g.mean

	return g
}

// Len returns the number of data points held by the receiver.
func (g GetisOrd) Len() int { return g.data.Len() }

// Gstar returns the Local Getis-Ord G* statistic for element i. The
// returned value is a z-score.
func (g GetisOrd) Gstar(i int) float64 {
	wi := g.locality.RowView(i)
	ws := g.mean * mat.Sum(wi)
	n := float64(g.data.Len())
	num := mat.Dot(wi, g.data) - ws
	den := g.s * math.Sqrt((n*mat.Dot(wi, wi)-ws*ws)/(n-1))
	return num / den
}

// GlobalMoransI performs Global Moran's I calculation of spatial autocorrelation.
// GlobalMoransI returns Moran's I, Var(I) and the z-score associated with those
// values.
//
// See https://en.wikipedia.org/wiki/Moran%27s_I.
func GlobalMoransI(data, weights []float64, locality mat.Matrix) (i, v, z float64) {
	if weights != nil {
		panic("spatial: weighted data not yet implemented")
	}
	if r, c := locality.Dims(); r != len(data) || c != len(data) {
		panic("spatial: data length mismatch")
	}
	mean := stat.Mean(data, nil)

	// Calculate Moran's I for the data.
	var num, den float64
	for i, xi := range data {
		zi := xi - mean
		den += zi * zi
		for j, xj := range data {
			zj := xj - mean
			num += locality.At(i, j) * zi * zj
		}
	}
	i = (float64(len(data)) * num) / (mat.Sum(locality) * den)

	// Calculate Moran's E(I) for the data.
	e := -1 / float64(len(data)-1)

	// Calculate Moran's Var(I) for the data.
	//  http://pro.arcgis.com/en/pro-app/tool-reference/spatial-statistics/h-how-spatial-autocorrelation-moran-s-i-spatial-st.htm
	//  http://pro.arcgis.com/en/pro-app/tool-reference/spatial-statistics/h-global-morans-i-additional-math.htm
	var s0, s1, s2 float64
	var var2, var4 float64
	for i, v := range data {
		v -= mean
		v *= v
		var2 += v
		var4 += v * v

		var p2 float64
		for j := range data {
			wij := locality.At(i, j)
			wji := locality.At(j, i)

			s0 += wij

			v := wij + wji
			s1 += v * v

			p2 += wij + wji
		}
		s2 += p2 * p2
	}
	s1 *= 0.5

	n := float64(len(data))
	a := n * ((n*n-3*n+3)*s1 - n*s2 + 3*s0*s0)
	c := (n - 1) * (n - 2) * (n - 3) * s0 * s0
	d := var4 / (var2 * var2)
	b := d * ((n*n-n)*s1 - 2*n*s2 + 6*s0*s0)

	v = (a-b)/c - e*e

	// Calculate z-score associated with Moran's I for the data.
	z = (i - e) / math.Sqrt(v)

	return i, v, z
}
