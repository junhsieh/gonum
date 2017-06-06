// Copyright ©2017 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package spatial // import "gonum.org/v1/gonum/stat/spatial"

import (
	"math"

	"gonum.org/v1/gonum/matrix/mat64"
	"gonum.org/v1/gonum/stat"
)

// GetisOrd performs Local Getis-Ord G*i statistic calculation.
type GetisOrd struct {
	data     *mat64.Vector
	locality *mat64.Dense

	mean, s float64
}

// NewGetisOrd returns a GetisOrd based on the provided data and locality matrix.
// NewGetisOrd will panic if locality is not a square matrix with dimensions the
// same as the length of data.
func NewGetisOrd(data []float64, locality mat64.Matrix) GetisOrd {
	r, c := locality.Dims()
	if r != len(data) || c != len(data) {
		panic("spatial: data length mismatch")
	}

	var g GetisOrd
	d := make([]float64, len(data))
	copy(d, data)
	g.data = mat64.NewVector(len(data), d)

	g.locality = mat64.DenseCopyOf(locality)
	g.mean = stat.Mean(data, nil)
	var ss float64
	for _, v := range data {
		ss += v * v
	}
	g.s = ss/float64(len(data)) - g.mean*g.mean

	return g
}

// Reset sets the receiver's data and locality. data and locality must match
// dimensions as for NewGetisOrd.
func (g *GetisOrd) Reset(data []float64, locality mat64.Matrix) {
	r, c := locality.Dims()
	if r != len(data) || c != len(data) {
		panic("spatial: data length mismatch")
	}
	// FIXME(kortschak): There is no way to grow a Vector.
	// Change this to reduce allocation load when that is added.
	d := append(g.data.RawVector().Data[:0], data...)
	g.data = mat64.NewVector(len(data), d)

	g.locality.Reset()
	g.locality = g.locality.Grow(r, c).(*mat64.Dense)
	g.locality.Copy(locality)

	g.mean = stat.Mean(data, nil)
	var ss float64
	for _, v := range data {
		ss += v * v
	}
	g.s = ss/float64(len(data)) - g.mean*g.mean
}

// SetData sets the receiver's data. The lenght of data must match the existing
// locality matrix held by the receiver.
func (g *GetisOrd) SetData(data []float64) {
	r, c := g.locality.Dims()
	if r != len(data) || c != len(data) {
		panic("spatial: data length mismatch")
	}
	copy(g.data.RawVector().Data, data)

	g.mean = stat.Mean(data, nil)
	var ss float64
	for _, v := range data {
		ss += v * v
	}
	g.s = ss/float64(len(data)) - g.mean*g.mean
}

// SetLocality sets the receiver's locality matrix. The dimensions of locality must
// match the existing data slice held by the receiver.
func (g *GetisOrd) SetLocality(locality mat64.Matrix) {
	r, c := locality.Dims()
	if r != g.data.Len() || c != g.data.Len() {
		panic("spatial: data length mismatch")
	}
	g.locality.Copy(locality)
}

// Len returns the number of data points held by the receiver.
func (g GetisOrd) Len() int { return g.data.Len() }

// Gstar returns the Local Getis-Ord G* statistic for element i. The
// returned value is a z-score.
func (g GetisOrd) Gstar(i int) float64 {
	wi := g.locality.RowView(i)
	ws := g.mean * mat64.Sum(wi)
	n := float64(g.data.Len())
	num := mat64.Dot(wi, g.data) - ws
	den := g.s * math.Sqrt((n*mat64.Dot(wi, wi)-ws*ws)/(n-1))
	return num / den
}

// Moran performs Global Moran's I calculation of spatial autocorrelation.
//
// See https://en.wikipedia.org/wiki/Moran%27s_I.
type Moran struct {
	data     []float64
	mean     float64
	locality *mat64.Dense
}

// NewMoran returns a new Moran based on the provided data and locality matrix.
// NewMoran will panic if locality is not a square matrix with dimensions the
// same as the length of data.
func NewMoran(data []float64, locality mat64.Matrix) Moran {
	r, c := locality.Dims()
	if r != len(data) || c != len(data) {
		panic("spatial: data length mismatch")
	}
	var m Moran
	m.data = make([]float64, len(data))
	copy(m.data, data)
	m.mean = stat.Mean(m.data, nil)
	m.locality = mat64.DenseCopyOf(locality)

	return m
}

// Reset sets the receiver's data and locality. data and locality must match
// dimensions as for NewMoran.
func (m *Moran) Reset(data []float64, locality mat64.Matrix) {
	r, c := locality.Dims()
	if r != len(data) || c != len(data) {
		panic("spatial: data length mismatch")
	}
	m.data = append(m.data[:0], data...)

	m.locality.Reset()
	m.locality = m.locality.Grow(r, c).(*mat64.Dense)
	m.locality.Copy(locality)

	m.mean = stat.Mean(m.data, nil)
}

// SetData sets the receiver's data. The lenght of data must match the existing
// locality matrix held by the receiver.
func (m *Moran) SetData(data []float64) {
	r, c := m.locality.Dims()
	if r != len(data) || c != len(data) {
		panic("spatial: data length mismatch")
	}
	copy(m.data, data)
	m.mean = stat.Mean(m.data, nil)
}

// SetLocality sets the receiver's locality matrix. The dimensions of locality must
// match the existing data slice held by the receiver.
func (m *Moran) SetLocality(locality mat64.Matrix) {
	r, c := locality.Dims()
	if r != len(m.data) || c != len(m.data) {
		panic("spatial: data length mismatch")
	}
	m.locality.Copy(locality)
}

// I returns Moran's I for the data represented by the receiver.
func (m Moran) I() float64 {
	var num, den float64
	for i, xi := range m.data {
		zi := xi - m.mean
		den += zi * zi
		for j, xj := range m.data {
			zj := xj - m.mean
			num += m.locality.At(i, j) * zi * zj
		}
	}
	return (float64(len(m.data)) * num) / (mat64.Sum(m.locality) * den)
}

// E returns Moran's E(I) for the data represented by the receiver.
func (m Moran) E() float64 {
	return -1 / float64(len(m.data)-1)
}

// Var returns Moran's Var(I) for the data represented by the receiver.
func (m Moran) Var() float64 {
	// From http://pro.arcgis.com/en/pro-app/tool-reference/spatial-statistics/h-how-spatial-autocorrelation-moran-s-i-spatial-st.htm
	// and http://pro.arcgis.com/en/pro-app/tool-reference/spatial-statistics/h-global-morans-i-additional-math.htm

	var s0, s1, s2 float64
	var var2, var4 float64
	for i, v := range m.data {
		v -= m.mean
		v *= v
		var2 += v
		var4 += v * v

		var p2 float64
		for j := range m.data {
			wij := m.locality.At(i, j)
			wji := m.locality.At(j, i)

			s0 += wij

			v := wij + wji
			s1 += v * v

			p2 += wij + wji
		}
		s2 += p2 * p2
	}
	s1 *= 0.5

	n := float64(len(m.data))
	a := n * ((n*n-3*n+3)*s1 - n*s2 + 3*s0*s0)
	c := (n - 1) * (n - 2) * (n - 3) * s0 * s0
	d := var4 / (var2 * var2)
	b := d * ((n*n-n)*s1 - 2*n*s2 + 6*s0*s0)

	e := m.E()

	return (a-b)/c - e*e
}

// Z returns the z-score associated with Moran's I for the data represented by the receiver.
func (m Moran) Z() float64 {
	return (m.I() - m.E()) / math.Sqrt(m.Var())
}
