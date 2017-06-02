// Copyright ©2017 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package spatial_test

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/matrix/mat64"
	"gonum.org/v1/gonum/stat/spatial"
)

type Euclid struct{ x, y int }

func (e Euclid) Dims() (r, c int) { return e.x * e.y, e.x * e.y }
func (e Euclid) At(i, j int) float64 {
	d := e.x * e.y
	if i < 0 || d <= i || j < 0 || d <= j {
		panic("bounds error")
	}
	if i == j {
		return 0
	}
	x := float64(j%e.x - i%e.x)
	y := float64(j/e.x - i/e.x)
	return 1 / math.Hypot(x, y)
}
func (e Euclid) T() mat64.Matrix { return mat64.Transpose{e} }

func ExampleMoran_2() {
	locality := Euclid{10, 10}

	data1 := []float64{
		1, 0, 0, 1, 0, 0, 1, 0, 0, 0,
		0, 1, 1, 0, 0, 1, 0, 0, 0, 0,
		1, 0, 0, 1, 0, 0, 0, 0, 1, 0,
		0, 0, 1, 0, 1, 0, 1, 0, 0, 0,
		1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
		0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
		0, 0, 1, 0, 0, 0, 1, 0, 1, 0,
		1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 1, 0, 1, 0, 1, 0, 0, 0,
		1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
	}
	m1 := spatial.NewMoran(data1, locality)

	data2 := []float64{
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
		0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
		0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
		0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
		0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
		0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
		0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
	}
	m2 := spatial.NewMoran(data2, locality)

	fmt.Printf("%v sparse points Moran's I=%.4v z-score=%.4v\n", floats.Sum(data1), m1.I(), m1.Z())
	fmt.Printf("%v clustered points Moran's I=%.4v z-score=%.4v\n", floats.Sum(data2), m2.I(), m2.Z())

	// Output:
	//
	// 24 sparse points Moran's I=-0.02999 z-score=-1.913
	// 24 clustered points Moran's I=0.09922 z-score=10.52
}
