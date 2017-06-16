// Copyright ©2017 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package spatial_test

import (
	"fmt"

	"gonum.org/v1/gonum/matrix/mat64"
	"gonum.org/v1/gonum/stat/spatial"
)

func ExampleGlobalMoransI_1() {
	data := []float64{0, 0, 0, 1, 1, 1, 0, 1, 0, 0}
	locality := mat64.NewDense(10, 10, []float64{
		0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
		1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
		0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
		0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
		0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
		0, 0, 0, 0, 0, 1, 0, 1, 0, 0,
		0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
		0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
		0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
	})

	i, _, z := spatial.GlobalMoransI(data, locality)

	fmt.Printf("Moran's I=%.4v z-score=%.4v\n", i, z)

	// Output:
	//
	// Moran's I=0.1111 z-score=0.6335
}

func ExampleGetisOrd() {
	data := []float64{0, 0, 0, 1, 1, 1, 0, 1, 0, 0}
	locality := mat64.NewDense(10, 10, []float64{
		1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
		1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
		0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
		0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
		0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
		0, 0, 0, 0, 0, 1, 1, 1, 0, 0,
		0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
		0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
		0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
	})

	g := spatial.NewGetisOrd(data, locality)

	for i, v := range data {
		fmt.Printf("v=%v G*i=% .4v\n", v, g.Gstar(i))
	}

	// Output:
	//
	// v=0 G*i=-2.273
	// v=0 G*i=-2.807
	// v=0 G*i=-0.4678
	// v=1 G*i= 1.871
	// v=1 G*i= 4.21
	// v=1 G*i= 1.871
	// v=0 G*i= 1.871
	// v=1 G*i=-0.4678
	// v=0 G*i=-0.4678
	// v=0 G*i=-2.273
}
