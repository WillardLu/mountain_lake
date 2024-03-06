// @copyright Copyright 2024 Willard Lu
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#ifndef MATH_RANDOM_H_
#define MATH_RANDOM_H_

#include <random>

void BetterRand(int *rn, int size, int min, int max);
void NormalDistr(float *rn, int size, float mean, float stddev);

#endif  // MATH_RANDOM_H_