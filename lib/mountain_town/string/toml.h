// @copyright Copyright 2024 Willard Lu
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#ifndef STRING_TOML_H_
#define STRING_TOML_H_

#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "basic.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;

string ReadSTOML(string file, std::unordered_map<string, string> &config);
void ReadSTOMLArr(string str, vector<string> &arr);

#endif  // STRING_TOML_H_