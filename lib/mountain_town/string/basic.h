// @copyright Copyright 2024 Willard Lu
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#ifndef STRING_BASIC_H_
#define STRING_BASIC_H_

#include <iostream>
#include <regex>
#include <vector>

using std::cerr;
using std::cout;
using std::endl;
using std::string;
using std::vector;

void StrReplace(string &ori, const string &old_str, const string &new_str);
vector<string> SplitStr(string ori, string delim_str);
vector<string> SplitStrRegex(string ori, string delim_str);
string LTrim(string ori, string str);
string RTrim(string ori, string str);
string Trim(string ori, string str);
string CharArrConvertToString(char *arr, int arr_size, int len, int start = 0);
string ConvertToLowerCase(const string& str);

#endif  // STRING_BASIC_H_