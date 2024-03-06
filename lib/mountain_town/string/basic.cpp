// @copyright Copyright 2024 Willard Lu
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "basic.h"

/// @brief 新新字符串替换源字符串中的旧字符串
/// @param ori 源字符串
/// @param old_str 旧字符串
/// @param new_str 新字符串
void StrReplace(string &ori, const string &old_str, const string &new_str) {
  if (old_str.empty() == true) return;  // 对于替换空值不作处理
  size_t pos = ori.find(old_str);
  while (pos != string::npos) {
    ori.replace(pos, old_str.length(), new_str);
    pos = ori.find(old_str, pos + new_str.length());
  }
}

/// @brief 拆分字符串
/// @param ori 源字符串
/// @param regex 分隔字符串
/// @return 拆分后的字符串
std::vector<std::string> SplitStr(string ori, string delim_str) {
  size_t pos_start = 0;
  size_t pos_end = delim_str.length();
  size_t delim_len = delim_str.length();
  string token;
  vector<string> res;
  // 当分隔字符串为空时返回内容也为空。有些拆分函数在这种情况下会把字符串拆分成单个的字符。
  if (delim_str.empty() == true) {
    return res;
  }

  while ((pos_end = ori.find(delim_str, pos_start)) != string::npos) {
    token = ori.substr(pos_start, pos_end - pos_start);
    pos_start = pos_end + delim_len;
    if (token.empty() == true) {
      continue;
    }
    res.push_back(token);
  }

  if (ori.substr(pos_start).empty() == false) {
    res.push_back(ori.substr(pos_start));
  }
  return res;
}

/// @brief 使用正则表达式拆分字符串
/// @param ori 源字符串
/// @param delim_str 正则表达式
/// @return 拆分后的字符串
/// @remark 使用正则表达式来拆分字符串比使用简单的字符串拆分更方便灵活，但代价是增加了很多运算量，
///         选用哪种方式来拆分应根据实际情况而定。
vector<string> SplitStrRegex(string ori, string delim_str) {
  std::regex delim{delim_str};
  std::vector<string> res;
  // 当分隔字符串为空时返回内容也为空。有些拆分函数在这种情况下会把字符串拆分成单个的字符。
  if (delim_str.empty() == true) {
    return res;
  }
  std::sregex_token_iterator end;
  std::sregex_token_iterator iter(ori.begin(), ori.end(), delim, -1);
  for (; iter != end; ++iter) {
    string split(*iter);
    if (split.size()) res.push_back(split);
  }
  return res;
}

/// @brief 去掉字符串左边指定的字符串
/// @param ori 源字符串
/// @param str 要去掉的字符串
/// @return 处理后的字符串
string LTrim(string ori, string str) {
  ori.erase(0, ori.find_first_not_of(str));
  return ori;
}

/// @brief 去掉字符串右边指定的字符串
/// @param ori 源字符串
/// @param str 要去掉的字符串
/// @return 处理后的字符串
string RTrim(string ori, string str) {
  ori.erase(ori.find_last_not_of(str) + 1);
  return ori;
}

/// @brief 去掉字符串左右两边指定的字符串
/// @param ori 源字符串
/// @param str 要去掉的字符串
/// @return 处理后的字符串
string Trim(string ori, string str) { return LTrim(RTrim(ori, str), str); }

/// @brief 转换字符数组为string
/// @param arr char数组
/// @param arr_size char数组尺寸
/// @param len 要转换的尺寸
/// @param start 起始位置，默认为0
/// @return 转换好的string
string CharArrConvertToString(char *arr, int arr_size, int len, int start) {
  if (start < 0 || start + len > arr_size) {
    return "";
  }
  string res;
  try {
    for (int i = start; i < start + len; i++) {
      res.append(1, arr[i]);
    }
  } catch (std::exception &e) {
    cout << e.what() << endl;
    return "";
  }
  return res;
}

/// @brief 把字符串字母转换为小写字母
/// @param str 源字符串
/// @return 转换后的字符串
string ConvertToLowerCase(const string &str) {
  string result = "";
  for (auto c : str) {
    result += std::tolower(c);
  }
  return result;
}