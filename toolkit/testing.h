// Copyright (C) 2014-2020 Russell Smith.
//
// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option)
// any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
// more details.

// Simple unit testing infrastructure.

#ifndef __TOOLKIT_TESTING_H__
#define __TOOLKIT_TESTING_H__

namespace testing {

  // Run all unit tests.
  void RunAll();

  // Process the command line arguments looking for flags of the form
  // "-runtest=name". If any are found, run only those tests.
  void ProcessCommandLineArguments(int argc, char **argv);

  // Internal function called by TEST_FUNCTION.
  void __RegisterTest(void (*fn)(), const char *name,
                      const char *filename, int line_number);

}

// Define a new test function.
#define TEST_FUNCTION(name) \
  void __RunTest_##name(); \
  static struct __RegisterTest_##name { \
    __RegisterTest_##name() { \
      testing::__RegisterTest(__RunTest_##name, #name, __FILE__, __LINE__); \
    } \
  } __register_test_##name; \
  void __RunTest_##name()

#endif
