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

#include "testing.h"
#include "random.h"
#include <stdio.h>
#include <string.h>
#include <vector>

struct TestFunction {
  void (*fn)() = 0;
  const char *name = 0, *filename = 0;
  int line_number = 0;
  bool runtest = false;     // True if test was mentioned on the command line
};

// This should be a pointer to a vector, not the vector itself, because
// otherwise this might be constructed *after* some calls to __RegisterTest
// which are also driven from global initializers.
static std::vector<TestFunction> *test_functions;

void testing::__RegisterTest(void (*fn)(), const char *name,
                             const char *filename, int line_number) {
  if (!test_functions) {
    test_functions = new std::vector<TestFunction>;
  }
  TestFunction t;
  t.fn = fn;
  t.name = name;
  t.filename = filename;
  t.line_number = line_number;
  test_functions->push_back(t);
}

void testing::RunAll() {
  // Make sure testing is deterministic:
  RandomSeed(0);

  // If any tests were mentioned on the command line then we will run only
  // those tests.
  bool run_subset = false;
  for (int i = 0; i < test_functions->size(); i++) {
    run_subset |= test_functions->at(i).runtest;
  }

  // Run all tests.
  for (int i = 0; i < test_functions->size(); i++) {
    if (!run_subset || test_functions->at(i).runtest) {
      printf("********** Test %s (%s:%d)\n", test_functions->at(i).name,
            test_functions->at(i).filename, test_functions->at(i).line_number);
      test_functions->at(i).fn();
    }
  }
  printf("%d tests passed!\n", (int) test_functions->size());
}

void testing::ProcessCommandLineArguments(int argc, char **argv) {
  if (!test_functions) {
    // No test functions are registered so nothing to do.
    return;
  }
  for (int i = 1; i < argc; i++) {
    if (strncmp(argv[i], "-runtest=", 9) == 0) {
      for (int j = 0; j < test_functions->size(); j++) {
        if (strcmp(test_functions->at(j).name, argv[i] + 9) == 0) {
          test_functions->at(j).runtest = true;
        }
      }
    }
  }
}

#ifdef __TOOLKIT_DEFINE_TESTING_MAIN__
int main(int argc, char **argv) {
  testing::ProcessCommandLineArguments(argc, argv);
  testing::RunAll();
  return 0;
}
#endif
