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

// Multithreading primitives.

#ifndef __TOOLKIT_THREAD_H__
#define __TOOLKIT_THREAD_H__

#include <functional>
#include <mutex>
#include "error.h"

// Ensure a lock is released when a MutexLock goes out of scope. This is a bit
// cleaner to use than std::lock_guard.
class MutexLock {
 public:
  explicit MutexLock(std::mutex *mutex) : mu_(mutex) { mu_->lock(); }
  ~MutexLock() { mu_->unlock(); }

 private:
  std::mutex *mu_;
  DISALLOW_COPY_AND_ASSIGN(MutexLock);
};

// Catch bug where variable name is omitted, e.g. MutexLock(&mu);
#define MutexLock(x) static_assert(0, "MutexLock missing variable name");

// Run the function 'fn' from 'n' threads with integer arguments in the range
// [first..last].
void ParallelFor(int first, int last, int n, std::function<void(int)> fn);

#endif
