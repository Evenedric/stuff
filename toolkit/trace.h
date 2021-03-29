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

// Simple performance monitoring, for measuring where all the time is spent.
// This is thread safe, although calling trace functions from multiple threads
// will not necessarily generate sensible results.

#ifndef __TRACE_H__
#define __TRACE_H__

#include <string>

// Start a new trace.
void TraceStart();

// Measure the time taken during the lifetime of this class, labelling it with
// 'what'.
class Trace {
 public:
  explicit Trace(const char *what);
  ~Trace();
 private:
  int slot_;
};

// Stop the trace and get the full trace report.
void TraceReport(std::string *report);

// Catch bug where variable name is omitted, e.g. Trace("foo");
#define Trace(x) static_assert(0, "Trace missing variable name");

#endif
