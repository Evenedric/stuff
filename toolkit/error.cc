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

#include "error.h"
#include "mystring.h"
#include "thread.h"
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <vector>

using std::vector;

//***************************************************************************
// The basic error handling mechanism.

void Error(const char *msg, ...) {
  va_list ap;
  va_start(ap, msg);
  GetErrorHandler()->HandleError(ErrorHandler::Error, msg, ap);
}

void Warning(const char *msg, ...) {
  va_list ap;
  va_start(ap, msg);
  GetErrorHandler()->HandleError(ErrorHandler::Warning, msg, ap);
}

void Message(const char *msg, ...) {
  va_list ap;
  va_start(ap, msg);
  GetErrorHandler()->HandleError(ErrorHandler::Message, msg, ap);
}

void Panic(const char *msg, ...) {
  va_list ap;
  va_start(ap, msg);
  GetErrorHandler()->HandleError(ErrorHandler::Panic, msg, ap);
  _exit(1);
}

void VError(const char *msg, va_list ap) {
  GetErrorHandler()->HandleError(ErrorHandler::Error, msg, ap);
}

void VWarning(const char *msg, va_list ap) {
  GetErrorHandler()->HandleError(ErrorHandler::Warning, msg, ap);
}

void VMessage(const char *msg, va_list ap) {
  GetErrorHandler()->HandleError(ErrorHandler::Message, msg, ap);
}

void VPanic(const char *msg, va_list ap) {
  GetErrorHandler()->HandleError(ErrorHandler::Panic, msg, ap);
  _exit(1);
}

ErrorHandler::~ErrorHandler() {
}

struct DefaultErrorHandler : public ErrorHandler {
  void HandleError(Type type, const char *msg, va_list ap);
};

void DefaultErrorHandler::HandleError(Type type, const char *msg, va_list ap) {
  fflush(stdout);
  fflush(stderr);
  if (type == ErrorHandler::Error) {
    fprintf(stderr, "Error: ");
  } else if (type == ErrorHandler::Warning) {
    fprintf(stderr, "Warning: ");
  } else if (type == ErrorHandler::Message) {
    fprintf(stderr, "Message: ");
  } else if (type == ErrorHandler::Panic) {
    fprintf(stderr, "Panic: ");
  } else {
    fprintf(stderr, "<unknown error type>: ");
  }
  vfprintf(stderr, msg, ap);
  fprintf(stderr, "\n");
  fflush(stderr);
  if (type == ErrorHandler::Panic) {
    // Call _exit() here, which immediately terminates the process, rather than
    // exit() which will do all kinds of housekeeping such as running global
    // destructors, closing files and calling atexit functions. The reason is
    // that the system might be in a bad state (we paniced after all) and doing
    // that housekeeping could result in bad stuff happening, e.g. infinite
    // recursions.
    _exit(1);
  }
}

static DefaultErrorHandler default_error_handler;
static ErrorHandler *error_handler = &default_error_handler;
static std::mutex error_handler_mutex;

ErrorHandler *SetErrorHandler(ErrorHandler *e) {
  MutexLock lock(&error_handler_mutex);
  ErrorHandler *ret = error_handler;
  error_handler = e;
  return ret;
}

ErrorHandler *GetErrorHandler() {
  MutexLock lock(&error_handler_mutex);
  return error_handler;
}

//***************************************************************************
// Complain-once mechanism.

static vector<bool*> complaint_flags;
static std::mutex complaint_flags_mutex;

bool ComplaintFlagToReset(bool *flag) {
  MutexLock lock(&complaint_flags_mutex);
  if (!*flag) {
    complaint_flags.push_back(flag);
    *flag = true;
    return true;
  } else {
    return false;
  }
}

void ResetComplaints() {
  MutexLock lock(&complaint_flags_mutex);
  for (int i = 0; i < complaint_flags.size(); i++) {
    *complaint_flags[i] = false;
  }
  complaint_flags.clear();
}

//***************************************************************************
// Time.

double Now() {
  struct timeval tp;
  gettimeofday(&tp, 0);
  return tp.tv_sec + tp.tv_usec * 1e-6;
}

//***************************************************************************
// qt error handling.

#ifdef QT_CORE_LIB

#include <QMessageBox>
#include "error_window.h"

qtErrorHandler::qtErrorHandler() {
  // We create one error window and then use it forever. This must be created
  // in the main GUI thread so the error windows has the correct thread
  // affinity, so we can reliably handle errors that originate from any thread.
  errwin = new ErrorWindow(0);
}

void qtErrorHandler::HandleError(Type type, const char *msg, va_list ap) {
  switch (type) {
    case ErrorHandler::Error:
      errwin->VAddLine(msg, ap, ErrorWindow::ERROR);
      break;
    case ErrorHandler::Warning:
      errwin->VAddLine(msg, ap, ErrorWindow::WARNING);
      break;
    case ErrorHandler::Message:
      errwin->VAddLine(msg, ap, ErrorWindow::MESSAGE);
      break;
    default: {
      // Print the error message to stderr first, in case we have lost the
      // ability to run the gui at this point. Then try to pop up a window,
      // then exit.
      std::string s;
      StringAppendV(&s, msg, ap);
      fprintf(stderr, "FATAL ERROR: ");
      fprintf(stderr, "%s\n", s.c_str());
      fflush(stdout);
      QMessageBox::critical(0, "Fatal error", s.c_str());
      // See the rationale in DefaultErrorHandler::HandleError() for this:
      _exit(1);
    }
  }
}

#endif  // QT_CORE_LIB
