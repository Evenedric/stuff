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

// The error handling and logging mechanism to be used by all toolkit code. The
// main functions called by the user are Error(), Warning(), Message() and
// Panic(), which all take printf()-style arguments. All functions below are
// thread safe.

#ifndef __TOOLKIT_ERROR_H__
#define __TOOLKIT_ERROR_H__

#include <stdarg.h>

// The error generating functions.
void Error(const char *msg, ...) __attribute__((format(printf, 1, 2)));
void Warning(const char *msg, ...) __attribute__((format(printf, 1, 2)));
void Message(const char *msg, ...) __attribute__((format(printf, 1, 2)));
void Panic(const char *msg, ...)
  __attribute__((format(printf, 1, 2), noreturn));

// V-versions of those functions that take a va_list.
void VError(const char *msg, va_list ap);
void VWarning(const char *msg, va_list ap);
void VMessage(const char *msg, va_list ap);
void VPanic(const char *msg, va_list ap);

// An error handler object that actually does the work of error handling.
struct ErrorHandler {
  enum Type { Error, Warning, Message, Panic };
  virtual ~ErrorHandler();
  virtual void HandleError(Type type, const char *msg, va_list ap) = 0;
};

// There is a single global error handler object. By default it emits messages
// to stderr, and exits the program on Panic(). In GUI environments it can be
// set to more useful things. The SetErrorHandler() function atomically sets
// the error handler and returns the previous error handler. It can be used to
// chain error handlers together in a thread safe way.
ErrorHandler *SetErrorHandler(ErrorHandler *e);
ErrorHandler *GetErrorHandler();

// An error handler for Qt.
#ifdef QT_CORE_LIB
// This must be created after the QApplication is constructed.
class ErrorWindow;
struct qtErrorHandler : public ErrorHandler {
  qtErrorHandler();
  void HandleError(Type type, const char *msg, va_list ap);
  ErrorWindow *errwin;
};
#endif

// CHECK is like assert(), but always runs regardless of debug settings.
#define CHECK_MSG(cond, msg) if (!(cond)) \
  { Panic("Check failed: %s: %s, at %s:%d", (msg), #cond, __FILE__, __LINE__); }
#define CHECK(cond) CHECK_MSG(cond, "")

// DBG_CHECK is like CHECK but is only used in debug builds.
#ifdef NDEBUG
  #define DBG_CHECK_MSG(cond)
  #define DBG_CHECK(cond)
#else
  #define DBG_CHECK_MSG(cond, msg) CHECK_MSG(cond, msg)
  #define DBG_CHECK(cond) CHECK(cond)
#endif

// Convenience macros to help prevent errors.
#define MUST_USE_RESULT __attribute__ ((warn_unused_result))
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete; \
  void operator=(const TypeName&) = delete
#define MAY_NOT_BE_USED __attribute__((unused))

// If an error occurs frequently but is redundant after the first occurence,
// then the COMPLAIN_ONCE family of macros can be used. The complaint will be
// issued one time, until the next ResetComplaints(). This feature is thread
// safe.
#define COMPLAIN_ONCE(Function, format, args...) { \
  static bool complaint_issued = false; \
  if (ComplaintFlagToReset(&complaint_issued)) { \
    Function(format, ##args); \
  } \
}

#define MESSAGE_ONCE(format, args...) COMPLAIN_ONCE(Message, format, ##args)
#define WARNING_ONCE(format, args...) COMPLAIN_ONCE(Warning, format, ##args)
#define ERROR_ONCE(format, args...) COMPLAIN_ONCE(Error, format, ##args)

// Used internally by COMPLAIN_ONCE. If the flag is false then we need to issue
// the complaint, so set the flag to true, add it to the reset list and return
// true. Otherwise, return false.
bool ComplaintFlagToReset(bool *flag);

// Allow all COMPLAIN_ONCE call sites to issue new complaints. This clears all
// flags on the reset list that was populated by ComplaintFlagToReset.
void ResetComplaints();

// Return the number of seconds since the epoch (often useful for error
// handling).
double Now();

#endif
