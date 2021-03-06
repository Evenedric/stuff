// Copyright (C) 2014-2021 Russell Smith.
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

#ifndef __TOOLKIT_STEP_H__
#define __TOOLKIT_STEP_H__

#include <vector>
#include <map>
#include <string>
#include "Eigen/Dense"

namespace step {

// The error handling strategy is to throw an Exception. In the Parser,
// whenever a file read or parse error occurs Error() is called internally to
// throw the Exception. All class and function calls must be wrapped in a
// try/catch block.

struct Exception : public std::exception {
  std::string message;
  int64_t instance_number;      // Message refers to this instance, if not 0
  Exception(const std::string &m = "", int64_t i = 0)
      : message(m), instance_number(i) {}
  const char* what() const noexcept override { return message.c_str(); }
};

// Tokenizer and parser.

class Parser {
 public:
  explicit Parser(const char *filename);
  ~Parser();

  // ********** Tokenizer.

  struct Token {
    bool is_string = false;     // Token is a string?
    bool is_number = false;     // Token is a number? (true also for integers)
    bool is_integer = false;    // That number is an int64_t compatible integer?
    bool is_end_of_file = false;  // Has end of file been reached?
    std::string word;           // If non empty, token is a word or single char
    std::string str;            // String, if token is a string (can be empty)
    double number = 0;          // Number, if token is a number or integer

    void Reset() {
      is_string = false;
      is_number = false;
      is_integer = false;
      is_end_of_file = false;
      word.clear();
      str.clear();
      number = 0;
    }
  };

  // Read the next token from the input stream into 'token_'.
  void NextToken();

  // Return the current token.
  const Token& GetToken() const { return token_; }

  // Return a string representation of the current token, for debugging.
  std::string TokenToString();

  // ********** Parser.

  struct DataInstance {
    std::string name;                   // Name of the instance
    std::vector<int64_t> refs;          // All references to other instances
    std::vector<double> numbers;        // All numbers
    bool boolean = false;               // The last boolean

    Eigen::Vector3d GetVector() const {
      if (numbers.size() != 3) {
        throw Exception("Expecting instance 3-vector");
      }
      return Eigen::Vector3d(numbers[0], numbers[1], numbers[2]);
    }
  };

  typedef std::map<int64_t, DataInstance> Database;

  void ParseFile(Database *database);

 private:
  FILE *fin_ = 0;               // Input file, closed when the end is reached
  int line_number_ = 1;         // Current line number
  Token token_;                 // Current token

  // Parser.
  void ParseHeaderSection();
  void ParseDataSection(Database *database);
  void ParseDataInstance(DataInstance *data);
  void ParseBracketedData(DataInstance *data);
  void ParseDataItemList(DataInstance *data);
  void ParseDataItem(DataInstance *data);
  void ParseUnknownInstance();
  void Expecting(const char *s);
  double ExpectingNumber();
  int64_t ExpectingInteger();
  int64_t ExpectingRef();

  // Log the error message, along with the line number and any errno value.
  // Close the input file, set fin_ to 0. Throw an exception.
  void Error(const char *msg, ...) __attribute__((noreturn));

  // Read the next input character, count lines.
  char GetC() {
    int c = getc_unlocked(fin_);
    line_number_ += (c == '\n');
    return c;
  }
};

// ********** Process the instance database in various ways.

// Write all data instance parent/child relationships to a 'dot' graph file.
// That file can be processed by:
//     dot -Tpng -ofoo.png filename.dot
// This generates a diagram of which instances refer to which other instances.
void WriteDotFile(const char *filename, const Parser::Database &database);

// A boundary edge in a face.
struct BoundaryEdge {
  Eigen::Vector3d start, end;           // Endpoints of edge, in boundary order
  Eigen::Vector3d center, normal;       // For circles and arcs

  struct Spline {
    std::vector<Eigen::Vector3d> p;       // Control points
    std::vector<int> knot_multiplicity;   // 4 at ends, 1 or 2 in middle
    std::vector<double> knot_position;    // 0..1, increasing
  };
  std::unique_ptr<Spline> spline;       // If nonempty, this is a spline

  // The type of geometry:
  //   * Lines go from start to end.
  //   * Circles are centered at 'center', with radius |start-center|, and the
  //     given normal vector.
  //   * Arcs go from start to end with the given center and normal. In many
  //     cases the normal can be inferred from the start-center-end plane,
  //     but this is not true for 180 degree arcs which is why the normal is
  //     given separately. Arcs can go clockwise or counterclockwise from the
  //     start. The radius of the arc is |start-center|.
  //   * If this is a spline, the spline pointer is nonnull.
  enum { LINE, ARC_CW, ARC_CCW, CIRCLE, SPLINE } type;
};

struct PlanarFace {
  Eigen::Vector3d normal;       // Normal vector of a planar face
  // All boundaries of the face.
  std::vector<std::vector<BoundaryEdge>> boundaries;

  void swap(PlanarFace &f) {
    normal.swap(f.normal);
    boundaries.swap(f.boundaries);
  }
};

// Extract all planar faces.
void ExtractPlanarFaces(const Parser::Database &database,
                        std::vector<PlanarFace> *faces);

}  // namespace step

#endif
