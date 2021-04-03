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

// Parser for STEP files. The STEP file format was largely reverse engineered
// from example files, so this is likely not a completely conformant parser.
//
// STEP files are lexically and grammatically simple, requiring in most cases
// just a single character (or token) of lookahead, so we use a simple
// hand-written recursive descent parser here.
//
// STEP file structure is simple enough. There is a header (that we ignore) and
// a data section. The data section is make of numbered data instances that can
// contain some data (e.g. a 3-vector) and can refer to other instances by
// their number, so the file is actually a tree (or perhaps a directed acyclic
// graph) of instances. Each data instance has a type name. It is easy to skip
// over instances that we don't know how to process, by just ignoring parts of
// the tree that originate from unknown type names. The entire file must be
// loaded into memory before processing, as e.g. the first instance could have
// a reference to the last. As a representation of 3D objects, STEP files are
// overly verbose, e.g. a LINE instance must refer to a VECTOR, a
// CARTESIAN_POINT and a DIRECTION, rather than encoding the information
// directly. However the schema, though not ideal, is standardized and stable,
// and easy enough to figure out.
//
// TODO:
//
// The implementation here strives for simplicity at the expense of having a
// relatively large memory footprint for loaded files. In the future there are
// some simple things that could be done to improve the memory footprint for
// large files:
//   * We spend a lot of memory on redundantly storing type names in instance
//     nodes. We should store string indexes instead, and the strings
//     themselves only once.
//   * We are not efficiently storing coordinate data that makes up the bulk of
//     the file, i.e. we use make small vectors in the DataInstance structures.
//     We could collapse the leaf nodes of the instance tree into something
//     more compact as we load it.
// File loading performance right now is about 60% in the tokenizer, of which
// about half is simply fscanf() for numbers and 12% is ungetc(). So, that's
// hard to speed up without writing more custom file read code. The remainder
// of the load time is spent in the parser mostly doing STL map and vector
// manipulations - that could be sped up by reducing the memory footprint, as
// above.

#include "step.h"
#include <stdio.h>
#include <errno.h>
#include <set>

using std::vector;
using std::string;
using Eigen::Vector3d;

namespace step {

//---------------------------------------------------------------------------
// Parser.

Parser::Parser(const char *filename) {
  fin_ = fopen(filename, "rb");
  if (!fin_) {
    Error("Can not open file '%s'", filename);
  }

  // For efficiency we do all file reads using the unlocked versions of file IO
  // functions where possible. That means we should lock the file here.
  flockfile(fin_);
}

Parser::~Parser() {
  if (fin_) {
    fclose(fin_);
  }
}

void Parser::NextToken() {
  if (!fin_) {
    Error("Internal: attempt to read beyond end of file");
  }
  token_.Reset();

  // In case we call Error() below for a non-file related error:
  errno = 0;

  while (true) {
    if (ferror_unlocked(fin_)) {
      Error("Can not read from input file");
    }
    if (feof_unlocked(fin_)) {
      fclose(fin_);
      fin_ = 0;
      token_.is_end_of_file = true;
      return;
    }

    int c = GetC();

    switch (c) {
      case EOF:
        break;                  // Handle end of file above

      case ' ': case '\t': case '\r': case '\n':
        // Ignore whitespace.
        break;

      case '/': {
        // Skip over comments
        if (GetC() != '*') {
          Error("Malformed comment");
        }
        bool last_was_star = false;
        while (true) {
          c = GetC();
          if (c == EOF) {
            Error("File interrupted while reading comment");
          }
          if (last_was_star && c == '/') {
            break;
          }
          last_was_star = (c == '*');
        }
        break;
      }

      case '#': case ';': case ',': case '=': case '(': case ')': case '*':
      case '$': {
        // Single character tokens.
        token_.word = c;
        return;
      }

      case '\'': {
        // Quoted string.
        bool escaping = false;
        while (true) {
          c = GetC();
          if (c == EOF) {
            Error("File interrupted while reading string token");
          } else if (c == '\'') {
            if (!escaping) {
              token_.is_string = true;
              return;
            }
            token_.str.push_back(c);
          } else if (c == '\\') {
            // Escape the next character. Actually since we don't do anything
            // useful with strings yet, we leave the escape sequence in there.
            token_.str.push_back('\\');
            escaping = true;
            continue;
          } else {
            token_.str.push_back(c);
          }
          escaping = false;
        }
        Error("Internal: this should be unreachable");
      }

      case '0' ... '9': case '-': case '.': {
        // Number. The dot . is the only character where one character of
        // lookahead can not be used to decide what the token is, because there
        // are word tokens like '.T.' (representing 'true'), etc. However the
        // character after the dot can be used to make this distinction.
        if (c == '.') {
          int c2 = GetC();
          if (c2 == EOF) {
            Error("File interrupted while reading dotted token");
          }
          ungetc(c2, fin_);             // No unlocked version exists
          if (c2 < '0' || c2 > '9') {
            // This is not a number, it's probably a word.
            goto read_word;     // 'c' will still have the value '.'
          }
        }
        ungetc(c, fin_);        // Put 'c' back on the stream for fscanf

        if (fscanf(fin_, "%lf", &token_.number) != 1) {
          Error("Could not parse number");
        }
        token_.is_number = true;
        token_.is_integer = int64_t(token_.number) == token_.number;
        return;
      }

      case 'A' ... 'Z': case 'a' ... 'z': case '_': read_word: {
        // Word.
        token_.word = c;
        while (true) {
          c = GetC();
          if (c == EOF) {
            if (!ferror_unlocked(fin_)) {
              return;           // Terminal word without whitespace
            }
            break;              // Otherwise handle error above
          } else if ( (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                      (c >= '0' && c <= '9') || c == '_' || c == '-' ||
                       c == '.') {
            token_.word.push_back(c);
          } else {
            ungetc(c, fin_);
            return;
          }
        }
      }

      default:
        if (isprint(c)) {
          Error("Invalid character '%c'", c);
        } else {
          Error("Invalid character 0x%02x", c);
        }
    }
  }
}

string Parser::TokenToString() {
  char *s = 0;
  while (true) {
    if (token_.is_end_of_file) {
      return "end of file";
    } else if (!token_.word.empty()) {
      return "word " + token_.word;
    } else if (token_.is_string) {
      return "string '" + token_.str + "'";
    } else if (token_.is_number && token_.is_integer) {
      asprintf(&s, "integer %.0f", token_.number);
      string r = s;
      free(s);
      return r;
    } else if (token_.is_number) {
      asprintf(&s, "number %.10g", token_.number);
      string r = s;
      free(s);
      return r;
    } else {
      Error("Internal: unknown token (%d,%d,%d,%d,%d)", token_.is_string,
            token_.is_number, token_.is_integer, token_.is_end_of_file,
            !token_.word.empty());
    }
  }
}

void Parser::ParseFile(Database *database) {
  NextToken();
  Expecting("ISO-10303-21");
  Expecting(";");
  ParseHeaderSection();
  ParseDataSection(database);
  Expecting("END-ISO-10303-21");
  Expecting(";");
  if (!token_.is_end_of_file) {
    Error("Expecting end of file");
  }
  line_number_ = 0;
}

void Parser::ParseHeaderSection() {
  Expecting("HEADER");
  Expecting(";");
  while (token_.word != "ENDSEC") {
    // We skip things like FILE_DESCRIPTION, FILE_NAME, FILE_SCHEMA.
    ParseUnknownInstance();
  }
  Expecting("ENDSEC");
  Expecting(";");
}

void Parser::ParseDataSection(Database *database) {
  Expecting("DATA");
  Expecting(";");
  while (token_.word != "ENDSEC") {
    Expecting("#");
    int64_t instance_number = ExpectingInteger();
    if (database->count(instance_number) > 0) {
      Error("Instance number defined twice");
    }
    Expecting("=");
    ParseDataInstance(&(*database)[instance_number]);
  }
  Expecting("ENDSEC");
  Expecting(";");
}

void Parser::ParseDataInstance(DataInstance *data) {
  // Parse a generic instance.
  if (token_.word.empty()) {
    Error("Expecting a word");
  }
  data->name = token_.word;
  if (token_.word == "(") {
    // Things are not separated by commas in this sort of list, just scan to
    // the end by matching brackets, but otherwise ignore.
    ParseUnknownInstance();
  } else {
    NextToken();
    ParseBracketedData(data);
    Expecting(";");
  }
}

void Parser::ParseBracketedData(DataInstance *data) {
  Expecting("(");
  if (token_.word != ")") {
    ParseDataItemList(data);
  }
  Expecting(")");
}

void Parser::ParseDataItemList(DataInstance *data) {
  ParseDataItem(data);
  while (token_.word == ",") {
    NextToken();
    ParseDataItem(data);
  }
}

void Parser::ParseDataItem(DataInstance *data) {
  if (token_.is_string) {
    // Ignore strings.
    NextToken();
  } else if (token_.word == "#") {
    data->refs.push_back(ExpectingRef());
  } else if (token_.is_number) {
    data->numbers.push_back(ExpectingNumber());
  } else if (token_.word == ".T.") {
    data->boolean = true;
    NextToken();
  } else if (token_.word == ".F.") {
    data->boolean = false;
    NextToken();
  } else if (token_.word == "(") {
    ParseBracketedData(data);
  } else if (!token_.word.empty()) {
    NextToken();
    if (token_.word == "(") {
      ParseBracketedData(data);
    }
  } else {
    Error("Unknown data item %s", TokenToString().c_str());
  }
}

void Parser::ParseUnknownInstance() {
  // Skip over an instance with an unknown grammar, just by matching brackets
  // and looking for the terminal ';'.
  if (token_.word.empty()) {
    Error("Expecting a word");
  }
  int depth = (token_.word == "(");
  while (true) {
    NextToken();
    if (token_.word == "(") {
      depth++;
    } else if (token_.word == ")") {
      depth--;
      if (depth < 0) {
        Error("Mismatched brackets");
      }
    } else if (token_.word == ";") {
      if (depth == 0) {
        NextToken();
        return;
      }
    }
  }
}

void Parser::Expecting(const char *s) {
  if (token_.word != s) {
    Error("Expecting '%s', instead got %s", s, TokenToString().c_str());
  }
  NextToken();
}

double Parser::ExpectingNumber() {
  if (!token_.is_number) {
    Error("Expecting a number, instead got %s", TokenToString().c_str());
  }
  double n = token_.number;
  NextToken();
  return n;
}

int64_t Parser::ExpectingInteger() {
  if (!token_.is_integer) {
    Error("Expecting an integer, instead got %s", TokenToString().c_str());
  }
  int64_t n = token_.number;
  NextToken();
  return n;
}

int64_t Parser::ExpectingRef() {
  Expecting("#");
  return ExpectingInteger();
}

void Parser::Error(const char *msg, ...) {
  int errno_value = errno;
  Exception e;

  if (fin_) {
    fclose(fin_);
    fin_ = 0;
  }

  va_list ap;
  va_start(ap, msg);
  char *s = 0;
  vasprintf(&s, msg, ap);
  e.message = s;
  free(s);

  if (line_number_ > 0) {
    asprintf(&s, " at line %d", line_number_);
    e.message += s;
    free(s);
  }
  if (errno_value != 0) {
    asprintf(&s, " (%s)", strerror(errno_value));
    e.message += s;
    free(s);
  }

  throw e;
}

//---------------------------------------------------------------------------
// Functions.

static void Validate(const Parser::Database &database) {
  for (auto &it : database) {
    // Check that ADVANCED_FACE is a list of FACE_BOUND instances followed by a
    // face geometry (of which we only check PLANE).
    if (it.second.name == "ADVANCED_FACE") {
      if (it.second.refs.size() < 2) {
        throw Exception("ADVANCED_FACE needs at least 2 references", it.first);
      }
      for (int i = 0; i < it.second.refs.size() - 1; i++) {
        auto &facebound = database.at(it.second.refs[i]);
        if (facebound.name != "FACE_BOUND" &&
            facebound.name != "FACE_OUTER_BOUND") {
          throw Exception("ADVANCED_FACE doesn't contain all FACE_BOUND",
                          it.first);
        }
      }
      // Make sure the face geometry reference exists.
      database.at(it.second.refs.back());
    }

    // Check that PLANEs are an AXIS2_PLACEMENT_3D
    if (it.second.name == "PLANE") {
      auto &plane = it.second;
      if (plane.refs.size() != 1) {
        throw Exception("PLANE invalid", it.first);
      }
      auto &axis2 = database.at(plane.refs[0]);
      if (axis2.name != "AXIS2_PLACEMENT_3D") {
        throw Exception("PLANE doesn't ref AXIS2_PLACEMENT_3D", it.first);
      }
    }

    // Check that AXIS2_PLACEMENT_3D contains a CARTESIAN_POINT and DIRECTION.
    if (it.second.name == "AXIS2_PLACEMENT_3D") {
      auto &axis2 = it.second;
      if (axis2.refs.size() != 3) {
        throw Exception("AXIS2_PLACEMENT_3D invalid", it.first);
      }
      auto &location = database.at(axis2.refs[0]);
      auto &normal   = database.at(axis2.refs[1]);
      auto &axis1    = database.at(axis2.refs[2]);
      if (location.name != "CARTESIAN_POINT" ||
          location.numbers.size() != 3 ||
          normal.name != "DIRECTION" || normal.numbers.size() != 3 ||
          axis1.name != "DIRECTION" || axis1.numbers.size() != 3) {
        throw Exception("AXIS2_PLACEMENT_3D children bad", it.first);
      }
    }

    // Starting with FACE_BOUND or FACE_OUTER_BOUND instances, validate the
    // schema of the instances below them in the tree, stopping when we get to
    // things that have been validated above.
    if (it.second.name == "FACE_BOUND" ||
        it.second.name == "FACE_OUTER_BOUND") {
      // A FACE_BOUND is a reference to an EDGE_LOOP, plus an orientation flag
      // that always seem to be true.
      if (it.second.boolean != true) {
        throw Exception("FACE_BOUND orientation unexpectedly false", it.first);
      }
      if (it.second.refs.size() != 1) {
        throw Exception("FACE_BOUND expecting one child", it.first);
      }
      auto &edge_loop = database.at(it.second.refs[0]);
      // The EDGE_LOOP is a list of references to ORIENTED_EDGE.
      if (edge_loop.name != "EDGE_LOOP") {
        throw Exception("FACE_BOUND child is not an EDGE_LOOP", it.first);
      }
      for (int i = 0; i < edge_loop.refs.size(); i++) {
        auto &oriented_edge = database.at(edge_loop.refs[i]);
        // An ORIENTED_EDGE is a reference to an EDGE_CURVE, plus an
        // orientation flag.
        if (oriented_edge.name != "ORIENTED_EDGE") {
          throw Exception("EDGE_LOOP child is not an ORIENTED_EDGE",
                          it.second.refs[0]);
        }
        if (oriented_edge.refs.size() != 1) {
          throw Exception("ORIENTED_EDGE expecting one child",
                          edge_loop.refs[i]);
        }
        auto &edge_curve = database.at(oriented_edge.refs[0]);
        // An EDGE_CURVE is three references (start, end, geometry) and a
        // same_sense flag.
        if (edge_curve.name != "EDGE_CURVE") {
          throw Exception("ORIENTED_EDGE child is not an EDGE_CURVE",
                          edge_loop.refs[i]);
        }
        if (edge_curve.refs.size() != 3) {
          throw Exception("EDGE_CURVE excepting three children",
                          oriented_edge.refs[0]);
        }
        auto &v1 = database.at(edge_curve.refs[0]);  // Edge start
        auto &v2 = database.at(edge_curve.refs[1]);  // Edge end
        if (v1.name != "VERTEX_POINT" || v1.refs.size() != 1 ||
            v2.name != "VERTEX_POINT" || v2.refs.size() != 1) {
          throw Exception("EDGE_CURVE start/end are not correct VERTEX_POINTs",
                          oriented_edge.refs[0]);
        }
        auto &p1 = database.at(v1.refs[0]);  // Edge start point
        auto &p2 = database.at(v2.refs[0]);  // Edge end point
        if (p1.name != "CARTESIAN_POINT" || !p1.refs.empty() ||
            p2.name != "CARTESIAN_POINT" || !p2.refs.empty() ||
            p1.numbers.size() != 3 || p2.numbers.size() != 3) {
          throw Exception("EDGE_CURVE start/end don't refer to correct "
                          "CARTESIAN_POINTs", oriented_edge.refs[0]);
        }
        // The EDGE_CURVE geometry can be a variety of things. We handle lines,
        // circles (which includes arcs), and splines.
        auto &geom = database.at(edge_curve.refs[2]);
        if (geom.name == "LINE") {
          // A LINE refers to a VECTOR, a CARTESIAN_POINT and a DIRECTION.
          // However those are redundant as we already know the line goes from
          // edge_start to edge_end. Don't bother validating below this point.
        } else if (geom.name == "CIRCLE") {
          if (geom.numbers.size() != 1 || geom.refs.size() != 1) {
            throw Exception("CIRCLE invalid", edge_curve.refs[2]);
          }
          auto &axis2 = database.at(geom.refs[0]);
          if (axis2.name != "AXIS2_PLACEMENT_3D") {
            throw Exception("AXIS2_PLACEMENT_3D invalid", geom.refs[0]);
          }
        } else if (geom.name == "B_SPLINE_CURVE_WITH_KNOTS") {
          // We handle splines of the form:
          //   B_SPLINE_CURVE_WITH_KNOTS('',
          //     degree,                    // Expected to be 3
          //     (ref_list_of_points),      // Control points
          //     .UNSPECIFIED.,             // curve_form
          //     .F.,                       // closed_curve
          //     .F.,                       // self_intersect
          //     (knot_multiplicity_list),  // 4 at ends, 1,2,3 in middle
          //     (knot_position_list),      // Knot positions (0..1)
          //     .UNSPECIFIED.);            // knot_spec
          //
          // @@@ Validate or handle curve_form, closed_curve, self_intersect,
          //     knot_spec. Handle degree==1 (these are straight lines).
          if (geom.numbers.size() < 5 || (geom.numbers.size() & 1) == 0 ||
              geom.numbers[0] != 3 /* degree == 3 */) {
            throw Exception("B_SPLINE_CURVE_WITH_KNOTS invalid (1)",
                              edge_curve.refs[2]);
          }
          int N = (geom.numbers.size() - 1) / 2;    // Number of knots
          int total_multiplicity = 0;
          for (int j = 0; j < N; j++) {
            int multiplicity = geom.numbers[1 + j];
            bool at_end = (j == 0 || j == N-1);
            if (!( (at_end && multiplicity == 4) ||
                   (!at_end && (multiplicity >= 1 && multiplicity <= 3)) )) {
              throw Exception("B_SPLINE_CURVE_WITH_KNOTS invalid (2)",
                                edge_curve.refs[2]);
            }
            total_multiplicity += multiplicity;
          }
          if (geom.refs.size() != total_multiplicity - 4) {
            throw Exception("B_SPLINE_CURVE_WITH_KNOTS invalid (3)",
                            edge_curve.refs[2]);
          }

          // Make sure all the refs are to CARTESIAN_POINT.
          for (int j = 0; j < geom.refs.size(); j++) {
            auto &p = database.at(geom.refs[j]);
            if (p.name != "CARTESIAN_POINT") {
              throw Exception("B_SPLINE_CURVE_WITH_KNOTS refs to non-point",
                              edge_curve.refs[2]);
            }
          }
        }
      }
    }
  }
}

void WriteDotFile(const char *filename, const Parser::Database &data) {
  FILE *fout = fopen(filename, "wb");
  if (!fout) {
    throw Exception("Can't write to file");
  }

  fprintf(fout, "digraph Foo {\n");
  fprintf(fout, "  node [shape=box,style=filled,fillcolor=\"#ffff80\"];\n");
  std::set<std::pair<string, string>> dotmap;
  for (auto &it : data) {
    for (int i = 0; i < it.second.refs.size(); i++) {
      int64_t ref = it.second.refs[i];
      if (data.count(ref) == 0) {
        throw Exception("Bad reference from instance");
      }
      dotmap.insert(std::make_pair(data.at(it.first).name, data.at(ref).name));
    }
  }
  for (auto &it : dotmap) {
    fprintf(fout, "  %s -> %s;\n",
            it.first == "(" ? "group" : it.first.c_str(),
            it.second == "(" ? "group" : it.second.c_str());
  }
  fprintf(fout, "}\n");
  fclose(fout);
}

void ExtractPlanarFaces(const Parser::Database &database,
                        std::vector<PlanarFace> *faces) {
  // We start with ADVANCED_FACE instances and recurse the trees below them.
  // Those trees are ridiculously verbose and have some oddities to deal with.
  // A face bound is a curve that is traversed in a particular direction. Each
  // ORIENTED_EDGE of the curve has an orientation flag that indicates if the
  // start and end vertices of the edge are swapped. Then each EDGE_CURVE has a
  // same_sense flag indicates whether the senses of the edge and the curve
  // defining the edge geometry are the same. The sense of an edge is from the
  // edge start vertex to the edge end vertex; the sense of a curve is in the
  // direction of increasing parameter.

  Validate(database);           // Validate all the assumptions made below
  for (auto &it : database) {
    if (it.second.name == "ADVANCED_FACE") {
      // ADVANCED_FACE is a list of FACE_BOUND instances followed by a face
      // geometry. We only accept PLANE.
      auto &advanced_face = it.second;
      auto &plane = database.at(advanced_face.refs.back());
      if (plane.name != "PLANE") {
        continue;
      }
      faces->resize(faces->size() + 1);
      auto &axis2 = database.at(plane.refs[0]);
      faces->back().normal = database.at(axis2.refs[1]).GetVector();
      for (int fi = 0; fi < advanced_face.refs.size() - 1; fi++) {
        auto &face_bound = database.at(advanced_face.refs[fi]);
        faces->back().boundaries.resize(faces->back().boundaries.size() + 1);
        vector<BoundaryEdge> &boundary = faces->back().boundaries.back();

        auto &edge_loop = database.at(face_bound.refs[0]);
        for (int i = 0; i < edge_loop.refs.size(); i++) {
          auto &oriented_edge = database.at(edge_loop.refs[i]);
          auto &edge_curve = database.at(oriented_edge.refs[0]);
          auto &geom = database.at(edge_curve.refs[2]);
          Vector3d edge_start = database.at(
                  database.at(edge_curve.refs[0]).refs[0]).GetVector();
          Vector3d edge_end = database.at(
                  database.at(edge_curve.refs[1]).refs[0]).GetVector();
          if (!oriented_edge.boolean) {
            edge_start.swap(edge_end);
          }

          if (geom.name == "CIRCLE") {
            // Circles and arcs. The normal and axis1 are assumed to be unit
            // length and perpendicular, the instance radius is assumed to be
            // |location-edge_start|. Don't actually check the radius though
            // because in some STEP files it can be a bit wrong (we've seen it
            // a factor of 1e-5 off).
            auto &axis = database.at(geom.refs[0]);
            Vector3d location = database.at(axis.refs[0]).GetVector();
            Vector3d normal = database.at(axis.refs[1]).GetVector();
            Vector3d axis1 = database.at(axis.refs[2]).GetVector();
            if (fabs(normal.norm() - 1) > 1e-9 ||
                fabs(axis1.dot(normal)) > 1e-9) {
              throw Exception("Bad CIRCLE", edge_curve.refs[2]);
            }
            boundary.resize(boundary.size() + 1);
            boundary.back().start = edge_start;
            boundary.back().end = edge_end;
            boundary.back().center = location;
            boundary.back().normal = normal;
            if (edge_start == edge_end) {
              boundary.back().type = BoundaryEdge::CIRCLE;
            } else if (edge_curve.boolean ^ (!oriented_edge.boolean)) {
              boundary.back().type = BoundaryEdge::ARC_CCW;
            } else {
              boundary.back().type = BoundaryEdge::ARC_CW;
            }
          } else if (geom.name == "B_SPLINE_CURVE_WITH_KNOTS") {
            int N = (geom.numbers.size() - 1) / 2;    // Number of knots
            boundary.resize(boundary.size() + 1);
            boundary.back().start = edge_start;
            boundary.back().end = edge_end;
            boundary.back().type = BoundaryEdge::SPLINE;
            auto *s = new BoundaryEdge::Spline;
            boundary.back().spline.reset(s);
            s->p.resize(geom.refs.size());
            s->knot_multiplicity.resize(N);
            s->knot_position.resize(N);
            for (int j = 0; j < geom.refs.size(); j++) {
              s->p[j] = database.at(geom.refs[j]).GetVector();
            }
            for (int j = 0; j < N; j++) {
              s->knot_multiplicity[j] = geom.numbers[1 + j];
              s->knot_position[j] = geom.numbers[1 + N + j];
            }
          } else {
            // Represent LINE plus all other unhandled geometry as line
            // segments.
            boundary.resize(boundary.size() + 1);
            boundary.back().start = edge_start;
            boundary.back().end = edge_end;
            boundary.back().type = BoundaryEdge::LINE;
          }
        }
      }
    }
  }
}

}  // namespace step
