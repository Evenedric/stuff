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

// Standard shaders. The standard shader variables are:
// * Uniform "transform", which should be set to the projection matrix times
//   the modelview matrix.
// * Vertex attribute "vertex", the vec3 coordinates of the vertex.
// * In shaders with per-vertex color, vec3 vertex attribute "vertex_color".

#ifndef __TOOLKIT_SHADERS_H__
#define __TOOLKIT_SHADERS_H__

#include "gl_utils.h"

namespace gl {

// Create a program from the given shader source code. Errors in the shader
// programs are fatal.
GLuint CreateProgramFromShaders(const char *vertex_shader,
                                const char *fragment_shader);

// Convenience class for creating programs from shader source. The idea is to
// create static instances of this class that compile the source only once.
class Shader {
 public:
  Shader(const char *vertex, const char *fragment) :
      vertex_(vertex), fragment_(fragment), program_(0) {}
  ~Shader() { GL(DeleteProgram)(program_); }
  void Use() {
    if (!program_) program_ = CreateProgramFromShaders(vertex_, fragment_);
    GL(UseProgram)(program_);
  }
 private:
  const char *vertex_, *fragment_;      // Program source
  GLuint program_;                      // Compiled program, 0 if none
};

// Convenience class for applying a new shader program then restoring the old
// program when we go out of scope. This also works when no program ("program
// 0") is currently being used. The new shader will have the current transforms
// reapplied.
class PushShader {
 public:
  explicit PushShader(Shader &shader) {
    GL(GetIntegerv)(GL_CURRENT_PROGRAM, &program_);
    shader.Use();
    ReapplyTransform();
  }
  ~PushShader() { GL(UseProgram)(program_); }
 private:
  GLint program_;
};

// Catch bug where a PushShader variable name is omitted
#define PushShader(x) static_assert(0, "PushShader missing variable name");

// A simple shader with a constant color (via a uniform vec3 called "color").
Shader &FlatShader();

// A simple shader with a color per vertex (via a vertex attribute called
// "vertex_color").
Shader &SmoothShader();

// A shader program that does per-pixel lighting for nice looking surfaces.
// Once this is enabled, just draw your geometry.
Shader &PerPixelLightingShader();

}  // namespace gl

#endif
