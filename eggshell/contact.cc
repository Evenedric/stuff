#include "contact.h"

#include <iostream>
#include <limits>

#include "constants.h"
#include "error.h"
#include "model.h"
#include "util.h"

namespace {
constexpr double kBoxFrictionBound = 100;
}

Vector3d Contact::ComputeError() const {
  Vector3d error(0, 0, -cg_.depth);
  return error;
}

// TODO:
// Explain this out loud to make sure the logic makes sense.
//
// 1. write constraints in a coordinate system that is aligned with contact
// normal
//    - find the rotation matrix that re-writes vector a in global frame
//      to one that is in contact normal frame.
//    - row 3 of contact constraint in new frame should equal the no-friction
//      contact constraint. check that normal and transverse are separated out
//      in different row. (how is this result different from 1 row of dot-n
//      constraint and 3 rows of cross-n constraint)
//    - does the alignment/direction of transverse axes matter in some cases?
// 2. implement LCP solver that takes in low and high limits
// 3. Schur complement alternative to solve the mixed constraint problem.
void Contact::ComputeJ(MatrixXd* J_b0, MatrixXd* J_b1, ArrayXb* C,
                       VectorXd* x_lo, VectorXd* x_hi) const {
  // Constraint in the direction of contact normal is an inequality,
  // constraints that are transverse to contact normal are equalities.

  // 1. Write out the ball-n-socket J for the contact. R * c = x - p
  // 2. Find rotation matrices R, such that R*v expresses v in the local
  //    coordinates where z aligns with contact normal.
  // 3. Return R * J.
  // Because of the z alignment with contact normal, row 0-1 are equality
  // constraints, row 2 is an inequality constraint

  Vector3d z_axis(0, 0, 1);
  Matrix3d R = AlignVectors(cg_.normal, z_axis);

  // TODO: debug only
  // std::cout << "R\n" << R << std::endl;

  if (b0_ == nullptr) {
    *J_b0 << Matrix3d::Zero(), Matrix3d::Zero();
  } else {
    Matrix3d J_v0 = -1 * Matrix3d::Identity();
    Matrix3d J_w0 = CrossMat(cg_.position - b0_->p());
    *J_b0 << R * J_v0, R * J_w0;
  }

  Matrix3d J_v1 = Matrix3d::Identity();
  Matrix3d J_w1 = -1 * CrossMat(cg_.position - b1_->p());
  *J_b1 << R * J_v1, R * J_w1;

  // TODO: debug only
  // std::cout << "J_b1 in contact frame = \n" << *J_b1 << std::endl;
  // std::cout << "J_b1 (global frame) *dot* contact normal = "
  //           << cg_.normal.transpose() * J_v1 << " "
  //           << cg_.normal.transpose() * J_w1 << std::endl;

  switch (f_) {
    case FrictionModel::NO_FRICTION:
      *C << 0, 0, 0;
      *x_lo << -1 * std::numeric_limits<double>::infinity(),
          -1 * std::numeric_limits<double>::infinity(), 0;
      *x_hi << std::numeric_limits<double>::infinity(),
          std::numeric_limits<double>::infinity(),
          std::numeric_limits<double>::infinity();
      break;
    case FrictionModel::INFINITE:
      *C << 1, 1, 0;
      *x_lo << 0, 0, 0;
      *x_hi << 0, 0, std::numeric_limits<double>::infinity();
      break;
    case FrictionModel::BOX:
      *C << 0, 0, 0;
      *x_lo << -1 * kBoxFrictionBound, -1 * kBoxFrictionBound, 0;
      *x_hi << kBoxFrictionBound, kBoxFrictionBound,
          std::numeric_limits<double>::infinity();
      break;
    default:
      Panic("Contact constraint encountered an unknown FrictionModel::%i", f_);
  }
}

void Contact::ComputeJDot(MatrixXd* Jdot_b0, MatrixXd* Jdot_b1) const {
  ComputeJDot_NoFriction(Jdot_b0, Jdot_b1);
}

void Contact::ComputeJDot_NoFriction(MatrixXd* Jdot_b0,
                                     MatrixXd* Jdot_b1) const {
  Panic(
      "ODE time stepper does not need to compute JdotV for contact "
      "constraints. Other integrators do not yet support contacts.");
  // // R * c = x - p
  // MatrixXd Jdot_w0 = -1 * cg_.normal.transpose() *
  //                    CrossMat(b0_->w_g().cross(cg_.position - b0_->p()));
  // Jdot_b0 << MatrixXd::Zero(1, 3), Jdot_w0;
  // if (b1_ == nullptr) {
  //   Jdot_b1 << MatrixXd::Zero(1, 3), MatrixXd::Zero(1, 3);
  // } else {
  //   MatrixXd Jdot_w1 = cg_.normal.transpose() *
  //                      CrossMat(b1_->w_g().cross(cg_.position - b1_->p()));
  //   Jdot_b1 << MatrixXd::Zero(1, 3), Jdot_w1;
  // }
}

void Contact::ComputeJDot_InfiniteFriction(MatrixXd* Jdot_b0,
                                           MatrixXd* Jdot_b1) const {
  Panic(
      "ODE time stepper does not need to compute JdotV for contact "
      "constraints. Other integrators do not yet support contacts.");
}

void Contact::ComputeJ_BoxFriction(MatrixXd* J_b0, MatrixXd* J_b1) const {}
void Contact::ComputeJDot_BoxFriction(MatrixXd* Jdot_b0,
                                      MatrixXd* Jdot_b1) const {}
void Contact::ComputeJ_CoulombPyramid(MatrixXd* J_b0, MatrixXd* J_b1) const {}
void Contact::ComputeJDot_CoulombPyramid(MatrixXd* Jdot_b0,
                                         MatrixXd* Jdot_b1) const {}

void Contact::Draw() const {
  DrawPoint(cg_.position);
  DrawLine(cg_.position, cg_.position + cg_.normal * 0.1);
}

std::string Contact::PrintInfo() const {
  std::ostringstream s;
  s << "Vector3d position = \n"
    << cg_.position << "\nVector3d normal = \n"
    << cg_.normal << "\ndepth = " << cg_.depth;
  return s.str();
}

// void Contact::CheckJ_InfiniteFriction() const {
//   // J and Jdot for no friction
//   MatrixXd J0_b0(1, 6), J0_b1(1, 6), Jdot0_b0(1, 6), Jdot0_b1(1, 6);
//   MatrixXd J0(1, 12), Jdot0(1, 12);
//   // J and Jdot for CFM with infinite friction
//   MatrixXd JCfm_b0(3, 6), JCfm_b1(3, 6), JdotCfm_b0(3, 6), JdotCfm_b1(3, 6);
//   MatrixXd JCfm(3, 12), JdotCfm(3, 12);
//   ArrayXb C(3);
//   VectorXd x_lo(3), x_hi(3);
//
//   ComputeJ_NoFriction(&J0_b0, &J0_b1);
//   ComputeJDot_NoFriction(&Jdot0_b0, &Jdot0_b1);
//   ComputeJ_InfiniteFriction(&JCfm_b0, &JCfm_b1, &C, &x_lo, &x_hi);
//   ComputeJDot_InfiniteFriction(&JdotCfm_b0, &JdotCfm_b1);
//   J0 << J0_b0, J0_b1;
//   Jdot0 << Jdot0_b0, Jdot0_b1;
//   JCfm << JCfm_b0, JCfm_b1;
//   JdotCfm << JdotCfm_b0, JdotCfm_b1;
//
//   // Check that J0 == JCfm.row(2);
//   std::cout << "J0: \n" << J0 << "\n";
//   std::cout << "JCfm: \n" << JCfm << "\n";
//   std::cout << "Jdot0: \n" << Jdot0 << "\n";
//   std::cout << "JdotCfm: \n" << JdotCfm << "\n";
//   CHECK(J0.row(0).isApprox(JCfm.row(2), kAllowNumericalError));
//   CHECK(Jdot0.row(0).isApprox(JdotCfm.row(2), kAllowNumericalError));
// }

std::ostream& operator<<(std::ostream* out, const Contact& c) {
  return *out << c.PrintInfo();
}
