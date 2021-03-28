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

// References:
//   [1] Timothy A. Davis and William W. Hager,
//       "Row modifications of a sparse cholesky factorization"
//       https://people.engr.tamu.edu/davis/publications_files/Row_modifications_of_a_sparse_Cholesky_factorization.pdf

#include <stdio.h>
#include <iostream>
#include <utility>
#include "lcp.h"
#include "testing.h"
#include "random.h"

using Eigen::MatrixXd;
using std::vector;

namespace lcp {

inline double sqr(double x) {
  return x * x;
}

//***************************************************************************
// More convenient access to Eigen's LLT Cholesky solver. This provides an
// important feature missing from Eigen: the ability to perform rank updates on
// already-factored matrices.

// Below we peek into some of Eigen's internals. If we upgrade Eigen, make sure
// that this code still works as expected.
#define EIGEN_VERSION_IS(x, y, z) (EIGEN_WORLD_VERSION == x && \
    EIGEN_MAJOR_VERSION == y && EIGEN_MINOR_VERSION == z)
#if !EIGEN_VERSION_IS(3, 3, 8) && !EIGEN_VERSION_IS(3, 3, 9)
  #error "Unexpected Eigen version. Update check after ensuring code is ok."
#endif

// Perform an in-place Cholesky factorization on an SPD matrix A to give a
// matrix L. Only the lower triangle is accessed.
inline void Cholesky(MatrixXd *A_and_L) {
  Eigen::LLT<Eigen::Ref<MatrixXd>> llt(*A_and_L);
}

// Solve in place: L*x = b, regarding just the top left n*n block of L. Only
// the lower triangle of L is read.
inline void LSolve(const MatrixXd &L, int n, VectorXd *x_and_b) {
  Eigen::Map<VectorXd> xb(x_and_b->data(), n);
  L.block(0, 0, n, n).triangularView<Eigen::Lower>().solveInPlace(xb);
}

// Solve in place: L'*x = b, regarding just the top left n*n block of L. Only
// the lower triangle of L is read.
inline void LTSolve(const MatrixXd &L, int n, VectorXd *x_and_b) {
  Eigen::Map<VectorXd> xb(x_and_b->data(), n);
  L.block(0, 0, n, n).adjoint().triangularView<Eigen::Upper>()
    .solveInPlace(xb);
}

// Solve L*L'*x = b, regarding just the top left n*n block of L. Only the lower
// triangle of L is read.
inline void LLTSolve(const MatrixXd &L, int n, VectorXd *x_and_b) {
  LSolve(L, n, x_and_b);
  LTSolve(L, n, x_and_b);
}

// Perform an in-place rank update on a block(i,i,p,p) of an already-factored
// L. Only the lower triangle of L is read.
inline void RankUpdate(int i, int p, const VectorXd &vec, double sigma,
                       MatrixXd *L) {
  int n = L->rows();
  Eigen::Map<MatrixXd, 0, Eigen::OuterStride<>> Lmap(L->data() + i + i*n, p, p,
                                                     Eigen::OuterStride<>(n));
  Eigen::internal::llt_rank_update_lower(Lmap, vec, sigma);
}

//***************************************************************************
// Modify factorizations.

// On entry the top left (n-1)*(n-1) block of L is a factorization of the top
// left (n-1)*(n-1) block of A. Fill in row n-1 of L so that now the n*n
// top-left block of L is a complete factorization of the n*n top-left block of
// A. Only the lower triangle of A and L is accessed.

static void AddCholeskyRow(const MatrixXd &A, int n, MatrixXd *L) {
  // Perform one further iteration of Choleksy decomposition.
  CHECK(n >= 0 && A.rows() >= n);
  if (n == 1) {
    (*L)(0, 0) = sqrt(A(0, 0));
  } else {
    VectorXd ell(A.block(n-1, 0, 1, n-1).transpose());
    LSolve(*L, n-1, &ell);
    L->block(n-1, 0, 1, n-1) = ell.transpose();
    (*L)(n-1, n-1) = sqrt(A(n-1, n-1) - ell.squaredNorm());
  }
}

// Given an n*n factorization L of an n*n matrix A, modify L so that it is an
// (n-1)*(n-1) factorization of A with row and column n-1 moved to i. A is not
// actually changed. Only the top left n*n blocks of A and L are considered, A
// and L can actually be larger. Only the lower triangle of A and L is
// accessed.

static void SwapCholeskyRows(const MatrixXd &A, int i, int n, MatrixXd *L) {
  // Implementation note: It is possible to delete a row/column of a Cholesky
  // factorization with a single rank 1 update (e.g. [1]). Here we use a rank 2
  // update. What gives? With the rank 1 deletion we would have to shuffle
  // memory for A and L to fill in the gap left behind. The rank 2 update swaps
  // indexes i and n-1, requiring much less data motion. Experiments show that
  // the rank 2 approach is faster.

  if (n <= 1 || i == n-1) {
    return;
  }

  if (i == 0) {
    // Perform a rank-2 update to L. Note that for scalar b and column vector
    // a, if wq=[b/2+1;a] and wr=[b/2-1;a] then:
    //   wq*wq'/2 - wr*wr'/2 == [b,a'; a,zeros]
    // So we can adjust the top/left of A with a rank 2 update by selecting b
    // and a appropriately.
    VectorXd wq = A.block(n-1, 0, 1, n-1).transpose() - A.block(0, 0, n-1, 1);
    wq[0] = (A(n-1, n-1) - A(0, 0))*0.5 + 1;
    RankUpdate(0, n-1, wq, 0.5, L);
    wq[0] = (A(n-1, n-1) - A(0, 0))*0.5 - 1;
    RankUpdate(0, n-1, wq, -0.5, L);
  } else {
    // Partitioning the factorization       |1 column
    //   [ L1  0  0 ] [ L1' l1 L2' ]   [ A  a B' ]
    //   [ l1' e  0 ] [  0  e  l2' ] = [ a' b c' ]  <-- 1 row
    //   [ L2  l2 E ] [  0  0   E' ]   [ B  c C  ]
    // We are changing a,b,c. Expanding L*L' and see what changes:
    //   L1 l1 = new_a                <-- solve for l1
    //   l1'l1 + e^2 = new_b          <-- solve for e
    //   L2 l1 + l2 e = new_c         <-- solve for l2
    //   L2 L2' + l2 l2' + E E' = C   <-- rank 2 update to E
    VectorXd l1 = A.block(n-1, 0, 1, i).transpose();
    LSolve(*L, i, &l1);
    auto e = sqrt(A(n-1, n-1) - l1.dot(l1));
    (*L)(i, i) = e;
    L->block(i, 0, 1, i) = l1.transpose();
    if (i < n-2) {
      // Rank update the original l2.
      RankUpdate(i+1, n-2-i, L->block(i+1, i, n-2-i, 1), 1, L);
      L->block(i+1, i, n-2-i, 1) = (A.block(n-1, i+1, 1, n-2-i).transpose() -
                                    L->block(i+1, 0, n-2-i, i) * l1) / e;
      // Rank update with the new l2.
      RankUpdate(i+1, n-2-i, L->block(i+1, i, n-2-i, 1), -1, L);
    }
  }
}

//***************************************************************************
// MatrixPermutation.

MatrixPermutation::MatrixPermutation(MatrixXd &A) : A_(A) {
  perm_.resize(A_.rows());
  iperm_.resize(A_.rows());
  for (int i = 0; i < A_.rows(); i++) {
    perm_[i] = i;
    iperm_[i] = i;
  }
}

void MatrixPermutation::SwapRowsAndColumns(int i, int j) {
  // Note that we physically move data around rather than simply adjusting a
  // permutation and doing later indirection. This is because data locality is
  // important for the speed of big matrix operations, so the cost of data
  // movement pays off.
  if (i == j) {
    return;
  }
  if (i > j) {
    std::swap(i, j);      // Enforce i < j
  }
  int i2 = perm_[i];      // perm_ maps current index to original
  int j2 = perm_[j];
  std::swap(perm_[i], perm_[j]);
  std::swap(iperm_[i2], iperm_[j2]);
  // For swapping in A_ we can't do the following since we only access the
  // lower triangle:
  //   A_.col(i).swap(A_.col(j));
  //   A_.row(i).swap(A_.row(j));
  int n = A_.rows();
  A_.block(i, 0, 1, i).swap(A_.block(j, 0, 1, i));
  A_.block(j+1, i, n-j-1, 1).swap(A_.block(j+1, j, n-j-1, 1));
  A_.block(i+1, i, j-i-1, 1).swap(A_.block(j, i+1, 1, j-i-1).transpose());
  std::swap(A_(i, i), A_(j, j));
}

void MatrixPermutation::Permute(const VectorXd &in, VectorXd *out) const {
  for (int i = 0; i < perm_.size(); i++) {
    (*out)[i] = in[perm_[i]];
  }
}

void MatrixPermutation::Unpermute(const VectorXd &in, VectorXd *out,
                                  int start_index) const {
  for (int i = start_index; i < perm_.size(); i++) {
    (*out)[perm_[i]] = in[i];
  }
}

//***************************************************************************
// LCP.

bool SolveLCP_BoxMurty(const Settings &settings,
                       MatrixXd &A, const VectorXd &b_arg,
                       const VectorXd &lo_arg, const VectorXd &hi_arg,
                       VectorXd *x_arg, VectorXd *w_arg) {
  // If lo=0 and hi=infinity this code can be simplified quite a bit, and
  // the x update at the start of each iteration can be twice as fast by
  // observing that if we add a row/column to an A x = b problem:
  //   [ A  a ] [ x1 ] = [ L  0 ] [ L' l ] [ x1 ] = [ b1 ]
  //   [ a' c ] [ x2 ] = [ l' e ] [ 0  e ] [ x2 ]   [ b2 ]
  // and we know oldx1 = inv(A) b1, the x1,x2 can be computed from
  //   x2 = (b2 - a' oldx1) / e^2
  //   x1 = oldx1 - inv(L') l x2
  // (and similarly for removing a row/column). This speeds things up a little,
  // but the result is still not competitive with BoxDantzig, so we don't
  // actually implement this.
  //
  // This can not handle unbounded variables with lo=-infinity and hi=infinity,
  // because of the way that x is initialized. It is assumed that unbounded
  // variables will be handled with a Schur complement method.

  int n = A.rows();

  // Permute A, b, lo, hi as we go.
  MatrixPermutation Aperm(A);
  VectorXd b = b_arg;
  VectorXd lo = lo_arg;
  VectorXd hi = hi_arg;

  // The factorization of Aperm that we build as we go.
  MatrixXd L(n, n);

  // 'index' is the size of the top left matrix in A that is currently
  // factorized into L. Indexes 0..index-1 are where w=0 and x is free.
  // Indexes index..i-1 are where w can take any value and x is clamped.
  int index = 0;

  // Initialize x. We can not put 'infinity' values here because they will
  // multiply other things and cause a mess.
  VectorXd x(n);
  for (int i = 0; i < n; i++) {
    if (lo[i] > -__DBL_MAX__)
      x[i] = lo[i];
    else if (hi[i] < __DBL_MAX__)
      x[i] = hi[i];
    else
      Panic("This can not handle unbounded variables");
  }

  VectorXd xhead, wtail;
  VectorXd BCx2 = A.selfadjointView<Eigen::Lower>() * x;
  for (int iteration = 0; iteration < settings.max_iterations; iteration++) {
    // Each iteration we solve the problem
    //   [ A B' ] [ x1 ] = [ b1 ] + [ 0  ]    (A is index x index)
    //   [ B C  ] [ x2 ] = [ b2 ] + [ w2 ]
    // Where x1 is 'free', x2 is clamped lo or hi. L is an incrementally
    // updated factorization of A. We compute:
    //   x1 = inv(A) (b1 - B' x2)
    //   w2 = B x1 + C x2 - b2
    // BCx2 is an incrementally computed [ 0 B' ] [ 0  ]
    //                                   [ 0 C  ] [ x2 ]

    // Recompute x.
    x.head(index) = b.head(index) - BCx2.head(index);
    LLTSolve(L, index, &x);

    // Look for violations in x and w. It is important to check indexes in a
    // consistent order otherwise we may enter cycles. We use the original
    // (unpermuted) order.
    wtail.resize(n - index);      // Computed incrementally
    for (int original_i = 0; original_i < n; original_i++) {
      int i = Aperm.PermutedIndexOf(original_i);
      if (i < index) {
        if (x[i] < lo[i] || x[i] > hi[i]) {
          // Take index i out of the set.
          x[index-1] = (x[i] < lo[i]) ? lo[i] : hi[i];
          SwapCholeskyRows(A, i, index, &L);
          Aperm.SwapRowsAndColumns(index-1, i);
          std::swap(b[index-1], b[i]);
          std::swap(lo[index-1], lo[i]);
          std::swap(hi[index-1], hi[i]);
          std::swap(BCx2[index-1], BCx2[i]);
          index--;
          // Incrementally update B'*x2 and C*x2
          BCx2.head(index) += x[index] * A.row(index).head(index).transpose();
          BCx2[index] = A.col(index).tail(n-index-1).dot(x.tail(n-index-1));
          BCx2.tail(n-index) += A.col(index).tail(n-index) * x[index];
          goto retry;
        }
      } else {
        // Compute w values as needed.
        wtail[i-index] = A.row(i).head(index).dot(x.head(index))
                         + BCx2[i] - b[i];
        if ( (wtail[i-index] < 0 && x[i] == lo[i]) ||
             (wtail[i-index] > 0 && x[i] == hi[i]) ) {
          // Put index i into the set.
          Aperm.SwapRowsAndColumns(index, i);
          std::swap(b[index], b[i]);
          std::swap(lo[index], lo[i]);
          std::swap(hi[index], hi[i]);
          std::swap(x[index], x[i]);
          std::swap(BCx2[index], BCx2[i]);
          AddCholeskyRow(A, index + 1, &L);
          // Incrementally update B'*x2 and C*x2.
          BCx2.head(index) -= x[index] * A.row(index).head(index).transpose();
          BCx2[index] = A.col(index).tail(n-index-1).dot(x.tail(n-index-1));
          BCx2.tail(n-index-1) -= A.col(index).tail(n-index-1) * x[index];
          index++;
          goto retry;
        }
      }
    }

    // We have a correct solution.
    {
      VectorXd wret(n);
      wret.setZero();
      wret.tail(n - index) = wtail;
      x_arg->resize(n);
      w_arg->resize(n);
      Aperm.Unpermute(x, x_arg);
      Aperm.Unpermute(wret, w_arg);
      return true;
    }
   retry:;
  }

  // We exceeded the maximum number of iterations.
  return false;
}

bool SolveLCP_BoxDantzig(const Settings &settings,
                         MatrixXd &A, const VectorXd &b,
                         const VectorXd &lo_arg, const VectorXd &hi_arg,
                         VectorXd *x_arg, VectorXd *w_arg) {
  // The restrictions: lo <= 0, hi >= 0, lo < hi make this algorithm a bit
  // simpler. We need lo < hi so that we can always tell whether x is
  // clamped low or high by looking at its value.

  // Current partial solution x and w.
  int n = A.rows();
  VectorXd x(n);
  VectorXd w(n);
  x.setZero();            // Not necessary, but it helps debugging
  w.setZero();
  VectorXd lo(lo_arg);    // Make a copy of lo, hi, since we permute them
  VectorXd hi(hi_arg);

  // Permute A as we go.
  MatrixPermutation Aperm(A);

  // The factorization of Aperm that we build as we go.
  MatrixXd L(n, n);

  // 'index' is the size of the top left matrix in A that is currently
  // factorized into L. Indexes 0..index-1 are where w=0 and lo < x < hi.
  // Indexes index..i-1 are where w can take any value and x is either lo or
  // hi.
  int index = 0;

  for (int i = 0; i < n; i++) {
    // Thus far we have not even been computing the w values for indexes >= i,
    // so compute w[i] and x[i] now.
    w[i] = A.row(i).head(i).dot(x.head(i)) - b[i];
    x[i] = 0;

    // Check if LCP conditions for index i are already satisfied.
    if (w[i] == 0) {
      // Degenerate case.
      continue;
    } else if (lo[i] == 0 & w[i] >= 0) {
      continue;   // Because x is already at the lo value
    } else if (hi[i] == 0 & w[i] <= 0) {
      continue;   // Because x is already at the hi value
    }

    // Find the direction to push on x[i]: delta_xS = -dir*inv(ASS)*AiS
    double dir = (w[i] <= 0) ? 1 : -1;
    VectorXd delta_xS = -dir * A.block(i, 0, 1, index).transpose();
    LLTSolve(L, index, &delta_xS);
    double delta_xi = dir;

    while (true) {
      // Push x[i] and w[i] to hit one of the three lines.
      // Calculate delta_w = A*delta_x, except that we don't need to calculate
      // the parts of delta_w that are going to be zero anyway.
      VectorXd delta_wNS = A.block(index, 0, i-index, index) * delta_xS +
                           A.block(i, index, 1, i-index).transpose() * dir;
      double delta_wi = A.row(i).head(index).dot(delta_xS) + A(i, i) * dir;

      // Find the largest step we can take for index i.
      double best_alpha = -w[i] / delta_wi;       // Steps to w[i] = 0
      int best_index = i;
      bool index_i_into_set = true;
      double index_i_limit = (dir > 0) ? hi[i] : lo[i];
      {
        double alpha = (index_i_limit - x[i]) / delta_xi;
        if (alpha > 0 && alpha < best_alpha) {
          best_alpha = alpha;
          best_index = i;
          index_i_into_set = false;
        }
      }

      // Find the largest step we can take for indexes in the set.
      VectorXd limit(index);
      for (int j = 0; j < index; j++) {
        // For this index we could have stepped to x=lo or x=hi in a previous
        // iteration (i.e. limit-x==0), so we should be careful not to take a
        // zero size spurious step (the alpha > 0 requirement).
        limit[j] = (delta_xS[j] > 0) ? hi[j] : lo[j];
        double alpha = (limit[j] - x[j]) / delta_xS[j];
        if (alpha > 0 && alpha < best_alpha) {
          best_alpha = alpha;
          best_index = j;
        }
      }

      // Find the largest step we can take for indexes not in the set.
      for (int j = index; j < i; j++) {
        // For this index we could have stepped to w=0 in a previous iteration,
        // so we should be careful not to take the spurious step w=0 --> w=0
        // (the alpha > 0 requirement).
        double alpha = -w[j] / delta_wNS[j-index];
        if (alpha > 0 && alpha < best_alpha) {
          best_alpha = alpha;
          best_index = j;
        }
      }
      CHECK(best_index >= 0);
      CHECK(best_alpha >= 0);

      // Do the step. Only modify the x,w values that are allowed to change.
      x.head(index) += best_alpha * delta_xS;
      x[i] += best_alpha * delta_xi;
      w.segment(index, i-index) += best_alpha * delta_wNS;
      w[i] += best_alpha * delta_wi;

      // Modify the factorization.
      index_i_into_set = (best_index == i && index_i_into_set);
      if (best_index < index) {
        x[best_index] = limit[best_index];
        // Swap best_index out of the set. Don't swap w because those are 0.
        SwapCholeskyRows(A, best_index, index, &L);
        Aperm.SwapRowsAndColumns(index-1, best_index);
        std::swap(x [index-1], x [best_index]);
        std::swap(lo[index-1], lo[best_index]);
        std::swap(hi[index-1], hi[best_index]);
        index--;

        // Recompute delta_xS = -dir*inv(ASS)*AiS
        // Note that dir is unchanged because the sign of w[i] is still the
        // same because the step can not push w[i] across zero.
        // TODO: There is a small optimization to take here. L generally only
        // changes at the bottom/right, so in LLTSolve() where we solve L y = b
        // then L' x = y, if we already have y then we can update only the
        // bottom part of it and save some work. This requires us to save the y
        // values though.
        delta_xS = -dir * A.block(i, 0, 1, index).transpose();  // -dir * AiS
        LLTSolve(L, index, &delta_xS);
      }
      else if (best_index < i || index_i_into_set) {
        w[index_i_into_set ? i : best_index] = 0;
        // Swap best_index into the set.
        Aperm.SwapRowsAndColumns(index, best_index);
        std::swap(x [index], x [best_index]);
        std::swap(w [index], w [best_index]);
        std::swap(lo[index], lo[best_index]);
        std::swap(hi[index], hi[best_index]);
        AddCholeskyRow(A, index + 1, &L);

        // Recompute delta_xS = -dir*inv(ASS)*AiS.
        // We can do this with a single LTSolve and save half the work compared
        // to the original computation of delta_xS. This is based on the fact
        // that if we add a row/column to an A x = b problem,
        //   [ A  a ] [ x1 ] = [ L  0 ] [ L' l ] [ x1 ] = [ b1 ]
        //   [ a' c ] [ x2 ] = [ l' e ] [ 0  e ] [ x2 ]   [ b2 ]
        // and we know oldx1 = inv(A) b1, the x1,x2 can be computed from
        //   x2 = (b2 - a' oldx1) / e^2
        //   x1 = oldx1 - inv(L') l x2
        if (best_index != i) {
          // If best_index == i we're done anyway so no need to do this update.
          double value =  // New value to append to delta_xS
              (-dir*A(i,index) - A.row(index).head(index).dot(delta_xS)) /
              sqr(L(index, index));
          delta_xS.conservativeResize(index + 1);
          delta_xS[index] = value;
          VectorXd v = L.block(index, 0, 1, index).transpose();
          LTSolve(L, index, &v);
          delta_xS.head(index) -= value * v;
        }
        index++;
      } else {
        x[i] = index_i_limit;
      }
      CHECK(index >= 0 && index <= A.rows());

      // See if we're done for index i.
      if (best_index == i) {
        break;
      }
    }
  }

  x_arg->resize(n);
  w_arg->resize(n);
  Aperm.Unpermute(x, x_arg);
  Aperm.Unpermute(w, w_arg);
  return true;
}

// If 'nub' is given as >= 0 it is the number of unbounded indexes at the start
// of lo and hi. If nub==-1 then nub will be determined by scanning lo and hi.
// This ability to control nub is used for testing. The global variable is an
// output used for testing only.
static int test_nub_from_SolveLCP_BoxSchur;

bool SolveLCP_BoxSchur(const Settings &settings,
                       MatrixXd &A, const VectorXd &b_arg,
                       const VectorXd &lo_arg, const VectorXd &hi_arg,
                       VectorXd *x, VectorXd *w, int nub = -1) {
  // To solve the linear system
  //   [ Z B' ] [ y ] = [ c ]
  //   [ B C  ] [ z ] = [ d ]
  // by the Schur complement we do this:
  //   L*L' = Z             // Factor Z
  //   Q = inv(L)*B'        // Note: solving B rows one by one is inefficient
  //   R = C - Q'*Q         // R = C - B*inv(Z)*B'. Only compute lower triangle.
  //   t = inv(Z)*c         // Z solve
  //   z = d - B*t          // z = d - B*inv(Z)*c
  //   z = inv(R)*z         // z = inv(C - B*inv(Z)*B') (d - B*inv(Z)*c)
  //   y = inv(Z)*(c - B'*z)
  // One advantage here is that we refer to Z purely by its ability to solve
  // for some right hand side, so we do not necessarily need a dense matrix
  // representation of Z. Another advantage is that we can solve for z as an
  // LCP problem.

  int n = A.rows();

  // Make copies of argument vectors since we might permute them below.
  VectorXd b(b_arg);
  VectorXd lo(lo_arg);
  VectorXd hi(hi_arg);

  // Permute the problem so that the unbounded variables come first. These will
  // make up the matrix Z.
  MatrixPermutation permutation(A);
  bool did_swaps = false;
  if (nub < 0) {
    nub = 0;        // Known unbounded indexes are [0,nub)
    int nb = n-1;   // Known bounded indexes are (nb,n]
    while (true) {
      // Advance nub as much as possible.
      for (; nub <= nb; nub++) {
        if (lo[nub] > -__DBL_MAX__ || hi[nub] < -__DBL_MAX__)
          break;                // This index is bounded
      }
      // Decrease nb as much as possible.
      for (; nb >= nub; nb--) {
        if (lo[nb] <= -__DBL_MAX__ && hi[nb] >= -__DBL_MAX__)
          break;                // This index is unbounded
      }
      if (nub > nb) {
        break;
      } else {
        // Swap these indexes so we can make further progress.
        CHECK(0 <= nub && nub < nb && nb < n);
        permutation.SwapRowsAndColumns(nub, nb);
        std::swap(b[nub], b[nb]);
        std::swap(lo[nub], lo[nb]);
        std::swap(hi[nub], hi[nb]);
        did_swaps = true;
      }
    }
  }
  test_nub_from_SolveLCP_BoxSchur = nub;

  // If this is not actually an LCP problem, just solve it directly here.
  if (nub == n) {
    *x = A.llt().solve(b);
    w->resize(A.rows());
    w->setZero();
    return true;
  }

  // If this is entirely an LCP problem, skip the Schur complement.
  if (nub == 0) {
    CHECK(!did_swaps);    // So A should be unchanged
    Settings settings2 = settings;
    settings2.schur_complement = false;
    return SolveLCP(settings2, A, b, lo, hi, x, w);
  }

  // Form the sub-matrices of the Schur complement. Only refer to the lower
  // triangle of A, which means we only refer to the lower triangle of Z and C.
  typedef Eigen::Map<const MatrixXd, 0, Eigen::OuterStride<>> Map;
  Map Z(A.data(), nub, nub, Eigen::OuterStride<>(n));
  Map B(A.data() + nub, n-nub, nub, Eigen::OuterStride<>(n));
  Map C(A.data() + nub + n*nub, n-nub, n-nub, Eigen::OuterStride<>(n));
  Map c(b.data(), nub, 1, Eigen::OuterStride<>(n));
  Map d(b.data() + nub, n-nub, 1, Eigen::OuterStride<>(n));
  Map lo2(lo.data() + nub, n-nub, 1, Eigen::OuterStride<>(n));
  Map hi2(hi.data() + nub, n-nub, 1, Eigen::OuterStride<>(n));

  // Schur complement with LCP. Note we only compute the lower triangle of R.
  MatrixXd L = Z;
  lcp::Cholesky(&L);
  MatrixXd Q = L.triangularView<Eigen::Lower>().solve(B.transpose());
  MatrixXd R = C.triangularView<Eigen::Lower>();
  R.selfadjointView<Eigen::Lower>().rankUpdate(Q.transpose(), -1);
  VectorXd t = c;
  lcp::LLTSolve(L, nub, &t);
  // The non-LCP variant is:
  //   lcp::Cholesky(&R);
  //   VectorXd z = d - B*t;
  //   lcp::LLTSolve(R, n-nub, &z);
  VectorXd rhs = d - B*t;
  VectorXd z, w2;
  Settings settings2 = settings;
  settings2.schur_complement = false;
  if (!SolveLCP(settings2, R, rhs, lo2, hi2, &z, &w2)) {
    return false;
  }
  VectorXd y = c - B.transpose()*z;
  lcp::LLTSolve(L, nub, &y);

  // Assemble results.
  VectorXd x_ret(n), w_ret(n);
  x_ret.head(nub) = y;
  x_ret.tail(n - nub) = z;
  w_ret.head(nub).setZero();
  w_ret.tail(n - nub) = w2;
  x->resize(n);
  w->resize(n);
  permutation.Unpermute(x_ret, x);
  permutation.Unpermute(w_ret, w);
  return true;
}

//***************************************************************************
// Drivers.

bool SolveLCP(const Settings &settings,
              MatrixXd &A, const VectorXd &b,
              const VectorXd &lo, const VectorXd &hi,
              VectorXd *x, VectorXd *w) {
  CHECK(A.rows() == A.cols());
  CHECK(A.rows() > 0);
  CHECK(b.size() == A.rows());
  CHECK(lo.size() == A.rows());
  CHECK(hi.size() == A.rows());

  MatrixXd Acopy;
  if (settings.test_tolerance > 0) {
    Acopy = A;
  }

  bool ok = false;
  if (settings.schur_complement) {
    ok = SolveLCP_BoxSchur(settings, A, b, lo, hi, x, w);
  } else if (settings.algorithm == MURTY) {
    ok = SolveLCP_BoxMurty(settings, A, b, lo, hi, x, w);
  } else if (settings.algorithm == COTTLE_DANTZIG) {
    ok = SolveLCP_BoxDantzig(settings, A, b, lo, hi, x, w);
  } else {
    Panic("Unknown LCP solver selection");
  }

  if (settings.test_tolerance > 0) {
    double error = (Acopy.selfadjointView<Eigen::Lower>() * *x - b - *w).norm();
    CHECK(error < settings.test_tolerance);
    for (int i = 0; i < A.rows(); i++) {
      CHECK( (((*x)[i] >= lo[i] && (*x)[i] <= hi[i]) && (*w)[i] == 0) ||
              ((*x)[i] == lo[i] && (*w)[i] >= 0) ||
              ((*x)[i] == hi[i] && (*w)[i] <= 0))
    }
  }
  return ok;
}

}  // namespace lcp

//***************************************************************************
// LCP testing.

namespace {

using Eigen::VectorXd;
using Eigen::MatrixXd;

const int N = 7;            // Problem sizes below

TEST_FUNCTION(Murty) {
  for (int iteration = 0; iteration < 1000; iteration++) {
    // Create a random positive definite LCP problem.
    MatrixXd A0 = MatrixXd::Random(N,N);
    MatrixXd A = A0*A0.transpose();
    VectorXd b = VectorXd::Random(N,1);

    {
      VectorXd x, w, lo(N), hi(N);

      // Solve a box LCP problem with random lo and hi vectors. Only pass in
      // the lower triangle of A, to ensure that the upper triangle is not
      // read.
      for (int i = 0; i < N; i++) {
        lo[i] = -RandomDouble() * 10.0;
        hi[i] = +RandomDouble() * 10.0;
        // Occasionally make lo or hi equal to zero.
        int r = RandomInt(100);
        if (r == 0) {
          lo[i] = 0;
        } else if (r == 1) {
          hi[i] = 0;
        }
      }
      MatrixXd Alower = A.triangularView<Eigen::Lower>();
      lcp::Settings settings;
      settings.test_tolerance = 1e-9;
      CHECK(lcp::SolveLCP_BoxMurty(settings, Alower, b, lo, hi, &x, &w));

      // Check the solution.
      //std::cout << "x:\n" << x << "\n";
      //std::cout << "w:\n" << w << "\n";
      for (int i = 0; i < N; i++) {
        CHECK( ((x[i] >= lo[i] && x[i] <= hi[i]) && w[i] == 0) ||
               (x[i] == lo[i] && w[i] >= 0) ||
               (x[i] == hi[i] && w[i] <= 0))
      }
      double error = (A*x - b - w).norm();
      printf("Error = %e\n", error);
      CHECK(error < 1e-6);
    }
  }
  printf("Success\n");
}

TEST_FUNCTION(Dantzig) {
  for (int iteration = 0; iteration < 1000; iteration++) {
    // Create a random positive definite LCP problem.
    MatrixXd A0 = MatrixXd::Random(N,N);
    MatrixXd A = A0*A0.transpose();
    A += MatrixXd::Identity(N, N) * 0.001;   // Stabilize condition number
    VectorXd b = VectorXd::Random(N,1);

    {
      VectorXd x, w, lo(N), hi(N);

      // Solve a box LCP problem with random lo and hi vectors. Only pass in
      // the lower triangle of A, to ensure that the upper triangle is not
      // read.

      // Test problem variants.
      double lo_range = 10;
      double hi_range = 10;
      switch (iteration % 6) {
        case 0: break;
        case 1: lo_range *= 10;  hi_range *= 10;  break;
        case 2: lo_range = 1e99; hi_range = 1e99; break;
        case 3: lo_range *= 0.1; hi_range *= 0.1; break;
        case 4: lo_range = 0; break;
        case 5: hi_range = 0; break;
      }

      for (int i = 0; i < N; i++) {
        lo[i] = -RandomDouble() * lo_range;
        hi[i] = +RandomDouble() * hi_range;
        // Occasionally make lo or hi equal to zero (but avoid the case
        // lo=hi=0).
        int r = RandomInt(100);
        if (r == 0) {
          if (hi[i] != 0)
            lo[i] = 0;
        } else if (r == 1) {
          if (lo[i] != 0)
            hi[i] = 0;
        }
      }

      MatrixXd Alower = A.triangularView<Eigen::Lower>();
      lcp::Settings settings;
      settings.test_tolerance = 1e-9;
      CHECK(lcp::SolveLCP_BoxDantzig(settings, Alower, b, lo, hi, &x, &w));

      // Check the solution.
      for (int i = 0; i < N; i++) {
        CHECK( ((x[i] >= lo[i] && x[i] <= hi[i]) && w[i] == 0) ||
               (x[i] == lo[i] && w[i] >= 0) ||
               (x[i] == hi[i] && w[i] <= 0))
      }
      double error = (A*x - b - w).norm();
      printf("Error = %e\n", error);
      CHECK(error < 1e-6);
    }
  }
  printf("Success\n");
}

TEST_FUNCTION(Cholesky) {
  MatrixXd A0 = MatrixXd::Random(N,N);
  MatrixXd A = A0 * A0.transpose();
  Eigen::LLT<MatrixXd> llt(A);
  lcp::Cholesky(&A);
  MatrixXd L1 = llt.matrixL();
  MatrixXd L2 = A.triangularView<Eigen::Lower>();
  double error = (L1 - L2).norm();
  printf("Error = %e\n", error);
  CHECK(error < 1e-10);
}

TEST_FUNCTION(LSolve) {
  for (int n = 1; n < N; n++) {
    // Form a triangular L and right hand side b
    MatrixXd A0 = MatrixXd::Random(N,N);
    MatrixXd A = A0 * A0.transpose();
    MatrixXd L = A.triangularView<Eigen::Lower>();
    lcp::Cholesky(&L);
    VectorXd b = MatrixXd::Random(n, 1);
    VectorXd x = b;
    lcp::LSolve(L, n, &x);
    double error = (L.block(0, 0, n, n)*x - b).norm();
    printf("Error = %e\n", error);
    CHECK(error < 1e-10);
  }
}

TEST_FUNCTION(AddCholeskyRow) {
  for (int n = 1; n <= N; n++) {
    // Form a positive definite A, factor, and then add a row.
    MatrixXd A0 = MatrixXd::Random(N,N);
    MatrixXd A = A0 * A0.transpose();
    MatrixXd L = A.triangularView<Eigen::Lower>();
    lcp::Cholesky(&L);
    MatrixXd Loriginal = L;
    L.block(n-1, 0, N-n+1, N).setZero();
    L.block(0, n-1, N, N-n+1).setZero();
    lcp::AddCholeskyRow(A, n, &L);
    double error = (L.block(0, 0, n, n) - Loriginal.block(0, 0, n, n)).norm();
    printf("Error = %e\n", error);
    CHECK(error < 1e-10);
  }
}

TEST_FUNCTION(SwapCholeskyRows) {
  for (int sz = N; sz < N+5; sz++) {
    for (int n = 0; n < N; n++) {
      // Form a positive definite A, factor, and then remove a row/column.
      MatrixXd A0 = MatrixXd::Random(sz, sz);
      MatrixXd A = A0 * A0.transpose();
      Eigen::LLT<MatrixXd> llt(A);
      MatrixXd L = llt.matrixL();
      MatrixXd Lorig = L;
      lcp::SwapCholeskyRows(A, n, N, &L);

      // Compare with explicit refactorization.
      MatrixXd A2 = A;
      A2.col(n).swap(A2.col(N-1));
      A2.row(n).swap(A2.row(N-1));
      Eigen::LLT<MatrixXd> llt2(A2);
      MatrixXd L2 = llt2.matrixL();
      double error = (L.block(0, 0, N-1, N-1) - L2.block(0, 0, N-1, N-1)).norm();
      printf("Removing row/col %d, error = %e\n", n, error);
      CHECK(error < 1e-9);

      // Check the parts of L that should not have been touched.
      if (sz > N) {
        CHECK(L.block(N, 0, sz-N, sz) == Lorig.block(N, 0, sz-N, sz))
      }
    }
  }
}

TEST_FUNCTION(SolveLCP_BoxSchur) {
  const int n = 20;
  MatrixXd A0 = MatrixXd::Random(n, n);
  MatrixXd A = A0 * A0.transpose();
  VectorXd b = VectorXd::Random(n);
  lcp::Settings settings;
  settings.algorithm = lcp::COTTLE_DANTZIG;
  settings.test_tolerance = 1e-9;

  const int M = 1;     // Number of repeats, for more accurate timing

  // Full solve.
  VectorXd x;
  {
    double t1 = Now();
    for (int i = 0; i < M; i++) {
      MatrixXd L = A;
      lcp::Cholesky(&L);
      x = b;
      lcp::LLTSolve(L, n, &x);
    }
    double t2 = Now();
    double error = (A*x - b).norm();
    printf("Full solve,            time = %f, error = %e\n", (t2-t1)/M, error);
    CHECK(error < 1e-6);
  }

  // Unbounded problem, test nub==n (i.e. just do a direct factor and solve).
  {
    VectorXd lo(n), hi(n), x2, w2;
    lo.setConstant(-__DBL_MAX__);
    hi.setConstant(__DBL_MAX__);
    double t1 = Now();
    for (int i = 0; i < M; i++) {
      // Only pass the lower triangle of A to the solver, to ensure that the upper
      // triangle is not read.
      MatrixXd Alower = A.triangularView<Eigen::Lower>();
      CHECK(lcp::SolveLCP_BoxSchur(settings, Alower, b, lo, hi, &x2, &w2, n));
    }
    double t2 = Now();
    double error = (x2 - x).norm();
    printf("Solve, nub=n,   time = %f, error = %e\n", (t2-t1)/M, error);
    CHECK(error < 1e-6);
    CHECK(w2.size() == x2.size());
    for (int i = 0; i < w2.size(); i++) {
      CHECK(w2[i] == 0);
    }
  }

  // Unbounded problem, test schur complement for nub==n/2.
  {
    VectorXd lo(n), hi(n), x3, w3;
    lo.setConstant(-__DBL_MAX__);
    hi.setConstant(__DBL_MAX__);
    double t1 = Now();
    for (int i = 0; i < M; i++) {
      MatrixXd Alower = A.triangularView<Eigen::Lower>();
      CHECK(lcp::SolveLCP_BoxSchur(settings, Alower, b, lo, hi, &x3, &w3, n/2));
    }
    double t2 = Now();
    double error = (x3 - x).norm();
    printf("Solve, nub=n/2, time = %f, error = %e\n", (t2-t1)/M, error);
    CHECK(error < 1e-6);
    CHECK(w3.size() == x3.size());
    for (int i = 0; i < w3.size(); i++) {
      CHECK(w3[i] == 0);
    }
  }

  // Solve a full box LCP problem, with nub = 0 and n/2.
  for (int pass = 0; pass < 2; pass++) {
    VectorXd lo(n), hi(n), x4, w4;
    lo.setConstant(-__DBL_MAX__);
    hi.setConstant(__DBL_MAX__);
    int start = 0, end = n;
    if (pass == 1) {
      start = n/4;
      end = start + n/2;
    }
    for (int i = start; i < end; i++) {
      lo[i] = -RandomDouble() * 10.0;
      hi[i] = +RandomDouble() * 10.0;
    }
    double t1 = Now();
    for (int i = 0; i < M; i++) {
      MatrixXd Alower = A.triangularView<Eigen::Lower>();
      CHECK(lcp::SolveLCP_BoxSchur(settings, Alower, b, lo, hi, &x4, &w4));
    }
    double t2 = Now();
    double error = (A*x4 - b - w4).norm();
    printf("Solve lcp,  time = %f, error = %e\n", (t2-t1)/M, error);
    CHECK(error < 1e-6);
    for (int i = 0; i < n; i++) {
      CHECK(x4[i] >= lo[i] && x4[i] <= hi[i]);
    }
    CHECK(lcp::test_nub_from_SolveLCP_BoxSchur == n - (end-start));
  }

  // Solve a full box LCP problem, with random nub.
  for (int pass = 0; pass < 100; pass++) {
    VectorXd lo(n), hi(n), x4, w4;
    lo.setConstant(-__DBL_MAX__);
    hi.setConstant(__DBL_MAX__);
    int nb = 0;
    for (int i = 0; i < n; i++) {
      if (RandomInt(2) == 1) {
        lo[i] = -RandomDouble() * 10.0;
        hi[i] = +RandomDouble() * 10.0;
        nb++;
      }
    }
    MatrixXd Alower = A.triangularView<Eigen::Lower>();
    CHECK(lcp::SolveLCP_BoxSchur(settings, Alower, b, lo, hi, &x4, &w4));
    double error = (A*x4 - b - w4).norm();
    printf("Solve lcp, nub = %d, error = %e\n", n - nb, error);
    CHECK(error < 1e-6);
    for (int i = 0; i < n; i++) {
      CHECK(x4[i] >= lo[i] && x4[i] <= hi[i]);
    }
    CHECK(lcp::test_nub_from_SolveLCP_BoxSchur == n - nb);
  }
}

}  // anonymous namespace
