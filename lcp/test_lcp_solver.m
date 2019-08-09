function test_lcp_solver(n, num_tests)
% n = size of A and b.

num_tests_failed = 0;
for ii = 1:num_tests
  ret = test_mixed_constraints_solver(n);
  if ret ~= 0
    num_tests_failed = num_tests_failed + 1;
  end
end
fprintf('%d out of %d tests passed.\n', num_tests - num_tests_failed, num_tests);

function ret = test_murty_pivot(n)
A = generate_spd_matrix(n);
b = rand(n,1);
[A_, b_, x, w, ret] = murty_principal_pivot(A, b);

function ret = test_mixed_constraints_solver(n)
A = generate_spd_matrix(n);
b = rand(n,1);
C = rand(n,1) < 0.5;
[x, w, ret] = mixed_constraints_solver(A, b, C);

function A = generate_spd_matrix(n)
% Generate a random SPD matrix A of size (n x n)
% TODO: rand generates elements between [0,1], test negative numbers?
m = rand(n);
A = m'*m;
