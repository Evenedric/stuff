
-- Check that a complex number is valid
function IsComplex(a)
  assert(getmetatable(a) == __complex_metatable__)
  assert(type(rawget(a, 1)) == 'number' or vec.IsVector(rawget(a, 1)))
  assert(type(rawget(a, 2)) == 'number' or vec.IsVector(rawget(a, 2)))
  assert(rawget(a, 'fn') == math or rawget(a, 'fn') == vec)
  return a
end

function TestComplex()
  print 'Test complex number (and vector) functionality in lua:'

  -- Complex number scalar tests.
  a = IsComplex(Complex(3, 4))
  assert(a.abs == 5)
  assert(math.abs(a.angle - 0.92729521800161) < 1e-8)
  b = IsComplex(Complex(2, -3))
  c = IsComplex(a + b)
  assert(c.re == 5 and c.im == 1)
  c = IsComplex(a - b)
  assert(c.re == 1 and c.im == 7)
  c = IsComplex(a * b)
  assert(c.re == 18 and c.im == -1)
  c = IsComplex(a / b)
  assert(c.re == -6/13 and c.im == 17/13)
  c = IsComplex(a ^ b)
  assert(math.abs(c.re + 398.060793444953) < 1e-9 and
         math.abs(c.im + 67.457050438711) < 1e-9)
  c = IsComplex(-a)
  assert(c.re == -3 and c.im == -4)

  -- Check combining scalar and complex arguments.
  c = IsComplex(a + 2)
  assert(c.re == 5)
  assert(c.im == 4)
  c = IsComplex(2 + a)
  assert(c.re == 5)
  assert(c.im == 4)

  -- Complex number vector tests.
  local function CheckVec(result, v1, v2)
    assert(vec.IsVector(result))
    assert(#result == 2)
    assert(math.abs(result[1] - v1) < 1e-8)
    assert(math.abs(result[2] - v2) < 1e-8)
  end
  a_re = Vector():Resize(2)
  a_im = Vector():Resize(2)
  a_re[1] = 3;
  a_im[1] = 4;
  a_re[2] = 1;
  a_im[2] = 2;
  b_re = Vector():Resize(2)
  b_im = Vector():Resize(2)
  b_re[1] = 2;
  b_im[1] = -3;
  b_re[2] = 5;
  b_im[2] = -4;
  a = IsComplex(Complex(a_re, a_im))
  b = IsComplex(Complex(b_re, b_im))
  assert(vec.IsVector(a.re))
  assert(vec.IsVector(a.im))
  assert(#a.re == 2)
  assert(#a.im == 2)
  c = a.abs
  CheckVec(c, 5, math.sqrt(5));
  c = a.angle
  CheckVec(c, 0.92729521800161, 1.10714871779409);
  c = IsComplex(a + b)
  CheckVec(c.re, 5, 6)
  CheckVec(c.im, 1, -2)
  c = IsComplex(a - b)
  CheckVec(c.re, 1, -4)
  CheckVec(c.im, 7, 6)
  c = IsComplex(a * b)
  CheckVec(c.re, 18, 13)
  CheckVec(c.im, -1, 6)
  c = IsComplex(a / b)
  CheckVec(c.re, -6/13, -3/41)
  CheckVec(c.im, 17/13, 14/41)
  c = IsComplex(a ^ b)
  CheckVec(c.re, -398.060793444953, -3180.20371857103)
  CheckVec(c.im, -67.457050438711, 3440.71717759186)
  c = IsComplex(-a)
  CheckVec(c.re, -3, -1)
  CheckVec(c.im, -4, -2)

  -- Check combining vector and complex arguments.
  c = IsComplex(a + 2)
  CheckVec(c.re, 3+2, 1+2)
  CheckVec(c.im, 4+0, 2+0)
  c = IsComplex(2 + a)
  CheckVec(c.re, 3+2, 1+2)
  CheckVec(c.im, 4+0, 2+0)
  d = Vector():Resize(2)
  d[1] = 2
  d[2] = 3
  c = IsComplex(a + d)
  CheckVec(c.re, 3+2, 1+3)
  CheckVec(c.im, 4+0, 2+0)
  -- This doesn't work as you might expect because the overloaded operator
  -- function that is used is from the vector and not from the complex table.
  --   c = d + a

  print 'Success'
end

-- A dummy configuration.
config = {
  type = 'Ez',
  unit = 'm',
  mesh_edge_length = 0.1,
  mesh_refines = 0,
  excited_port = 1,
  frequency = 60e9,
  depth = 0,
  cd = Rectangle(0, 0, 1, 1),
  test = TestComplex,
}