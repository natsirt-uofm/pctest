# LINEAR ALGEBRA CHEAT SHEET

> **Comprehensive reference for linear transformations, matrix operations, and systems of equations.**

---

## Table of Contents

1. [Linearity (Additivity & Homogeneity)](#1-linearity-additivity--homogeneity)
2. [Matrix Multiplication](#2-matrix-multiplication)
3. [Systems of Equations](#3-systems-of-equations)
4. [Augmented Matrix](#4-augmented-matrix)
5. [Gaussian Elimination](#5-gaussian-elimination)
6. [Row Echelon Form (REF)](#6-row-echelon-form-ref)
7. [Reduced Row Echelon Form (RREF)](#7-reduced-row-echelon-form-rref)
8. [Free Variables & Solution Sets](#8-free-variables--solution-sets)
9. [Determinant](#9-determinant)
10. [Inverse Matrix (2×2)](#10-inverse-matrix-2x2)
11. [Inverse Matrix (n×n)](#11-inverse-matrix-nxn)
12. [Inverse Properties](#12-inverse-properties)
13. [Invertible Matrix Theorem](#13-invertible-matrix-theorem)
14. [Common Patterns](#14-common-patterns)
15. [Common Mistakes](#15-common-mistakes)
16. [Test-Taking Strategy](#16-test-taking-strategy)
17. [Quick Reference Formulas](#17-quick-reference-formulas)

---

## 1. Linearity (Additivity & Homogeneity)

### What Makes Something "Linear"?

A transformation T is **linear** if and only if it satisfies **two properties**:

#### Property 1: Additivity

```
T(u + v) = T(u) + T(v)
```

*In plain English:* Add inputs first then transform = transform each then add.

#### Property 2: Homogeneity (Scaling)

```
T(c * u) = c * T(u)
```

*In plain English:* Scaling the input by c scales the output by the same c.

#### Combined Super-Property

```
T(c1*u + c2*v) = c1*T(u) + c2*T(v)
```

If this holds for **all** vectors u, v and **all** scalars c1, c2 → T is linear.

> **⚠️ Quick Disqualifier:** If T(0) ≠ 0, then T is **NOT** linear. Period.
>
> *Why?* By homogeneity: T(0) = T(0·v) = 0·T(v) = 0. Any linear transformation **must** send the zero vector to the zero vector.

---

### Proof: Showing a Function IS Linear

**Claim:** T(x, y) = (3x − y, 2x + 5y) is linear.

```
Let u = (u1, u2), v = (v1, v2), and c1, c2 be arbitrary scalars.

LEFT SIDE:
  c1*u + c2*v = (c1*u1 + c2*v1,  c1*u2 + c2*v2)

  T(c1*u + c2*v) = ( 3(c1*u1 + c2*v1) − (c1*u2 + c2*v2),
                     2(c1*u1 + c2*v1) + 5(c1*u2 + c2*v2) )

                 = ( c1(3u1 − u2) + c2(3v1 − v2),
                     c1(2u1 + 5u2) + c2(2v1 + 5v2) )

RIGHT SIDE:
  c1*T(u) + c2*T(v) = ( c1(3u1 − u2) + c2(3v1 − v2),
                         c1(2u1 + 5u2) + c2(2v1 + 5v2) )

LEFT = RIGHT ✅  →  T is linear.
```

### Disproof: Showing a Function is NOT Linear

**Claim:** T(x, y) = (x², y) is **not** linear.

```
Let u = (1, 0) and c = 2.

T(c·u) = T(2, 0) = (4, 0)
c·T(u) = 2·T(1,0) = 2·(1, 0) = (2, 0)

(4, 0) ≠ (2, 0)  ❌  →  Homogeneity fails  →  NOT linear.
```

---

### Linear vs. Non-Linear Examples

| Function | Linear? | Reason |
|---|---|---|
| T(x, y) = (2x, 3y) | ✅ YES | Only scaling of components |
| T(x, y) = (x + y, x − y) | ✅ YES | Only addition/subtraction |
| T(x, y, z) = (x + z, y − z) | ✅ YES | Add/subtract components only |
| T(x, y) = (0, 0) | ✅ YES | Zero transformation (valid) |
| T(x, y) = (x + 1, y) | ❌ NO | T(0,0) = (1,0) ≠ (0,0) |
| T(x) = x² | ❌ NO | T(2x) = 4x² ≠ 2x² |
| T(x, y) = (xy, 0) | ❌ NO | Multiplying variables together |
| T(x) = \|x\| | ❌ NO | T(−1) = 1 but −1·T(1) = −1 |
| T(x) = sin(x) | ❌ NO | sin(a+b) ≠ sin(a) + sin(b) |

> **💡 Rules of Thumb:**
> - **Linear:** only addition, subtraction, and scalar multiplication of variables
> - **Not linear:** squaring, products of variables, absolute value, trig, adding constants, roots

---

## 2. Matrix Multiplication

### What is a Matrix?

A matrix is a rectangular grid of numbers with **m rows** and **n columns** (an *m × n* matrix):

```
A = [ 1  2  3 ]    ← 2×3 matrix
    [ 4  5  6 ]      (2 rows, 3 columns)
```

Every linear transformation T: Rⁿ → Rᵐ can be written as **T(x) = Ax** for some m × n matrix A.

### The Size Rule

```
A is (m × n)  and  B is (n × p)
          ^              ^
          These MUST match!

Result AB is (m × p)
```

If inner dimensions don't match → **multiplication is impossible**.

### Worked Example 1: 2×2 times 2×2

```
A = [ 1  2 ]    B = [ 5  6 ]
    [ 3  4 ]        [ 7  8 ]

Position (1,1): (1)(5) + (2)(7) = 5 + 14  = 19
Position (1,2): (1)(6) + (2)(8) = 6 + 16  = 22
Position (2,1): (3)(5) + (4)(7) = 15 + 28 = 43
Position (2,2): (3)(6) + (4)(8) = 18 + 32 = 50

        [ 19  22 ]
AB  =   [ 43  50 ]
```

### Worked Example 2: 2×3 times 3×1

```
A = [ 1  2  3 ]    B = [ 2 ]
    [ 4  5  6 ]        [ 1 ]
                       [ 3 ]

A is 2×3, B is 3×1 → inner dims match → result is 2×1

Row 1 · Col 1: (1)(2) + (2)(1) + (3)(3) = 2 + 2 + 9  = 13
Row 2 · Col 1: (4)(2) + (5)(1) + (6)(3) = 8 + 5 + 18 = 31

        [ 13 ]
AB  =   [ 31 ]
```

### Properties of Matrix Multiplication

| Property | True? | Note |
|---|---|---|
| AB = BA | ❌ NO | **Not** commutative |
| A(BC) = (AB)C | ✅ YES | Associative |
| A(B + C) = AB + AC | ✅ YES | Distributive |
| AI = IA = A | ✅ YES | Identity acts like 1 |

### Why AB ≠ BA (Counterexample)

```
A = [ 1  0 ]    B = [ 0  1 ]
    [ 0  0 ]        [ 0  0 ]

AB = [ 0  1 ]    BA = [ 0  0 ]     ← Completely different!
     [ 0  0 ]         [ 0  0 ]
```

### Identity Matrix

```
I₂ = [ 1  0 ]      I₃ = [ 1  0  0 ]
     [ 0  1 ]            [ 0  1  0 ]
                         [ 0  0  1 ]
```

Multiplying any matrix by I leaves it unchanged: **AI = IA = A**.

---

## 3. Systems of Equations

### Three Possible Outcomes

When you solve a system of linear equations you will get **exactly one** of:

1. ✅ **Exactly one solution** — lines/planes intersect at a single point
2. ♾️ **Infinitely many solutions** — equations describe the same line/plane (dependent)
3. ❌ **No solution** — lines/planes are parallel, never meet (inconsistent)

### How to Identify from RREF

| RREF Pattern | Solutions | Geometry |
|---|---|---|
| `[1 0 \| a]` / `[0 1 \| b]` | One unique solution | Point |
| `[1 * \| *]` / `[0 0 \| 0]` | Infinitely many (1 free var) | Line |
| `[1 * * \| *]` / `[0 0 0 \| 0]` | Infinitely many (2 free vars) | Plane |
| `[0 0 \| c]` where c ≠ 0 | No solution | Empty set |

> **Key rule:** A row of the form `[0  0  …  0 | c]` with **c ≠ 0** means **no solution** (inconsistent system).

---

## 4. Augmented Matrix

### How to Construct

Write coefficients in a matrix, then append the constants after a vertical bar `|`:

**System:**
```
3x + 5y = 13
2x + 6y = 14
```

**Augmented matrix:**
```
[ 3  5 | 13 ]
[ 2  6 | 14 ]
```

### Example: 3 Variables

**System:**
```
x +  y +  z =  6
x + 2y + 3z = 14
2x +  y + 2z = 10
```

**Augmented matrix:**
```
[ 1  1  1 |  6 ]
[ 1  2  3 | 14 ]
[ 2  1  2 | 10 ]
```

---

## 5. Gaussian Elimination

### The Three Row Operations

1. **Swap two rows:** R1 ↔ R2
2. **Multiply a row by a nonzero scalar:** k·Rᵢ → Rᵢ (k ≠ 0)
3. **Add a multiple of one row to another:** Rᵢ + k·Rⱼ → Rᵢ

These operations **never change the solution set**.

### Step-by-Step Strategy

1. Write the augmented matrix `[A | b]`
2. Find the leftmost nonzero column (the **pivot column**)
3. Swap rows if needed so the pivot position is nonzero
4. Scale the pivot row to make the pivot = 1 (optional for REF, required for RREF)
5. Eliminate all entries **below** the pivot using row operations
6. Repeat for the next pivot column (ignoring rows already processed)
7. For RREF: eliminate entries **above** each pivot as well

### Complete Worked Example

**System:**
```
x +  y +  z =  6
x + 2y + 3z = 14
2x +  y + 2z = 10
```

**Step 1: Write augmented matrix**
```
[ 1  1  1 |  6 ]
[ 1  2  3 | 14 ]
[ 2  1  2 | 10 ]
```

**Step 2: Eliminate below first pivot (column 1)**

R2 → R2 − R1:
```
[ 1  1  1 |  6 ]
[ 0  1  2 |  8 ]
[ 2  1  2 | 10 ]
```

R3 → R3 − 2·R1:
```
[ 1  1  1 |  6 ]
[ 0  1  2 |  8 ]
[ 0 -1  0 | -2 ]
```

**Step 3: Eliminate below second pivot (column 2)**

R3 → R3 + R2:
```
[ 1  1  1 | 6 ]
[ 0  1  2 | 8 ]
[ 0  0  2 | 6 ]    ← This is REF!
```

**Step 4: Back-substitution from REF**

```
From Row 3:  2z = 6   →  z = 3
From Row 2:  y + 2(3) = 8  →  y = 2
From Row 1:  x + 2 + 3 = 6  →  x = 1

Solution: (x, y, z) = (1, 2, 3)
```

**Step 5: Continue to RREF**

R3 → (1/2)·R3:
```
[ 1  1  1 | 6 ]
[ 0  1  2 | 8 ]
[ 0  0  1 | 3 ]
```

R2 → R2 − 2·R3, R1 → R1 − R3:
```
[ 1  1  0 | 3 ]
[ 0  1  0 | 2 ]
[ 0  0  1 | 3 ]
```

R1 → R1 − R2:
```
[ 1  0  0 | 1 ]
[ 0  1  0 | 2 ]    ← RREF: read directly x=1, y=2, z=3
[ 0  0  1 | 3 ]
```

---

## 6. Row Echelon Form (REF)

### Pattern

```
[ #  *  *  * ]    # = pivot (leading nonzero)
[ 0  #  *  * ]    * = any value
[ 0  0  #  * ]    0 = must be zero
[ 0  0  0  0 ]    zero rows at the bottom
```

### Requirements

- [ ] Each pivot is strictly to the **right** of the pivot in the row above
- [ ] All entries **below** a pivot are zero
- [ ] Zero rows are at the **bottom**

### Example

```
[ 1  3  0  2 ]
[ 0  0  2  5 ]    ← pivot in column 3 (skips column 2)
[ 0  0  0  1 ]
[ 0  0  0  0 ]
```

> **💡 TIP:** REF is not unique — you can have many valid REF forms for the same matrix. RREF is unique.

---

## 7. Reduced Row Echelon Form (RREF)

### Requirements (Stricter than REF)

- [ ] All REF requirements
- [ ] Every pivot equals **1**
- [ ] Every entry **above** a pivot is also **zero**

### Pattern

```
[ 1  0  0 | a ]
[ 0  1  0 | b ]
[ 0  0  1 | c ]
```

Read answers directly: x = a, y = b, z = c.

### RREF vs REF Comparison

| Feature | REF | RREF |
|---|---|---|
| Zeros below pivots | ✅ YES | ✅ YES |
| Zeros **above** pivots | ❌ Not required | ✅ YES |
| Pivots equal to 1 | ❌ Not required | ✅ YES |
| Need back-substitution? | ✅ YES | ❌ NO (read directly) |
| Unique result? | ❌ NO | ✅ YES |

### Reading Solutions from RREF

**Example RREF with free variable:**
```
[ 1  0  3 | 5 ]
[ 0  1 -2 | 1 ]
[ 0  0  0 | 0 ]
```

- Column 1 → pivot (x₁ is a **basic variable**)
- Column 2 → pivot (x₂ is a **basic variable**)
- Column 3 → no pivot (x₃ is a **free variable**)

```
x₁ + 3x₃ = 5   →   x₁ = 5 − 3t
x₂ − 2x₃ = 1   →   x₂ = 1 + 2t
x₃ = t               (free, any real number)
```

---

## 8. Free Variables & Solution Sets

### What is a Free Variable?

A variable corresponding to a **non-pivot column** in RREF. It can take **any** real value, making infinitely many solutions.

### One Free Variable → Line

**RREF:**
```
[ 1  0  3 | 5 ]
[ 0  1 -2 | 1 ]
```

**Parametric form:** Let x₃ = t

```
x₁ = 5 − 3t
x₂ = 1 + 2t
x₃ = t
```

**Vector form:**
```
[ x₁ ]   [ 5 ]       [ -3 ]
[ x₂ ] = [ 1 ] + t · [  2 ]    (a line in 3D space)
[ x₃ ]   [ 0 ]       [  1 ]
```

### Two Free Variables → Plane

**RREF:**
```
[ 1  2  0  3 | 4 ]
[ 0  0  1  5 | 2 ]
```

Let x₂ = s, x₄ = t (free variables):

```
x₁ = 4 − 2s − 3t
x₂ = s
x₃ = 2 − 5t
x₄ = t
```

**Vector form:**
```
[ x₁ ]   [ 4 ]       [ -2 ]       [ -3 ]
[ x₂ ] = [ 0 ] + s · [  1 ] + t · [  0 ]
[ x₃ ]   [ 2 ]       [  0 ]       [ -5 ]
[ x₄ ]   [ 0 ]       [  0 ]       [  1 ]
```

This is a **plane** in 4D space.

### Solution Geometry Summary

| Situation | What you get | Geometry |
|---|---|---|
| Every variable is a pivot | Unique solution | Point |
| 1 free variable | Infinitely many | Line |
| 2 free variables | Infinitely many | Plane |
| Row `[0 0 0 \| c]`, c ≠ 0 | No solution | Empty set |

---

## 9. Determinant

### 2×2 Formula

```
      [ a  b ]
A  =  [ c  d ]

det(A) = ad − bc
```

### 3×3 Formula (Cofactor Expansion along Row 1)

```
      [ a  b  c ]
A  =  [ d  e  f ]
      [ g  h  i ]

det(A) = a(ei − fh) − b(di − fg) + c(dh − eg)
```

**Memory trick:** Alternating signs: **+ − +**

### 3×3 Worked Example

```
A = [ 1  2  3 ]
    [ 4  5  6 ]
    [ 7  8  9 ]

det(A) = 1·(5·9 − 6·8) − 2·(4·9 − 6·7) + 3·(4·8 − 5·7)
       = 1·(45 − 48) − 2·(36 − 42) + 3·(32 − 35)
       = 1·(−3) − 2·(−6) + 3·(−3)
       = −3 + 12 − 9
       = 0
```

det = 0 → matrix is **singular** (not invertible).

### Key Properties of Determinants

| Property | Formula |
|---|---|
| Determinant of product | det(AB) = det(A)·det(B) |
| Determinant of inverse | det(A⁻¹) = 1/det(A) |
| Determinant of transpose | det(Aᵀ) = det(A) |
| Row swap | det changes sign |
| Row scaling by k | det multiplied by k |
| Row of zeros | det = 0 |
| Triangular matrix | det = product of diagonal entries |

> **Key fact:** A matrix is invertible **if and only if** det(A) ≠ 0.

---

## 10. Inverse Matrix (2×2)

### Formula

```
      [ a  b ]              1      [  d  -b ]
A  =  [ c  d ]    A⁻¹ =  ──────  [ -c   a ]
                           ad−bc
```

**In words:** Swap the main diagonal, negate the off-diagonal, divide by the determinant.

### Step-by-Step Process

1. Compute det(A) = ad − bc
2. Check det(A) ≠ 0 (otherwise no inverse exists)
3. Swap a ↔ d
4. Negate b and c (flip signs)
5. Multiply the whole matrix by 1/det(A)

### Worked Example

```
A = [ 4  7 ]
    [ 2  6 ]

det(A) = (4)(6) − (7)(2) = 24 − 14 = 10  (nonzero ✅)

A⁻¹ = (1/10) · [  6  -7 ]  =  [  0.6  -0.7 ]
               [ -2   4 ]     [ -0.2   0.4 ]
```

### Verification

```
A · A⁻¹ = [ 4  7 ] · [  0.6  -0.7 ]
           [ 2  6 ]   [ -0.2   0.4 ]

        = [ (4)(0.6)+(7)(−0.2)   (4)(−0.7)+(7)(0.4) ]
          [ (2)(0.6)+(6)(−0.2)   (2)(−0.7)+(6)(0.4) ]

        = [ 2.4−1.4   −2.8+2.8 ]   = [ 1  0 ]  ✅
          [ 1.2−1.2   −1.4+2.4 ]     [ 0  1 ]
```

---

## 11. Inverse Matrix (n×n)

### Row Reduction Method

Set up the augmented matrix **[A | I]**, then row-reduce the left side to I:

```
[ A | I ]   →   row reduce   →   [ I | A⁻¹ ]
```

If the left side **cannot** be reduced to I (you get a zero row), then A is **not invertible**.

### Complete 3×3 Example

```
A = [ 1  1  0 ]
    [ 0  1  1 ]
    [ 1  0  1 ]
```

**Set up [A | I]:**
```
[ 1  1  0 | 1  0  0 ]
[ 0  1  1 | 0  1  0 ]
[ 1  0  1 | 0  0  1 ]
```

R3 → R3 − R1:
```
[ 1  1  0 |  1  0  0 ]
[ 0  1  1 |  0  1  0 ]
[ 0 -1  1 | -1  0  1 ]
```

R3 → R3 + R2:
```
[ 1  1  0 |  1  0  0 ]
[ 0  1  1 |  0  1  0 ]
[ 0  0  2 | -1  1  1 ]
```

R3 → (1/2)·R3:
```
[ 1  1  0 |  1     0     0   ]
[ 0  1  1 |  0     1     0   ]
[ 0  0  1 | -1/2   1/2   1/2 ]
```

R2 → R2 − R3:
```
[ 1  1  0 |  1     0     0   ]
[ 0  1  0 |  1/2   1/2  -1/2 ]
[ 0  0  1 | -1/2   1/2   1/2 ]
```

R1 → R1 − R2:
```
[ 1  0  0 |  1/2  -1/2   1/2 ]
[ 0  1  0 |  1/2   1/2  -1/2 ]
[ 0  0  1 | -1/2   1/2   1/2 ]
```

**Result:**
```
        [  1/2  -1/2   1/2 ]
A⁻¹ =  [  1/2   1/2  -1/2 ]
        [ -1/2   1/2   1/2 ]
```

### When a Matrix is NOT Invertible

A matrix is **not invertible** (singular) when:
- det(A) = 0
- RREF of A has a row of zeros (≠ I)
- Rows (or columns) are linearly dependent

**Example:**
```
B = [ 1  2 ]     det(B) = (1)(4) − (2)(2) = 0
    [ 2  4 ]

NOT invertible — Row 2 = 2 × Row 1.
```

---

## 12. Inverse Properties

### Key Properties

| Property | Formula |
|---|---|
| Inverse of inverse | (A⁻¹)⁻¹ = A |
| Product inverse | **(AB)⁻¹ = B⁻¹A⁻¹** (order reverses!) |
| Transpose inverse | (Aᵀ)⁻¹ = (A⁻¹)ᵀ |
| Scalar inverse | (kA)⁻¹ = (1/k)A⁻¹ |
| Determinant | det(A⁻¹) = 1/det(A) |

> **⚠️ COMMON MISTAKE:** (AB)⁻¹ = B⁻¹A⁻¹, **NOT** A⁻¹B⁻¹. The order reverses!

### Solving Systems Using the Inverse

If A is invertible and **Ax = b**, then:

```
x = A⁻¹ · b
```

**Example:**
```
A = [ 2  1 ]    b = [ 5 ]
    [ 1  3 ]        [ 7 ]

det(A) = 6 − 1 = 5

A⁻¹ = (1/5) · [  3  -1 ]
               [ -1   2 ]

x = A⁻¹b = (1/5) · [  3  -1 ] · [ 5 ]  =  (1/5) · [ 8  ]  =  [ 8/5  ]
                    [ -1   2 ]   [ 7 ]               [ 9  ]     [ 9/5  ]
```

---

## 13. Invertible Matrix Theorem

For an *n × n* matrix A, the following are **ALL equivalent** — either all true or all false:

- [ ] A is invertible
- [ ] A⁻¹ exists
- [ ] det(A) ≠ 0
- [ ] RREF of A = I (the identity matrix)
- [ ] Ax = 0 has **only** the trivial solution (x = 0)
- [ ] Ax = b has **exactly one solution** for every b
- [ ] The columns of A are **linearly independent**
- [ ] The columns of A **span** Rⁿ
- [ ] rank(A) = n
- [ ] null space of A = {0} (nullity = 0)

> **If ANY one condition is true, they are ALL true. If any one fails, they ALL fail.**

---

## 14. Common Patterns

### Same Line (Dependent — Infinitely Many Solutions)

```
2x + y = 4
4x + 2y = 8     ← Row 2 = 2 × Row 1

Augmented: [ 2  1 | 4 ]    After elimination:  [ 1  1/2 | 2 ]
           [ 4  2 | 8 ]                         [ 0  0   | 0 ]

→ Zero row → infinitely many solutions → y is free
```

### Parallel Lines (Inconsistent — No Solution)

```
x + 2y = 4
x + 2y = 7     ← Same left side, different right side

Augmented: [ 1  2 | 4 ]    After elimination:  [ 1  2 | 4 ]
           [ 1  2 | 7 ]                         [ 0  0 | 3 ]

→ [0 0 | 3] means 0 = 3 → IMPOSSIBLE → no solution
```

### Intersecting Lines (Unique Solution)

```
3x + 5y = 13
2x + 6y = 14

After RREF:  [ 1  0 | 1 ]
             [ 0  1 | 2 ]

→ Unique solution: (x, y) = (1, 2)
```

---

## 15. Common Mistakes

- ❌ **Forgetting to check T(0) = 0** before testing linearity — it's the fastest disqualifier.
- ❌ **Reversing the order in (AB)⁻¹** — it's B⁻¹A⁻¹, not A⁻¹B⁻¹.
- ❌ **Assuming AB = BA** — matrix multiplication is generally NOT commutative.
- ❌ **Wrong size check for multiplication** — the *inner* dimensions must match (m×**n** times **n**×p).
- ❌ **Misidentifying free variables** — a variable is free only if its column has **no pivot** in RREF.
- ❌ **Forgetting the sign pattern in 3×3 determinants** — it's +, −, + along the first row.
- ❌ **Using det = 0 to mean "zero matrix"** — det = 0 means singular/non-invertible, not that A is zero.
- ❌ **Stopping at REF instead of RREF** — REF still requires back-substitution; RREF gives answers directly.
- ❌ **Row operation errors** — be careful with signs when subtracting multiples of rows.
- ❌ **Multiplying rows instead of adding multiples** — only the third row operation allows combining two rows.

---

## 16. Test-Taking Strategy

### Quick Approach by Problem Type

| Problem Type | First Step | Key Formula/Method |
|---|---|---|
| Is T linear? | Check T(0) = 0 first | Then test additivity & homogeneity |
| Matrix product size? | Check inner dimensions match | Result is outer dimensions |
| Solve system? | Write augmented matrix | Row reduce to RREF |
| How many solutions? | Look at RREF pattern | Zero row = ∞ or none; `[0 0 ... 0 \| c≠0]` = none |
| Find determinant (2×2)? | Use ad − bc | Check sign! |
| Find determinant (3×3)? | Cofactor expansion Row 1 | Alternate signs: +, −, + |
| Find A⁻¹ (2×2)? | Compute det first | Swap diag, negate off-diag, ÷ det |
| Find A⁻¹ (n×n)? | Set up [A \| I] | Row reduce to [I \| A⁻¹] |
| Is matrix invertible? | Check det(A) ≠ 0 OR RREF = I | Invertible Matrix Theorem |

### General Tips

> **💡 TIP:** Always verify your answer if time permits — multiply A · A⁻¹ to check you get I, or substitute your solution back into the original equations.

> **💡 TIP:** For linearity proofs, write T(c₁u + c₂v) on the left and c₁T(u) + c₂T(v) on the right, then show they are equal algebraically.

> **💡 TIP:** When row reducing, write down every operation you perform (e.g., "R2 → R2 − 2R1") to avoid confusion and earn partial credit.

> **💡 TIP:** If a row reduces to all zeros on the left of `|` but a nonzero value on the right, **stop** — the system is inconsistent.

---

## 17. Quick Reference Formulas

### Linearity

```
Additivity:    T(u + v)     = T(u) + T(v)
Homogeneity:   T(c·u)       = c·T(u)
Combined:      T(c₁u + c₂v) = c₁T(u) + c₂T(v)
Disqualifier:  T(0) ≠ 0  →  NOT linear
```

### Matrix Sizes

```
(m×n) · (n×p) = (m×p)    [inner must match, result = outer]
```

### Determinants

```
2×2:  det[ a  b ] = ad − bc
         [ c  d ]

3×3:  det(A) = a(ei−fh) − b(di−fg) + c(dh−eg)
      where A = [ a  b  c ]
                [ d  e  f ]
                [ g  h  i ]
```

### 2×2 Inverse

```
A = [ a  b ]  →  A⁻¹ = ──1──  [  d  −b ]
    [ c  d ]          ad−bc   [ −c   a ]
```

### Inverse Properties

```
(A⁻¹)⁻¹  = A
(AB)⁻¹   = B⁻¹A⁻¹       ← order reverses!
(Aᵀ)⁻¹   = (A⁻¹)ᵀ
det(A⁻¹) = 1/det(A)
```

### Solving with Inverse

```
Ax = b  →  x = A⁻¹b
```

### Solution Types (by RREF)

```
Every column has a pivot              →  Unique solution
One or more non-pivot columns         →  Infinitely many solutions (free variables)
Row [ 0  0  …  0 | c ] with c ≠ 0    →  No solution (inconsistent)
```

### Row Reduction Target

```
[A | I]  →  row reduce  →  [I | A⁻¹]
```

---

*End of Cheat Sheet — Good luck on your exam!* 🎓
