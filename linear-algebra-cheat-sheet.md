# Linear Algebra Cheat Sheet

## Linearity
- A function is linear if it satisfies:
  - Additivity:  f(x + y) = f(x) + f(y)
  - Homogeneity:  f(ax) = af(x)

## Matrix Multiplication
- If A is an m × n matrix and B is an n × p matrix, the product AB is an m × p matrix.
- Formula:  (AB)_{ij} = \sum_{k=1}^{n} A_{ik}B_{kj}

## Systems of Equations
1. **Consistent**: At least one solution.
2. **Inconsistent**: No solution.
3. **Dependent**: Infinite solutions.

## Augmented Matrices
- Used to represent systems of equations. E.g., for:
  - 3x + 4y = 5  
  - 2x + 5y = 8  

  The augmented matrix is:
  \[ \begin{pmatrix} 3 & 4 & | & 5 \\ 2 & 5 & | & 8 \end{pmatrix} \]

## Gaussian Elimination
- A method to solve linear systems by transforming the augmented matrix to Row Echelon Form (REF).

## REF (Row Echelon Form)
- Each leading entry of a row is in a column to the right of the leading entry of the previous row.
- All entries below each leading entry are zero.

## RREF (Reduced Row Echelon Form)
- In RREF, every leading entry is 1, and is the only non-zero entry in its column.

## Free Variables
- Variables that can take any value in a system of equations. Occur when there are fewer pivot columns than variables.

## Solution Sets
- The set of all possible solutions to a system of equations, which can be expressed in terms of free variables.

## Determinants
- A scalar value that is a function of a square matrix. Measures the volume scaling factor of the linear transformation.
- To compute Det(A):
  - For 2x2: Det(A) = ad - bc for A = \[ \begin{pmatrix} a & b \\ c & d \end{pmatrix} \]

## Inverse Matrices
- A matrix A has an inverse (denoted A^{-1}) if: A * A^{-1} = I.
- For 2x2: \[ A^{-1} = \frac{1}{ad-bc} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix} \]

## Inverse Properties
1. (A^{-1})^{-1} = A
2. (AB)^{-1} = B^{-1}A^{-1}
3. (A^T)^{-1} = (A^{-1})^T

## Invertible Matrix Theorem
- A square matrix A is invertible if and only if:
  - its determinant is non-zero,  
  - the columns of A span R^n,
  - the system Ax = 0 has only the trivial solution.

## Common Patterns
- **Eigenvalues & Eigenvectors**: Solutions to the equation Ax = λx.

## Common Mistakes
1. Confusing linear dependence with independence.
2. Miscalculating determinants.
3. Not identifying free variables correctly.

## Test-Taking Strategies
1. Review common formulas and theorems.
2. Practice with previous exam problems.
3. Allocate time wisely between questions.
