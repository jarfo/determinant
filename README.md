## Symbolic, exact and division-free computation algorithms for the determinant and the characteristic polynomial ##

### **A simple expression for the determinant and characteristic polynomial** ###
Following the Dynamic Programming formulation of the clow-based method, the
Mahajan–Vinay DP computes the signed sum of clow sequences of *every* length
simultaneously. These signed sums are exactly the coefficients of the
characteristic polynomial, so the whole computation is a single time-invariant
linear recurrence — that is, repeated application of one fixed matrix.

$$P_{A}(\lambda) := \det(\lambda I - A) = q_{n}\lambda^{n} + q_{n-1}\lambda^{n-1} + \cdots + q_{1}\lambda + q_{0}$$

with

$$q_{i} = r^T\tilde{A}^{n-i}s$$

where $\tilde{A}$ is the *transition matrix* of the dynamic algorithm (as
described in [Rote]), $s$ encodes the initial conditions, and $r$ collects the
final accumulator node.

The determinant is the special case $i = 0$. Because $q_0$ is the constant term
of $\det(\lambda I - A)$, it carries the standard sign:

$$q_{0} = r^T\tilde{A}^{n}s = (-1)^{n}\det(A), \qquad\text{equivalently}\qquad \det(A) = (-1)^{n} r^T\tilde{A}^{n}s.$$

For $n=3$ we have:

$$r^T=\left[\begin{array}{lllllll}
0 & 0 & 0 & 0 & 0 & 0 & 1
\end{array}\right]$$


$$s^T=\left[\begin{array}{lllllll}
1 & 0 & 0 & 1 & 0 & 1 & 1
\end{array}\right]$$

and

$$\tilde{A} = \left[\begin{array}{ccccccc}
0 & 0 & 0 & 0 & 0 & 0 & 0 \\
a_{10} & a_{11} & a_{12} & 0 & 0 & 0 & 0 \\
a_{20} & a_{21} & a_{22} & 0 & 0 & 0 & 0 \\
-a_{00} & -a_{01} & -a_{02} & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & a_{21} & a_{22} & 0 & 0 \\
-a_{00} & -a_{01} & -a_{02} & -a_{11} & -a_{12} & 0 & 0 \\
-a_{00} & -a_{01} & -a_{02} & -a_{11} & -a_{12} & -a_{22} & 0
\end{array}\right]$$
  
### **Remark: matrix-pencil form** ###
The construction $A \mapsto \tilde{A}$ is linear, so
$\widetilde{\lambda I-A} = \lambda\tilde{I}-\tilde{A}$. Applying the expression
above to the pencil $\lambda I-A$ therefore restates it as a single polynomial
identity:

$$P_{A}(\lambda) = (-1)^{n} r^T(\lambda \tilde{I}-\tilde{A})^ns = r^T(\lambda^{n}\tilde{A}^0 + \lambda^{n-1}\tilde{A}^1 + \cdots + \lambda\tilde{A}^{n-1} + \tilde{A}^n)s$$

where $\tilde{I}$ is, for $n=3$,

$$\tilde{I} = \left[\begin{array}{ccccccc}
0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 0 \\
-1 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 \\
-1 & 0 & 0 & -1 & 0 & 0 & 0 \\
-1 & 0 & 0 & -1 & 0 & -1 & 0
\end{array}\right]$$
  
### **Corollary: resolvent form and spectral factorization** ###
The transition matrix is built so that $r^T\tilde{A}^{k}s = 0$ for every $k>n$
(no clow of length greater than $n$ reaches the accumulator node). Hence the
Neumann series of the resolvent of $\tilde{A}$ collapses to a finite sum, and by
the expression above it reproduces $\det(I-\lambda A)$, the *reversed*
characteristic polynomial of $A$:

$$r^T(I-\lambda\tilde{A})^{-1}s = \sum_{k=0}^{n} \left(r^T\tilde{A}^{k}s\right)\lambda^{k} = \sum_{k=0}^{n} q_{n-k}\,\lambda^{k} = \lambda^{n}P_A(1/\lambda) = \det(I-\lambda A).$$

In systems-theoretic terms, $(r,\tilde{A},s)$ is a finite-impulse-response
realization whose Markov parameters are the characteristic-polynomial
coefficients of $A$. The realization is far from minimal: every non-nilpotent
mode of $\tilde{A}$ is cancelled by the vectors $r$ and $s$, and these cancelled
modes are precisely the eigenvalues of the *trailing* principal submatrices of
$A$. Writing $A^{(i)}$ for the submatrix obtained by deleting the first $i$ rows
and columns (so $A^{(0)}=A$), the characteristic polynomial of $\tilde{A}$
factors as

$$\det(\lambda I-\tilde{A}) = \lambda^{\,n+1}\prod_{i=1}^{n-1}\det\big(\lambda I-A^{(i)}\big)$$

— the eigenvalue $0$ with multiplicity $n+1$, together with the spectra of
$A^{(1)},\dots,A^{(n-1)}$. Clearing the denominator of the resolvent through
$\operatorname{adj}(I-\lambda\tilde{A}) = \det(I-\lambda\tilde{A})\,(I-\lambda\tilde{A})^{-1}$
then gives the closed form

$$r^T\operatorname{adj}(I-\lambda\tilde{A})\,s = \det(I-\lambda A)\,\det(I-\lambda\tilde{A}) = \prod_{i=0}^{n-1}\det\big(I-\lambda A^{(i)}\big),$$

and matching powers of $\lambda$ yields explicit convolution relations between
the characteristic-polynomial coefficients of $A$ and those of its trailing
submatrices (relations of this type were anticipated by Rote, who derives one
such identity from his extended recursion system and remarks that it "may be an
interesting identity but is of little use for computing anything" [Rote, §2.5,
p. 124]). The $A^{(i)}$ are exactly the matrices processed by Samuelson-type
vertex-at-a-time recursions (see *CVcoefs*), so the "extra" spectrum of the
dynamic-programming transition matrix is the spectrum that those methods work
through explicitly.

*Proof of the factorization.* Group the states of the dynamic program by the
head of the current clow: block $i$ ($0\le i\le n$) collects the states with
head $i$, ordered by current vertex $v = i, i+1, \dots, n-1$, and block $n$ is
the single accumulator state — exactly the layout produced by *DPmatrix*
(visible in the $n=3$ example above as diagonal blocks of sizes $3,2,1,1$).
A transition either extends the current clow, keeping the head fixed, or closes
it and opens a new clow at a strictly larger head; heads never decrease, so
$\tilde{A}$ is block lower triangular and its characteristic polynomial is the
product of those of its diagonal blocks. Within block $i$, extending a clow
moves the current vertex to some $w \ge i+1$ (the head is the unique minimal
vertex of a clow), so the block's first state $(i,i)$ is never re-entered from
inside the block:

$$\tilde{A}_{ii} = \begin{pmatrix} 0 & 0 \\ A[\,i{+}1{:},\,i\,] & A^{(i+1)} \end{pmatrix},$$

with first row zero, and block $n$ is the $1\times 1$ zero matrix. Cofactor
expansion along the first row of $\lambda I-\tilde{A}_{ii}$ gives
$\det(\lambda I-\tilde{A}_{ii}) = \lambda\det\big(\lambda I-A^{(i+1)}\big)$, and the
product over the $n+1$ diagonal blocks ($A^{(n)}$ is empty, with determinant
$1$) is the stated factorization. $\blacksquare$

### **Consequence: one factorization, three algorithms** ###
Combining the resolvent form with the spectral factorization makes the
determinant identity telescope over the trailing submatrices:

$$\det(I-\lambda A)=\prod_{i=0}^{n-1}\frac{\det\big(I-\lambda A^{(i)}\big)}{\det\big(I-\lambda A^{(i+1)}\big)}=\prod_{i=0}^{n-1}\Big(1-\lambda\, a_{ii}-\lambda^{2}\,A[\,i,\,i{+}1{:}\,]\,\big(I-\lambda A^{(i+1)}\big)^{-1}A[\,i{+}1{:},\,i\,]\Big),$$

where each factor is simultaneously a Schur complement and the generating
function $1-C_i(\lambda)$ of the clows with head $i$ (a clow sequence chooses at
most one clow per head, independently, so the signed sum factors). The single
factor, with both of these readings, is due to Rote: the clow-splitting
recursion $P_A(\lambda)=F(\lambda)\,P_{A^{(1)}}(\lambda)$ [Rote, §§2.5–2.6,
eqs. (6)–(9)] and its algebraic derivation via the bordered determinant and the
resolvent of the trailing submatrix [Rote, §2.7, eqs. (14)–(15)]; the product
above is that recursion fully telescoped. Evaluating
this one product in the three natural ways recovers three classical algorithms:

- **with divisions**, $\lambda$-free: the pivots
  $a_{ii}-A[\,i,\,i{+}1{:}\,]\big(A^{(i+1)}\big)^{-1}A[\,i{+}1{:},\,i\,]$ give the
  Gaussian-elimination (LDU) pivot product for $\det(A)$;
- **division-free, one factor at a time**, each expanded as a power series
  truncated at $\lambda^{n}$: the Samuelson–Berkowitz recursion — this is
  *CVcoefs*, whose list $D$ holds precisely the coefficients
  $-1,\,a_{ii},\,rs,\,rMs,\dots$ of one factor;
- **all factors at once**, as the single sandwich
  $r^T(I-\lambda\tilde{A})^{-1}s$: the matrix-power method *MPcoefs*.

So the matrix-power and Samuelson-type methods are the same factorization read
in two directions, and both are division-free shadows of the pivot product.
As a practical byproduct, $\tilde{A}$ has only $\Theta(n^3)$ nonzero entries
(the diagonal blocks and closing rows), so performing the MP iteration
blockwise lowers its cost from $O(n^5)$ to $O(n^4)$ — recovering the arc count
of the original dynamic program, which Rote already gives as $O(n^4)$ [Rote,
§2.4], and matching Berkowitz, as it must, since the blockwise MP iteration is
the Berkowitz recursion reorganized.

### **Source code** ###
The [*DPmatrix*](https://github.com/jarfo/determinant/blob/ad5c47832fb23dbb504501e92c7f5e27e91b72af/determinant/determinant.py?plain=1#L103) function computes the $\tilde{A}$ matrix from $A$, and the *MPdet* and *MPcharpoly* functions use this method to compute the determinant or the characteristic polynomial.

### **References** ###
- [Meena Bhaskar Mahajan, V Vinay, Determinant: Combinatorics, algorithms, and complexity. Chicago J. Theor. Comput. Sci., Vol. 1997, Article no. 1997-5, 26 pp.](https://eccc.weizmann.ac.il/eccc-reports/1997/TR97-036/index.html)
- [Mahajan M., Vinay V. (1998) Determinant: Old algorithms, new insights. In: Arnborg S., Ivansson L. (eds) Algorithm Theory — SWAT'98. SWAT 1998. Lecture Notes in Computer Science, vol 1432. Springer, Berlin, Heidelberg.](https://doi.org/10.1007/BFb0054375)
- [Rote G. (2001) Division-Free Algorithms for the Determinant and the Pfaffian: Algebraic and Combinatorial Approaches. In: Alt H. (eds) Computational Discrete Mathematics. Lecture Notes in Computer Science, vol 2122. Springer, Berlin, Heidelberg.](https://doi.org/10.1007/3-540-45506-X_9)
- [Ikenmeyer C. (2025) On the gradient of the coefficient of the characteristic polynomial. arXiv:2511.04954.](https://arxiv.org/abs/2511.04954)
- [Bird R.S. (2011) A simple division-free algorithm for computing determinants. Information Processing Letters, 111(21–22), 1072–1074.](https://doi.org/10.1016/j.ipl.2011.08.006)
- [Strassen V. (1973) Vermeidung von Divisionen. Journal für die reine und angewandte Mathematik, 264, 184–202.](https://doi.org/10.1515/crll.1973.264.184)
- [Kaltofen E. (1992) On computing determinants of matrices without divisions. In: Proceedings of the 1992 International Symposium on Symbolic and Algebraic Computation (ISSAC '92), 342–349. ACM.](https://doi.org/10.1145/143242.143350)

