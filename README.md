## Symbolic, exact and division-free computation algorithms for the determinant and the characteristic polynomial ##
  
  

### **A simple expression for the determinant and characteristic polynomial** ###
If we follow the Dynamic Programming approach of the clow-based method, we can obtain a simple Matrix power expression for the determinant and all the coefficients of the characteristic polynomial.

$$\mathrm{det}(A)=r\tilde{A}^ns$$

$$P_{A}(\lambda):=\mathrm{det}(\lambda I-A)=q_{n} \lambda^{n}+q_{n-1} \lambda^{n-1}+\cdots+q_{1} \lambda+q_{0}$$

with

$$q_{i} = r\tilde{A}^{n-i}s$$

where $\tilde{A}$ is the 'transition matrix' of the dynamic algorithm as described in [Rote], $s$ corresponds to the initial conditions and $r$ to the final collecting node.

For $n=3$ we have:

$$r=\left[\begin{array}{lllllll}
0 & 0 & 0 & 0 & 0 & 0 & 1
\end{array}\right]$$


$$s=\left[\begin{array}{lllllll}
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
  
### **Corollary** ###
$$P_{A}(\lambda):=\mathrm{det}(\lambda I-A)=r\widetilde{(\lambda I-A)}^ns =r(\lambda \tilde{I}-\tilde{A})^ns = r(\lambda^{n}+\tilde{A} \lambda^{n-1}+\cdots+\tilde{A}^{n-1} \lambda + \tilde{A}^n)s$$

where $\tilde{I}$ is, for n=3,

$$\tilde{I} = \left[\begin{array}{ccccccc}
0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 0 \\
-1 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 \\
-1 & 0 & 0 & -1 & 0 & 0 & 0 \\
-1 & 0 & 0 & -1 & 0 & -1 & 0
\end{array}\right]$$
  
### **Source code** ##
  \
The [*DPmatrix*](https://github.com/jarfo/determinant/blob/ad5c47832fb23dbb504501e92c7f5e27e91b72af/determinant/determinant.py?plain=1#L103) function computes the $\tilde{A}$ matrix from $A$, and the *MPdet* and *MPcharpoly* functions use this method to compute the determinant or the characteristic polynomial.

### **References** ###
- [Meena Bhaskar Mahajan, V Vinay, Determinant: Combinatorics, algorithms, and complexity. Chicago J. Theor. Comput. Sci., Vol. 1997, Article no. 1997-5, 26 pp.](https://eccc.weizmann.ac.il/eccc-reports/1997/TR97-036/index.html)
- [Mahajan M., Vinay V. (1998) Determinant: Old algorithms, new insights. In: Arnborg S., Ivansson L. (eds) Algorithm Theory — SWAT'98. SWAT 1998. Lecture Notes in Computer Science, vol 1432. Springer, Berlin, Heidelberg.](https://doi.org/10.1007/BFb0054375)
- [Rote G. (2001) Division-Free Algorithms for the Determinant and the Pfaffian: Algebraic and Combinatorial Approaches. In: Alt H. (eds) Computational Discrete Mathematics. Lecture Notes in Computer Science, vol 2122. Springer, Berlin, Heidelberg.](https://doi.org/10.1007/3-540-45506-X_9)
