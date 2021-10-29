## Symbolic, exact and division-free computation algorithms for the determinant and the characteristic polynomial ##


### Explicit Matrix power method for the Clow-based algorithm ###
If we follow the Dynamic Programming approach of the clow-based method, we obtain a simple expression for the determinant and all the coefficients of the charasteristic polynomial.

<img src="https://render.githubusercontent.com/render/math?math=P_{A}(\lambda):=\operatorname{det}(\lambda I-A)=q_{n} \lambda^{n}%2Bq_{n-1} \lambda^{n-1}%2B\cdots%2Bq_{1} \lambda%2Bq_{0}">

with

<img src="https://render.githubusercontent.com/render/math?math=q_{i} = rM^{n-i}s">

where M is matrix is the 'transition matrix' of the dynamic algorithm as described in [Rote], s corresponds to the initial conditions and r to the final collecting node.

For n=3 we have:

<img src="https://render.githubusercontent.com/render/math?math=r = [0, 0, 0, 0, 0, 0, 1]">
<img src="https://render.githubusercontent.com/render/math?math=s = [0, 0, 1, 0, 1, 1, 1]^t">

and M is:

![M](M3.png)



### References ###
- [Meena Bhaskar Mahajan, V Vinay, Determinant: Combinatorics, algorithms, and complexity. Chicago J. Theor. Comput. Sci., Vol. 1997, Article no. 1997-5, 26 pp.](https://eccc.weizmann.ac.il/eccc-reports/1997/TR97-036/index.html)
- [Mahajan M., Vinay V. (1998) Determinant: Old algorithms, new insights. In: Arnborg S., Ivansson L. (eds) Algorithm Theory — SWAT'98. SWAT 1998. Lecture Notes in Computer Science, vol 1432. Springer, Berlin, Heidelberg.](https://doi.org/10.1007/BFb0054375)
- [Rote G. (2001) Division-Free Algorithms for the Determinant and the Pfaffian: Algebraic and Combinatorial Approaches. In: Alt H. (eds) Computational Discrete Mathematics. Lecture Notes in Computer Science, vol 2122. Springer, Berlin, Heidelberg.](https://doi.org/10.1007/3-540-45506-X_9)
