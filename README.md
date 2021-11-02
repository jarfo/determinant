## Symbolic, exact and division-free computation algorithms for the determinant and the characteristic polynomial ##


### Explicit Matrix power method for the Clow-based algorithm ###
If we follow the Dynamic Programming approach of the clow-based method, we obtain a simple expression for the determinant and all the coefficients of the charasteristic polynomial.

<img src="https://render.githubusercontent.com/render/math?math=P_%7BA%7D(%5Clambda)%3A%3D%5Cmathrm%7Bdet%7D(%5Clambda%20I-A)%3Dq_%7Bn%7D%20%5Clambda%5E%7Bn%7D%2Bq_%7Bn-1%7D%20%5Clambda%5E%7Bn-1%7D%2B%5Ccdots%2Bq_%7B1%7D%20%5Clambda%2Bq_%7B0%7D">

with

<img src="https://render.githubusercontent.com/render/math?math=q_{i} = rM^{n-i}s">

where M is the 'transition matrix' of the dynamic algorithm as described in [Rote], s corresponds to the initial conditions and r to the final collecting node.

For n=3 we have:

<img src="https://render.githubusercontent.com/render/math?math=r%3D%5Cleft%5B%5Cbegin%7Barray%7D%7Blllllll%7D%0A0%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%201%0A%5Cend%7Barray%7D%5Cright%5D">,


<img src="https://render.githubusercontent.com/render/math?math=s%3D%5Cleft%5B%5Cbegin%7Barray%7D%7Blllllll%7D%0A1%20%26%200%20%26%200%20%26%201%20%26%200%20%26%201%20%26%201%0A%5Cend%7Barray%7D%5Cright%5D">,

and

<img src="https://render.githubusercontent.com/render/math?math=M%20%3D%20%5Cleft%5B%5Cbegin%7Barray%7D%7Bccccccc%7D%0A0%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%5C%5C%0Aa_%7B10%7D%20%26%20a_%7B11%7D%20%26%20a_%7B12%7D%20%26%200%20%26%200%20%26%200%20%26%200%20%5C%5C%0Aa_%7B20%7D%20%26%20a_%7B21%7D%20%26%20a_%7B22%7D%20%26%200%20%26%200%20%26%200%20%26%200%20%5C%5C%0A-a_%7B00%7D%20%26%20-a_%7B01%7D%20%26%20-a_%7B02%7D%20%26%200%20%26%200%20%26%200%20%26%200%20%5C%5C%0A0%20%26%200%20%26%200%20%26%20a_%7B21%7D%20%26%20a_%7B22%7D%20%26%200%20%26%200%20%5C%5C%0A-a_%7B00%7D%20%26%20-a_%7B01%7D%20%26%20-a_%7B02%7D%20%26%20-a_%7B11%7D%20%26%20-a_%7B12%7D%20%26%200%20%26%200%20%5C%5C%0A-a_%7B00%7D%20%26%20-a_%7B01%7D%20%26%20-a_%7B02%7D%20%26%20-a_%7B11%7D%20%26%20-a_%7B12%7D%20%26%20-a_%7B22%7D%20%26%200%0A%5Cend%7Barray%7D%5Cright%5D">



### References ###
- [Meena Bhaskar Mahajan, V Vinay, Determinant: Combinatorics, algorithms, and complexity. Chicago J. Theor. Comput. Sci., Vol. 1997, Article no. 1997-5, 26 pp.](https://eccc.weizmann.ac.il/eccc-reports/1997/TR97-036/index.html)
- [Mahajan M., Vinay V. (1998) Determinant: Old algorithms, new insights. In: Arnborg S., Ivansson L. (eds) Algorithm Theory — SWAT'98. SWAT 1998. Lecture Notes in Computer Science, vol 1432. Springer, Berlin, Heidelberg.](https://doi.org/10.1007/BFb0054375)
- [Rote G. (2001) Division-Free Algorithms for the Determinant and the Pfaffian: Algebraic and Combinatorial Approaches. In: Alt H. (eds) Computational Discrete Mathematics. Lecture Notes in Computer Science, vol 2122. Springer, Berlin, Heidelberg.](https://doi.org/10.1007/3-540-45506-X_9)
