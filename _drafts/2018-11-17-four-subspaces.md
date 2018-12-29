---
title:  "The Four Fundamental Subspaces"
header:
  teaser: /assets/images/linear_combination.png
---
{% include mathjax.html %}

As we saw, vector spaces can be “contained” in others, i.e. vector spaces can be subspaces of other vector spaces. As it turns out, the matrix $$A$$  in a system in a linear equations naturally defines four vector spaces:

- The column space, or the space spanned by the columns of $$A$$

- The row space, or the space spanned by the rows of $$A$$

- The nullspace, or the set of vectors $$v$$ for which $$Av=0$$

- The left nullspace, or the set of vectors $$v$$ for which $$A^Tv=0$$, or equivalently $$v^TA=0$$

The number of columns in a matrix is the sum of the rank and the nullity

This is called the rank-nullity theorem. Its proof is relatively straightforward: we first convert the matrix to its Gauss-Jordan form using Gaussian elimination (which only involves row operations, which doesn’t change the rank). Then we only need to prove the statement for Gauss-Jordan matrices, which we’ll see how to do in the next problem.

Recall that a Gauss-Jordan matrix has a leading 1 in every nonzero row. Suppose there are columns containing a leading one and  column that do not. Which of the following is true?



Lastly, I want to mention that sometimes Gaussian Elimination and LU decomposition methods will sometimes fail. This can happen in multiple ways but the two most common ways for Gaussian Elimination in particular are zeroes in the pivot position and values in the pivot that are very close to zero. The first one is to be expected since we cannot define a $$c$$ because we need to divide something by zero and any computer will throw an error or put out infinity as an answer. The second way is more subtle, the reason of failure in a case when we have a very small pivot is because of loss of information in numerical rounding, also called swamping.

Consider the example:

$$
\begin{bmatrix}
\epsilon & 1 & \vdots &1+\epsilon\\
2 & 3 & \vdots & 5
\end{bmatrix}
$$

with the solution $$ x = \begin{bmatrix} 1\\
1 \end{bmatrix}$$

If we set $$\epsilon = 10^{-16}$$ and apply Gaussian Elimination we get

$$
\begin{align}
E_1\\
E_2 -(2/\epsilon)E_1&\rightarrow E_2
\end{align}
\begin{bmatrix}
\epsilon & 1 & \vdots &1+\epsilon\\
0 & 3-2*10^{-16} & \vdots & 5-2*10^{-16}
\end{bmatrix}
$$

Which when rounded-off becomes

$$
\begin{bmatrix}
\epsilon & 1 & \vdots &1\\
0 & -2*10^{-16} & \vdots & -2*10^{-16}
\end{bmatrix}
$$

and leads us to a wrong answer $$x = \begin{bmatrix} 0\\1\end{bmatrix}$$.

```python
e = 1 * 10**-16
A = np.array([[e,1],
              [2,3]])
b = np.array([1+e,5])
gauss_elim(A,b)
```

```python
array([0., 1.])
```

Luckily there is a solution. We can just swap the rows

```python
A = A = np.array([[2,3],
                  [e,1]])
b = np.array([5,1+e])
gauss_elim(A,b)
```

```python
array([1., 1.])
```

Diagonal dominance
