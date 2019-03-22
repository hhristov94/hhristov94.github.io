---
title:  "Notes on Numerical Elimination"
header:
  teaser: /assets/images/sym_permut.png
---
{% include mathjax.html %}

In a past post I discussed some of the numerical ways to decompose into two other matrices, one lower triangular $$L$$ and one upper triangular $$U$$. The methods to do so, however, rely on numerical elimination. The thing is, when implementing such elimination in practice on should consider a few things. To demonstrate I'll be using the some of the code from my previous post.
\\
When using elimination, sometimes the order in which the equations are presented has a profound effect on the results. This is because division is involved. In the case of elimination we need need to divide by the number in the pivot position at some point. But if this number is zero we get an undefined answer. In fact, even if the number is very close to zero we encounter a problem. Consider the example:

$$
\begin{bmatrix}
\epsilon & 1 & \vdots & 1+\epsilon \\
2 & 3 & \vdots & 5\\
\end{bmatrix}
$$

with the exact solution $$x_1 = 1$$ and $$x_2 = 1$$. If we set $$\epsilon = 10^{-16}$$ and try the familiar way of say Gaussian elimination look what happens.

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

We get a wrong answer. The main reason behind is that computers have limited precision(in our case 16 digits) and after a certain point round off the number. This is also known as swamping.
Essentially after the first step the matrix looks like this:

$$
\begin{bmatrix}
\epsilon & 1 & \vdots & 1+\epsilon \\
0 & 3-2*10^{16} & \vdots & 5-2*10^{16}\\
\end{bmatrix}
$$

But when we round to our finite precision this is what we get:

$$
\begin{bmatrix}
\epsilon & 1 & \vdots & 1 \\
0 & -2*10^{16} & \vdots & -2*10^{16} \\
\end{bmatrix}
$$

Which leads to the wrong answer $$x_1 = 0$$ and $$x_2 = 1$$.

\\
This can be overcomed with a simple trick. We can just rearrange the order of the rows so that the system has the highest number in the column in the pivot position.
When we start with

$$
\begin{bmatrix}
2 & 3 & \vdots & 5\\
\epsilon & 1 & \vdots & 1+\epsilon \\
\end{bmatrix}
$$

Gaussian ellimination leads to the system:

$$
\begin{bmatrix}
2 & 3 & \vdots & 5\\
0 & 1-\epsilon/2 & \vdots & 1-5\epsilon/2 \\
\end{bmatrix}
$$

Which then rounds off to

$$
\begin{bmatrix}
2 & 3 & \vdots & 5\\
0 & 1 & \vdots & 1 \\
\end{bmatrix}
$$

```python
A = A = np.array([[2,3],
                  [e,1]])
b = np.array([5,1+e])
gauss_elim(A,b)
```

```python
array([1., 1.])
```

And we have the right answer. This is simple technique is called Partial Pivoting and gist of it is to just compare all the numbers in a column, swap with the one that is the biggest and proceed with the elimination.

Sadly, only swapping the rows doesn't solve all of our swamping problems. Suppose we multiply the first row of our original system by $$\frac{10^4}{e}$$.

$$
\begin{bmatrix}
10^4 & \frac{10^4}{\epsilon} & \vdots & (1+\epsilon)\frac{10^4}{\epsilon} \\
2 & 3 & \vdots & 5\\
\end{bmatrix}
$$

Now by the rules of partial pivoting the system doesn't need any rearrangements. If we proceed to do ellimination however, we will again get a wrong answer.

```python
e = 1 * 10**-16
A = np.array([[10**4,10**4/e],
              [2,3]])
b = np.array([(1+e)*10**4/e,5])
gauss_elim(A,b)
```

```python
array([0., 1.])
```

Swamping again occurs and leads the algorithm astray. The system becomes

$$
\begin{bmatrix}
10^4 & \frac{10^4}{\epsilon} & \vdots & (1+\epsilon)\frac{10^4}{\epsilon} \\
0 & 3-\frac{2}{\epsilon} & \vdots & 5-(1+\epsilon)\frac{2}{\epsilon}\\
\end{bmatrix}
$$

which when rounded becomes

$$
\begin{bmatrix}
10^4 & \frac{10^4}{\epsilon} & \vdots & \frac{10^4}{\epsilon} \\
0 & -2*10^{16} & \vdots & -2*10^{16}\\
\end{bmatrix}
$$

and we get our wrong answer.

It would seem that we would need something different than just comparing the numbers inside a column. Here comes the idea that Gaussian elimination actually works best if the equation matrix $$A$$ is __diagonally dominant__. A matrix is said to be diagonally dominant if each diagonal element is larger than the sum of the other elements in the same row in absolute terms.
Or otherwise:

$$
|A_{ii}| > \sum_{\substack{j=1 \\ j \neq i}}|A_{ij}| \qquad (i=1,2,3,...,n)
$$

So, if we rearrange our matrix so that the pivot element is as large as possible in comparison to other elements in the pivot row the elimination should work fine. In order to do the said comparison we establish an array __s__ with the elements:

$$
s_i  = \max_{j}|A_{ij}|, \qquad i=1,2,...,n
$$

```python
for i in range(n):
    s[i] = max(abs(a[i,:]))
```

After that we use it to define the relative size of an element $$A_{ij}$$ that represents the candidate row:

$$
r_{ij} = \frac{|A_{ij}|}{s_i}
$$

At each iteration we naturally choose the element $$A_{pk}$$ that has
the largest relative size. That is, we choose $$p$$ such that

$$
r_{pk} = \max_{j}r_{jk}, \qquad j \geq k
$$

and do the swap:

```python
def swapRows(v,i,j):
    if len(v.shape) == 1:
        v[i],v[j] = v[j],v[i]
    else: 
        v[[i,j],:] = v[[j,i],:]
```

The whole operation in code would look like that:

```python
def gaussPivot(A,b):
    n = len(b)
    #Set up scale factors
    s = np.zeros(n)
    for i in range(n):#for every row check the biggest
        s[i] = max(np.abs(A[i,:]))
    for j in range(0,n-1):
        #Row interchange, if needed
        p = np.argmax(np.abs(A[j:n,j])/s[j:n]) + j
        if p != j:
            swapRows(b,j,p)
            swapRows(s,j,p)
            swapRows(A,j,p)
        #Elimination
        for i in range(j+1,n):
            c = A[j,i]/A[i,i]
            A[j,i:n] = A[j,i:n] - c*A[i,i:n]
            b[j] = b[j] - c*b[i]
    return backward(A,b)
```

Example:

```python
e = 1 * 10**-16
A = np.array([[10**4,10**4/e],
              [2,3]])
b = np.array([(1+e)*10**4/e,5])
gaussPivot(A,b)
```

```python
array([1., 1.])
```