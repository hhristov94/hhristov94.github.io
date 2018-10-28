---
title: Linear Independence and Basis
---
{% include mathjax.html %}

Last time I finished with a illustration of a sample linear combination of two vectors : $$\hat{i} = \begin{bmatrix}1\\0 \end{bmatrix}$$ and $$\hat{j} = \begin{bmatrix}0\\1 \end{bmatrix}$$ and their 
$$ 
 span(\hat{i},\hat{j}) = \mathbb{R^2} 
$$ which formed a plane.

Let's see another example however. The set of vectors

$$
 (\begin{bmatrix}1\\1 \end{bmatrix},
 \begin{bmatrix}2\\2 \end{bmatrix},
 \begin{bmatrix}4\\4 \end{bmatrix})
$$

You can immediately see that  they are multiples of each other and if we plot them we can argue that their span is a just line passing through the origin (think about what happens when you scale by a negative number).

![png](/assets/images/linearly_dependent.png)

In this example, three vectors formed a mere line, while the two vectors from the first example managed to form a whole plane which is substantionally more 'spacial'. It would seem that some of the three vectors were 'redundant' in a sense that they weren't adding additional space to the overall span. Mathematically speaking this is known as __linear dependence__ and is defined as follows:  

A set of vectors $$\{v_1,v_2,...,v_n\}$$ is linearly dependent if there exist constant $$c_1,c_2,...,c_n$$ not all 0 such that

$$
  c_1v_1+c_2v_2+...+c_nv_n=0
$$

Otherwise they are __linearly independent__.

In the example above the constants that prove the linear dependency between the vectors would be $$2,1$$ and $$-2$$.

$$
2\begin{bmatrix}1\\1 \end{bmatrix} +
1\begin{bmatrix}2\\2 \end{bmatrix} + 
-1\begin{bmatrix}4\\4 \end{bmatrix} = 0
$$

But why is it good to know if a set of vectors is linearly dependent or not. Well, one application is that you can check if a system of equations have an unique solution or not. Consider this system of equations:

$$
  \begin{align}
  x + 2y = 3 \\
  2x + 6y = 4
  \end{align}
$$

If you solve it you will get the unique solution of $$[x,y] = [5,-1]$$
But now consider the system 

$$
  \begin{align}
  x + 2y = 3 \\
  2x + 4y = 6
  \end{align}
$$

After a glance at this one, you will most probably conclude that it has an infinite amount of solutions. What you can also notice is that the row vectors $$
\begin{bmatrix}1\\2\\3 \end{bmatrix},
\begin{bmatrix}2\\4\\6 \end{bmatrix}
$$are actually linearly dependent also.

We can compare the two systems geometrically to get an even better understanding. Let's first plot the "row picture" of the systems by rearranging and expressing $$y$$ in each of the equations and letting $$x$$ range.

![png](/assets/images/solution.png)

You can clearly see that the unique solution of the first system lies exactly where the two lines intersect in the 2D plane. The second system on the otherhand has infinitely many solutions since the two lines lie ontop of eachother.

We can also get a complimentary "column picture" using the column vectors. In the first system for example they look like this: $$\begin{bmatrix}1\\2 \end{bmatrix}$$,$$\begin{bmatrix}2\\6 \end{bmatrix}$$ and $$\begin{bmatrix}3\\4\end{bmatrix}$$.


![png](/assets/images/column_pic.png)

Here by looking at the column vectors we can see that the solution lies in the exact scalar values that multiplied with the first two black vectors will equal the third orange one. 

Comparing this to the "column picture" of the second equation we can see that the linear dependence of the column vectors leads to infinite solutions.

![png](/assets/images/column_2.png)

We can conclude that linear independence implies some sort of uniqueness. This can be also proven but to do this I need to introduce the matrix form. The first system for example looks like this.

$$
\begin{bmatrix}
1&2\\
2&6 
\end{bmatrix}
\begin{bmatrix}
x\\
y 
\end{bmatrix} = 
\begin{bmatrix}
3\\
4 
\end{bmatrix}
$$

$$
Ax=b
$$

To prove that 



Thinking about the concept of linear independence naturally leads to 


A set of elements(vectors) in a vector space $$V$$ is called a basis or a set of basis vectors if the vectors are linearly independent and every vector in the vector space is a linear combination of this set, meaning that it spans the whole $$V$$.