---
title: Linear Independence and Basis
---
{% include mathjax.html %}

Last time I finished with a illustration of a sample linear combination of two vectors : $$\hat{i} = \begin{bmatrix}1\\0 \end{bmatrix}$$ and $$\hat{j} = \begin{bmatrix}0\\1 \end{bmatrix}$$ and their 
$$ 
 span(\hat{i},\hat{j}) = \mathbb{R^2} 
$$ 

which formed a plane.

Let's see another example however. The set of vectors

$$
 (\begin{bmatrix}1\\1 \end{bmatrix},
 \begin{bmatrix}2\\2 \end{bmatrix},
 \begin{bmatrix}4\\4 \end{bmatrix})
$$

You can immediately see that  they are multiples of each other and if we plot them we can argue that their span is a ust line passing through the origin (think about what happens when you scale by a negative number).

![png](/assets/images/linearly_dependent.png)

In this example, three vectors formed a mere line, while the two vectors from the first example managed to form a whole plane which substantionally more 'spacial'. It would seem that some of the three vectors were 'redundant' in a sense that they weren't adding additional space of the overall span. Mathematically speaking this is known as __linear dependence__ and is defined as follows:  

A set of vectors $$\{v_1,v_2,...,v_n\}$$ is linearly dependent if there exist constant $$c_1,c_2,...,c_n$$ not all 0 such that

$$
  c_1v_1+c_2v_2+...+c_nv_n=0
$$

Otherwise they are __linearly independent__.

In the example above these constants would be $$2,1$$ and $$-2$$.
But why is it good to know if a set of vectors is linearly dependent or not. Let's see an example with a simple system of equations.

$$
  \begin{align}
  x + 2y = 3 \\
  2x + 6y = 4
  \end{align}
$$
draw planes intersecting here.