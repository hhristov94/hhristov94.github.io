---
title:  "Vector Spaces"
---
{% include mathjax.html %}

Formally, a vector space  is a set of objects, called vectors. However, it is important to note that word “vector” here has a much broader sense than just pointy arrows with direction and magnitute. A vector just means an element of a vector space. A suitable candidate for such and element might very well be a matrix or a function. 

Thinking in such and abstract way is can be extremely useful but we need to have some boundaries in the definition in order to make sense and be useful to others. To qualify as a vector space, the set of objects, which I would refer to as $$V$$, and the operations which are used on them must adhere to a number of requirements called axioms:
1. Associativity of addition :
$$u + (v + w) = (u + v) + w$$
2. Commutativity of addition :
$$u + v = v + u$$
3. Identity element of addition :
There exists an element $$0 \in V$$, called the zero vector, such that $$v + 0 = v$$ for all $$v \in V$$.
4. Inverse elements of addition :
For every $$v \in V$$, there exists an element $$−v \in V$$, called the additive inverse of $$v$$, such that $$v + (−v) = 0$$.
5. Compatibility of scalar multiplication with field multiplication:
$$a(bv) = (ab)v$$
6. Identity element of scalar multiplication :
$$1v = v$$
7. Distributivity of scalar multiplication with respect to vector addition :
$$a(u + v) = au + av$$
8. Distributivity of scalar multiplication with respect to field addition :
$$(a + b)v = av + bv$$

These axioms seem a bit tedious but they have the important task of defining a rigorous frame that if followed correctly is sure to work.
Now that everything is defined we can look at some examples. Some of the most important vector spaces are denoted by $$\mathbb{R^n}$$. This is also known as "n-dimensional real space". Each space of $$\mathbb{R^n}$$ consists of whole collection of real number vectors. For example $$\mathbb{R^5}$$ contains all columns vectors with five components and one of these vectors is :

$$v = \begin{pmatrix}
    1 \\
    2 \\
    3 \\
    4 \\
    5 \\
    \end{pmatrix}
$$

Five dimentional space is a bit hard to explore though, since we are living in the three dimentional world and are not used to thinking so abstract. Let's go to Flatland for a moment and see how everything works in 2D. 

If we have two vectors $$v$$ and $$w$$ in the same vector space $$\mathbb{R^2}$$, every linear combination of them in the form $$x = cv + dw$$ for some scalars $$c$$ and $$d$$ should also be in $$\mathbb{R^2}$$.

```python
%matplotlib inline 
# Magics that tells the plots appear inside the notebook
import numpy as np # Linear Algebra module
import matplotlib.pyplot as plt # Data Visualization module
import seaborn as sns
sns.set()
#Define Scalars
c = 1.5
d = -1
# Start and end coordinates of the vectors
v = [0,0,1,3]
w = [0,0,2,1]
colors = ['k','silver','silver','orange','k']

plt.figure(figsize=(20,6))
plt.subplot(1,2,1)
plt.quiver([w[0],c*v[0],d*w[0],c*v[0]+d*w[0],v[0]],
           [w[1],c*v[1],d*w[1],c*v[1]+d*w[1],v[1]],
           [w[2],c*v[2],d*w[2],c*v[2]+d*w[2],v[2]],
           [w[3],c*v[3],d*w[3],c*v[3]+d*w[3],v[3]],
           angles='xy', scale_units='xy', scale=1, color = colors)
plt.xlim(-5, 5)
plt.ylim(-2, 5)
# Draw axes
plt.axvline(x=0, color='#A9A9A9')
plt.axhline(y=0, color='#A9A9A9')
plt.scatter(4,7,marker='x',s=50)
# Draw the name of the vectors
plt.text(0.2, 2, r'$\vec{v}$', size=18)
plt.text(1.25, 0.00, r'$\vec{w}$',  size=18)
plt.text(-1, 2, r'$\vec{x}$',  size=18)

plt.show()
```
![png](/assets/images/linear_combination.png)
