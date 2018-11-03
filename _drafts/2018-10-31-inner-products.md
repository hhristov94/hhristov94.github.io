---
title: Norms and Inner Products
---
{% include mathjax.html %}

If you were to describe a vector you most probably would use its basis to orient it in space and then describe its length and direction. I want to focus on the former property first though - vector length. The concept of length in our three-dimensional physical space is pretty simple but when we are thinking abstractly this can be somewhat elusive. That's why it is useful to define it mathematically using someting called the vector __norm__ and it is denoted as $$\|x\|$$.

A norm is a function that assigns a strictly positive length or size to each vector in a vector space—except for the zero vector, which is assigned a length of zero. This concept can be extended to something called a __seminorm__, which on the other hand, is allowed to assign zero length to some non-zero vectors (in addition to the zero vector).

A norm must also satisfy certain properties pertaining to scalability and additivity. If for instance we have a nonnegative-valued scalar function $$p: V → [0,+\infty)$$

1. Being subadditive:

$$p(u + v) ≤ p(u) + p(v)$$
2. Being absolutely scalable:

$$p(av) = |a| p(v)$$
3. Positive-definiteness:

If $$p(v) = 0$$ then $$v=0$$ is the zero vector.

A seminorm on V is a function $$p : V → R$$ with the properties 1 and 2 above.

Let's look at an example of a norm. The so called __p-norm__ is defined as follows:

$$\|x\|_p = (\sum_{i=1}^n |x_i|^p)^{1/p}$$

For $$p = 1$$ we get the $$L^1$$ norm which is also called the taxicab or Manhattan norm because if we take two arbitrary vectors,say $$x$$ and $$y$$, the distance between them 

$$\|x-y\|_1 = (\sum_{i=1}^n |x_i-y_i|)$$

will emulate moving between two points as though you are moving through the streets of Manhattan in a taxi cab. The cab will have to take corners to take you to your destination.

For $$p = 2$$ we get the $$L^2$$ norm which is also called the Euclidian and is a straight application of Pythagoras theorem. In the language of the first analogy if we had our two points it would be a straight raod connecting them as if you were flying over your destination.

And also for  $$p = \infty$$ we get the infinity norm denoted as $$L^\infty$$ and defined a bit differently as
$$\|x\|_{\infty }=\max _{i}(|x_{i}|)$$

The concept of unit circle (the set of all vectors of norm 1) is different in different norms: for the 1-norm the unit circle in \mathbb{R^2} is a square, for the 2-norm (Euclidean norm) it is the well-known unit circle, while for the infinity norm it is a different square. For any p-norm it is a superellipse (with congruent axes).

{% capture x %}
x = np.linspace(-1.0, 1.0, 1000)
y = np.linspace(-1.0, 1.0, 1000)
X, Y = np.meshgrid(x,y)
F = np.abs(X) + np.abs(Y) - 1
cls = ['#ff9a17','#ff8316','#ff6f0f','#ff6918','#ff5613']
plt.figure(figsize=(8,8))
plt.contour(X,Y,F,0, colors = 'orange')

for i in range(2,7):
    F = X**i + Y**i - 1
    plt.contour(X,Y,F,0, colors = cls[i-2])

plt.xlim(-1.25, 1.25)
plt.ylim(-1.25, 1.25)
plt.axvline(x=0, color='#A9A9A9')
plt.axhline(y=0, color='#A9A9A9')
plt.text(0.32, 0.4, r'$p=1$',  size=14)
plt.text(0.59, 0.55, r'$p=2$',  size=14)
plt.text(1, 0.73, r'$p=6$',  size=14)
plt.show()
{% endcapture %}
{% include accordion.html toggle-text=x button-text="Vector Graph" %}

![png](/assets/images/unit_norm.png)

After exploring vector length I want to go back to the idea of direction of a vector. This can be described as an angle relative to another vector which can even a basis one.
cuachy swatz
