# Example 1 - shuffle of a boring palindrom

## Parameters
- $n$: Number of shuffled words
- $m$: Length of $a$-segements

We consider the following words 
$$w_m = a^m b a^m$$ 

And the corresponding language/process tree
$$T[m,n] = w_m \oplus w_m \oplus \cdots \oplus w_m$$

Clearly, each word $w \in L(T[m,n])$ consists precisely of $n$ many $b$ and $2mn$ many $a$

In other words, such a word has the form 
$$ w = a^{i_0} b a^{i_1} b a^{i_3} b \cdots a^{i_{n-1}} b a^{i_n}$$
where $i_0 + i_1 + \cdots + i_n = 2mn$ and $i_0, i_1, \ldots, i_n \geq 0$.
Moreoveor, to be part of $L(T[m,n])$, the indices must statisfy the following conditions:
- $\sum_{k = 0}^j i_k \geq (j+1) m$
- $\sum_{k= 0 }^j i_k \leq nm + jm$
  
for all $j \in \{0,1,\ldots,n\}$.

To generate random traces of the language, we randomly choose exponents $i_0, i_1, \ldots, i_{n-1} \in [\lfloor 1.5m \rfloor, 2m]$ and $i_n$ such that the above conditions are satisfied.