# Tests

## Initial conditions
We start with a network with 3 layers (one input, one hidden and one output).
They have sizes: 3, 2, and 3.
The weights look like this: \
$\omega^{[1]} = \begin{bmatrix}-0.75 & 0.5 & -0.25\\\ 0.25 & -0.5 & 0.75 \end{bmatrix}$ \
$\omega^{[2]} = \begin{bmatrix}-0.75 & 0.75 \\\ 0.5 & -0.5 \\\ -0.25 & 0.25 \end{bmatrix}$\
Then we will have biases: \
$b^{[1]} = \begin{bmatrix} 0 \\\ 1 \end{bmatrix}$ \
$b^{[2]} = \begin{bmatrix} 0.75 \\\ 0.5 \\\ 0.25 \end{bmatrix}$ \
We will have mock data: $\begin{bmatrix} 0.25 \\\  0.5 \\\ 0.75 \end{bmatrix}$, and expect an output of $\begin{bmatrix} 0 \\\ 1 \\\ 0\end{bmatrix}$.

## Doing forward pass
Here we test if the forward pass works.
For $L_1$, we expect:
$$
\begin{aligned}
L_1 & = ReLU((\omega^{[1]} \times \begin{bmatrix} 0.25 \\\  0.5 \\\ 0.75 \end{bmatrix}) + b^{[1]}) \\
& = ReLU(\begin{bmatrix}-0.75 & 0.5 & -0.25\\\ 0.25 & -0.5 & 0.75 \end{bmatrix} \times \begin{bmatrix} 0.25 \\\  0.5 \\\ 0.75 \end{bmatrix} + \begin{bmatrix} 0 \\\ 1 \end{bmatrix})\\
& = ReLU(\begin{bmatrix} -0.125 \\\ 0.375 \end{bmatrix} + \begin{bmatrix} 0 \\\ 1 \end{bmatrix})\\
& = ReLU(\begin{bmatrix} -0.125 \\\ 1.375 \end{bmatrix}) \\ 
& =  \begin{bmatrix} -0.025 \\\ 1.375 \end{bmatrix}
\end{aligned}
$$
And for $L_2$:
$$
\begin{aligned}
L_2 & = \sigma((\omega^{[2]} \times L_1) + b^{[2]}) \\
& = \sigma(\begin{bmatrix}-0.75 & 0.75 \\\ 0.5 & -0.5 \\\ -0.25 & 0.25 \end{bmatrix} \times \begin{bmatrix} -0.025 \\\ 1.375 \end{bmatrix} + \begin{bmatrix} 0.75 \\\ 0.5 \\\ 0.25 \end{bmatrix})\\
& = \sigma(\begin{bmatrix} 1.05 \\\ -0.7 \\\ 0.35 \end{bmatrix} + \begin{bmatrix} 0.75 \\\ 0.5 \\\ 0.25 \end{bmatrix})\\
& = \sigma(\begin{bmatrix} 1.8 \\\ -0.2 \\\ 0.6 \end{bmatrix}) \\ 
& =  \begin{bmatrix} 0.858 \\\ 0.450 \\\ 0.646 \end{bmatrix}
\end{aligned}
$$
Note that we use both leaky ReLU and the logistical function (leaky ReLU for hidden layers and logistical function for output). The leaky ReLU function we use has a slope of $0.2$.

## Doing backprop
#### Output layer
##### Testing the calculate cost function
First we must attain the costs vector which we do with our resultant $L_2$:
$$
C = \begin{bmatrix} 0.858 \\\ 0.450 \\\ 0.646 \end{bmatrix} -  \begin{bmatrix} 0 \\\ 1 \\\ 0 \end{bmatrix} = \begin{bmatrix} 0.858 \\\ -0.550 \\\ 0.646 \end{bmatrix}
$$

##### Deltas for output layer
Let us calculate $2(a_j - \hat{a}_j) \times \sigma\prime(z^{[2]}_{j})$ for $j \in [0, 2]$  (i.e. for all of the output nodes).
Note this value is called delta in the code as it is generally in literature.
Also, it is obvious that $aji - \hat{a}_j = C[j]$.
We use $\odot$ to represent the Hadamard product and $z^{[2]}_{j}$ is just taken from the final value of $L_2$.
$$
\begin{aligned}
\delta &^{[2]} = 2C \odot \sigma\prime(\begin{bmatrix} 0.858 \\\ 0.450 \\\ 0.646 \end{bmatrix}) \\
& = \begin{bmatrix} 1.7163 \\\ -0.9000 \\\ 1.2913 \end{bmatrix} \odot \begin{bmatrix} \sigma(0.858)(1 - \sigma(0.858)) \\\ \sigma(0.450)(1 - \sigma(0.450)) \\\ \sigma(0.646)(1 - \sigma(0.646)) \end{bmatrix} \\
& = \begin{bmatrix} 0.359 \\\ -0.261 \\\ 0.291 \end{bmatrix}
\end{aligned}
$$

##### Change in biases for output layer
Conveniently (and perhaps via the benifit of hindsight) $\frac{\partial C}{\partial b^{[L]}_j} = 2(a_i - \hat{a}_i) \times \sigma\prime(z^{[L]}_{i}) = \delta$. So, we can say that the change in biases for layer 2, is $\delta^{[2]}$.
So:
$$
\Delta b^{[2]} = \delta^{[2]} = \begin{bmatrix} 0.359 \\\ -0.261 \\\ 0.291 \end{bmatrix}
$$

##### Change in weights for output layer
We can determine that $\frac{\partial C}{\partial \omega^{[L]}_{jk}} = 2(a_j - \hat{a}_j) \times \sigma\prime(z^{[L]}_{j})\times a^{[L-1]}_k = \delta^{[L]}[j] \times a^{[L-1]}_k$, where $\omega^{[L]}_{jk}$ represents the weight from node $k$ on the previous layer to node $j$ on the output layer and $a^{[L-1]}_k$ is the value of the node $a_k$ on the previous layer. We have already calculated $\delta^{[2]}$, so we can use it here, $\delta^{[2]}[j]$ corresponds to the $j$th output node's delta. As we iterate over all of $j$ and $k$ we find that the matrix of the desired derivatives is:
$$
\begin{bmatrix} \delta^{[2]}[0]a^{[1]}_0 & \delta^{[2]}[0]a^{[1]}_1 \\\ \delta^{[2]}[1]a^{[1]}_0 & \delta^{[2]}[1]a^{[1]}_1 \\\ \delta^{[2]}[2]a^{[1]}_0 & \delta^{[2]}[2]a^{[1]}_1 \end{bmatrix}
$$
We spot this is the outer product, i.e. $\delta^{[L]} \otimes L^{[L-1]}$, so we can say:

$$
\begin{aligned}
\frac{\partial C}{\partial \omega^{[2]}} = \Delta \omega^{[2]} & = \delta^{[2]} \otimes {L^{[2-1]}}\\
& = \delta^{[2]} \otimes {L^{[1]}}^T \\
& = \begin{bmatrix} 0.359 \\\ -0.261 \\\ 0.291 \end{bmatrix} \begin{bmatrix} -0.025 & 1.375 \end{bmatrix} \\
& = \begin{bmatrix} -0.0089713 & 0.49342246 \\\ 0.00653616 & -0.35948870 \\\ -0.00728476 & 0.40066166 \end{bmatrix}
\end{aligned}
$$

#### Hidden layer
##### Costs of hidden layer
We use the output layer to determine the costs of the nodes in the hidden layer (i.e. $\frac{\partial C}{\partial a^{[1]}_k}$). This is as $\frac{\partial C}{\partial a^{[L]}_k} =  \sum_j(2(a_j - \hat{a}_j) \times \sigma\prime(z^{[L]}_{j}) \times \omega^{[L]}_{jk}) = \sum_j \delta^{[2]}[j] \times \omega^{[L]}_{jk}$. This is the sum of the weights of the nodes connections with the outer nodes multiplied by the outer nodes deltas. As we iterate over $j$ and $k$ we find that the matrix of the desired derivative is:
$$
\begin{bmatrix}
\delta^{[2]}[0]\omega^{[2]}_{00} + \delta^{[2]}[1]\omega^{[2]}_{10} + \delta^{[2]}[2]\omega^{[2]}_{20} \\\
\delta^{[2]}[0]\omega^{[2]}_{01} + \delta^{[2]}[1]\omega^{[2]}_{11} + \delta^{[2]}[2]\omega^{[2]}_{21}\end{bmatrix}
$$
We spot that this is equal to the product of ${\omega^{[2]}}^T$ and $\delta^{[2]}$.
So:
$$
\begin{aligned}
\frac{\partial C}{\partial a^{[2]}_k} = \Delta a^{[1]} & = {\omega^{[2]}}^T \delta^{[2]}\\
& = \begin{bmatrix}-0.75 & 0.75 \\\ 0.5 & -0.5 \\\ -0.25 & 0.25 \end{bmatrix}^T \begin{bmatrix} 0.359 \\\ -0.261 \\\ 0.291 \end{bmatrix} \\
& = \begin{bmatrix} -0.75 & 0.5 & -0.25 \\\ 0.75 & -0.5 & 0.25 \end{bmatrix} \begin{bmatrix} 0.359 \\\ -0.261 \\\ 0.291 \end{bmatrix} \\\
& = \begin{bmatrix} -0.47269 \\\ 0.47269 \end{bmatrix}
\end{aligned}
$$

##### Deltas for hidden layer
As we discussed earlier, we need to calculate the deltas now we have the costs of the hidden layer. However, now we have to use the derivative of leaky relu as that is what we used in the forward pass. I will denote this $f \prime$.
$$
\begin{aligned}
\delta^{[1]} & = 2 \Delta a^{[1]} \odot f\prime(L_1) \\
& = 2 \begin{bmatrix} -0.47269 \\\ 0.47269 \end{bmatrix} \odot f\prime(\begin{bmatrix} -0.025 \\\ 1.375 \end{bmatrix}) \\
& = \begin{bmatrix} -0.94538 \\\ 0.94538 \end{bmatrix} \odot \begin{bmatrix} 0.2 \\\  1 \end{bmatrix} \\
& = \begin{bmatrix} -0.18908 \\\ 0.94538 \end{bmatrix}
\end{aligned}
$$

##### Change in biases for hidden layer
$$
\Delta b^{[1]} = \delta^{[1]} = \begin{bmatrix} -0.18908 \\\ 0.94538 \end{bmatrix}
$$

##### Change in weights for hidden layer
$$
\begin{aligned}
\frac{\partial C}{\partial \omega^{[1]}} = \Delta \omega^{[1]} & = \delta^{[1]} \otimes {L^{[1-1]}}\\
& = \delta^{[1]} \otimes {L^{[0]}}^T \\
& = \begin{bmatrix} -0.18908 \\\ 0.94538 \end{bmatrix} \otimes \begin{bmatrix} 0.25 \\\  0.5 \\\ 0.75 \end{bmatrix} ^T \\
& = \begin{bmatrix} -0.18908 \\\ 0.94538 \end{bmatrix} \otimes \begin{bmatrix} 0.25 & 0.5 & 0.75 \end{bmatrix} \\
& = \begin{bmatrix} -0.04727103& 0.23635514& -0.09454206 \\\ 0.47271028& -0.14181308& 0.70906544 \end{bmatrix}
\end{aligned}
$$

### And that's all!
