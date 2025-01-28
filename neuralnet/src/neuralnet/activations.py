"""
This module provides implementations of the main activation functions used in neural networks. Each activation function is represented as a Python class with two methods: `predict` and `derivative`.

Classes:
    Sigmoid: Implements the Sigmoid activation function.
        Methods:
            predict(x): Computes the Sigmoid activation of the input x.
            derivative(x): Computes the derivative of the Sigmoid function at x.

    Tanh: Implements the hyperbolic tangent (tanh) activation function.
        Methods:
            predict(x): Computes the tanh activation of the input x.
            derivative(x): Computes the derivative of the tanh function at x.

    ReLU: Implements the Rectified Linear Unit (ReLU) activation function.
        Methods:
            predict(x): Computes the ReLU activation of the input x.
            derivative(x): Computes the derivative of the ReLU function at x.

    LeakyReLU: Implements the Leaky Rectified Linear Unit (LeakyReLU) activation function.
        Methods:
            predict(x): Computes the LeakyReLU activation of the input x.
            derivative(x): Computes the derivative of the LeakyReLU function at x.

    ELU: Implements the Exponential Linear Activation (ELA) function.
        Methods:
            predict(x): Computes the ELA activation of the input x.
            derivative(x): Computes the derivative of the ELA function at x.
"""
import numpy as np

class Sigmoid:
    """
    Sigmoid activation function class.

    Methods
    -------
    predict(z):
        Computes the sigmoid of z.
    
    derivative():
        Computes the derivative of the sigmoid function with respect to z.
    """
    def predict(self,z:np.ndarray)-> np.ndarray:
        r"""
        Compute the sigmoid of z.
        $$ g(z) =\frac{1}{1+e^{-z}}$$

        Parameters
        ----------
        z : float or np.ndarray
            Input value or array of values.

        Returns
        -------
        float or np.ndarray
            Sigmoid of the input.
        """
        self.out = 1/(1+np.exp(-z))
        return self.out
    @property
    def derivative(self)-> np.ndarray:
        r"""
        Compute the derivative of the sigmoid function with respect to z.
        $$ g'(z) = (1-g(z))*g(z)$$

        Returns
        -------
        float or np.ndarray
            Derivative of the sigmoid function with respect to the input.
        """
        return self.out*(1-self.out)
class Tanh:
    """
    Hyperbolic tangent activation function class.

    Methods
    -------
    predict(z):
        Computes the hyperbolic tangent of z.
    
    derivative():
        Computes the derivative of the hyperbolic tangent function with respect to z.
    """
    def predict(self,z:np.ndarray)-> np.ndarray:
        r"""
        Compute the hyperbolic tangent of z.
        $$ g(z) =\frac{e^{z}-e^{-z}}{e^{z}+e^{-z}}$$

        Parameters
        ----------
        z : float or np.ndarray
            Input value or array of values.

        Returns
        -------
        float or np.ndarray
            Hyperbolic tangent of the input.
        """
        self.out = (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
        return self.out
    @property
    def derivative(self)-> np.ndarray:
        r"""
        Compute the derivative of the hyperbolic tangent function with respect to z.
        $$ g'(z) = 1-g(z)^2$$

        Returns
        -------
        float or np.ndarray
            Derivative of the hyperbolic tangent function with respect to the input.
        """
        return 1-self.out**2
class ReLU:
    """
    Rectified Linear Unit activation function class.

    Methods
    -------
    predict(z):
        Computes the ReLU of z.
    
    derivative():
        Computes the derivative of the ReLU function with respect to z.
    """
    def predict(self,z:np.ndarray)-> np.ndarray:
        r"""
        Compute the ReLU of z.
        $$ g(z) = max(0,z)$$

        Parameters
        ----------
        z : float or np.ndarray
            Input value or array of values.

        Returns
        -------
        float or np.ndarray
            ReLU of the input.
        """
        self.out = np.maximum(0,z)
        return self.out
    @property
    def derivative(self)-> np.ndarray:
        r"""
        Compute the derivative of the ReLU function with respect to z.
        $$ g'(z) = 1 if z>0 else 0$$

        Returns
        -------
        float or np.ndarray
            Derivative of the ReLU function with respect to the input.
        """
        return np.where(self.out>0,1,0)
class LeakyReLU :
    """
    Leaky Rectified Linear Unit activation function class.

    Methods
    -------
    predict(z):
        Computes the Leaky ReLU of z.
    
    derivative():
        Computes the derivative of the Leaky ReLU function with respect to z.
    """
    def __init__(self,alpha:float = 0.01):
        self.alpha = alpha
    def predict(self,z:np.ndarray)-> np.ndarray:
        r"""
        Compute the Leaky ReLU of z.
        $$ g(z) = max(\alpha*z,z)$$

        Parameters
        ----------
        z : float or np.ndarray
            Input value or array of values.

        Returns
        -------
        float or np.ndarray
            Leaky ReLU of the input.
        """
        self.out = np.maximum(self.alpha*z,z)
        return self.out
    @property
    def derivative(self)-> np.ndarray:
        r"""
        Compute the derivative of the Leaky ReLU function with respect to z.
        $$ g'(z) = 1 if z>0 else \alpha$$

        Returns
        -------
        float or np.ndarray
            Derivative of the Leaky ReLU function with respect to the input.
        """
        return np.where(self.out>0,1,self.alpha)
class ELU:
    def __init__(self,alpha:float = 1):
        self.alpha = alpha
    def predict(self,z:np.ndarray)-> np.ndarray:
        r""" 
        Compute the Exponential Linear Unit of z.
        $$
\begin{cases} 
\alpha(\exp{z}-1) & \text{if} & z \le 0\\
z & \text{if} & z > 0
\end{cases}
        $$
        Parameters
        ----------
        z : float or np.ndarray
            Input value or array of values.

        Returns
        -------
        float or np.ndarray
            ELU of the input.
        """
        self.out = np.where(z<0,self.alpha*(np.exp(z)-1),z)
        return self.out
    @property
    def derivative(self)-> np.ndarray:
        r"""
        computes the derivative with respect to the output
        Returns
        -------
        np.ndarray
        """
        return np.where(self.out>0,1,self.alpha*np.exp(self.out))