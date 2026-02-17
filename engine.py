from math import *

class Value:
  def __init__(self, data, _children=(), _op=''):
    self.data = float(data)
    self.grad = 0.0
    self._backward_fn = lambda: None
    self._prev = set(_children)

  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    res = Value(self.data + other.data, (self, other), '+')

    def _backward():
      self.grad += 1.0 * res.grad
      other.grad += 1.0 * res.grad

    res._backward_fn = _backward
    return res

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    res = Value(self.data * other.data, (self, other), '*')

    def _backward():
      self.grad += other.data * res.grad
      other.grad += self.data * res.grad

    res._backward_fn = _backward
    return res

  def __pow__(self, other):
    assert isinstance(other, (int, float))
    res = Value(self.data**other, (self,), f'**{other}')

    def _backward():
      self.grad += (other * self.data**(other - 1)) * res.grad

    res._backward_fn = _backward
    return res

  def exp(self):
    res = Value(exp(self.data), (self, ), 'exp')

    def _backward():
      self.grad += res.data * res.grad

    res._backward_fn = _backward
    return res

  def relu(self):
    res = Value(self.data if self.data > 0 else 0, (self, ), 'ReLU')

    def _backward():
      self.grad += (1.0 if self.data > 0 else 0) * res.grad

    res._backward_fn = _backward
    return res


  def __truediv__(self, other): return self * (other**-1)
  def __sub__(self, other): return self + (other * -1)
  def __radd__(self, other): return self + other
  def __rmul__(self, other): return self * other

  def backward(self):
    tp = list()
    visited = set()

    def f(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          f(child)
        tp.append(v)
    f(self)
    self.grad = 1.0
    for node in reversed(tp):
      node._backward_fn()
