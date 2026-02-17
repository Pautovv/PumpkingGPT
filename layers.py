import random
from math import *
from engine import Value
from utils import sum_vectors, matvec_prod, dot_product

class Normalization:
    def __init__(self, d):
        self.gamma = [Value(1.0) for _ in range(d)]
        self.beta = [Value(0.0) for _ in range(d)]
        self.eps = 1e-5

    def __call__(self, X):
        ans = list()
        for x in X:
            mu = sum(x, Value(0.0)) / len(x)
            var = sum(((xi - mu)**2 for xi in x), Value(0.0)) / len(x)
            std = (var + self.eps)**0.5
            ans.append([(xi - mu) / std * g + b for xi, g, b in zip(x, self.gamma, self.beta)])
        return ans

    def parameters(self): return self.gamma + self.beta
    
class FeedForwardLayer:
    def __init__(self, d, ffl_dim):
        self.W1 = [[Value(random.uniform(-0.1, 0.1)) for _ in range(ffl_dim)] for _ in range(d)]
        self.b1 = [Value(0.0) for _ in range(ffl_dim)]
        self.W2 = [[Value(random.uniform(-0.1, 0.1)) for _ in range(d)] for _ in range(ffl_dim)]
        self.b2 = [Value(0.0) for _ in range(d)]

    def __call__(self, X):
        return [sum_vectors(matvec_prod([h.relu() for h in sum_vectors(matvec_prod(x, self.W1), self.b1)], self.W2), self.b2) for x in X]

    def parameters(self):
        return [p for m in [self.W1, self.W2] for r in m for p in r] + self.b1 + self.b2

class QKVAttention:
    def __init__(self, head_count, d):
        self.d = d
        self.head_count = head_count
        self.W_q = [[[Value(random.uniform(-0.1, 0.1)) for _ in range(d)] for _ in range(d)] for _ in range(head_count)]
        self.W_k = [[[Value(random.uniform(-0.1, 0.1)) for _ in range(d)] for _ in range(d)] for _ in range(head_count)]
        self.W_v = [[[Value(random.uniform(-0.1, 0.1)) for _ in range(d)] for _ in range(d)] for _ in range(head_count)]

    def softmax(self, scores):
        m = max(s.data for s in scores)
        exps = [(s - m).exp() for s in scores]
        summ = sum(exps, Value(0.0))
        return [e / summ for e in exps]

    def __call__(self, X):
        ans = list()
        for i in range(len(X)):
            head = list()
            for h in range(self.head_count):
                q = matvec_prod(X[i], self.W_q[h])
                K = [matvec_prod(X[j], self.W_k[h]) for j in range(i + 1)]
                V = [matvec_prod(X[j], self.W_v[h]) for j in range(i + 1)]

                weights = self.softmax([dot_product(q, k) / (self.d**0.5) for k in K])

                a = [Value(0.0) for _ in range(self.d)]
                for j, w in enumerate(weights):
                    a = sum_vectors(a, [v * w for v in V[j]])
                head.append(a)

            A = head[0]
            for h in range(1, self.head_count):
                A = sum_vectors(A, head[h])
            ans.append(A)
        return ans

    def parameters(self):
        return [p for heads in [self.W_q, self.W_k, self.W_v] for mat in heads for row in mat for p in row]

class PositionalEncoding:
    def __init__(self, d, max_len=500):
        self.pe = list()
        for pos in range(max_len):
            row = list()
            for i in range(d):
                if i % 2 == 0: row.append(sin(pos / (10000 ** (i / d))))
                else: row.append(cos(pos / (10000 ** ((i - 1) / d))))
            self.pe.append(row)

    def __call__(self, X):
        res = list()
        for i, x in enumerate(X):
            row = [xi + pi for xi, pi in zip(x, self.pe[i])]
            res.append(row)
        return res

class Embedding:
    def __init__(self, v_size, d):
        self.W = [[Value(random.uniform(-0.1, 0.1)) for _ in range(d)] for _ in range(v_size)]
    def __call__(self, X):
        return [[Value(w.data, (w,)) for w in self.W[x]] for x in X]
    def parameters(self): return [p for r in self.W for p in r]

class Dropout:
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, X, training=True):
        if not training:
            return X
        res = list()
        for x in X:
            mask = 0 if random.random() < self.p else 1
            res.append([xi * mask for xi in x])
        return res