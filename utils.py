from engine import Value

def sum_vectors(v1, v2):
    return [a + b for a, b in zip(v1, v2)]

def matvec_prod(vector, matrix):
    res = [Value(0.0) for _ in range(len(matrix[0]))]
    for j in range(len(matrix[0])):
        for i in range(len(matrix)):
            res[j] = res[j] + vector[i] * matrix[i][j]
    return res

def dot_product(v1, v2):
    return sum((a * b for a, b in zip(v1, v2)), Value(0.0))