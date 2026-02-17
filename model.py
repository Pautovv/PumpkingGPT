import random
from engine import Value
from utils import sum_vectors, matvec_prod
from layers import QKVAttention, FeedForwardLayer, Normalization, Dropout, Embedding, PositionalEncoding

class DecoderBlock:
    def __init__(self, token_len, head_count, ffl_dim, dropout_rate):
        self.attention = QKVAttention(head_count, token_len)
        self.ffl = FeedForwardLayer(token_len, ffl_dim)
        self.normal_1 = Normalization(token_len)
        self.normal_2 = Normalization(token_len)
        self.dropout_1 = Dropout(dropout_rate)
        self.dropout_2 = Dropout(dropout_rate)

    def __call__(self, X, training=True):
        attn = self.attention(X)
        X = [sum_vectors(xi, di) for xi, di in zip(X, self.dropout_1(attn, training))]
        X = self.normal_1(X)

        ffl = self.ffl(X)
        X = [sum_vectors(xi, di) for xi, di in zip(X, self.dropout_2(ffl, training))]
        X = self.normal_2(X)

        return X

    def parameters(self):
        return self.attention.parameters() + self.ffl.parameters() + self.normal_1.parameters() + self.normal_2.parameters()

class GPT:
    def __init__(self, layers_count, token_len, head_count, ffl_dim, dropout_rate, vocab_size):
        self.embedding = Embedding(vocab_size, token_len)
        self.pos_enc = PositionalEncoding(token_len)
        self.dropout = Dropout(dropout_rate)

        self.blocks = [DecoderBlock(token_len, head_count, ffl_dim, dropout_rate) for _ in range(layers_count)]

        self.W_final = [[Value(random.uniform(-0.1, 0.1)) for _ in range(vocab_size)] for _ in range(token_len)]
        self.b_final = [Value(0.0) for _ in range(vocab_size)]

    def __call__(self, X, training=True):
        x = self.embedding(X)
        x = self.pos_enc(x)
        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(x, training=training)

        ans = list()
        for xi in x:
            out = sum_vectors(matvec_prod(xi, self.W_final), self.b_final)
            ans.append(out)
        return ans

    def parameters(self):
        p = self.embedding.parameters()
        for block in self.blocks:
            p += block.parameters()
        p += [v for row in self.W_final for v in row] + self.b_final
        return p