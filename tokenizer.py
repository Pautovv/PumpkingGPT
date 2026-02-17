from collections import Counter

class Tokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocab = set()
        self.merges = list()
        self.token2id = dict()
        self.id2token = dict()
  
    def train(self, text):
        tokens = list(text)
        self.vocab.update(tokens)

        while len(self.vocab) < self.vocab_size:
            pairs = Counter(zip(tokens, tokens[1:]))
            if not pairs: break

            best_pair = pairs.most_common(1)[0][0]
            self.merges.append(best_pair)
            
            new_token = "".join(best_pair)
            self.vocab.add(new_token)

            new_tokens = list()
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        self.token2id = {token: i for i, token in enumerate(sorted(self.vocab))}
        self.id2token = {i: token for token, i in self.token2id.items()}
  
    def encode(self, text):
        tokens = list(text)
        for merge in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if (tokens[i], tokens[i+1]) == merge:
                    tokens[i] = "".join(merge)
                    del tokens[i+1]
                else: i += 1
        return [self.token2id[t] for t in tokens if t in self.token2id]
  
    def decode(self, tokens_ids):
        return "".join([self.id2token[i] for i in tokens_ids])