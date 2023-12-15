import torch

class tikzTokenizer:
    def __init__(self, max_seq_length=50):
        letters = ['a', 'b', 'c', 'x', 'y', 'z']
        digits = [str(x) for x in range(10)]
        self.vocab = list(letters)
        for l in letters:
            for s in ['_', '^']:
                for d in digits:
                    self.vocab.append(l + s + d)
        self.vocab += ['{', '}', ',', '->']
        self.vocab += ['<pad>', '<s>', '</s>', '<unk>']
        self.token2idx = {t: i for i, t in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

        self.pad_token_id = self.token2idx['<pad>']
        self.eos_token_id = self.token2idx['</s>']
        self.bos_token_id = self.token2idx['<s>']
        self.unk_token_id = self.token2idx['<unk>']
        self.max_length = max_seq_length

    def __getitem__(self, token):
        return self.token2idx.get(token, self.unk_token_id)

    def save_pretrained(self, *args, **kwargs):
        pass

    def tokens2indices(self, sents):
        """
        Convert list of lists of tokens to list of lists of ids.
        """
        return [[self.bos_token_id] + [self[t] for t in s] + [self.eos_token_id] for s in sents]
    
    def pad_sents(self, sents):
        sents_padded = []

        max_len = len(max(sents, key=lambda x: len(x)))
        for sent in sents:
            if len(sent) < max_len:
                sents_padded.append(list(sent) + [self.pad_token_id] * (max_len - len(sent)))
            else:
                sents_padded.append(list(sent))
        return sents_padded

    def batch_tokenize(self, strings):
        """
        Takes list of sentences, returns list of lists of tokens.
        """
        return [s.split(' ') for s in strings]

    def batch_encode(self, sents):
        """
        sents is a list of lists of tokens (sentences)
        """
        token_ids = self.tokens2indices(sents)
        token_ids_padded = self.pad_sents(token_ids)
        return torch.tensor(token_ids_padded, dtype=torch.long)
    
    def __call__(self, sents):
        """
        Tokenizes, then encodes. Takes list of sentences, returns
        list of lists of token ids.
        """
        sents = self.batch_tokenize(sents)
        input_ids = self.batch_encode(sents)
        attention_mask = input_ids.ne(self.pad_token_id).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def batch_decode(self, token_ids):
        """
        token_ids is a tensor of size (batch, max_len)
        """
        sents = []
        batch_size, max_len = token_ids.shape
        for i in range(batch_size):
            curr_sent = []
            for j in range(max_len):
                if token_ids[i, j].item() == self.eos_token_id:
                    break
                token_idx = token_ids[i, j]
                curr_sent.append(self.vocab[token_idx])
            sents.append(curr_sent)
        return [' '.join(s) for s in sents]