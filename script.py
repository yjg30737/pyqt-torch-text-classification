import torch
import torch.nn as nn


class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_len):
        super(TextClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Embedding(vocab_size, embedding_dim),
            nn.Flatten(),
            nn.Linear(embedding_dim * max_len, 1)
        )

    def forward(self, x):
        x = self.model(x)
        return x


class TextPredictor:
    def __init__(self, model_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # metadata in model (which is not saved in result.pth, so it has to be defined in static like this)
        self.vocab = {'very': 2, 'it': 3, 'than': 4, "I'm": 5, 'the': 6, 'more': 7, 'again.': 8, 'boring': 9, 'not': 10, 'good.': 11, 'sure.': 12, 'well-made': 13, 'is': 14, 'a': 15, 'fun.': 16, 'best.': 17, 'The': 18, 'thought.': 19, 'acting': 20, 'movie': 21, "It's": 22, 'see': 23, 'recommend.': 24, 'movie.': 25, 'awkward.': 26, 'want': 27, 'so': 28, 'Well,': 29, 'I': 30, 'really': 31, 'to': 32, '<unk>': 0, '<pad>': 1}
        self.embedding_dim = 100
        self.max_len = 7

        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        model = TextClassifier(len(self.vocab), self.embedding_dim, self.max_len).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def pad_sequences(self, sequences, max_len, pad_value):
        padded_sequences = []
        for sequence in sequences:
            if len(sequence) < max_len:
                sequence += [pad_value] * (max_len - len(sequence))
            padded_sequences.append(sequence)
        return padded_sequences

    def text_to_tensor(self, inputs, vocab, max_len):
        words = inputs.split()
        indices = [vocab[word] for word in words if word in vocab]
        indices = self.pad_sequences([indices], max_len, vocab['<pad>'])
        tensor_inputs = torch.Tensor(indices).long().to(self.device)
        return tensor_inputs

    def predict_text(self, text):
        tensor_inputs = self.text_to_tensor(text, self.vocab, self.max_len)
        with torch.no_grad():
            outputs = self.model(tensor_inputs)
            # binary text classification
            prediction = torch.sigmoid(outputs)  # apply sigmoid activation function
            predicted_labels = (prediction > 0.5).int()  # decide the binary label based on 0.5
            return predicted_labels.item()


