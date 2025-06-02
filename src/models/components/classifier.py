from torch import nn


class Classifier(nn.Module):
    def __init__(self, input_size: int, dropout_value=0.3, hidden_size=128):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Dropout(dropout_value),
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, 7)
        )

    def forward(self, x):
        return self.fc(x)
