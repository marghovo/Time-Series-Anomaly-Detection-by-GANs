from torch import nn


class DNet(nn.Module):
    def __init__(self):
        super(DNet, self).__init__()
        n_features = 12  # might be changed
        n_out = 1

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 60),
            nn.Tanh(),
            nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(60, 120),
            nn.Tanh(),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(120, 60),
            nn.Tanh(),
            nn.Dropout(0.3)
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(60, 36),
            nn.Tanh(),
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            nn.Linear(36, n_out),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.out(x)
        return x


class GNet(nn.Module):
    def __init__(self):
        super(GNet, self).__init__()
        n_features = 12
        n_out = 12

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 36),
            nn.Tanh()
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(36, 60),
            nn.Tanh()
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(60, 120),
            nn.Tanh()
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(120, 60),
            nn.Tanh()
        )
        self.hidden4 = nn.Sequential(
            nn.Linear(60, 36),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(36, n_out),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.out(x)
        return x