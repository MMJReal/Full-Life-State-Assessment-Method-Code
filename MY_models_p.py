import torch
import torch.nn as nn
from MY_lstm_cell import CustomLSTMCell


class Classification_Dann(nn.Module):
    def __init__(self, output_size, step):
        super(Classification_Dann, self).__init__()
        self.hidden_size = 250
        self.num_layers = 3

        activation = 'leaky_relu' #
        input_size = 32 * 4
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=False)

        self.lstm_cells = nn.ModuleList()
        for _ in range(self.num_layers):
            self.lstm_cells.append(CustomLSTMCell(input_size, self.hidden_size, activation=activation))
            input_size = self.hidden_size

        self.encoder = nn.Sequential(
            nn.Conv1d(step, 16, kernel_size=2, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1, bias=True),
            nn.LeakyReLU(inplace=True),
            )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 32, kernel_size=3, stride=2, padding=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(32),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=0, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(16),
            nn.ConvTranspose1d(16, step, kernel_size=2, stride=2, padding=1, output_padding=0, bias=False),
            nn.LeakyReLU(inplace=True),
        )


        self.classer = nn.Sequential(
            nn.Linear(in_features=250, out_features=250, bias=True),
            nn.Dropout(0.2),  # drop 50% of the neuron
            nn.LeakyReLU(True),
            nn.InstanceNorm1d(250),
            nn.Linear(in_features=250, out_features=120, bias=True),
            nn.LeakyReLU(True),
            nn.Linear(120, output_size),
        )

    def forward(self, x_s, x_t):
        features_cv = self.encoder(x_s)
        features_dcv = self.decoder(features_cv)


        features_s = features_cv.view(features_cv.shape[0], 1, -1)
        batch_size, seq_len, _ = features_s.size()
        h0 = torch.zeros(batch_size, self.hidden_size).cuda()
        c0 = torch.zeros(batch_size, int(self.hidden_size)).cuda()
        h = [h0] * self.num_layers
        c = [c0] * self.num_layers
        outputs = []
        for t in range(seq_len):
            input_t = features_s[:, t, :]
            for layer in range(self.num_layers):
                h[layer], c[layer] = self.lstm_cells[layer](input_t, (h[layer], c[layer]))
                input_t = h[layer]
            outputs.append(h[-1])
        features_s = torch.stack(outputs, dim=1)
        features_s = features_s.view(features_s.shape[0],-1)
        outs = self.classer(features_s)
        feature_cv_t = self.encoder(x_t)
        feature_dcv_t = self.decoder(feature_cv_t)
        features_t = feature_cv_t.view(feature_cv_t.shape[0], 1, -1)
        batch_size, seq_len, _ = features_t.size()
        h0 = torch.zeros(batch_size, self.hidden_size).cuda()
        c0 = torch.zeros(batch_size, int(self.hidden_size)).cuda()
        h = [h0] * self.num_layers
        c = [c0] * self.num_layers
        output_t = []
        for t in range(seq_len):
            input_t = features_t[:, t, :]
            for layer in range(self.num_layers):
                h[layer], c[layer] = self.lstm_cells[layer](input_t, (h[layer], c[layer]))
                input_t = h[layer]
            output_t.append(h[-1])


        return outs, features_dcv, feature_dcv_t


class DANN_Encoder(nn.Module):
    def __init__(self, output_size, num_layers, hidden_size, step):
        super(DANN_Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoder = nn.Sequential(
            nn.Conv1d(step, 16, kernel_size=2, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=0, bias=True),
            nn.BatchNorm1d(32),

        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 32, kernel_size=3, stride=2, padding=1,output_padding=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(32),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=0, output_padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(16),
            nn.ConvTranspose1d(16, step, kernel_size=2, stride=2, padding=0, output_padding=0, bias=False),
            nn.BatchNorm1d(step),
        )

        activation = 'relu'
        input_size = 32*12
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)        

        self.lstm_cells = nn.ModuleList()
        for _ in range(num_layers):
            self.lstm_cells.append(CustomLSTMCell(input_size, hidden_size, activation=activation))
            input_size = hidden_size
        self.class_classifier = nn.Sequential(
            nn.Linear(in_features=250, out_features=250, bias=True),
            nn.Dropout(0.2),  # drop 50% of the neuron
            nn.BatchNorm1d(250),
            nn.LeakyReLU(True),
            nn.Linear(in_features=250, out_features=120, bias=True),
            nn.LeakyReLU(True),
            nn.Linear(120, output_size),
        )
        self.relu = nn.LeakyReLU()



    def forward(self, src, tara):
        src = src  
        tara = tara  
        # -------------source----
        feature_s = self.encoder(src)
        De_s = self.decoder(feature_s)
        feature_s = feature_s.view(feature_s.shape[0], 1, -1)

        batch_size, seq_len, _ = feature_s.size()
        h0 = torch.zeros(batch_size, self.hidden_size).to(src.device)
        c0 = torch.zeros(batch_size, int(self.hidden_size)).to(src.device)
        h = [h0] * self.num_layers
        c = [c0] * self.num_layers
        outputs = []
        for t in range(seq_len):
            input_t = feature_s[:, t, :]
            for layer in range(self.num_layers):
                h[layer], c[layer] = self.lstm_cells[layer](input_t, (h[layer], c[layer]))
                input_t = h[layer]
            outputs.append(h[-1])
        feature_s = torch.stack(outputs, dim=1)
        feature_s = feature_s.view(feature_s.shape[0], -1)

        # --------Target---------
        feature_t = self.encoder(tara)
        De_t = self.decoder(feature_t)

        feature_t = feature_t.view(feature_t.shape[0], 1, -1)

        batch_size, seq_len, _ = feature_t.size()
        h0 = torch.zeros(batch_size, self.hidden_size).to(tara.device)
        c0 = torch.zeros(batch_size, int(self.hidden_size)).to(tara.device)
        h = [h0] * self.num_layers
        c = [c0] * self.num_layers
        outputs = []
        for t in range(seq_len):
            input_t = feature_t[:, t, :]
            for layer in range(self.num_layers):
                h[layer], c[layer] = self.lstm_cells[layer](input_t, (h[layer], c[layer]))
                input_t = h[layer]
            outputs.append(h[-1])
        feature_t = torch.stack(outputs, dim=1)

        feature_t = feature_t.view(feature_t.shape[0], -1)

        x_src_mmd = self.relu(feature_s)
        x_tar_mmd = self.relu(feature_t)
        y_src = self.class_classifier(feature_s)

        return y_src, x_src_mmd, x_tar_mmd, De_s, De_t

