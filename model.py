import torch
import torch.nn as nn

from utils import draw, attention_plot
from modules import Module, LSTMCell, Linear, LSTM, bivariate_gaussian_nll, mixture_of_bivariate_normal_sample


class GaussianAttention(Module):
    def __init__(self, input_size=400, output_size=30):
        super().__init__()

        self.linear = Linear(in_features=input_size, out_features=output_size)

    def forward(self, h_t, kappa_prev, text_mask=None, onehots=None, params=None, episode=None):
        mixture_params = self.linear(h_t, params, episode)
        mixture_params = torch.exp(mixture_params)[:, None]

        alpha, beta, kappa = torch.chunk(mixture_params, 3, dim=-1)
        kappa = kappa + kappa_prev.unsqueeze(1)

        num_chars = onehots.shape[1]
        batch_size = onehots.shape[0]
        u = torch.arange(num_chars, dtype=torch.float32).to(onehots.device)
        u = u[None, :, None].repeat(batch_size, 1, 1)
        
        phi = torch.sum(alpha * torch.exp(-beta * torch.pow((kappa - u), 2)), dim=-1)
        phi = phi * text_mask # multiply by the text mask to omit results from padded elements

        w_t = torch.sum(phi.unsqueeze(-1) * onehots, dim=1)

        alpha = alpha.squeeze(1)
        beta = beta.squeeze(1)
        kappa = kappa.squeeze(1)
        
        return w_t, alpha, beta, kappa, phi
    
class HandwritingSynthesisNetwork(Module):
    def __init__(self, dictionary_size):
        super().__init__()
        self.dictionary_size = dictionary_size
        self.lstm1 = LSTMCell(input_size=dictionary_size+3, hidden_size=400, bias=True)
        self.lstm2 = LSTM(input_size=403 + dictionary_size, hidden_size=400, bias=True, batch_first=True)
        self.window = GaussianAttention()

        ### Change here for mixture component
        self.fc = Linear(in_features=800, out_features=121)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def init_hidden(self, batch_size):
        """
        Function for initialisation of input hidden states and soft window vector at time t=0

        Args:
            batch_size (int): Batch Size
        """
        h0 = torch.zeros((batch_size, 400), dtype=torch.float32, device=self.device)
        c0 = torch.zeros((batch_size, 400), dtype=torch.float32, device=self.device)

        h1 = torch.zeros((1, batch_size, 400), dtype=torch.float32, device=self.device)
        c1 = torch.zeros((1, batch_size, 400), dtype=torch.float32, device=self.device)

        kappa_0 = torch.zeros((batch_size, 10), dtype=torch.float32, device=self.device)
        w0 = torch.zeros((batch_size, self.dictionary_size), dtype=torch.float32, device=self.device)
        return h0, c0, h1, c1, kappa_0, w0
    
    def parse_outputs(self, output_params):
        ### Change here for mixture component
        mu, log_sigma, pi, rho, eos = output_params.split([40, 40, 20, 20, 1], -1)

        # Activations
        rho = torch.tanh(rho)
        pi = torch.nn.functional.softmax(pi, dim=-1)
        eos = torch.sigmoid(-eos)

        ### Change here for mixture component
        mu = mu.view(mu.shape[:-1] + (20, 2))
        sigma = log_sigma.view(log_sigma.shape[:-1] + (20, 2))

        return pi, mu, sigma, rho, eos

    def forward(self, strokes, onehots=None, text_mask=None, hidden_states=None, params=None, episode=None):
        outputs = {}

        # initialize hidden states
        batch_size = strokes.shape[0]

        if hidden_states is None:
            h0, c0, h1, c1, kappa, w_t = self.init_hidden(batch_size)
        else:
            h0, c0, h1, c1, kappa, w_t = hidden_states

        w_t = onehots[:, 0, :]

        T = strokes.shape[1]
        lstm1_output = []
        attention_output = []
        for i in range(T):
            stroke_input = strokes[:, i, :]
            lstm1_input = torch.cat([stroke_input, w_t], dim=-1)

            # fist lstm layer
            h0, c0 = self.lstm1(lstm1_input, (h0, c0), params, episode)

            # window layer
            w_t, alpha, beta, kappa, phi = self.window(h0, kappa, text_mask, onehots)

            lstm1_output.append(h0)
            attention_output.append(w_t)
            
        lstm1_output = torch.stack(lstm1_output, 1)
        attention_output = torch.stack(attention_output, 1)

        # second lstm layer
        lstm2_input = torch.cat([strokes, attention_output, lstm1_output], dim=-1)
        lstm2_output, (h1, c1) = self.lstm2(lstm2_input, (h1, c1))

        # fully-connected layer
        fc_input = torch.cat([lstm1_output, lstm2_output], dim=-1)
        output_params = self.fc(fc_input)

        # compute mixture params
        mixture_params = self.parse_outputs(output_params)

        pi, mu, sigma, rho, eos = mixture_params

        return pi, mu, sigma, rho, eos, w_t, kappa, phi, h0, c0, h1, c1
    
    def sample(self, onehots, text_mask, maxlen=1000):
        last_idx = (text_mask.sum(-1) - 2)
        h0, c0, h1, c1, kappa, w_t = self.init_hidden(onehots.shape[0])
        x_t = torch.zeros(onehots.shape[0], 3).float().cuda()

        strokes = []
        phis = []
        alphas = []
        betas = []
        kappas = []

        for i in range(maxlen):
            h0, c0 = self.lstm1(torch.cat([x_t, w_t], -1), (h0, c0))

            w_t, alpha, beta, kappa, phi = self.window(h0, kappa, text_mask, onehots)

            phis.append(phi)
            kappas.append(kappa)
            betas.append(beta)
            alphas.append(alpha)

            _, (h1, c1) = self.lstm2(torch.cat([x_t, w_t, h0], 1).unsqueeze(1), (h1, c1))

            output_params = self.fc(torch.cat([h0, h1.squeeze(0)], dim=-1))

            mixture_params = self.parse_outputs(output_params)

            pi, mu, sigma, rho, eos = mixture_params

            x_t = torch.cat([
                eos.bernoulli(),
                mixture_of_bivariate_normal_sample(pi, mu, sigma, rho, bias=3.0)
            ], dim=1)

            strokes.append(x_t)
            phis.append(phi)

            #####################
            ### Exit Condition ###
            ######################
            phi_t = phi[0, :].unsqueeze(0)
            if phi_t.argmax(dim=1) == phi_t.shape[1] - 1:
                break
        phis = torch.stack(phis, dim=1)

        return torch.stack(strokes, 1), phis
    