import numpy as np
import torch
from model import HandwritingSynthesisNetwork
from utils import draw, attention_plot
np.random.seed(5340)
torch.manual_seed(4045)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

synthesis_weight_path = "experiments/20_mixture_components.ckpt"
model = HandwritingSynthesisNetwork(dictionary_size=60)

trained_params = torch.load(synthesis_weight_path, map_location=torch.device('cpu'))['state_dict']
state_dict = {}
for param in trained_params.keys():
    state_dict[param[6:]] = trained_params[param]

model.load_state_dict(state_dict=state_dict)
char_to_code = torch.load('char_to_code.pt')
model.to(device)

def build_one_hots(text):
    onehots = np.zeros((len(text), len(char_to_code) + 1))
    for _ in range(len(text)):
        try:
            onehots[_][char_to_code[text[_]]] = 1
        except:
            onehots[_][-1] = 1
    onehots = torch.from_numpy(onehots).unsqueeze(0).float()
    text_mask = torch.ones((onehots.shape[0], onehots.shape[1])).float()
    return onehots, text_mask


def synthesize_handwriting(text, bias):
    text = text + " " # append a character at the end so termination works better
    onehots, text_mask = build_one_hots(text)
    onehots = onehots.to(device)
    text_mask = text_mask.to(device)
    out, phis = model.sample(text_mask=text_mask, onehots=onehots, maxlen=2000)
    out = out.detach().cpu().numpy()
    phis = phis.detach().cpu().numpy()
    draw(out[0],save_file=f'./generated_{bias}.jpg')
    attention_plot(phis[0])


if __name__ == "__main__":
    text = "The quick brown fox jumps over the lazy dog"
    # text = "do not go gentle into that good night"
    # text = "The truth is rarely pure and never simple"
    # text = "hello world"
    for bias in [5.0]: #0, 0.1, 0.5, 1, 2, 5, 
        synthesize_handwriting(text, bias=bias)