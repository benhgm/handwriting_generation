import torch
import numpy as np
import matplotlib.pyplot as plt

def draw(offsets, ascii_seq=None, save_file=None):
    strokes = np.concatenate(
        [offsets[:, 0:1], np.cumsum(offsets[:, 1:], axis=0)],
        axis=1
    )

    fig, ax = plt.subplots(figsize=(12, 3))

    stroke = []
    for eos, x, y in strokes:
        stroke.append((x, y))
        if eos == 1:
            xs, ys = zip(*stroke)
            ys = np.array(ys)
            ax.plot(xs, ys, 'k')
            stroke = []

    if stroke:
        xs, ys = zip(*stroke)
        ys = np.array(ys)
        ax.plot(xs, ys, 'k')
        stroke = []

    ax.set_xlim(-100, 600)
    ax.set_ylim(-30, 30)
    ax.axis('off')

    ax.set_aspect('equal')
    ax.tick_params(
        axis='both', left=False, right=False,
        top=False, bottom=False,
        labelleft=False, labeltop=False,
        labelright=False, labelbottom=False
    )

    if ascii_seq is not None:
        if not isinstance(ascii_seq, str):
            ascii_seq = ''.join(list(map(chr, ascii_seq)))
        plt.title(ascii_seq)

    if save_file is not None:
        plt.savefig(save_file)
    plt.close('all')
    return 

def generate_text_from_onehots(onehots, text_len):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    onehots = onehots[0]
    idxs = torch.arange(0, onehots.shape[-1], device=device).float()
    text = ''
    char_to_code = torch.load("char_to_code.pt")

    for i in range(text_len):
        idx = torch.dot(onehots[i], idxs).item()
        for k in char_to_code:
            if int(idx) == char_to_code[k]:
                text += k   

    return text


def attention_plot(phis):
    fig, ax = plt.subplots(figsize=(12, 3))
    phis = phis/(np.sum(phis, axis = 0, keepdims=True))
    plt.xlabel('handwriting generation')
    plt.ylabel('text scanning')
    plt.imshow(phis, cmap='hot', interpolation='nearest', aspect='auto')
    plt.savefig('./attention_plot.jpg')
    plt.close('all')