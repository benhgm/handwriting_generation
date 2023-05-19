import torch
import numpy as np
import matplotlib.pyplot as plt

def draw_dataset2(stroke, save_name=None):
    # Plot a single example.
    f, ax = plt.subplots(figsize=(12, 3))

    x = np.cumsum(stroke[:, 1])
    y = -np.cumsum(stroke[:, 2])

    size_x = x.max() - x.min() + 1.
    size_y = y.max() - y.min() + 1.

    f.set_size_inches(5. * size_x / size_y, 5.)

    cuts = np.where(stroke[:, 0] == 1)[0]
    start = 0

    for cut_value in cuts:
        ax.plot(x[start:cut_value], y[start:cut_value],
                'k-', linewidth=3)
        start = cut_value + 1

    ax.set_xlim(-1000, 7000)
    ax.set_ylim(-1000, 1000)
    ax.axis('off')

    ax.set_aspect('equal')
    ax.tick_params(
        axis='both', left=False, right=False,
        top=False, bottom=False,
        labelleft=False, labeltop=False,
        labelright=False, labelbottom=False
    )

    if save_name is None:
        plt.show()
    else:
        try:
            plt.savefig(
                save_name,
                bbox_inches='tight',
                pad_inches=0.5)
        except Exception:
            print ("Error building image!: " + save_name)

    plt.close()

def config_inner_args(inner_args):
    if inner_args is None:
        inner_args = dict()
    
    inner_args['reset_model'] = inner_args.get('reset_model') or False
    inner_args['n_step'] = inner_args.get('n_step') or 5
    inner_args['lstm_lr'] = inner_args.get('lstm_lr') or 0.01
    inner_args['fc_lr'] = inner_args.get('fc_lr') or 0.01
    inner_args['momentum'] = inner_args.get('weight_decay') or 0
    inner_args['first_order'] = inner_args.get('first_order') or False
    inner_args['frozen'] = inner_args.get('frozen') or []

    return inner_args

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

    # ax.set_xlim(-200, 300)
    # ax.set_ylim(-30, 30)
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