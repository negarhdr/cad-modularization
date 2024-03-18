SELECTED_GPUS = [2]

import os

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu_number) for gpu_number in SELECTED_GPUS])

import sys
import torch

from dgl.data.utils import load_graphs
from torch import FloatTensor
from uvnet.models import Classification, UVNetClassifier


if __name__ == '__main__':
    lightning_model = Classification.load_from_checkpoint('results/classification/1113/124546/best.ckpt')
    torch.save(lightning_model.model.state_dict(), "results/classification/1113/124546/best.pt")
    torch_model = UVNetClassifier(26)
    torch_model.load_state_dict(torch.load("results/classification/1113/124546/best.pt"))
    representation = None
    def hook(model, input_, output):
        global representation
        representation = input_[0]
    torch_model.clf.register_forward_hook(hook)
    torch_model.cuda()
    torch_model.eval()
    letters_dir = 'SolidLetters/graph_with_eattr'
    output_dir = 'SolidLetters/representations'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, file_name in enumerate(os.listdir(letters_dir)):
        if file_name.endswith('.bin'):
            sys.stdout.write('\r%d' % (i + 1))
            sys.stdout.flush()
            graph = load_graphs(os.path.join(letters_dir, file_name))[0][0]
            graph.ndata["x"] = graph.ndata["x"].type(FloatTensor)
            graph.edata["x"] = graph.edata["x"].type(FloatTensor)
            graph.ndata["x"] = graph.ndata["x"].permute(0, 3, 1, 2)
            graph.edata["x"] = graph.edata["x"].permute(0, 2, 1)
            graph = graph.to("cuda:0")
            torch_model(graph)
            assert len(file_name.split('.')) == 2
            torch.save(representation, os.path.join(output_dir, file_name.split('.')[0] + '.pt'))
