#!/usr/bin/env python3

"""
Usage:
$ pip install onnx2pytorch
$ ./onnx2pt <onnx_model.onnx> <pytorch_model_name.pt>
"""

import sys
import torch
import onnx
from onnx2pytorch import ConvertModel
from k_fold_training_data import KFoldTrainingData


onnx_model = onnx.load(sys.argv[1])
pytorch_model = ConvertModel(onnx_model)

domain = "../../tasks/blocksworld_ipc/probBLOCKS-12-0/domain.pddl"
problems = ["../../tasks/blocksworld_ipc/probBLOCKS-12-0/p1.pddl"]
domain_max_value = 327
kfold = KFoldTrainingData(domain, problems, domain_max_value, batch_size=1, num_folds=1, shuffle=False)
train_dataloader, _ = kfold.get_fold(0)

example_input = train_dataloader.dataset[0][0]
print(example_input)
traced_model = torch.jit.trace(pytorch_model, example_input)
traced_model.save(sys.argv[2])
