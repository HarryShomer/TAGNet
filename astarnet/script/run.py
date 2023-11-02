import os
import sys
import math
import numpy as np

import torch

from torchdrug import core, tasks
from torchdrug.utils import comm, pretty

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from reasoning import dataset, layer, model, task, util

from time import perf_counter


def train_and_validate(cfg, solver):
    if cfg.train.num_epoch == 0:
        return

    if hasattr(cfg.train, "batch_per_epoch"):
        step = 1
    else:
        step = math.ceil(cfg.train.num_epoch / 10)
    best_result = float("-inf")
    best_epoch = -1

    for i in range(0, cfg.train.num_epoch, step):
        kwargs = cfg.train.copy()
        kwargs["num_epoch"] = min(step, cfg.train.num_epoch - i)
        
        solver.train(**kwargs)
 
        # print("\n# Msgs per Sample (Running):", flush=True)
        # print(    "A*:", np.mean(solver.model.model.NUM_ASTAR_MSGS), flush=True)
        # print(    "Ours:", np.mean(solver.model.model.NUM_OUR_MSGS), flush=True)
        # print(    "Both:", np.mean(solver.model.model.NUM_BOTH_MSGS), flush=True)

        for k, v in layer.GeneralizedRelationalConv(1, 1, 1, 1).ALL_TIMES.items():
            print(f"{k} = {np.mean(v)}")
        print("", flush=True)

        solver.save("model_epoch_%d.pth" % solver.epoch)
        
        metric = solver.evaluate("valid")

        result = metric[cfg.metric]
        if result > best_result:
            best_result = result
            best_epoch = solver.epoch

    # TODO: Throws error
    # solver.load("model_epoch_%d.pth" % best_epoch)

    return solver


def test(cfg, solver):
    solver.evaluate("valid")
    solver.evaluate("test")


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + comm.get_rank())

    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pretty.format(cfg))

    dataset = core.Configurable.load_config_dict(cfg.dataset)
    solver = util.build_solver(cfg, dataset)

    train_and_validate(cfg, solver)
    test(cfg, solver)