import os
import sys
import ast
import time
import logging
import argparse

import yaml
import jinja2
import easydict
from jinja2 import meta

import dgl
import torch
from torch import distributed as dist
from torch_geometric.data import Data
from torch_geometric.utils import degree

from src import models
from src import datasets

logger = logging.getLogger(__file__)



def create_dgl_graph(edge_index):
    """
    DGL graph from edge_index
    """
    return dgl.graph((edge_index[0], edge_index[1]))


def detect_variables(cfg_file):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    env = jinja2.Environment()
    tree = env.parse(raw)
    vars = meta.find_undeclared_variables(tree)
    return vars


def load_config(cfg_file, context=None):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    template = jinja2.Template(raw)
    instance = template.render(context)
    cfg = yaml.safe_load(instance)
    cfg = easydict.EasyDict(cfg)
    return cfg


def literal_eval(string):
    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        return string


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file", required=True)
    parser.add_argument("-s", "--seed", help="random seed for PyTorch", type=int, default=1024)
    parser.add_argument("-d", "--delta", help="Delta param", type=int, default=3)

    parser.add_argument("--sample", help="Percentage edges to sample", type=float, default=None)
    parser.add_argument("--checkpoint", help="File of model to load", type=str)
    parser.add_argument("--export-att", help="Export attention values during testing", action="store_true", default=False)
    parser.add_argument("--eval-on", help="Which split to eval on", type=str, default="test")
    parser.add_argument("--testing", help="Flag when == True don't create dir to save models", action="store_true", default=False)

    args, unparsed = parser.parse_known_args()
    # get dynamic arguments defined in the config file
    vars = detect_variables(args.config)
    parser = argparse.ArgumentParser()
    for var in vars:
        parser.add_argument("--%s" % var, required=True)
    vars = parser.parse_known_args(unparsed)[0]
    vars = {k: literal_eval(v) for k, v in vars._get_kwargs()}

    return args, vars


def get_root_logger(file=True):
    format = "%(asctime)-10s %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(format=format, datefmt=datefmt)
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)

    if file:
        handler = logging.FileHandler("log.txt")
        format = logging.Formatter(format, datefmt)
        handler.setFormatter(format)
        logger.addHandler(handler)

    return logger


def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    return 0


def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    return 1


def synchronize():
    if get_world_size() > 1:
        dist.barrier()


def get_device(cfg):
    if cfg.train.gpus:
        device = torch.device(cfg.train.gpus[get_rank()])
    else:
        device = torch.device("cpu")
    return device


def create_working_directory(cfg):
    file_name = "working_dir.tmp"
    world_size = get_world_size()

    # No idea why this happens
    if isinstance(cfg.train.gpus, int):
        cfg.train.gpus = [cfg.train.gpus]

    if cfg.train.gpus is not None and len(cfg.train.gpus) != world_size:
        error_msg = "World size is %d but found %d GPUs in the argument"
        if world_size == 1:
            error_msg += ". Did you launch with `python -m torch.distributed.launch`?"
        raise ValueError(error_msg % (world_size, len(cfg.train.gpus)))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl", init_method="env://")

    working_dir = os.path.join(os.path.expanduser(cfg.output_dir),
                               cfg.model["class"], cfg.dataset["class"], time.strftime("%Y-%m-%d-%H-%M-%S"))

    # synchronize working directory
    if get_rank() == 0:
        with open(file_name, "w") as fout:
            fout.write(working_dir)
        os.makedirs(working_dir)
    synchronize()
    if get_rank() != 0:
        with open(file_name, "r") as fin:
            working_dir = fin.read()
    synchronize()
    if get_rank() == 0 and os.path.isfile(file_name):
        os.remove(file_name)

    os.chdir(working_dir)
    return working_dir


def build_dataset(cfg):
    """
    Build the specific dataset
    """
    cls = cfg.dataset.pop("class")

    if cls == "FB15k-237":
        dataset = datasets.build_fb15k237(cfg)
    elif cls == "WN18RR":
        dataset = datasets.build_wn18rr(cfg)
    elif cls.startswith("Ind"):
        dataset = datasets.IndRelLinkPredDataset(name=cls[3:], **cfg.dataset)
    else:
        raise ValueError("Unknown dataset `%s`" % cls)

    if get_rank() == 0:
        logger.warning("%s dataset" % cls)
        logger.warning("#Entities: %d, #Relations %d" % (int(dataset.data.edge_index.max()) + 1, dataset.num_relations))
        logger.warning("#train: %d, #valid: %d, #test: %d" %
                       (dataset[0].target_edge_index.shape[1], dataset[1].target_edge_index.shape[1],
                        dataset[2].target_edge_index.shape[1]))

    return dataset


def build_model(cfg, dist_dataset, args, dataset, is_inductive):
    """
    """
    device = get_device(cfg)
    dgl_graph = create_dgl_graph(dataset[0].edge_index.to(device))
    train_degree = degree(dataset[0].edge_index[1], dataset[0].num_nodes)

    if is_inductive:
        test_degree = degree(dataset[2].edge_index[1], dataset[2].num_nodes) 
    else:
        test_degree = train_degree

    kwargs = {
        "dist_dataset": dist_dataset,
        "dgl_graph": dgl_graph,
        "train_degree": train_degree,
        "test_degree": test_degree,
        "edge_index": dataset[0].edge_index,
        "export_att": args.export_att
    }
    model = models.TAGNet(**cfg.model, **kwargs)

    if args.checkpoint is not None:
        print("Loading model from", args.checkpoint)
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])

    return model



