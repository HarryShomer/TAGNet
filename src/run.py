import os
import sys
import json
import math
import pprint
import random
import numpy  as np
from collections import defaultdict
from time import time, perf_counter


import torch
from torch import nn
from tqdm import tqdm
import torch_geometric
from torch import optim
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src import tasks, util, datasets_ours


separator = ">" * 30
line = "-" * 30


def set_seed(seed):
    """
    Set all seeds
    """
    random.seed(seed)  
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_and_validate(cfg, model, train_data, valid_data, filtered_data=None, seed=0, log=True):
    if cfg.train.num_epoch == 0:
        return  

    logger = util.get_root_logger()
    world_size = util.get_world_size()
    rank = util.get_rank()
    device = util.get_device(cfg)

    train_triplets = torch.cat([train_data.target_edge_index, train_data.target_edge_type.unsqueeze(0)]).t()
    sampler = torch_data.DistributedSampler(train_triplets, world_size, rank, seed=torch.initial_seed())
    train_loader = torch_data.DataLoader(train_triplets, cfg.train.batch_size, sampler=sampler, worker_init_fn=np.random.seed(seed), num_workers=0)

    cls = cfg.optimizer.pop("class")
    optimizer = getattr(optim, cls)(model.parameters(), **cfg.optimizer)

    if 'decay' in cfg.train and cfg.train.decay:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: cfg.train.decay ** e)
    else:
        lr_scheduler = None

    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
    else:
        parallel_model = model

    best_epoch = -1
    best_result = float("-inf")

    step = cfg.train.get("val_every", 2) # Defaut to 2 if not specifie
    step = cfg.train.num_epoch if step == -1 else step  # If -1 then implies last epoch

    for epoch in range(1, cfg.train.num_epoch+1):
        if util.get_rank() == 0 and log:
            logger.warning(separator)
            logger.warning("Epoch %d begin" % epoch)

        losses = []
        batch_id = 0
        sampler.set_epoch(epoch)
        max_batches_per_epoch = len(train_loader)

        for batch in tqdm(train_loader, f"Epoch {epoch}", total=max_batches_per_epoch, file=sys.stdout):
            parallel_model.train()
            batch = tasks.negative_sampling(train_data, batch, cfg.task.num_negative, strict=cfg.task.strict_negative) 

            pred = parallel_model(train_data, batch)

            target = torch.zeros_like(pred)
            target[:, 0] = 1

            loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

            neg_weight = torch.ones_like(pred)
            if cfg.task.adversarial_temperature > 0:
                with torch.no_grad():
                    neg_weight[:, 1:] = F.softmax(pred[:, 1:] / cfg.task.adversarial_temperature, dim=-1)
            else:
                neg_weight[:, 1:] = 1 / cfg.task.num_negative
            loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
            loss = loss.mean()

            # if util.get_rank() == 0 and batch_id % cfg.train.log_interval == 0:
            #     logger.warning(separator)
            #     logger.warning("binary cross entropy: %g" % loss)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
    
            losses.append(loss.item())
            batch_id += 1

        for k, v in model.layers[0].ALL_TIMES.items():
            print(f"{k} = {np.mean(v)}")
        print("", flush=True)


        # Decay!
        if lr_scheduler is not None:
            lr_scheduler.step()

        if util.get_rank() == 0 and log:
            avg_loss = sum(losses) / len(losses)
            logger.warning("average binary cross entropy: %g" % avg_loss)
                
        # Only eval every 'step' epochs or last one
        if (epoch % step == 0 or epoch == cfg.train.num_epoch) and rank == 0:
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }

            torch.save(state, "model_epoch_%d.pth" % epoch)
            util.synchronize()

            if log:
                logger.warning("Save checkpoint to model_epoch_%d.pth" % epoch)
                logger.warning(separator)
                logger.warning("Evaluate on valid")

            result = test(cfg, model, valid_data, split="valid", filtered_data=filtered_data)

            if result > best_result:
                best_result = result
                best_epoch = epoch
        

    if rank == 0 and log:
        logger.warning("Load checkpoint from model_epoch_%d.pth" % best_epoch)

    state = torch.load("model_epoch_%d.pth" % best_epoch, map_location=device)
    model.load_state_dict(state["model"])
    util.synchronize()

    if log:
        logger.warning("Save best checkpoint as model_final.pth")

    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(state, "model_final.pth")


@torch.no_grad()
def test(cfg, model, test_data, split="test", filtered_data=None):
    world_size = util.get_world_size()
    rank = util.get_rank()
    device = util.get_device(cfg)
    logger = util.get_root_logger()

    test_triplets = torch.cat([test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)]).t()
    
    sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
    test_loader = torch_data.DataLoader(test_triplets, cfg.train.batch_size, sampler=sampler)
    
    model.eval()

    rankings = []
    num_negatives = []


    for batch in tqdm(test_loader, "Testing", file=sys.stdout):
        # t_batch = predicting tails  ;  h_batch = predicting heads
        t_batch, h_batch = tasks.all_negative(test_data, batch)

        pos_h_index, pos_t_index, r_index = batch.t()

        # t_pred means predicting tail...
        t_pred = model(test_data, t_batch, true_ent=pos_t_index, testing = split == "test")
        h_pred = model(test_data, h_batch, true_ent=pos_h_index, testing = split == "test")

        # t_mask = 1 for strict negative **tails** and same for h_mask
        if filtered_data is None:
            t_mask, h_mask = tasks.strict_negative_mask(test_data, batch)
        else:
            t_mask, h_mask = tasks.strict_negative_mask(filtered_data, batch)

        t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
        h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)
        num_t_negative = t_mask.sum(dim=-1)
        num_h_negative = h_mask.sum(dim=-1)

        rankings += [t_ranking, h_ranking]
        num_negatives += [num_t_negative, num_h_negative]

    ranking = torch.cat(rankings)
    num_negative = torch.cat(num_negatives)
    all_size = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size[rank] = len(ranking)
    if world_size > 1:
        dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
    cum_size = all_size.cumsum(0)
    all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_ranking[cum_size[rank] - all_size[rank]: cum_size[rank]] = ranking
    all_num_negative = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_num_negative[cum_size[rank] - all_size[rank]: cum_size[rank]] = num_negative
    if world_size > 1:
        dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)

    all_scores = {}

    if rank == 0:
        for metric in cfg.task.metric:
            if metric == "mr":
                score = all_ranking.float().mean()
            elif metric == "mrr":
                score = (1 / all_ranking.float()).mean()
            elif metric.startswith("hits@"):
                values = metric[5:].split("_")
                threshold = int(values[0])
                if len(values) > 1:
                    num_sample = int(values[1])
                    # unbiased estimation
                    fp_rate = (all_ranking - 1).float() / all_num_negative
                    score = 0
                    for i in range(threshold):
                        # choose i false positive from num_sample - 1 negatives
                        num_comb = math.factorial(num_sample - 1) / math.factorial(i) / math.factorial(num_sample - i - 1)
                        score += num_comb * (fp_rate ** i) * ((1 - fp_rate) ** (num_sample - i - 1))
                    score = score.mean()
                else:
                    score = (all_ranking <= threshold).float().mean()
                
                all_scores[metric] = score

            logger.warning("%s: %g" % (metric, score))
    mrr = (1 / all_ranking.float()).mean()

    return all_scores['hits@10'] if 'hits@10_50' in all_scores else mrr


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    set_seed(args.seed)

    if not args.testing:
        working_dir = util.create_working_directory(cfg)

    logger = util.get_root_logger()
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
        logger.warning("Delta: %s" % args.delta)
        logger.warning(pprint.pformat(cfg))
    
    dataset_name = cfg.dataset["class"]
    is_inductive = cfg.dataset["class"].startswith("Ind")
    dataset = util.build_dataset(cfg)
    
    # Distances class
    dist_dataset = datasets_ours.DistDataset(dataset_name, args.delta, device=util.get_device(cfg))

    cfg.model.num_nodes = dataset[0].num_nodes
    cfg.model.num_relation = dataset.num_relations
    model = util.build_model(cfg, dist_dataset, args, dataset, is_inductive)

    device = util.get_device(cfg)
    model = model.to(device)

    model.train_degree = model.train_degree.to(device)
    model.test_degree = model.test_degree.to(device)

    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    test_data = test_data.to(device)

    if is_inductive:
        # for inductive setting, use only the test fact graph for filtered ranking
        filtered_data = None
    else:
        # for transductive setting, use the whole graph for filtered ranking
        filtered_data = Data(edge_index=dataset.data.target_edge_index, edge_type=dataset.data.target_edge_type)
        filtered_data = filtered_data.to(device)

    ### Test saved model
    if args.checkpoint is not None:
        train_eval = args.eval_on.lower() == "train"
        eval_data = train_data if train_eval else test_data
        test(cfg, model, eval_data, split="test", filtered_data=filtered_data)
        exit() 

    train_and_validate(cfg, model, train_data, valid_data, filtered_data=filtered_data, seed=args.seed)
    
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on valid")
    test(cfg, model, valid_data, split="valid", filtered_data=filtered_data)

    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on test")
    test(cfg, model, test_data, filtered_data=filtered_data)

