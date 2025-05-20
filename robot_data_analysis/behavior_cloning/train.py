import os
# import wandb
import argparse
import numpy as np
import yaml
import time
import pdb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam, AdamW
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from dataset import BCDataset
from drlvo_model import CustomCNN
import tqdm
import itertools
import torch.nn.functional as F

def _compute_losses(
    #dist_label: torch.Tensor,
    action_label: torch.Tensor,
    #dist_pred: torch.Tensor,
    action_pred: torch.Tensor,
    #alpha: float,
    learn_angle: bool,
    action_mask: torch.Tensor = None,
):
    """
    Compute losses for distance and action prediction.

    """
    #dist_loss = F.mse_loss(dist_pred.squeeze(-1), dist_label.float())

    def action_reduce(unreduced_loss: torch.Tensor):
        # Reduce over non-batch dimensions to get loss per batch element
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
        return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

    # Mask out invalid inputs (for negatives, or when the distance between obs and goal is large)
    assert action_pred.shape == action_label.shape, f"{action_pred.shape} != {action_label.shape}"
    action_loss = action_reduce(F.mse_loss(action_pred, action_label, reduction="none"))

    action_waypts_cos_similairity = action_reduce(F.cosine_similarity(
        action_pred[:, :, :2], action_label[:, :, :2], dim=-1
    ))
    multi_action_waypts_cos_sim = action_reduce(F.cosine_similarity(
        torch.flatten(action_pred[:, :, :2], start_dim=1),
        torch.flatten(action_label[:, :, :2], start_dim=1),
        dim=-1,
    ))

    results = {
        #"dist_loss": dist_loss,
        "action_loss": action_loss,
        "action_waypts_cos_sim": action_waypts_cos_similairity,
        "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim,
    }

    if learn_angle:
        action_orien_cos_sim = action_reduce(F.cosine_similarity(
            action_pred[:, :, 2:], action_label[:, :, 2:], dim=-1
        ))
        multi_action_orien_cos_sim = action_reduce(F.cosine_similarity(
            torch.flatten(action_pred[:, :, 2:], start_dim=1),
            torch.flatten(action_label[:, :, 2:], start_dim=1),
            dim=-1,
            )
        )
        results["action_orien_cos_sim"] = action_orien_cos_sim
        results["multi_action_orien_cos_sim"] = multi_action_orien_cos_sim

    total_loss = action_loss #alpha * 1e-2 * dist_loss + (1 - alpha) * action_loss
    results["total_loss"] = total_loss

    return results

def train(
    model: nn.Module,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    epoch: int,
    alpha: float = 0.5,
    learn_angle: bool = True,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
    use_tqdm: bool = True,
    goal_type: str = "image",
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        learn_angle: whether to learn the angle of the action
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
        use_tqdm: whether to use tqdm
    """
    model.train()
    tqdm_iter = tqdm.tqdm(
        dataloader,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Training epoch {epoch}",
    )
    loss_log = {}
    for i, data in enumerate(tqdm_iter):
        (
            pedestrian_map,
            scan,
            action_label,
            dist_label,
            goal_pos,
            dataset_index,
            action_mask
        ) = data
        # print(f"ped map shape: {pedestrian_map.shape}")
        # print(f"scan shape: {scan.shape}")
        # print(f"action label shape: {action_label.shape}")
        # print(f"goal pos shape: {goal_pos.shape}")
        pedestrian_map = pedestrian_map.to(device)
        scan = scan.to(device)
        goal_pos = goal_pos.to(device)
        model_outputs = model(pedestrian_map,scan,goal_pos)

        dist_label = dist_label.to(device)
        action_label = action_label.to(device)
        action_mask = action_mask.to(device)

        optimizer.zero_grad()
      
        action_pred = model_outputs #these are deltas 
        action_pred = action_pred.reshape(action_pred.shape[0], action_label.shape[1],action_label.shape[2])
        action_pred[:,:,:2] = torch.cumsum(
            action_pred[:,:,:2],dim=1
        )
        if learn_angle:
            action_pred[:,:,2:] = F.normalize(
                action_pred[:,:,2:].clone(),dim=-1
            )
             
        losses = _compute_losses(
            #dist_label=dist_label,
            action_label=action_label,
            #dist_pred=dist_pred,
            action_pred=action_pred,
            #alpha=alpha,
            learn_angle=learn_angle,
            action_mask=action_mask,
        )

        losses["total_loss"].backward()
        optimizer.step()

        for key, value in losses.items():
            #if key in loggers:
            #    logger = loggers[key]
            #    logger.log_data(value.item())
            if key not in loss_log:
                loss_log[key] = []
            loss_log[key].append(value.item())
            if i % print_log_freq == 0 and print_log_freq != 0:
                print(f"(epoch {epoch}) {key} {np.mean(loss_log[key]):.4f}")
    
    for k,v in loss_log.items():
        loss_log[k] = np.mean(v)     
    return loss_log

def evaluate(
    eval_type: str,
    model: nn.Module,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    epoch: int = 0,
    alpha: float = 0.5,
    learn_angle: bool = True,
    num_images_log: int = 8,
    print_log_freq: int = 1,
    use_wandb: bool = True,
    eval_fraction: float = 1.0,
    use_tqdm: bool = True,
    goal_type: str = "image"
):
    """
    Evaluate the model on the given evaluation dataset.

    Args:
        eval_type (string): f"{data_type}_{eval_type}" (e.g. "recon_train", "gs_test", etc.)
        model (nn.Module): model to evaluate
        dataloader (DataLoader): dataloader for eval
        transform (transforms): transform to apply to images
        device (torch.device): device to use for evaluation
        project_folder (string): path to project folder
        epoch (int): current epoch
        alpha (float): weight for action loss
        learn_angle (bool): whether to learn the angle of the action
        num_images_log (int): number of images to log
        use_wandb (bool): whether to use wandb for logging
        eval_fraction (float): fraction of data to use for evaluation
        use_tqdm (bool): whether to use tqdm for logging
    """
    model.eval()
    num_batches = len(dataloader)
    num_batches = max(int(num_batches * eval_fraction), 1)

    #viz_obs_image = None
    loss_log = {}
    with torch.no_grad():
        tqdm_iter = tqdm.tqdm(
            itertools.islice(dataloader, num_batches),
            total=num_batches,
            disable=not use_tqdm,
            dynamic_ncols=True,
            desc=f"Evaluating {eval_type} for epoch {epoch}",
        )
        for i, data in enumerate(tqdm_iter):
            (
                pedestrian_map,
                scan,
                action_label,
                dist_label,
                goal_pos,
                dataset_index,
                action_mask
            ) = data

            pedestrian_map = pedestrian_map.to(device)
            scan = scan.to(device)
            goal_pos = goal_pos.to(device)
            model_outputs = model(pedestrian_map,scan,goal_pos)

            dist_label = dist_label.to(device)
            action_label = action_label.to(device)
            action_mask = action_mask.to(device)
            
            action_pred = model_outputs #these are deltas 
            action_pred = action_pred.reshape(action_pred.shape[0], action_label.shape[1],action_label.shape[2])
            action_pred[:,:,:2] = torch.cumsum(
                action_pred[:,:,:2],dim=1
            )
            if learn_angle:
                action_pred[:,:,2:] = F.normalize(
                    action_pred[:,:,2:].clone(),dim=-1
                )

            losses = _compute_losses(
                #dist_label=dist_label,
                action_label=action_label,
                #dist_pred=dist_pred,
                action_pred=action_pred,
                #alpha=alpha,
                learn_angle=learn_angle,
                action_mask=action_mask,
            )

            for key, value in losses.items():
                #if key in loggers:
                #    logger = loggers[key]
                #    logger.log_data(value.item())
                if key not in loss_log:
                    loss_log[key] = []
                loss_log[key].append(value.item())
                if i % print_log_freq == 0 and print_log_freq != 0:
                    print(f"(epoch {epoch}) {key} {np.mean(loss_log[key]):.4f}")

    # Log data to wandb/console, with visualizations selected from the last batch

    return loss_log

def main(config):
    assert config["distance"]["min_dist_cat"] < config["distance"]["max_dist_cat"]
    assert config["action"]["min_dist_cat"] < config["action"]["max_dist_cat"]

    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if "gpu_ids" not in config:
            config["gpu_ids"] = [0]
        elif type(config["gpu_ids"]) == int:
            config["gpu_ids"] = [config["gpu_ids"]]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(x) for x in config["gpu_ids"]]
        )
        print("Using cuda devices:", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print("Using cpu")

    first_gpu_id = config["gpu_ids"][0]
    device = torch.device(
        f"cuda:{first_gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True

    cudnn.benchmark = True  # good if input sizes don't vary

    # Load the data
    train_dataset = []
    test_dataloaders = {}

    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]
        
        if "goals_per_obs" not in data_config:
            data_config["goals_per_obs"] = 1
        if "end_slack" not in data_config:
            data_config["end_slack"] = 0
        if "waypoint_spacing" not in data_config:
            data_config["waypoint_spacing"] = 1

        for data_split_type in ["train", "test"]:
            if data_split_type in data_config:
                    dataset = BCDataset(
                        data_folder=data_config["data_folder"],
                        data_split_folder=data_config[data_split_type],
                        dataset_name=dataset_name,
                        image_size=config["image_size"],
                        waypoint_spacing=data_config["waypoint_spacing"],
                        min_dist_cat=config["distance"]["min_dist_cat"],
                        max_dist_cat=config["distance"]["max_dist_cat"],
                        min_action_distance=config["action"]["min_dist_cat"],
                        max_action_distance=config["action"]["max_dist_cat"],
                        len_traj_pred=config["len_traj_pred"],
                        learn_angle=config["learn_angle"],
                        context_size=config["context_size"],
                        end_slack=data_config["end_slack"],
                        normalize=config["normalize"],
                    )
                    if data_split_type == "train":
                        train_dataset.append(dataset)
                    else:
                        dataset_type = f"{dataset_name}_{data_split_type}"
                        if dataset_type not in test_dataloaders:
                            test_dataloaders[dataset_type] = {}
                        test_dataloaders[dataset_type] = dataset

    # combine all the datasets from different robots
    train_dataset = ConcatDataset(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=False,
        persistent_workers=True,
    )

    if "eval_batch_size" not in config:
        config["eval_batch_size"] = config["batch_size"]

    for dataset_type, dataset in test_dataloaders.items():
        test_dataloaders[dataset_type] = DataLoader(
            dataset,
            batch_size=config["eval_batch_size"],
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

    model = CustomCNN(
        features_dim = config["features_dim"],
        output_dim = config["len_traj_pred"] * (2+2*config["learn_angle"]),
    )

    if config["clipping"]:
        print("Clipping gradients to", config["max_norm"])
        for p in model.parameters():
            if not p.requires_grad:
                continue
            p.register_hook(
                lambda grad: torch.clamp(
                    grad, -1 * config["max_norm"], config["max_norm"]
                )
            )

    lr = float(config["lr"])
    config["optimizer"] = config["optimizer"].lower()
    if config["optimizer"] == "adam":
        optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    elif config["optimizer"] == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr)
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Optimizer {config['optimizer']} not supported")

    current_epoch = 0
    if "load_run" in config:
        load_project_folder = os.path.join("logs", config["load_run"])
        print("Loading model from ", load_project_folder)
        latest_path = os.path.join(load_project_folder, "latest.pth")
        latest_checkpoint = torch.load(latest_path) #f"cuda:{}" if torch.cuda.is_available() else "cpu")
        loaded_model = latest_checkpoint["model"]
        try:
            state_dict = loaded_model.module.state_dict()
            model.load_state_dict(state_dict, strict=False)
        except AttributeError as e:
            print("Error loading model, trying to load without module")
            state_dict = loaded_model.state_dict()
            model.load_state_dict(state_dict, strict=False)
        
        if "epoch" in latest_checkpoint:
            current_epoch = latest_checkpoint["epoch"] + 1

    scheduler = None
    if config["scheduler"] is not None:
        config["scheduler"] = config["scheduler"].lower()
        if config["scheduler"] == "cosine":
            print("Using cosine annealing with T_max", config["epochs"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["epochs"]
            )
        elif config["scheduler"] == "cyclic":
            print("Using cyclic LR with cycle", config["cyclic_period"])
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=lr / 10.,
                max_lr=lr,
                step_size_up=config["cyclic_period"] // 2,
                cycle_momentum=False,
            )
        elif config["scheduler"] == "plateau":
            print("Using ReduceLROnPlateau")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=config["plateau_factor"],
                patience=config["plateau_patience"],
                verbose=True,
            )
        elif config["scheduler"] == "step":
            print("Using step LR with step size", config["step_size"])
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config["step_size"],
                gamma=config["step_gamma"],
                last_epoch = current_epoch - 1,
            )
        if config["warmup"]:
            print("Using warmup scheduler")
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=config["warmup_epochs"],
                after_scheduler=scheduler,
            )
    
    # Multi-GPU
    if len(config["gpu_ids"]) > 1:
        model = nn.DataParallel(model, device_ids=config["gpu_ids"])
    model = model.to(device)

    if "load_run" in config:  # load optimizer and scheduler after data parallel
        if "optimizer" in latest_checkpoint:
            optimizer.load_state_dict(latest_checkpoint["optimizer"].state_dict())
        if scheduler is not None and "scheduler" in latest_checkpoint:
            scheduler.load_state_dict(latest_checkpoint["scheduler"].state_dict())
    
    #tensorboard logging
    writer = SummaryWriter(
        os.path.join(config["project_folder"], "tensorboard")
    )
    
    #### train-eval loop ####
    assert 0 <= config['alpha'] <= 1
    project_folder=config["project_folder"]
    normalized=config["normalize"]
    print_log_freq=config["print_log_freq"]
    image_log_freq=config["image_log_freq"]
    num_images_log=config["num_images_log"]
    wandb_log_freq=config["wandb_log_freq"]
    goal_type = config["goal_type"]
    current_epoch=current_epoch
    learn_angle=config["learn_angle"]
    alpha=config["alpha"]
    use_wandb=config["use_wandb"]
    eval_fraction=config["eval_fraction"]
    epochs = config["epochs"]
    train_model = config["train_model"]
    
    transform = ([])
    transform = transforms.Compose(transform) 
    latest_path = os.path.join(project_folder, f"latest.pth")

    for epoch in range(current_epoch, current_epoch + epochs):
        if train_model:
            print(
            f"Start ViNT Training Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            training_loss = train(
                model=model,
                optimizer=optimizer,
                dataloader=train_loader,
                transform=transform,
                device=device,
                project_folder=project_folder,
                normalized=normalized,
                epoch=epoch,
                alpha=alpha,
                learn_angle=learn_angle,
                print_log_freq=print_log_freq,
                wandb_log_freq=wandb_log_freq,
                image_log_freq=image_log_freq,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
            )
            writer.add_scalar(
                "train/lr", optimizer.param_groups[0]["lr"], epoch
            )
            for k,v in training_loss.items():
                writer.add_scalar(
                    f"train/{k}", np.mean(v), epoch
                )
            
        avg_total_test_loss = []
        for dataset_type in test_dataloaders:
            print(
                f"Start {dataset_type} ViNT Testing Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            loader = test_dataloaders[dataset_type]

            testing_loss = evaluate(
                eval_type=dataset_type,
                model=model,
                dataloader=loader,
                transform=transform,
                device=device,
                project_folder=project_folder,
                normalized=normalized,
                epoch=epoch,
                alpha=alpha,
                learn_angle=learn_angle,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                print_log_freq=print_log_freq,
                eval_fraction=eval_fraction
            )

            avg_total_test_loss.append(testing_loss["total_loss"])
            for k,v in testing_loss.items():
                writer.add_scalar(
                    f"test/{dataset_type}/{k}", np.mean(v), epoch
                )

        checkpoint = {
            "epoch": epoch,
            "model": model,
            "optimizer": optimizer,
            "avg_total_test_loss": np.mean(avg_total_test_loss),
            "scheduler": scheduler
        }

        if scheduler is not None:
            # scheduler calls based on the type of scheduler
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(np.mean(avg_total_test_loss))
            else:
                scheduler.step()
                
        # wandb.log({
        #     "avg_total_test_loss": np.mean(avg_total_test_loss),
        #     "lr": optimizer.param_groups[0]["lr"],
        # }, commit=False)

        numbered_path = os.path.join(project_folder, f"{epoch}.pth")
        torch.save(checkpoint, latest_path)
        torch.save(checkpoint, numbered_path)  # keep track of model at every epoch
    
    print("FINISHED TRAINING")
    
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Visual Navigation Transformer")

    # project setup
    parser.add_argument(
        "--config",
        "-c",
        default="config/vint.yaml",
        type=str,
        help="Path to the config file in train_config folder",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    config["project_folder"] = os.path.join(
        "logs", config["project_name"], config["run_name"]
    )
    os.makedirs(
        config[
            "project_folder"
        ],  # should error if dir already exists to avoid overwriting and old project
    )

    # if config["use_wandb"]:
    #     wandb.login()
    #     wandb.init(
    #         project=config["project_name"],
    #         settings=wandb.Settings(start_method="fork"),
    #         entity="gnmv2", # TODO: change this to your wandb entity
    #     )
    #     wandb.save(args.config, policy="now")  # save the config file
    #     wandb.run.name = config["run_name"]
    #     # update the wandb args with the training configurations
    #     if wandb.run:
    #         wandb.config.update(config)

    print(config)
    main(config)