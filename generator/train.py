import yaml
import wandb
import os
from diffusion.mdm import MDM
from diffusion.mdm_climb_motion_sampler import MDMClimbMotionSampler
import sys
import pickle
from pathlib import Path

def train_mdm(config, input_mdm=None):
    use_wandb = config["use_wandb"]

    # Since loading the sampler can be pretty slow
    sampler_file_path = Path(config["sampler_save_filepath"])
    try:
        sampler_save_dir = sampler_file_path.parent
        print("making new directory:", sampler_save_dir)
        os.makedirs(sampler_save_dir, exist_ok=True)
    except:
        print("could not make directory")

    if sampler_file_path.is_file() and False:
        print("loading sampler: ", sampler_file_path)
        motion_sampler = pickle.load(sampler_file_path.open("rb"))
        motion_sampler.update_old_sampler()
    else:
        # motion_sampler = MDMHeightfieldContactMotionSampler(cfg=config)
        motion_sampler = MDMClimbMotionSampler(cfg=config)
        sampler_file_path.write_bytes(pickle.dumps(motion_sampler))

    config['seq_len'] = motion_sampler.get_seq_len()

    if input_mdm is None:
        if "input_model_path" in config:
            input_model_path = Path(config["input_model_path"])
            diffusion_model = pickle.load(input_model_path.open("rb"))
            diffusion_model.update_old_mdm()
            diffusion_model._use_wandb = use_wandb
        else:
            diffusion_model = MDM(cfg=config)
    else:
        diffusion_model = input_mdm

    output_dir = Path(config['output_dir'])
    checkpoint_dir = output_dir / "checkpoints"
    print("making new directory:", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print("making new directory:", checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if use_wandb:
        wandb.login()
        run = wandb.init(
            project="train-mdm",
            config=config
        )

    diffusion_model.train(motion_sampler, checkpoint_dir, stats_filepath=None)

    output_model_path = output_dir / "model.pkl"
    diffusion_model.save(output_model_path)
    print("saved diffusion model:", output_model_path)

    if use_wandb:
        run.finish()

    return diffusion_model

if __name__ == "__main__":

    if len(sys.argv) == 3:
        assert sys.argv[1] == "--config"
        cfg_path = Path(sys.argv[2])
        print("loading mdm training config from", cfg_path)
    else:
        cfg_path = Path("cfg/generator_default.yaml")

    try:
        config = yaml.safe_load(cfg_path.open("r"))
    except IOError:
        print("error opening file:", cfg_path)
        exit()

    train_mdm(config)