import torch
import json
import os
import sys
import time
import deepspeed
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from stable_audio_tools.data.dataset import create_dataloader_from_config
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict, remove_weight_norm_from_model
from stable_audio_tools.training.diffusion import DiffusionCondTrainingWrapper, DiffusionCondDemoCallback
from stable_audio_tools.training.utils import copy_state_dict
from stable_audio_tools.interface.aeiou import audio_spectrogram_image
from stable_audio_tools.training.utils import log_audio, log_image

def get_all_args():
    parser = argparse.ArgumentParser(description='DeepSpeed Training Script')
    
    # 基本参数
    parser.add_argument('--name', type=str, default='stable_audio_tools_rwkv',
                      help='name of the run')
    parser.add_argument('--project', type=str, default='server2_4090_audio_diffusion',
                      help='name of the project')
    parser.add_argument('--batch-size', type=int, default=2,
                      help='the batch size')
    parser.add_argument('--recover', type=bool, default=False,
                      help='If true, attempts to resume training from latest checkpoint')
    parser.add_argument('--save-top-k', type=int, default=-1,
                      help='Save top K model checkpoints during training')
    parser.add_argument('--num-nodes', type=int, default=1,
                      help='number of nodes to use for training')
    parser.add_argument('--strategy', type=str, default='deepspeed',
                      help='Multi-GPU strategy for PyTorch Lightning')
    parser.add_argument('--precision', type=str, default='bf16-mixed',
                      help='Precision to use for training')
    parser.add_argument('--num-workers', type=int, default=6,
                      help='number of CPU workers for the DataLoader')
    parser.add_argument('--seed', type=int, default=42,
                      help='the random seed')
    parser.add_argument('--accum-batches', type=int, default=1,
                      help='Batches for gradient accumulation')
    
    # 检查点相关参数
    parser.add_argument('--checkpoint-every', type=int, default=10000,
                      help='Number of steps between checkpoints')
    parser.add_argument('--val-every', type=int, default=-1,
                      help='Number of steps between validation runs')
    parser.add_argument('--ckpt-path', type=str, default='',
                      help='trainer checkpoint file to restart training from')
    parser.add_argument('--pretrained-ckpt-path', type=str, default='',
                      help='model checkpoint file to start a new training run from')
    parser.add_argument('--pretransform-ckpt-path', type=str, default='',
                      help='Checkpoint path for the pretransform model if needed')
    
    # 配置文件路径
    parser.add_argument('--model-config', type=str, 
                      default='stable_audio_tools/configs/model_configs/txt2audio/stable_audio_2_0_rwkv.json',
                      help='configuration model specifying model hyperparameters')
    parser.add_argument('--dataset-config', type=str,
                      default='stable_audio_tools/configs/dataset_configs/parquet_dataset.json',
                      help='configuration for datasets')
    parser.add_argument('--val-dataset-config', type=str, default='',
                      help='configuration for validation datasets')
    
    # 训练相关参数
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                      help='directory to save the checkpoints in')
    parser.add_argument('--gradient-clip-val', type=float, default=0.0,
                      help='gradient_clip_val passed into PyTorch Lightning Trainer')
    parser.add_argument('--remove-pretransform-weight-norm', type=str, default='',
                      help='remove the weight norm from the pretransform model')
    
    # 日志相关参数
    parser.add_argument('--logger', type=str, default='wandb',
                      help='Logger type to use')
    
    # DeepSpeed相关参数
    parser.add_argument('--deepspeed-config', type=str, default='ds_config.json',
                      help='DeepSpeed config file')
    parser.add_argument('--local_rank', type=int, default=-1,
                      help='Local rank for distributed training')
    
    # 训练包装器参数
    parser.add_argument('--lr', type=float, default=1e-4,
                      help='learning rate')
    parser.add_argument('--mask-padding', action='store_true',
                      help='whether to mask padding in loss calculation')
    parser.add_argument('--mask-padding-dropout', type=float, default=0.0,
                      help='dropout probability for mask padding')
    parser.add_argument('--use-ema', action='store_true',
                      help='whether to use exponential moving average')
    parser.add_argument('--log-loss-info', action='store_true',
                      help='whether to log loss information')
    parser.add_argument('--pre-encoded', action='store_true',
                      help='whether to use pre-encoded inputs')
    parser.add_argument('--cfg-dropout-prob', type=float, default=0.0,
                      help='probability of classifier-free guidance dropout')
    parser.add_argument('--timestep-sampler', type=str, default='uniform',
                      help='timestep sampling strategy')
    parser.add_argument('--validation-timesteps', type=int, default=100,
                      help='number of timesteps for validation')
    

    
    args = parser.parse_args()
    return args

class DeepSpeedTrainer:
    def __init__(self, args):
        self.args = args
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.global_rank = int(os.environ.get('RANK', 0))
        self.ds_config = args.deepspeed_config
        
        # 设置随机种子
        seed = args.seed
        if os.environ.get("SLURM_PROCID") is not None:
            seed += int(os.environ.get("SLURM_PROCID"))
        torch.manual_seed(seed)
        
        # 加载配置
        with open(args.model_config) as f:
            self.model_config = json.load(f)
            
        with open(args.dataset_config) as f:
            self.dataset_config = json.load(f)
            
        # 创建模型
        self.model = create_model_from_config(self.model_config)
        
        # 加载预训练权重
        if args.pretrained_ckpt_path:
            copy_state_dict(self.model, load_ckpt_state_dict(args.pretrained_ckpt_path))
            
        if args.remove_pretransform_weight_norm == "pre_load":
            remove_weight_norm_from_model(self.model.pretransform)
            
        if args.pretransform_ckpt_path:
            self.model.pretransform.load_state_dict(load_ckpt_state_dict(args.pretransform_ckpt_path))
            
        if args.remove_pretransform_weight_norm == "post_load":
            remove_weight_norm_from_model(self.model.pretransform)
            
        # 创建数据加载器
        self.train_dl = create_dataloader_from_config(
            self.dataset_config,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sample_rate=self.model_config["sample_rate"],
            sample_size=self.model_config["sample_size"],
            audio_channels=self.model_config.get("audio_channels", 2),
        )
        
        # 创建验证数据加载器
        self.val_dl = None
        if args.val_dataset_config:
            with open(args.val_dataset_config) as f:
                val_dataset_config = json.load(f)
                
            self.val_dl = create_dataloader_from_config(
                val_dataset_config,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                sample_rate=self.model_config["sample_rate"],
                sample_size=self.model_config["sample_size"],
                audio_channels=self.model_config.get("audio_channels", 2),
                shuffle=False
            )
            
        # 创建训练包装器
        self.training_wrapper = DiffusionCondTrainingWrapper(
            self.model,
            lr=args.lr,
            mask_padding=args.mask_padding,
            mask_padding_dropout=args.mask_padding_dropout,
            use_ema=args.use_ema,
            log_loss_info=args.log_loss_info,
            optimizer_configs=self.model_config["training"]["optimizer_configs"],
            pre_encoded=args.pre_encoded,
            cfg_dropout_prob=args.cfg_dropout_prob,
            timestep_sampler=args.timestep_sampler
        )
        
        # 创建演示回调

        training_config = model_config.get('training', None)
        demo_config = training_config.get("demo", {})
        self.demo_callback = DiffusionCondDemoCallback(
            demo_every=demo_config.get("demo_every", 2000), 
            sample_size=model_config["sample_size"],
            sample_rate=model_config["sample_rate"],
            demo_steps=demo_config.get("demo_steps", 250), 
            num_demos=demo_config["num_demos"],
            demo_cfg_scales=demo_config["demo_cfg_scales"],
            demo_conditioning=demo_config.get("demo_cond", {}),
            demo_cond_from_batch=demo_config.get("demo_cond_from_batch", False),
            display_audio_cond=demo_config.get("display_audio_cond", False),
            cond_display_configs=demo_config.get("cond_display_configs", None),
        )
        
        # 初始化DeepSpeed
        with open(self.ds_config, 'r') as f:
            config = json.load(f)
        self.model_engine, self.optimizer, self.train_dl, _ = deepspeed.initialize(
            model=self.training_wrapper,
            model_parameters=self.training_wrapper.parameters(),
            training_data=self.train_dl,
            config=config
        )
        
    def train(self):
        self.model_engine.train()
        global_step = 0
        
        while True:
            for batch in self.train_dl:
                # 训练步骤
                loss = self.model_engine(batch)
                
                # 更新模型
                self.model_engine.backward(loss)
                self.model_engine.step()
                
                # 记录日志
                if self.global_rank == 0:
                    self.log_metrics(global_step, loss)
                    
                # 演示生成
                if self.global_rank == 0 and global_step % self.args.demo_every == 0:
                    self.demo_callback.on_train_batch_end(
                        None,  # trainer
                        self.training_wrapper,  # module
                        None,  # outputs
                        batch,  # batch
                        None  # batch_idx
                    )
                    
                global_step += 1
                
    def log_metrics(self, global_step, loss):
        if self.args.logger == 'wandb':
            import wandb
            wandb.log({
                'train/loss': loss.item(),
                'train/global_step': global_step
            })
        elif self.args.logger == 'comet':
            from comet_ml import Experiment
            experiment = Experiment()
            experiment.log_metric('train/loss', loss.item(), step=global_step)
            
def main():
    args = get_all_args()
    print(f'args: {args}')
    # 初始化分布式环境
    deepspeed.init_distributed()
    
    # 创建训练器
    trainer = DeepSpeedTrainer(args)
    
    # 开始训练
    trainer.train()
    
if __name__ == '__main__':
    main() 