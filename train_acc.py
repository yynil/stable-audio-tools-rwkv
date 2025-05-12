import torch
import json
import os
import pytorch_lightning as pl

from prefigure.prefigure import get_all_args, push_wandb_config
from stable_audio_tools.data.dataset import create_dataloader_from_config, fast_scandir
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict, remove_weight_norm_from_model
from stable_audio_tools.training import create_training_wrapper_from_config, create_demo_callback_from_config
from stable_audio_tools.training.utils import copy_state_dict
import accelerate
from stable_audio_tools.training.utils import create_optimizer_from_config, create_scheduler_from_config
from functools import partial
import wandb
class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        if module.device == 'cuda:0':
            #print module's datatype
            for name, param in module.named_parameters():
                print(f'{name}: {param.dtype}, is_grad :{param.requires_grad}, device: {param.device}')
        print(f'type(err): {type(err)}')
        print(f'{type(err).__name__}: {err}')

class ModelConfigEmbedderCallback(pl.Callback):
    def __init__(self, model_config):
        self.model_config = model_config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["model_config"] = self.model_config

def save_checkpoint(training_wrapper, checkpoint_dir, global_step, accelerator, max_keep=3, is_epoch_end=False, epoch=None):
    """保存检查点并限制保留最近的几个检查点
    
    Args:
        training_wrapper: 训练包装器
        checkpoint_dir: 检查点保存目录
        global_step: 当前全局步数
        accelerator: accelerate 实例
        max_keep: 最多保留的检查点数量
        is_epoch_end: 是否是 epoch 结束时的检查点
        epoch: 当前 epoch 数
    """
    # 确保目录存在
    os.makedirs(checkpoint_dir, exist_ok=True)
    unwrapped_model = accelerator.unwrap_model(training_wrapper)
    
    if is_epoch_end:
        # epoch 结束时的检查点，使用 epoch 编号命名
        save_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt")
    else:
        # 普通检查点，使用步数命名
        save_path = os.path.join(checkpoint_dir, f"model_step_{global_step}.pt")
    
    # 保存完整的模型权重
    torch.save({"state_dict": unwrapped_model.state_dict()}, save_path)
    
    # 只对普通检查点进行清理，保留最近的 max_keep 个
    if not is_epoch_end:
        checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith("model_step_")])
        if len(checkpoints) > max_keep:
            for old_checkpoint in checkpoints[:-max_keep]:
                os.remove(os.path.join(checkpoint_dir, old_checkpoint))

def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = get_all_args()
    seed = args.seed

    # Set a different seed for each process if using SLURM
    if os.environ.get("SLURM_PROCID") is not None:
        seed += int(os.environ.get("SLURM_PROCID"))

    pl.seed_everything(seed, workers=True)

    #Get JSON config from args.model_config
    with open(args.model_config) as f:
        model_config = json.load(f)

    with open(args.dataset_config) as f:
        dataset_config = json.load(f)
    print(f'model_config: {model_config}')
    print(f'dataset_config: {dataset_config}')
    print(f'args.dataset_config: {args.dataset_config}')
    train_dl = create_dataloader_from_config(
        dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 2),
    )

    val_dl = None
    val_dataset_config = None

    if args.val_dataset_config:
        with open(args.val_dataset_config) as f:
            val_dataset_config = json.load(f)

        val_dl = create_dataloader_from_config(
            val_dataset_config,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sample_rate=model_config["sample_rate"],
            sample_size=model_config["sample_size"],
            audio_channels=model_config.get("audio_channels", 2),
            shuffle=False
        )

    model = create_model_from_config(model_config)

    if args.pretrained_ckpt_path:
        copy_state_dict(model, load_ckpt_state_dict(args.pretrained_ckpt_path))

    if args.remove_pretransform_weight_norm == "pre_load":
        remove_weight_norm_from_model(model.pretransform)

    if args.pretransform_ckpt_path:
        model.pretransform.load_state_dict(load_ckpt_state_dict(args.pretransform_ckpt_path))

    # Remove weight_norm from the pretransform if specified
    if args.remove_pretransform_weight_norm == "post_load":
        remove_weight_norm_from_model(model.pretransform)

    

    training_wrapper = create_training_wrapper_from_config(model_config, model)
    optimizer,scheduler = training_wrapper.configure_optimizers()
    diffusion_opt_config = training_wrapper.optimizer_configs['diffusion']
    opt_diff = create_optimizer_from_config(diffusion_opt_config['optimizer'], training_wrapper.diffusion.parameters())

    if "scheduler" in diffusion_opt_config:
        sched_diff = create_scheduler_from_config(diffusion_opt_config['scheduler'], opt_diff)
        sched_diff_config = {
                "scheduler": sched_diff,
                "interval": "step"
        }
    exc_callback = ExceptionCallback()

    checkpoint_dir = args.save_dir if args.save_dir else "checkpoints"

    if args.val_dataset_config:
        demo_callback = create_demo_callback_from_config(model_config, demo_dl=val_dl)
    else:
        demo_callback = create_demo_callback_from_config(model_config, demo_dl=train_dl)

    #Combine args and config dicts
    args_dict = vars(args)
    args_dict.update({"model_config": model_config})
    args_dict.update({"dataset_config": dataset_config})
    args_dict.update({"val_dataset_config": val_dataset_config})


    # 使用 accelerate 准备训练
    from accelerate import Accelerator
    from accelerate.utils import AORecipeKwargs
    def filter_linear_layers(module, fqn: str) -> bool:
        """
        A function which will check if `module` is:
        - a `torch.nn.Linear` layer
        - has in_features and out_features divisible by 16
        - is not part of `layers_to_filter`

        Args:
            module (`torch.nn.Module`):
                The module to check.
            fqn (`str`):
                The fully qualified name of the layer.
        """
        if isinstance(module, torch.nn.Linear):
            if module.in_features % 16 != 0 or module.out_features % 16 != 0:
                print(f'{fqn} is not divisible by 16,skip')
                return False
        else:
            print(f'{fqn} is not a linear layer,skip')
            return False
        if 'model.model.rwkv.layers.' in fqn and '.ffn.' in fqn:
            print(f'convert {fqn} to fp8')
            return True
        else:
            print(f'{fqn} is not a ffn layer,skip')
            return False
    
    kwargs = [AORecipeKwargs(module_filter_func=filter_linear_layers)]
    accelerator = Accelerator(mixed_precision="fp8", kwargs_handlers=kwargs)
    
    # 初始化 accelerate
    # accelerator = Accelerator()
    
    # 准备模型、优化器、数据加载器和调度器
    if "scheduler" in diffusion_opt_config:
        training_wrapper, opt_diff, train_dl, sched_diff = accelerator.prepare(
            training_wrapper, opt_diff, train_dl, sched_diff
        )
    else:
        training_wrapper, opt_diff, train_dl = accelerator.prepare(
            training_wrapper, opt_diff, train_dl
        )
    
    # 如果有验证数据集，也准备它
    if val_dl is not None:
        val_dl = accelerator.prepare(val_dl)
    
    # 打印训练信息
    accelerator.print(f"开始训练，设备数量: {accelerator.num_processes}")
    accelerator.print(f"混合精度: {accelerator.mixed_precision}")
    
    if args.logger == 'wandb' and accelerator.is_main_process:
        wandb.init(project=args.project,name=args.name)
    # 训练循环
    num_epochs = model_config["training"]["num_epochs"]
    global_step = 0
    if accelerator.is_main_process:
        from tqdm import tqdm
        pbar = tqdm(total=num_epochs*len(train_dl), desc="训练进度")
    for epoch in range(num_epochs):
        training_wrapper.train()
        
        for batch_idx, batch in enumerate(train_dl):
            # 前向传播
            loss = training_wrapper.training_step(batch, batch_idx)
            if accelerator.is_main_process:
                log_dict = training_wrapper.this_step_dictionary
                if args.logger == 'wandb':
                    for key, value in log_dict.items():
                        wandb.log({key: value, "step": global_step})
                pbar.set_postfix(log_dict)
                pbar.update(1)
            # 反向传播
            accelerator.backward(loss)
            
            # 更新参数
            opt_diff.step()
            if "scheduler" in diffusion_opt_config:
                sched_diff.step()
            opt_diff.zero_grad()
            
            # 更新 EMA
            if hasattr(training_wrapper, 'diffusion_ema') and training_wrapper.diffusion_ema is not None:
                training_wrapper.diffusion_ema.update()
            
            # 更新全局步数
            global_step += 1
            
            
            # 生成demo
            if global_step % model_config["training"]["demo"]["demo_every"] == 0:
                if accelerator.is_main_process:
                    accelerator.print(f"生成demo step {global_step}")
                    demo_callback.on_train_batch_end(None, training_wrapper, None, batch, batch_idx)
            
            # 检查点保存
            if args.checkpoint_every > 0 and global_step % args.checkpoint_every == 0:
                if accelerator.is_main_process:
                    accelerator.print(f"保存检查点 step {global_step}")
                    save_checkpoint(training_wrapper, checkpoint_dir, global_step, accelerator)
            
            # 如果有验证数据集，定期进行验证
            if val_dl is not None and args.val_every > 0 and global_step % args.val_every == 0:
                training_wrapper.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for val_batch_idx, val_batch in enumerate(val_dl):
                        val_step_loss = training_wrapper.validation_step(val_batch, val_batch_idx)
                        val_loss += val_step_loss.item()
                
                val_loss /= len(val_dl)
                accelerator.print(f"验证损失: {val_loss}")
                training_wrapper.train()
            
            # 如果达到最大步数，结束训练
            if args.max_steps > 0 and global_step >= args.max_steps:
                accelerator.print(f"达到最大步数 {args.max_steps}，结束训练")
                return
        
        # 每个 epoch 结束时保存检查点
        if accelerator.is_main_process:
            accelerator.print(f"保存 epoch {epoch} 的检查点")
            save_checkpoint(training_wrapper, checkpoint_dir, global_step, accelerator, is_epoch_end=True, epoch=epoch)

if __name__ == '__main__':
    main()
