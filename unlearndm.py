
import logging
import os
import shutil
import datetime
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# --- 核心库 ---
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed, ProjectConfiguration
from omegaconf import DictConfig, OmegaConf
import hydra
from tqdm.auto import tqdm

# --- Diffusers, Transformers, Datasets ---
import diffusers
import transformers
import datasets
from diffusers import DDPMPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import (
    check_min_version,
    is_accelerate_version,
    is_tensorboard_available,
    is_wandb_available,
)
from Fed_Unlearn_dm import Task
from losses.ddpm_deletion_loss import DDPMDeletionLoss 
from data.utils.infinite_sampler import InfiniteSampler
from data.utils.repeat_sampler import RepeatedSampler
from evaluate import Evaluator

check_min_version("0.27.0.dev0")
logger = get_logger(__name__, log_level="INFO")

class DDPMDeletionLoss:
    def __init__(self, gamma, sigma):
        self.all_gamma = gamma
        self.all_sigma = sigma

    def ffgu(self, unet, initial_unet, timesteps, noise, conditioning, all_samples_dict, deletion_samples_dict, **kwargs):
        gamma = self.all_gamma[timesteps].view(-1, 1, 1, 1)
        sigma = self.all_sigma[timesteps].view(-1, 1, 1, 1)

        # 1. 计算保留集损失 (loss_x): 标准的 DDPM 损失
        all_noise_preds = unet(all_samples_dict['noisy_latents'], timesteps, **conditioning, return_dict=False)[0]
        loss_x = F.mse_loss(all_noise_preds, noise, reduction='none')

        # 2. 计算遗忘集损失 (loss_a): 引导性遗忘损失
        deletion_noise_preds = unet(deletion_samples_dict['noisy_latents'], timesteps, **conditioning, return_dict=False)[0]
        
        # 使用“保留”集数据的方向来构造修改后的噪声目标
        new_noises = (deletion_samples_dict['noisy_latents'] - gamma * all_samples_dict['og_latents']) / sigma
        modified_noises = 2 * new_noises - noise
        loss_a = F.mse_loss(deletion_noise_preds, modified_noises, reduction='none')

        # 组合总损失
        loss = loss_x.mean() + 3.0 * loss_a.mean()

        # 返回与原始脚本结构一致的元组
        return loss, loss_x, loss_a, None, None, None, None

class FederatedClient:
    def __init__(self, deletion_dataset, noise_scheduler, device, cfg):
        self.noise_scheduler = noise_scheduler
        self.device = device
        
        sampler = RepeatedSampler(deletion_dataset, num_repeats=cfg.training_steps * cfg.gradient_accumulation_steps * cfg.train_batch_size)
        self.dataloader = torch.utils.data.DataLoader(
            deletion_dataset, batch_size=cfg.train_batch_size, num_workers=cfg.dataloader_num_workers, sampler=sampler
        )
        self.data_iterator = iter(self.dataloader)
        logger.info(f"客户端初始化完成，持有 {len(deletion_dataset)} 张遗忘图片。")

    def _get_next_batch(self):
        try:
            return next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.dataloader)
            return next(self.data_iterator)

    def process_data_for_unlearning(self, timesteps, noise):
        deletion_images = self._get_next_batch()
        deletion_images = deletion_images.to(self.device)
        noisy_deletion_images = self.noise_scheduler.add_noise(deletion_images, noise, timesteps)
        
        return {"og_latents": deletion_images, "noisy_latents": noisy_deletion_images}

class DeleteCeleb(Task):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def run(self):
    
        logging_dir = os.path.join(self.cfg.output_dir, self.cfg.logging.logging_dir)

        if self.cfg.logging.logger == "tensorboard":
            if not is_tensorboard_available():
                raise ImportError(
                    "Make sure to install tensorboard if you want to use it for logging during training."
                )
        elif self.cfg.logging.logger == "wandb":
            if not is_wandb_available():
                raise ImportError(
                    "Make sure to install wandb if you want to use it for logging during training."
                )
            import wandb

        accelerator_project_config = ProjectConfiguration(project_dir=self.cfg.output_dir, logging_dir=logging_dir)
        kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=7200))
        accelerator = Accelerator(
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            mixed_precision=self.cfg.mixed_precision,
            log_with=self.cfg.logging.logger,
            project_config=accelerator_project_config,
            kwargs_handlers=[kwargs],
        )
        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:
            wandb_conf = OmegaConf.to_container(self.cfg)  # fully converts to dict (even nested keys)
            accelerator.init_trackers(project_name=self.cfg.project_name, config=wandb_conf)

        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)
        if accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        # If passed along, set the training seed now.
        if self.cfg.random_seed is not None:
            set_seed(self.cfg.random_seed)

        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if self.cfg.ema.use_ema:
                    ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))

                # breakpoint()
                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        accelerator.register_save_state_pre_hook(save_model_hook)
        print(f'Loading checkpoint {self.cfg.checkpoint_path}')
        ddpm_pipeline = DDPMPipeline.from_pretrained(self.cfg.checkpoint_path)
        unet = ddpm_pipeline.unet
        noise_scheduler = ddpm_pipeline.scheduler
        
        initial_unet = DDPMPipeline.from_pretrained(self.cfg.checkpoint_path).unet.to(accelerator.device)
        initial_unet.requires_grad_(False)

        ema_model = None
        if self.cfg.ema.use_ema:
            ema_model = EMAModel.from_pretrained(os.path.join(self.cfg.checkpoint_path, self.cfg.subfolders.unet_ema))

        optimizer = hydra.utils.instantiate(self.cfg.optimizer, unet.parameters())
        
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
            self.cfg.mixed_precision = accelerator.mixed_precision
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
            self.cfg.mixed_precision = accelerator.mixed_precision

        transform = hydra.utils.instantiate(self.cfg.transform)
        dataset_all = hydra.utils.instantiate(self.cfg.dataset_all, transform=transform) # 服务器的“保留”数据
        dataset_deletion = hydra.utils.instantiate(self.cfg.dataset_deletion, transform=transform) # 客户端的“遗忘”数据

        sampler_all = InfiniteSampler(dataset_all)
        dataloader_all = torch.utils.data.DataLoader(dataset_all, batch_size=self.cfg.train_batch_size, num_workers=self.cfg.dataloader_num_workers, sampler=sampler_all)
        
        client = FederatedClient(dataset_deletion, noise_scheduler, accelerator.device, self.cfg)
        
        dataset_all_iterator = iter(dataloader_all)

        # 使用配置文件中的 'lr_scheduler' 键名，而不是 'lr_scheduler.name'
        lr_scheduler = get_scheduler(self.cfg.lr_scheduler, optimizer=optimizer, num_warmup_steps=self.cfg.warmup_steps, num_training_steps=self.cfg.training_steps)

        unet, optimizer, lr_scheduler, dataset_all_iterator = accelerator.prepare(unet, optimizer, lr_scheduler, dataset_all_iterator)

        if self.cfg.ema.use_ema:
            ema_model.to(accelerator.device)

        logger.info("***** Running training *****")
        logger.info(f"Num all examples: {len(dataset_all)}")
        logger.info(f"Num deletion examples: {len(dataset_deletion)}")
        logger.info(f"  Num training steps = {self.cfg.training_steps}")
        logger.info(
            f"  Instantaneous batch size per device = {self.cfg.train_batch_size}"
        )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {self.cfg.gradient_accumulation_steps}"
        )
        # ----------------------------------------------------------
        #  4. 评估、损失函数等 (完全保留原始逻辑)
        # ----------------------------------------------------------
        global_step = 0
        gamma = noise_scheduler.alphas_cumprod ** 0.5
        gamma = gamma.to(accelerator.device)
        sigma = (1 - noise_scheduler.alphas_cumprod) ** 0.5
        sigma = sigma.to(accelerator.device)

        # 直接实例化内部定义的损失类
        loss_class = DDPMDeletionLoss(gamma=gamma, sigma=sigma)
        # 仍然根据配置文件中的 'loss_fn' 字段获取具体方法
        loss_fn = getattr(loss_class, self.cfg.deletion.loss_fn)

        def evaluate_model(unet, num_imgs_to_generate=self.cfg.eval_batch_size, batch_size=self.cfg.eval_batch_size,
                           set_generator=False):
            unet.eval()
            unet = accelerator.unwrap_model(unet)

            if self.cfg.ema.use_ema:
                ema_model.store(unet.parameters())
                ema_model.copy_to(unet.parameters())

            evaluator = Evaluator(self.cfg)
            evaluator.load_model(unet, noise_scheduler)  # 加载模型和扩散调度器 调度器决定了“扩散模型如何去噪生成图像”

            generated_images = []
            remaining_images = num_imgs_to_generate

            while remaining_images > 0:
                current_batch_size = min(batch_size, remaining_images)
                images = evaluator.sample_images(current_batch_size, set_generator=set_generator)
                generated_images.append(images)
                remaining_images -= current_batch_size

            generated_images = np.concatenate(generated_images, axis=0)

            if self.cfg.ema.use_ema:
                ema_model.restore(unet.parameters())  # 恢复unet原始参数

            unet.train()
            return generated_images
        
        def evaluate_unlearning_timestep(unet, timestep, clean_image, num_imgs_to_generate=self.cfg.eval_batch_size,
                                         batch_size=self.cfg.eval_batch_size):
            unet.eval()
            unet = accelerator.unwrap_model(unet)

            if self.cfg.ema.use_ema:
                ema_model.store(unet.parameters())
                ema_model.copy_to(unet.parameters())

            evaluator = Evaluator(self.cfg)
            evaluator.load_model(unet, noise_scheduler)

            generated_images = []
            remaining_images = num_imgs_to_generate
            generator = torch.Generator(device=accelerator.device).manual_seed(self.cfg.random_seed)

            while remaining_images > 0:
                current_batch_size = min(batch_size, remaining_images)

                noise = torch.randn((current_batch_size, *clean_image.shape), generator=generator,
                                    device=accelerator.device)
                timestep_expanded = torch.full((current_batch_size,), timestep, device=accelerator.device)
                noisy_image_batch = noise_scheduler.add_noise(clean_image, noise, timestep_expanded)
                images = evaluator.denoise_images(noisy_image_batch, timestep)
                # breakpoint()
                generated_images.append(images)
                remaining_images -= current_batch_size

            generated_images = np.concatenate([image.cpu() for image in generated_images], axis=0)

            if self.cfg.ema.use_ema:
                ema_model.restore(unet.parameters())

            unet.train()
            return generated_images

        # 计算成员损失 在不同时间步计算loss 观察unet遗忘能力 绘制mloss曲线 可视化删除数据的恢复难度 如果mloss低说明数据仍能恢复

        if self.cfg.metrics.membership_loss:
            membership_class = hydra.utils.instantiate(self.cfg.metrics.membership_loss.class_cfg,
                                                       dataset_all=dataset_all, dataset_deletion=dataset_deletion,
                                                       noise_scheduler=noise_scheduler, unet=unet,
                                                       device=accelerator.device)
            membership_class.sample_images()
            membership_class.sample_noises()

            plot_params = self.cfg.metrics.membership_loss.plot_params
            if plot_params is not None:
                time_freq = plot_params.time_frequency
                timesteps = list(range(0, self.cfg.scheduler.num_train_timesteps, time_freq))
                losses = membership_class.compute_membership_losses(timesteps)
                all_membership_losses = [loss[0].cpu() for loss in losses]

                # Create plot
                fig, ax = plt.subplots()
                ax.plot(timesteps, all_membership_losses)

                # Set the labels and title
                ax.set_xlabel('Timestep')
                ax.set_ylabel('All membership loss')
                ax.set_title('Loss over time')

                # Save the plot to a file
                plt.savefig(plot_params.save_path)

                exit()

            # 用于计算分类准确率、Inception Score（IS）
        if self.cfg.metrics.classifier_cfg:
            classifier = hydra.utils.instantiate(self.cfg.metrics.classifier_cfg, device=accelerator.device)

        if self.cfg.metrics.fid:
            fid_calculator = hydra.utils.instantiate(self.cfg.metrics.fid.class_cfg, device=accelerator.device)
            fid_calculator.load_celeb()
        # 测试特定噪声时间步 t 下模型的恢复能力。
        if self.cfg.metrics.denoising_injections:
            injection_timestep = torch.tensor(self.cfg.metrics.denoising_injections.timestep, device=accelerator.device)
            target_image = transform(Image.open(self.cfg.metrics.denoising_injections.img_path)).to(accelerator.device)
        # 似然计算（Likelihood Estimation）衡量生成数据的概率密度。
        if self.cfg.metrics.likelihood:
            likelihood_calculator = hydra.utils.instantiate(self.cfg.metrics.likelihood.class_cfg)

        deletion_tracker = {
            'deletion_reached': False,
            'deletion_steps': None
        }

        def log_metrics(unet, *args, **kwargs):
            wandb_batch = {}
            if global_step % self.cfg.sampling_steps == 0:
                images = evaluate_model(unet, set_generator=True)  # fix randomness
                grid = Evaluator.make_grid_from_images(images)
                wandb_batch["Sampled Images"] = wandb.Image(grid)

                images = torch.from_numpy(images).permute(0, 3, 1, 2).to(
                    accelerator.device)  # convert to (N, C, H, W) format
                if self.cfg.metrics.fraction_deletion:
                    frac_deletion = classifier.compute_class_frequency(images, self.cfg.deletion.class_label)
                    wandb_batch["metrics/deletion_class_fraction"] = frac_deletion
                    if frac_deletion == 0 and not deletion_tracker['deletion_reached']:
                        wandb.run.summary['deletion_steps'] = global_step
                        deletion_tracker['deletion_steps'] = global_step
                        deletion_tracker['deletion_reached'] = True

                if self.cfg.metrics.denoising_injections:
                    targeted_image_generations = evaluate_unlearning_timestep(unet, injection_timestep, target_image)
                    target_image_grid = Evaluator.make_grid_from_images(targeted_image_generations)
                    wandb_batch[f"Target Image Generations (t={injection_timestep})"] = wandb.Image(target_image_grid)

                if self.cfg.metrics.likelihood and (global_step % self.cfg.metrics.likelihood.step_frequency == 0):
                    print('Computing likelihood...')
                    likelihood = likelihood_calculator.evaluate_likelihood(unet, target_image.unsqueeze(0))[0][0].item()
                    wandb_batch["metrics/likelihood"] = likelihood
                    print(f'Likelihood: {likelihood}')

            if self.cfg.metrics.membership_loss and (
                    global_step % self.cfg.metrics.membership_loss.step_frequency == 0):
                timesteps = self.cfg.metrics.membership_loss.timesteps
                membership_losses = membership_class.compute_membership_losses(timesteps)
                for idx, timestep in enumerate(timesteps):
                    wandb_batch[f"membership_loss/all_membership_loss_t={timestep}"] = membership_losses[idx][0]
                    wandb_batch[f"membership_loss/deletion_membership_loss_t={timestep}"] = membership_losses[idx][1]
                    wandb_batch[f"membership_loss/membership_ratio_t={timestep}"] = membership_losses[idx][1] / \
                                                                                    membership_losses[idx][0]

            if self.cfg.metrics.inception_score and (
                    global_step % self.cfg.metrics.inception_score.step_frequency == 0 or deletion_tracker[
                'deletion_steps'] == global_step):
                images = evaluate_model(unet,
                                        num_imgs_to_generate=self.cfg.metrics.inception_score.num_imgs_to_generate,
                                        batch_size=self.cfg.metrics.inception_score.batch_size)
                images = torch.from_numpy(images).permute(0, 3, 1, 2).to(
                    accelerator.device)  # convert to (N, C, H, W) format

                inception_calculator = hydra.utils.instantiate(self.cfg.metrics.inception_score.class_cfg,
                                                               classifier=classifier)
                inception_calculator.update(images)
                is_mean, is_std = inception_calculator.compute()
                wandb_batch['metrics/is_mean'] = is_mean.item()
                wandb_batch['metrics/is_std'] = is_std.item()
                print(f'IS_mean: {is_mean}')
                print(f'IS_std: {is_std}')

            if self.cfg.metrics.fid:
                should_compute_fid = self.cfg.metrics.fid.step_frequency and (
                            global_step % self.cfg.metrics.fid.step_frequency == 0)
                if should_compute_fid or deletion_tracker['deletion_steps'] == global_step:
                    print('Computing FID...')
                    images = evaluate_model(unet, num_imgs_to_generate=self.cfg.metrics.fid.num_imgs_to_generate,
                                            batch_size=self.cfg.metrics.fid.batch_size)
                    images = torch.from_numpy(images).permute(0, 3, 1, 2).to(
                        accelerator.device)  # convert to (N, C, H, W) format

                    fid_calculator.add_fake_images(images)
                    fid = fid_calculator.compute(verbose=True)
                    wandb_batch['metrics/fid'] = fid.item()
                    print(f'FID: {fid}')

            if self.cfg.logging.logger == "wandb":
                accelerator.get_tracker("wandb").log(wandb_batch, step=global_step)
        
        # ----------------------------------------------------------
        #  5. 联邦遗忘训练主循环 (服务器驱动)
        # ----------------------------------------------------------
        logger.info("***** Running Federated Unlearning Training *****")
        total_steps = self.cfg.training_steps
        progress_bar = tqdm(total=total_steps, disable=not accelerator.is_local_main_process, desc="Federated Steps")
        global_step = 0

        while global_step < total_steps:
            unet.train()
            all_images = next(dataset_all_iterator).to(device=accelerator.device, dtype=weight_dtype)
            bsz = all_images.shape[0]
            noise = torch.randn(all_images.shape, dtype=weight_dtype, device=all_images.device)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=all_images.device).long()

            deletion_samples_dict = client.process_data_for_unlearning(timesteps, noise)
            all_samples_dict = {"og_latents": all_images, "noisy_latents": noise_scheduler.add_noise(all_images, noise, timesteps)}
            
            with accelerator.accumulate(unet):
                items = loss_fn(unet, initial_unet, timesteps, noise, {}, all_samples_dict, deletion_samples_dict, **self.cfg.deletion.loss_params)
                loss, _, _, _, _, weighted_loss_x, weighted_loss_a = items
                   
                if self.cfg.scheduler.prediction_type == "epsilon":
                    #
                    items = loss_fn(unet,initial_unet,timesteps, noise, {}, all_samples_dict, deletion_samples_dict,
                                    **self.cfg.deletion.loss_params)
                    loss, loss_x, loss_a, importance_weight_x, importance_weight_a, weighted_loss_x, weighted_loss_a = (
                        items
                    )
                    batch_stats = {}
                    if loss is not None:
                        print("loss:", loss.mean().item())
                        batch_stats["loss/mean"] = loss.mean().item()
                        batch_stats["loss/max"] = loss.mean(dim=[1, 2, 3]).max().item()
                        batch_stats["loss/min"] = loss.mean(dim=[1, 2, 3]).min().item()
                        batch_stats["loss/std"] = loss.mean(dim=[1, 2, 3]).std().item()

                    if loss_x is not None:
                        batch_stats["loss_x/mean"] = loss_x.mean().item()
                        batch_stats["loss_x/max"] = loss_x.mean(dim=[1, 2, 3]).max().item()
                        batch_stats["loss_x/min"] = loss_x.mean(dim=[1, 2, 3]).min().item()
                        batch_stats["loss_x/std"] = loss_x.mean(dim=[1, 2, 3]).std().item()

                    if loss_a is not None:
                        batch_stats["loss_a/mean"] = loss_a.mean().item()
                        batch_stats["loss_a/max"] = loss_a.mean(dim=[1, 2, 3]).max().item()
                        batch_stats["loss_a/min"] = loss_a.mean(dim=[1, 2, 3]).min().item()
                        batch_stats["loss_a/std"] = loss_a.mean(dim=[1, 2, 3]).std().item()

                    if importance_weight_x is not None:
                        batch_stats["importance_weight_x/mean"] = importance_weight_x.mean().item()
                        batch_stats["importance_weight_x/max"] = importance_weight_x.max().item()
                        batch_stats["importance_weight_x/min"] = importance_weight_x.min().item()
                        batch_stats["importance_weight_x/std"] = importance_weight_x.std().item()

                    if importance_weight_a is not None:
                        batch_stats["importance_weight_a/mean"] = importance_weight_a.mean().item()
                        batch_stats["importance_weight_a/max"] = importance_weight_a.max().item()
                        batch_stats["importance_weight_a/min"] = importance_weight_a.min().item()
                        batch_stats["importance_weight_a/std"] = importance_weight_a.std().item()

                    if "superfactor" in self.cfg.deletion.loss_params:
                        batch_stats["superfactor"] = self.cfg.deletion.loss_params.superfactor
                        decay = self.cfg.deletion.superfactor_decay
                        if decay is not None:
                            self.cfg.deletion.loss_params.superfactor *= self.cfg.deletion.superfactor_decay
                    wandb.log(batch_stats, step=global_step)
                elif self.cfg.scheduler.prediction_type == "sample":
                    print( " 用到了model_output" )
                    alpha_t = _extract_into_tensor(
                        noise_scheduler.alphas_cumprod,
                        timesteps,
                        (all_images.shape[0], 1, 1, 1),
                    )
                    snr_weights = alpha_t / (1 - alpha_t)
                    # use SNR weighting from distillation paper
                    loss = snr_weights * F.mse_loss(
                        model_output.float(), all_images.float(), reduction="none"
                    )
                    loss = loss.mean()
                else:
                    raise ValueError(
                        f"Unsupported prediction type: {self.cfg.scheduler.prediction_type}"
                    )
                
                
                if loss is not None:
                    accelerator.backward(loss)
                else:
                    weighted_loss_x = weighted_loss_x.sum() / self.cfg.train_batch_size
                    weighted_loss_a = weighted_loss_a.sum() / self.cfg.train_batch_size
                    retain_graph = self.cfg.deletion.loss_fn == "importance_sampling_with_mixture"
                    accelerator.backward(weighted_loss_x, retain_graph=retain_graph)
                    gradients_loss_x = {name: param.grad.clone() for name, param in unet.named_parameters() if param.grad is not None}
                    accelerator.backward(weighted_loss_a)
                    for name, param in unet.named_parameters():
                        if name in gradients_loss_x:
                            true_grad = param.grad.clone() - gradients_loss_x[name]
                            if name not in accum_loss_a: accum_loss_a[name] = true_grad
                            else: accum_loss_a[name] += true_grad

                if accelerator.sync_gradients:
                    # print('prepare to step!')
                    if loss is None:
                        for name, param in unet.named_parameters():
                            accum_loss_x[name] = param.grad.clone() - accum_loss_a[name]

                        # Initialize variables to hold the sum of squared norms for both accum_loss_x and accum_loss_a
                        total_l2_norm_x = 0.0
                        total_l2_norm_a = 0.0

                        # Compute L2 norm for accum_loss_x
                        for grad in accum_loss_x.values():
                            total_l2_norm_x += torch.norm(grad, p=2) ** 2  # Squared L2 norm for each gradient tensor

                        # Compute L2 norm for accum_loss_a
                        for grad in accum_loss_a.values():
                            total_l2_norm_a += torch.norm(grad, p=2) ** 2  # Squared L2 norm for each gradient tensor

                        # Take the square root of the sum to get the final L2 norms
                        total_l2_norm_x = torch.sqrt(total_l2_norm_x)
                        total_l2_norm_a = torch.sqrt(total_l2_norm_a)

                        print(f"L2 norm of accum_loss_x: {total_l2_norm_x}")
                        print(f"L2 norm of accum_loss_a: {total_l2_norm_a}")

                        if self.cfg.deletion.loss_fn == 'erasediff':
                            scaling_factor = self.cfg.deletion.eta - sum(
                                torch.sum(accum_loss_x[name] * accum_loss_a[name]) for name, _ in
                                unet.named_parameters()) / (total_l2_norm_a ** 2)
                            scaling_factor = -max(scaling_factor, 0)
                        else:
                            # gradient norm fixing
                            scaling_factor = self.cfg.deletion.scaling_norm / total_l2_norm_a  # if total_l2_norm_a > self.cfg.deletion.loss_params.clipping_norm else 1

                        wandb.log({'gradient/norm_loss_x': total_l2_norm_x, 'gradient/norm_loss_a': total_l2_norm_a,
                                   'gradient/scaling_factor': scaling_factor}, step=global_step)
                        for name, param in unet.named_parameters():
                            param.grad = accum_loss_x[name] - scaling_factor * accum_loss_a[
                                name]  # Set the gradient to the clipped difference

                        accum_loss_x = {}
                        accum_loss_a = {}

                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                if self.cfg.ema.use_ema: ema_model.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % self.cfg.sampling_steps == 0:
                        self.log_metrics(unet, accelerator, ema_model, noise_scheduler, global_step, classifier, fid_calculator, likelihood_calculator, deletion_tracker, target_image, injection_timestep, membership_class)
                    if self.cfg.checkpointing_steps and global_step % self.cfg.checkpointing_steps == 0:
                        if self.cfg.checkpoints_total_limit is not None:
                            checkpoints = sorted([d for d in os.listdir(self.cfg.output_dir) if d.startswith("checkpoint")], key=lambda x: int(x.split("-")[1]))
                            if len(checkpoints) >= self.cfg.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - self.cfg.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]
                                for removing_checkpoint in removing_checkpoints:
                                    shutil.rmtree(os.path.join(self.cfg.output_dir, removing_checkpoint))
                        save_path = os.path.join(self.cfg.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            
            logs = {"step": global_step, "lr": lr_scheduler.get_last_lr()[0]}
            if loss is not None: logs["loss"] = loss.detach().item()
            progress_bar.set_postfix(**logs)
            if self.cfg.logging.logger == "wandb": accelerator.log(logs, step=global_step)
        
            accelerator.wait_for_everyone()
            progress_bar.close()
            logger.info("Federated unlearning training finished.")
            accelerator.end_training()