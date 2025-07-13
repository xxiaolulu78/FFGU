import copy
import os
import click
import legacy
import dnnlib
import torch
import random
import pickle
import copy
import lpips
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from FL_server import Server
from torch import optim
from tqdm import tqdm
from training.triplane import TriPlaneGenerator
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils.misc import copy_params_and_buffers
from arcface import IDLoss
import numpy as np
import torch
from torch import nn
from dataset import get_client_dataloader
from PIL import Image
import swanlab


class Client(object):
    def __init__(self, args, cid):
        torch.manual_seed(0)
        self.cid = cid
        self.args = args
        self.target_id = args.target_cid
        # self.epoch = args.local_epoch

        self.model = copy.deepcopy(args.model)
        self.learning_rate = args.lr
        self.device = args.device
        self.num_clients = args.num_clients
        self.device = args.device
        self.N_client = args.N_client

        self.encoder = None

    def _get_dataloader(self, client_id):
       
        return get_client_dataloader(client_id)

    def train(self):
      
        swanlab.init(
            project="my-first-ml",
            config={'learning-rate': 1e-4},
        )
        print(f"current client is（client.py）: {self.cid}")
        device = torch.device("cuda")
        print(f"Checkpoint path （client.py）: {self.args.pretrained_ckpt}")
        with dnnlib.util.open_url(self.args.pretrained_ckpt) as f:
            init_global_generator = legacy.load_network_pkl(f)["G_ema"].to(device)

        generator = TriPlaneGenerator(*init_global_generator.init_args,
                                      **init_global_generator.init_kwargs).requires_grad_(False).to(device)  
        copy_params_and_buffers(init_global_generator, generator, require_all=True)  
        generator.neural_rendering_resolution = init_global_generator.neural_rendering_resolution  
        generator.rendering_kwargs = init_global_generator.rendering_kwargs  
        generator.load_state_dict(init_global_generator.state_dict(), strict=False)  
        for param in generator.parameters():
            param.requires_grad = True
        init_global_generator = copy.deepcopy(
            generator) 
        for name, param in init_global_generator.named_parameters():
            param.requires_grad = False

        for name, param in generator.named_parameters():
            if "backbone.synthesis" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        exp = self.args.exp
        exp_dir = f"experiments/{exp}"
        ckpt_dir = f"experiments/{exp}/checkpoints"
        image_dir = f"experiments/{exp}/training/images"
        result_dir = f"experiments/{exp}/training/results"

        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)


        intrinsics = FOV_to_intrinsics(self.args.fov_deg, device=device)
        cam_pivot = torch.tensor(generator.rendering_kwargs.get("avg_cam_pivot", [0, 0, 0]), device=device)
        cam_radius = generator.rendering_kwargs.get("avg_cam_radius", 2.7)
        conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2, cam_pivot, radius=cam_radius,
                                                               device=device)
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)
        front_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2 - 0.2, cam_pivot, radius=cam_radius, device=device)
        camera_params_front = torch.cat([front_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)
       
        optimizer = optim.Adam(generator.parameters(), lr=self.args.lr)
        w_avg = torch.load("./files/w_avg_ffhqrebalanced512-128.pt", map_location=device)  # [1, 14, 512]
        w, flag, length = self.compute_latent_vectors(self.cid)
        target_idx = self.args.target_idx
        print(f"（client.py）target_idx is: {target_idx}")
        w_gen = w[[target_idx], :, :] + w_avg  
        with torch.no_grad():
            w_id = w[[target_idx], :, :]
            w_target = w_id + w_avg


        lpips_fn = lpips.LPIPS(net="vgg").to(device) 
        id_fn = IDLoss().to(device)
        pbar = tqdm(range(self.args.iter))
        for i in pbar:
            angle_y = np.random.uniform(-self.args.angle_y_abs, self.args.angle_y_abs)
            cam2world_pose = LookAtPoseSampler.sample(np.pi / 2 + angle_y, np.pi / 2 + self.args.angle_p, cam_pivot,
                                                      radius=cam_radius, device=device)
            camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)

            loss = torch.tensor(0.0, device=device)
            loss_dict = {}

            if self.args.local:
                
                loss_local = torch.tensor(0.0, device=device)
                feat_u = generator.get_planes(w_gen)  
                feat_target = init_global_generator.get_planes(w_target)  
                loss_local_mse = F.mse_loss(feat_u, feat_target)
                loss_local = loss_local + self.args.loss_local_mse_lambda * loss_local_mse
               
                img_u = generator.synthesis(w_gen, camera_params)["image"]
                img_target = init_global_generator.synthesis(w_target, camera_params)["image"]
                loss_local_lpips = lpips_fn(img_u, img_target).mean()
                loss_local = loss_local + self.args.loss_local_lpips_lambda * loss_local_lpips
                
                loss_local_id = id_fn(img_u, img_target)
                loss_local = loss_local + self.args.loss_local_id_lambda * loss_local_id
                loss = loss + loss_local
                loss_dict["loss_local"] = loss_local.item()

            if self.args.glob:
                loss_global = torch.tensor(0.0, device=device)
                for _ in range(self.args.loss_global_batch):
                    z_rg = torch.randn(1, 512, device=device)
                    w_rg = generator.mapping(z_rg, conditioning_params, truncation_psi=self.args.truncation_psi,
                                             truncation_cutoff=self.args.truncation_cutoff)

                    img_u = generator.synthesis(w_rg, camera_params)["image"]
                    img_target = init_global_generator.synthesis(w_rg, camera_params)["image"]
                    loss_global_lpips = lpips_fn(img_u, img_target).mean()
                    loss_global = loss_global + loss_global_lpips
                loss = loss + self.args.loss_global_lambda * loss_global
                loss_dict["loss_global"] = loss_global.item()
            optimizer.zero_grad()
            loss.backward()  
            optimizer.step()
            swanlab.log({"train/loss": loss.item()})
            pbar.set_postfix(loss=loss.item(), **loss_dict)
            if i % 20 == 0: 
                with torch.no_grad():
                    generator.eval()
                    img_u_save = generator.synthesis(w_gen, camera_params_front)["image"]
                    img_u_save = self.tensor_to_image(img_u_save)
                    img_u_save.save(os.path.join(image_dir, f"img_u_{str(i).zfill(5)}.png"))

                    img_target_save = init_global_generator.synthesis(w_target, camera_params_front)["image"]
                    img_target_save = self.tensor_to_image(img_target_save)
                    img_target_save.save(os.path.join(image_dir, f"img_target_{str(i).zfill(5)}.png"))
                    generator.train()
                del img_u_save, img_target_save
            
        generator_copy = copy.deepcopy(generator).eval().requires_grad_(False).cpu()   
        snapshot_data = {
            "G_ema": generator_copy.state_dict(),  
        }
        
        del generator, generator_copy
        torch.cuda.empty_cache()

        return snapshot_data  
      

    def invert_image_to_latent(self,image_path, encoder, transform, device):
        img = self.image_to_tensor(Image.open(image_path).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            w, _ = encoder(img)  
        return w

    def image_to_tensor(self,i, size=256):
        i = i.resize((size, size))
        i = np.array(i)
        i = i.transpose(2, 0, 1)
        i = torch.from_numpy(i).to(torch.float32).to("cuda") / 127.5 - 1
        return i

    def convert_tensor(self,t):
        t = (t.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return Image.fromarray(t[0].cpu().numpy(), "RGB")

    def load_encoder(self,encoder_ckpt, device):
        from goae import GOAEncoder 
        from goae.swin_config import get_config
        swin_config = get_config()  
        stage_list = [10000, 20000,
                      30000]  
        encoder = GOAEncoder(swin_config=swin_config, mlp_layer=2, stage_list=stage_list).to(
            device)  
        encoder.load_state_dict(
            torch.load(encoder_ckpt, map_location=device))
        encoder.eval()
        return encoder

    def tensor_to_image(self, t):
        t = (t.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return Image.fromarray(t[0].cpu().numpy(), "RGB")

    def image_to_tensor(self, i, size=256):
        i = i.resize((size, size))
        i = np.array(i)
        i = i.transpose(2, 0, 1)
        i = torch.from_numpy(i).to(torch.float32).to("cuda") / 127.5 - 1
        return i

    def upload_encoder(self, encoder):
        self.encoder = copy.deepcopy(encoder).to(self.device)

    def compute_latent_vectors(self, target_cid):

        dataloader, length = self._get_dataloader(target_cid)
        self.encoder.to(self.device)  
        self.encoder.eval() 
        print("dataloader's length ", length)
        flag = False
        all_w = []
        with torch.no_grad():
           
            for batch_idx, (batch_imgs, _) in enumerate(dataloader):
                batch_imgs = batch_imgs.to(self.device)
                batch_w, _ = self.encoder(batch_imgs)
                all_w.append(batch_w)

            all_w = torch.cat(all_w, dim=0)
            return all_w, flag, length  