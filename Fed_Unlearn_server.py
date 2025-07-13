import math
from thop import profile, clever_format
import torch.functional as F
import time
from FL_server import Server
from FL_client import Client
import os
import legacy
import dnnlib
import torch
import random
import pickle
import copy
import lpips
import datetime
import torch.nn.functional as F
import numpy as np

from torch import optim
from tqdm import tqdm
from training.triplane import TriPlaneGenerator
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils.misc import copy_params_and_buffers
from PIL import Image
from arcface import IDLoss



class FFGU(Server):
    def __init__(self, args):
        super(FFGU, self).__init__(args)
        self.tensor_to_image = self.tensor_to_image
        self.image_to_tensor = self.image_to_tensor
        self.clients = [] 
        total_clients = args.num_clients 
        for cid in range(total_clients):
            client = Client(args, cid)
            self.clients.append(client)



    def finetune_models(self):
        device = torch.device("cuda")
        exp = self.args.exp
        from goae import GOAEncoder
        from goae.swin_config import get_config
        swin_config = get_config()  
        stage_list = [10000, 20000,
                      30000]  
        encoder_ckpt = "./files/encoder_FFHQ.pt"  
        encoder = GOAEncoder(swin_config=swin_config, mlp_layer=2, stage_list=stage_list).to(
            device)  
        encoder.load_state_dict(
            torch.load(encoder_ckpt, map_location=device))
        num_clients = len(self.clients)
        client_snapshots = []


        for cid in range(1,num_clients):
            client = Client(self.args, cid)
            client.upload_encoder(encoder)  
            client_snapshot = client.train()
            client_snapshots.append(client_snapshot["G_ema"])
            del client, client_snapshot
            torch.cuda.empty_cache()


        averaged_state_dict = self.average_params(client_snapshots)
        pretrained_ckpt = "./files/ffhqrebalanced512-128.pkl"
        with dnnlib.util.open_url(pretrained_ckpt) as f:
             global_generator = legacy.load_network_pkl(f)["G_ema"].to(device)
        for name, param in global_generator.named_parameters():
            if name in averaged_state_dict:
                param.data.copy_(averaged_state_dict[name].data)
        ckpt_dir = f"experiments/{exp}/checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)
        snapshot_data = dict() 
        snapshot_data["G_ema"] = copy.deepcopy(global_generator).eval().requires_grad_(False).cpu()
        with open(os.path.join(ckpt_dir, "finetune_final.pkl"), "wb") as f:
            pickle.dump(snapshot_data, f)



    def baseline_neggrade(self):
        device = torch.device("cuda")
        from goae import GOAEncoder
        from goae.swin_config import get_config
        swin_config = get_config() 
        stage_list = [10000, 20000,
                      30000] 
        encoder_ckpt = "./files/encoder_FFHQ.pt"  
        encoder = GOAEncoder(swin_config=swin_config, mlp_layer=2, stage_list=stage_list).to(
            device) 
        encoder.load_state_dict(
            torch.load(encoder_ckpt, map_location=device))
        num_clients = len(self.clients)
        client_snapshots = []
        client = Client(self.args, 0)
        client.upload_encoder(encoder)  
        client_snapshot = client.neggrad()
        client_snapshots.append(client_snapshot["G_ema"])
        pretrained_ckpt = "./files/ffhqrebalanced512-128.pkl"

        with dnnlib.util.open_url(pretrained_ckpt) as f:
            global_generator = legacy.load_network_pkl(f)["G_ema"].to(device)
        for name, param in global_generator.named_parameters():
            if name in client_snapshot:
                param.data.copy_(client_snapshot[name].data)

        exp = self.args.exp
        ckpt_dir = f"experiments/{exp}/checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)
        snapshot_data = dict() 
        snapshot_data["G_ema"] = copy.deepcopy(global_generator).eval().requires_grad_(False).cpu()
        with open(os.path.join(ckpt_dir, "fu_negrad.pkl"), "wb") as f:
            pickle.dump(snapshot_data, f)

    def neggrade(self):
        device = torch.device("cuda")
        with dnnlib.util.open_url(self.args.pretrained_ckpt) as f:
            init_global_generator = legacy.load_network_pkl(f)["G_ema"].to(device)
        generator = TriPlaneGenerator(*init_global_generator.init_args,
                                      **init_global_generator.init_kwargs).requires_grad_(
            False).to(
            device) 
        copy_params_and_buffers(init_global_generator, generator, require_all=True)  
        generator.neural_rendering_resolution = init_global_generator.neural_rendering_resolution  
        generator.rendering_kwargs = init_global_generator.rendering_kwargs 
        generator.load_state_dict(init_global_generator.state_dict(), strict=False)
        generator.train() 

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
        conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2, cam_pivot,
                                                               radius=cam_radius,
                                                               device=device)
        conditioning_params = torch.cat(
            [conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)
        front_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2 - 0.2, cam_pivot, radius=cam_radius,
                                              device=device)
        camera_params_front = torch.cat([front_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)
      
        optimizer = optim.Adam(generator.parameters(), lr=self.args.lr)
        w_avg = torch.load("./files/w_avg_ffhqrebalanced512-128.pt", map_location=device)  # [1, 14, 512]
        inversion = self.args.inversion
        with torch.no_grad():  
            if inversion is not None:
                assert inversion in ["goae"]
                if inversion == "goae":
                    from goae import GOAEncoder
                    from goae.swin_config import get_config
                    swin_config = get_config() 
                    stage_list = [10000, 20000,
                                  30000] 
                    encoder_ckpt = "./files/encoder_FFHQ.pt" 
                    encoder = GOAEncoder(swin_config=swin_config, mlp_layer=2, stage_list=stage_list).to(
                        device) 
                    encoder.load_state_dict(
                        torch.load(encoder_ckpt, map_location=device))  
                    print(f"target_cid is: {self.args.target_cid}")
                    target_client = self.clients[self.args.target_cid]
                    target_client.upload_encoder(encoder)
                    w, flag, length = target_client.compute_latent_vectors(self.args.target_cid)
                    all_w_list = []
                    selected_image_paths = self.select_random_image_from_each_subfolder()
                    for image_path in selected_image_paths:
                        img = self.image_to_tensor(Image.open(image_path).convert("RGB")).unsqueeze(0)
                        client_w, _ = encoder(img)
                        all_w_list.append(client_w)
                    all_w_tensor = torch.cat(all_w_list, dim=0)
                    average_w = torch.mean(all_w_tensor, dim=0)
                    average_w = self.add_gaussian_noise(average_w)

                    if not flag:
                        N = w.shape[0]
                        target_idx = 3
                        print(f"target_idx is: {target_idx}")
                        w_origin = w + w_avg  
                        w_u = w[[target_idx], :, :] + w_avg  
                    else: 
                        w_u = w + w_avg
                    w_u = self.add_gaussian_noise(w_u)
                else:
                    raise NotImplementedError

            generator.eval()
            if flag is False:  
                for i in range(length):
                    for view, angle_y in enumerate(
                            np.linspace(-self.args.angle_y_abs, self.args.angle_y_abs,
                                        self.args.sample_views)):
                        cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y,
                                                                        np.pi / 2 + self.args.angle_p,
                                                                        cam_pivot, radius=cam_radius,
                                                                        device=device)
                        camera_params_view = torch.cat(
                            [cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)],
                            dim=1)

                        img_origin = generator.synthesis(w_origin[[i]], camera_params_view)["image"]
                        img_origin = self.tensor_to_image(img_origin)
                        img_origin.save(os.path.join(result_dir, f"unlearn_before_{i}_{view}.png"))
                del img_origin
            else:  
                for view, angle_y in enumerate(
                        np.linspace(-self.args.angle_y_abs, self.args.angle_y_abs, self.args.sample_views)):
                    cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y,
                                                                    np.pi / 2 + self.args.angle_p,
                                                                    cam_pivot,
                                                                    radius=cam_radius, device=device)
                    camera_params_view = torch.cat(
                        [cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)],
                        dim=1)
                    img_u = generator.synthesis(w_u, camera_params_view)["image"]
                    img_u = self.tensor_to_image(img_u)
                    img_u.save(os.path.join(result_dir, f"unlearn_before_0_{view}.png"))
                del img_u
            generator.train()

        if self.args.target == "random":
            z_rg = torch.randn(1, 512, device=device)
            w_target = generator.mapping(z_rg, conditioning_params, truncation_psi=self.args.truncation_psi,
                                         truncation_cutoff=self.args.truncation_cutoff)
            print("w_target is random")
        elif self.args.target == "neg":
            w_target = w_u
        elif self.args.target == "ffgu":
            with torch.no_grad():
                if self.args.inversion is not None:
                    w_id = w[[target_idx], :, :]
                else:
                    w_id = w_u - w_avg
                w_target = w_avg - w_id / w_id.norm(p=2) * self.args.target_d
        elif self.args.target == "custom_dp":
            w_custom = average_w + w_avg
            print(f"w_custom is :{ w_custom }")
            if self.args.inversion is not None:
                w_id = w[[target_idx], :, :]
            else:
                w_id = w_u - w_avg
            w_target = w_custom - w_id / w_id.norm(p=2) * self.args.target_d
            print("w_target is dp")
        elif self.args.target == "custom": 
            with torch.no_grad():
                w_custom = average_w + w_avg
                w_id = w[[target_idx], :, :]
                w_target = w_custom - w_id / w_id.norm(p=2) * self.args.target_d 

        lpips_fn = lpips.LPIPS(net="vgg").to(device) 
        id_fn = IDLoss().to(device)
        pbar = tqdm(range(self.args.iter))
        for i in pbar:
            angle_y = np.random.uniform(-self.args.angle_y_abs, self.args.angle_y_abs)
            cam2world_pose = LookAtPoseSampler.sample(np.pi / 2 + angle_y, np.pi / 2 + self.args.angle_p,
                                                      cam_pivot,
                                                      radius=cam_radius, device=device)
            camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)

            loss = torch.tensor(0.0, device=device)
            loss_dict = {}
            if self.args.local:
                loss_local = torch.tensor(0.0, device=device)
                feat_u = generator.get_planes(w_u) 
                feat_target = init_global_generator.get_planes(w_target)  
                loss_local_mse = F.mse_loss(feat_u, feat_target)
                loss_local = loss_local + self.args.loss_local_mse_lambda * loss_local_mse
                loss_dict["local_mse"] = loss_local_mse.item()
                img_u = generator.synthesis(w_u, camera_params)["image"]
                img_target = init_global_generator.synthesis(w_target, camera_params)["image"]
                loss_local_lpips = lpips_fn(img_u, img_target).mean()
                loss_local = loss_local + self.args.loss_local_lpips_lambda * loss_local_lpips
               
                loss_local_id = id_fn(img_u, img_target)
                loss_local = loss_local + self.args.loss_local_id_lambda * loss_local_id
                loss = loss + loss_local

                loss_local_l1 = torch.nn.L1Loss()(feat_u, feat_target)
                loss_local = loss_local + self.args.loss_local_l1_lambda * loss_local_l1  
                loss_dict["loss_local"] = loss_local.item()

            if self.args.adj:
                loss_adj = torch.tensor(0.0, device=device)
                for _ in range(self.args.loss_adj_batch):
                    z_ra = torch.randn(1, 512, device=device) 
                    w_ra = generator.mapping(z_ra, conditioning_params, truncation_psi=self.args.truncation_psi,
                                             truncation_cutoff=self.args.truncation_cutoff) 
                    if self.args.loss_adj_alpha_range_max is not None:
                        loss_adj_alpha = torch.from_numpy(
                            np.random.uniform(self.args.loss_adj_alpha_range_min,
                                              self.args.loss_adj_alpha_range_max,
                                              size=1)).unsqueeze(
                            1).unsqueeze(1).to(device)
                  
                    deltas = loss_adj_alpha * (w_ra - w_u) / (w_ra - w_u).norm(p=2)
                    w_u_adj = w_u + deltas 
                    w_target_adj = w_target + deltas  
    
                    feat_u = generator.get_planes(w_u_adj)
                    feat_target = init_global_generator.get_planes(w_target_adj)
                    loss_adj_mse = F.mse_loss(feat_u, feat_target)
                    loss_adj = loss_adj + self.args.loss_adj_mse_lambda * loss_adj_mse
                  
                    img_u = generator.synthesis(w_u_adj, camera_params)["image"]
                    img_target = init_global_generator.synthesis(w_target_adj, camera_params)["image"]
                   
                    loss_adj_lpips = lpips_fn(img_u, img_target).mean()
                    loss_adj = loss_adj + self.args.loss_adj_lpips_lambda * loss_adj_lpips
                    
                    loss_adj_id = id_fn(img_u, img_target)
                    loss_adj = loss_adj + self.args.loss_adj_id_lambda * loss_adj_id
               
                loss = loss + self.args.loss_adj_lambda * loss_adj
                loss_dict["loss_adj"] = loss_adj.item()

            loss_fn = -loss
          
            optimizer.zero_grad()
            loss_fn.backward()  
            optimizer.step()
            pbar.set_postfix(loss=loss.item(), **loss_dict)

            if i % 5 == 0: 
                with torch.no_grad():
                    generator.eval()
                    img_u_save = generator.synthesis(w_u, camera_params_front)["image"]
                    img_u_save = self.tensor_to_image(img_u_save)
                    img_u_save.save(os.path.join(image_dir, f"img_u_{str(i).zfill(5)}.png"))

                    img_target_save = init_global_generator.synthesis(w_target, camera_params_front)["image"]
                    img_target_save = self.tensor_to_image(img_target_save)
                    img_target_save.save(os.path.join(image_dir, f"img_target_{str(i).zfill(5)}.png"))
                    generator.train()
                del img_u_save, img_target_save
           
        with torch.no_grad():
            generator.eval()
            img_u_save = generator.synthesis(w_u, camera_params)["image"]
            img_target_save = init_global_generator.synthesis(w_target, camera_params)["image"]
            img_u_save = self.tensor_to_image(img_u_save)
            img_target_save = self.tensor_to_image(img_target_save)
            img_u_save.save(os.path.join(image_dir, f"img_u_{str(i).zfill(5)}.png"))
            img_target_save.save(os.path.join(image_dir, f"img_target_{str(i).zfill(5)}.png"))
            generator.train()
       
        with torch.no_grad():
            generator.eval()
            if flag is False:  
                for i in range(length):
                    for view, angle_y in enumerate(
                            np.linspace(-self.args.angle_y_abs, self.args.angle_y_abs,
                                        self.args.sample_views)):
                        cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y,
                                                                        np.pi / 2 + self.args.angle_p,
                                                                        cam_pivot, radius=cam_radius,
                                                                        device=device)
                        camera_params_view = torch.cat(
                            [cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)],
                            dim=1)
                        img_origin = generator.synthesis(w_origin[[i]], camera_params_view)["image"]
                        img_origin = self.tensor_to_image(img_origin)
                        img_origin.save(os.path.join(result_dir, f"unlearn_after_{i}_{view}.png"))
            else:  
                for view, angle_y in enumerate(
                        np.linspace(-self.args.angle_y_abs, self.args.angle_y_abs, self.args.sample_views)):
                    cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y,
                                                                    np.pi / 2 + self.args.angle_p,
                                                                    cam_pivot,
                                                                    radius=cam_radius, device=device)
                    camera_params_view = torch.cat(
                        [cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)],
                        dim=1)
                    img_u = generator.synthesis(w_u, camera_params_view)["image"]
                    img_u = self.tensor_to_image(img_u)
                    img_u.save(os.path.join(result_dir, f"unlearn_after_0_{view}.png"))
            generator.train()

        snapshot_data = dict()
        snapshot_data["G_ema"] = copy.deepcopy(generator).eval().requires_grad_(False).cpu()
        current_time = datetime.datetime.now()
       
        print("current time:", current_time)
        with open(os.path.join(ckpt_dir, f"last.pkl"), "wb") as f:
            pickle.dump(snapshot_data, f)

    def average_params(self, client_snapshots):
        averaged_params = {}
        model_keys = set()  
        for snapshot in client_snapshots:
            model_keys.update(snapshot.keys())  
        for key in model_keys:
            stacked_params = torch.stack([snapshot[key] for snapshot in client_snapshots if key in snapshot])
            averaged_params[key] = torch.mean(stacked_params, dim=0)

        return averaged_params

    def tensor_to_image(self, t):
        t = (t.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return Image.fromarray(t[0].cpu().numpy(), "RGB")

    def image_to_tensor(self, i, size=256):
        i = i.resize((size, size))
        i = np.array(i)
        i = i.transpose(2, 0, 1)
        i = torch.from_numpy(i).to(torch.float32).to("cuda") / 127.5 - 1
        return i

    def fuffgu(self):
        device = torch.device("cuda")
        with dnnlib.util.open_url(self.args.pretrained_ckpt) as f:
            init_global_generator = legacy.load_network_pkl(f)["G_ema"].to(device)
        generator = TriPlaneGenerator(*init_global_generator.init_args,
                                      **init_global_generator.init_kwargs).requires_grad_(
            False).to(
            device) 
        copy_params_and_buffers(init_global_generator, generator, require_all=True) 
        generator.neural_rendering_resolution = init_global_generator.neural_rendering_resolution 
        generator.rendering_kwargs = init_global_generator.rendering_kwargs  
        generator.load_state_dict(init_global_generator.state_dict(), strict=False)  
        generator.train() 

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
        inversion = self.args.inversion
        with torch.no_grad():  
            if inversion is not None:
                assert inversion in ["goae"]
                if inversion == "goae":
                    from goae import GOAEncoder
                    from goae.swin_config import get_config
                    swin_config = get_config() 
                    stage_list = [10000, 20000,
                                  30000] 
                    encoder_ckpt = "./files/encoder_FFHQ.pt"  
                    encoder = GOAEncoder(swin_config=swin_config, mlp_layer=2, stage_list=stage_list).to(
                        device)  
                    encoder.load_state_dict(
                        torch.load(encoder_ckpt, map_location=device))  

                    print(f"target_cid is: {self.args.target_cid}")
                    target_client = self.clients[self.args.target_cid]
                    target_client.upload_encoder(encoder)
                    w, flag, length = target_client.compute_latent_vectors(self.args.target_cid)

                    if not flag:
                        N = w.shape[0]
                        target_idx = 3
                        print(f"target_idx is: {target_idx}")
                        w_origin = w + w_avg 
                        w_u = w[[target_idx], :, :] + w_avg  
                    else:  
                        w_u = w + w_avg
                else:
                    raise NotImplementedError
            generator.eval()

            if flag is False:  
                for i in range(length):
                    for view, angle_y in enumerate(np.linspace(-self.args.angle_y_abs, self.args.angle_y_abs, self.args.sample_views)):
                        cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y, np.pi / 2 + self.args.angle_p,
                                                                        cam_pivot, radius=cam_radius, device=device)
                        camera_params_view = torch.cat(
                            [cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)],
                            dim=1)

                        img_origin = generator.synthesis(w_origin[[i]], camera_params_view)["image"]
                        img_origin = self.tensor_to_image(img_origin)
                        img_origin.save(os.path.join(result_dir, f"unlearn_before_{i}_{view}.png"))
                del img_origin
            else:  
                for view, angle_y in enumerate(np.linspace(-self.args.angle_y_abs, self.args.angle_y_abs, self.args.sample_views)):
                    cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y, np.pi / 2 + self.args.angle_p,
                                                                    cam_pivot,
                                                                    radius=cam_radius, device=device)
                    camera_params_view = torch.cat([cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)],
                                                    dim=1)
                    img_u = generator.synthesis(w_u, camera_params_view)["image"]
                    img_u = self.tensor_to_image(img_u)
                    img_u.save(os.path.join(result_dir, f"unlearn_before_0_{view}.png"))
                del img_u
            generator.train()

        if self.args.target == "random":
            z_rg= torch.randn(1, 512, device=device)
            w_target = generator.mapping(z_rg, conditioning_params, truncation_psi=self.args.truncation_psi,
                                     truncation_cutoff=self.args.truncation_cutoff)
            print("w_target is random")
        elif self.args.target == "ffgu":
            with torch.no_grad():
                if self.args.inversion is not None:
                    w_id = w[[target_idx], :, :]
                else:
                    w_id = w_u - w_avg
                w_target = w_avg - w_id / w_id.norm(p=2) * self.args.target_d
            
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
                feat_u = generator.get_planes(w_u) 
                feat_target = init_global_generator.get_planes(w_target)  
                loss_local_mse = F.mse_loss(feat_u, feat_target)
                loss_local = loss_local + self.args.loss_local_mse_lambda * loss_local_mse
             
                img_target = init_global_generator.synthesis(w_target, camera_params)["image"]
                loss_local_lpips = lpips_fn(img_u, img_target).mean()
                loss_local = loss_local + self.args.loss_local_lpips_lambda * loss_local_lpips

                loss_local_id = id_fn(img_u, img_target)
                loss_local = loss_local + self.args.loss_local_id_lambda * loss_local_id
                loss = loss + loss_local
                loss_dict["loss_local"] = loss_local.item()


            if self.args.adj:
                loss_adj = torch.tensor(0.0, device=device)
                for _ in range(self.args.loss_adj_batch):
                    z_ra = torch.randn(1, 512, device=device) 
                    w_ra = generator.mapping(z_ra, conditioning_params, truncation_psi=self.args.truncation_psi,
                                             truncation_cutoff=self.args.truncation_cutoff)  
                    if self.args.loss_adj_alpha_range_max is not None:
                        loss_adj_alpha = torch.from_numpy(
                            np.random.uniform(self.args.loss_adj_alpha_range_min,
                                              self.args.loss_adj_alpha_range_max,
                                              size=1)).unsqueeze(
                            1).unsqueeze(1).to(device)
           
                    deltas = loss_adj_alpha * (w_ra - w_u) / (w_ra - w_u).norm(p=2)
                    w_u_adj = w_u + deltas  
                    w_target_adj = w_target + deltas  
                    feat_u = generator.get_planes(w_u_adj)
                    feat_target = init_global_generator.get_planes(w_target_adj)
                    loss_adj_mse = F.mse_loss(feat_u, feat_target)
                    loss_adj = loss_adj + self.args.loss_adj_mse_lambda * loss_adj_mse
                    img_u = generator.synthesis(w_u_adj, camera_params)["image"]
                    img_target = init_global_generator.synthesis(w_target_adj, camera_params)["image"]

                    loss_adj_lpips = lpips_fn(img_u, img_target).mean()
                    loss_adj = loss_adj + self.args.loss_adj_lpips_lambda * loss_adj_lpips

                    loss_adj_id = id_fn(img_u, img_target)
                    loss_adj = loss_adj + self.args.loss_adj_id_lambda * loss_adj_id
                loss = loss + self.args.loss_adj_lambda * loss_adj
                loss_dict["loss_adj"] = loss_adj.item()

 
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
            pbar.set_postfix(loss=loss.item(), **loss_dict)

            if i % 100 == 0:  
                with torch.no_grad():
                    generator.eval()
                    img_u_save = generator.synthesis(w_u, camera_params_front)["image"]
                    img_u_save = self.tensor_to_image(img_u_save)
                    img_u_save.save(os.path.join(image_dir, f"img_u_{str(i).zfill(5)}.png"))

                    img_target_save = init_global_generator.synthesis(w_target, camera_params_front)["image"]
                    img_target_save = self.tensor_to_image(img_target_save)
                    img_target_save.save(os.path.join(image_dir, f"img_target_{str(i).zfill(5)}.png"))
                    generator.train()
                del img_u_save, img_target_save

        with torch.no_grad():
            generator.eval()
            img_u_save = generator.synthesis(w_u, camera_params)["image"]
            img_target_save = init_global_generator.synthesis(w_target, camera_params)["image"]
            img_u_save = self.tensor_to_image(img_u_save)
            img_target_save = self.tensor_to_image(img_target_save)
            img_u_save.save(os.path.join(image_dir, f"img_u_{str(i).zfill(5)}.png"))
            img_target_save.save(os.path.join(image_dir, f"img_target_{str(i).zfill(5)}.png"))
            generator.train()

        with torch.no_grad():
            generator.eval()
            if self.args.inversion is None:  
                for view, angle_y in enumerate(
                        np.linspace(-self.args.angle_y_abs, self.args.angle_y_abs, self.args.sample_views)):
                    cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y,
                                                                   np.pi / 2 + self.args.angle_p,
                                                                   cam_pivot,
                                                                   radius=cam_radius, device=device)
                    camera_params_view = torch.cat([cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)],
                                                   dim=1)
                    img_u = generator.synthesis(w_u, camera_params_view)["image"]
                    img_u = self.tensor_to_image(img_u)
                    img_u.save(os.path.join(result_dir, f"unlearn_after_0_{view}.png"))
            else:
                if flag is False:  
                    for i in range(length):
                        for view, angle_y in enumerate(
                                np.linspace(-self.args.angle_y_abs, self.args.angle_y_abs, self.args.sample_views)):
                            cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y, np.pi / 2 + self.args.angle_p,
                                                                           cam_pivot, radius=cam_radius,
                                                                           device=device)
                            camera_params_view = torch.cat(
                                [cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)],
                                dim=1)
                            img_origin = generator.synthesis(w_origin[[i]], camera_params_view)["image"]
                            img_origin = self.tensor_to_image(img_origin)
                            img_origin.save(os.path.join(result_dir, f"unlearn_after_{i}_{view}.png"))
                else:  
                    for view, angle_y in enumerate(
                            np.linspace(-self.args.angle_y_abs, self.args.angle_y_abs, self.args.sample_views)):
                        cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y,
                                                                       np.pi / 2 + self.args.angle_p,
                                                                       cam_pivot,
                                                                       radius=cam_radius, device=device)
                        camera_params_view = torch.cat(
                            [cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)],
                            dim=1)
                        img_u = generator.synthesis(w_u, camera_params_view)["image"]
                        img_u = self.tensor_to_image(img_u)
                        img_u.save(os.path.join(result_dir, f"unlearn_after_0_{view}.png"))
            generator.train()

        snapshot_data = dict() 
        snapshot_data["G_ema"] = copy.deepcopy(generator).eval().requires_grad_(False).cpu()
        with open(os.path.join(ckpt_dir, f"last.pkl"), "wb") as f:
            pickle.dump(snapshot_data, f)

    def select_random_image_from_each_subfolder(self,folder='./Deep3DFaceRecon/checkpoints/model_name/casia'):
        subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
        selected_images = []
        for subfolder in subfolders:
            image_files = [f.path for f in os.scandir(subfolder) if
                           f.is_file() and f.name.endswith(('.jpg', '.jpeg', '.png'))]
            if image_files:  
                random_image_path = random.choice(image_files)
                selected_images.append(random_image_path)

        return selected_images

    def ffgu(self):
        device = torch.device("cuda")
        with dnnlib.util.open_url(self.args.pretrained_ckpt) as f:
            init_global_generator = legacy.load_network_pkl(f)["G_ema"].to(device)
        generator = TriPlaneGenerator(*init_global_generator.init_args,
                                      **init_global_generator.init_kwargs).requires_grad_(
            False).to(
            device) 
        copy_params_and_buffers(init_global_generator, generator, require_all=True) 
        generator.neural_rendering_resolution = init_global_generator.neural_rendering_resolution  
        generator.rendering_kwargs = init_global_generator.rendering_kwargs  
        generator.load_state_dict(init_global_generator.state_dict(), strict=False)  
        generator.train() 

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
        conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2, cam_pivot,
                                                               radius=cam_radius,
                                                               device=device)
        conditioning_params = torch.cat(
            [conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)
        front_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2 - 0.2, cam_pivot, radius=cam_radius,
                                              device=device)
        camera_params_front = torch.cat([front_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)
        optimizer = optim.Adam(generator.parameters(), lr=self.args.lr)
        w_avg = torch.load("./files/w_avg_ffhqrebalanced512-128.pt", map_location=device)  # [1, 14, 512]
        inversion = self.args.inversion
        with torch.no_grad():  
            if inversion is not None:
                assert inversion in ["goae"]
                if inversion == "goae":
                    from goae import GOAEncoder
                    from goae.swin_config import get_config
                    swin_config = get_config()  
                    stage_list = [10000, 20000,
                                  30000]  
                    encoder_ckpt = "./files/encoder_FFHQ.pt"  
                    encoder = GOAEncoder(swin_config=swin_config, mlp_layer=2, stage_list=stage_list).to(
                        device)  
                    encoder.load_state_dict(
                        torch.load(encoder_ckpt, map_location=device)) 
                    print(f"target_cid is: {self.args.target_cid}")
                    target_client = self.clients[self.args.target_cid]
                    target_client.upload_encoder(encoder)
                    w, flag, length = target_client.compute_latent_vectors(self.args.target_cid)
                    all_w_list = []
                    selected_image_paths = self.select_random_image_from_each_subfolder()
                    for image_path in selected_image_paths:
                        img = self.image_to_tensor(Image.open(image_path).convert("RGB")).unsqueeze(0)
                        client_w, _ = encoder(img)
                        # print(img.shape)
                        # flops, params = profile(encoder, inputs=(img,))
                        # flops, params = clever_format([flops, params], "%.3f")
                        # print(f"encoder FLOPs: {flops}")
                        # print(f"encoder Params: {params}")
                        all_w_list.append(client_w)
                    all_w_tensor = torch.cat(all_w_list, dim=0)
                    average_w = torch.mean(all_w_tensor, dim=0)
                    average_w = self.add_gaussian_noise(average_w)
                    if not flag: 
                        N = w.shape[0]
                        target_idx = 3
                        print(f"target_idx is: {target_idx}")
                        w_origin = w + w_avg  
                        w_u = w[[target_idx], :, :] + w_avg 
                    else:  
                        w_u = w + w_avg
                    w_u = self.add_gaussian_noise(w_u)
                else:
                    raise NotImplementedError

            generator.eval()
          
            if flag is False:  
                for i in range(length):
                    for view, angle_y in enumerate(
                            np.linspace(-self.args.angle_y_abs, self.args.angle_y_abs,
                                        self.args.sample_views)):
                        cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y,
                                                                        np.pi / 2 + self.args.angle_p,
                                                                        cam_pivot, radius=cam_radius,
                                                                        device=device)
                        camera_params_view = torch.cat(
                            [cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)],
                            dim=1)

                        img_origin = generator.synthesis(w_origin[[i]], camera_params_view)["image"]
                        img_origin = self.tensor_to_image(img_origin)
                        img_origin.save(os.path.join(result_dir, f"unlearn_before_{i}_{view}.png"))
                del img_origin
            else:  
                for view, angle_y in enumerate(
                        np.linspace(-self.args.angle_y_abs, self.args.angle_y_abs, self.args.sample_views)):
                    cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y,
                                                                    np.pi / 2 + self.args.angle_p,
                                                                    cam_pivot,
                                                                    radius=cam_radius, device=device)
                    camera_params_view = torch.cat(
                        [cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)],
                        dim=1)
                    img_u = generator.synthesis(w_u, camera_params_view)["image"]
                    img_u = self.tensor_to_image(img_u)
                    img_u.save(os.path.join(result_dir, f"unlearn_before_0_{view}.png"))
                del img_u
        if self.args.target == "random":
            z_rg = torch.randn(1, 512, device=device)
            w_target = generator.mapping(z_rg, conditioning_params, truncation_psi=self.args.truncation_psi,
                                         truncation_cutoff=self.args.truncation_cutoff)
        elif self.args.target == "ffgu":
            with torch.no_grad():
                if self.args.inversion is not None:
                    w_id = w[[target_idx], :, :]
                else:
                    w_id = w_u - w_avg
                w_target = w_avg - w_id / w_id.norm(p=2) * self.args.target_d
        elif self.args.target == "custom_dp":
            w_custom = average_w + w_avg
            print(f"w_custom is :{ w_custom }")
            if self.args.inversion is not None:
                w_id = w[[target_idx], :, :]
            else:
                w_id = w_u - w_avg
            w_target = w_custom - w_id / w_id.norm(p=2) * self.args.target_d

        elif self.args.target == "custom": 
            with torch.no_grad():

                w_custom = average_w + w_avg
                w_id = w[[target_idx], :, :]
                w_target = w_custom - w_id / w_id.norm(p=2) * self.args.target_d 
            print("w_target is custom")
        lpips_fn = lpips.LPIPS(net="vgg").to(device)
        id_fn = IDLoss().to(device)
        pbar = tqdm(range(self.args.iter))
        for i in pbar:
            angle_y = np.random.uniform(-self.args.angle_y_abs, self.args.angle_y_abs)
            cam2world_pose = LookAtPoseSampler.sample(np.pi / 2 + angle_y, np.pi / 2 + self.args.angle_p,
                                                      cam_pivot,
                                                      radius=cam_radius, device=device)
            camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)

            loss = torch.tensor(0.0, device=device)
            loss_dict = {}
            if self.args.local:
                loss_local = torch.tensor(0.0, device=device)
                feat_u = generator.get_planes(w_u) 
                feat_target = init_global_generator.get_planes(w_target)  
                loss_local_mse = F.mse_loss(feat_u, feat_target)
                loss_local = loss_local + self.args.loss_local_mse_lambda * loss_local_mse

                img_u = generator.synthesis(w_u, camera_params)["image"]
                img_target = init_global_generator.synthesis(w_target, camera_params)["image"]
              
                loss_local_lpips = lpips_fn(img_u, img_target).mean()
                loss_local = loss_local + self.args.loss_local_lpips_lambda * loss_local_lpips
               
                loss_local_l1 = torch.nn.L1Loss()(feat_u, feat_target)
                loss_local = loss_local + self.args.loss_local_l1_lambda * loss_local_l1  
                loss = loss + loss_local
                loss_dict["local_mse"] = loss_local_mse.item()
                loss_dict["local_lpips"] = loss_local_lpips.item()
                loss_dict["local_l1"] = loss_local_l1.item()
            if self.args.adj:
                loss_adj = torch.tensor(0.0, device=device)
                for _ in range(self.args.loss_adj_batch):
                    z_ra = torch.randn(1, 512, device=device) 
                    w_ra = generator.mapping(z_ra, conditioning_params, truncation_psi=self.args.truncation_psi,
                                             truncation_cutoff=self.args.truncation_cutoff) 
                    if self.args.loss_adj_alpha_range_max is not None:
                        loss_adj_alpha = torch.from_numpy(
                            np.random.uniform(self.args.loss_adj_alpha_range_min,
                                              self.args.loss_adj_alpha_range_max,
                                              size=1)).unsqueeze(
                            1).unsqueeze(1).to(device)        
                    deltas = loss_adj_alpha * (w_ra - w_u) / (w_ra - w_u).norm(p=2)
                    w_u_adj = w_u + deltas 
                    w_target_adj = w_target + deltas            
                    feat_u = generator.get_planes(w_u_adj)
                    feat_target = init_global_generator.get_planes(w_target_adj)
                    loss_adj_mse = F.mse_loss(feat_u, feat_target)
                    loss_adj = loss_adj + self.args.loss_adj_mse_lambda * loss_adj_mse                  
                    img_u = generator.synthesis(w_u_adj, camera_params)["image"]
                    img_target = init_global_generator.synthesis(w_target_adj, camera_params)["image"]
                    loss_adj_lpips = lpips_fn(img_u, img_target).mean()
                    loss_adj = loss_adj + self.args.loss_adj_lpips_lambda * loss_adj_lpips
                    # loss_adj_id = id_fn(img_u, img_target)
                    # loss_adj = loss_adj + self.args.loss_adj_id_lambda * loss_adj_id
                    loss_adj_l1 = torch.nn.L1Loss()(feat_u, feat_target)
                    loss_adj = loss_adj + self.args.loss_adj_l1_lambda * loss_adj_l1                
                loss = loss + self.args.loss_adj_lambda * loss_adj
                loss_dict["loss_adj"] = loss_adj.item()
                loss_dict["adj_mse"] = loss_adj_mse.item()
                loss_dict["adj_lpips"] = loss_adj_lpips.item()
                loss_dict["adj_l1"] = loss_adj_l1.item()

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
            pbar.set_postfix(loss=loss.item(), **loss_dict)
            if i % 100 == 0: 
                with torch.no_grad():
                    generator.eval()
                    img_u_save = generator.synthesis(w_u, camera_params_front)["image"]
                    img_u_save = self.tensor_to_image(img_u_save)
                    img_u_save.save(os.path.join(image_dir, f"img_u_{str(i).zfill(5)}.png"))

                    img_target_save = init_global_generator.synthesis(w_target, camera_params_front)["image"]
                    img_target_save = self.tensor_to_image(img_target_save)
                    img_target_save.save(os.path.join(image_dir, f"img_target_{str(i).zfill(5)}.png"))
                    generator.train()
                del img_u_save, img_target_save
        with torch.no_grad():
            generator.eval()
            img_u_save = generator.synthesis(w_u, camera_params)["image"]
            img_target_save = init_global_generator.synthesis(w_target, camera_params)["image"]
            img_u_save = self.tensor_to_image(img_u_save)
            img_target_save = self.tensor_to_image(img_target_save)
            img_u_save.save(os.path.join(image_dir, f"img_u_{str(i).zfill(5)}.png"))
            img_target_save.save(os.path.join(image_dir, f"img_target_{str(i).zfill(5)}.png"))
            generator.train()
        with torch.no_grad():
            generator.eval()
            if self.args.inversion is None: 
                for view, angle_y in enumerate(
                        np.linspace(-self.args.angle_y_abs, self.args.angle_y_abs, self.args.sample_views)):
                    cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y,
                                                                   np.pi / 2 + self.args.angle_p,
                                                                   cam_pivot,
                                                                   radius=cam_radius, device=device)
                    camera_params_view = torch.cat(
                        [cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)],
                        dim=1)
                    img_u = generator.synthesis(w_u, camera_params_view)["image"]
                    img_u = self.tensor_to_image(img_u)
                    img_u.save(os.path.join(result_dir, f"unlearn_after_0_{view}.png"))
                    flops, params = profile(generator, inputs=(w_u, camera_params_view))
                    flops, params = clever_format([flops, params], "%.3f")
                    print(f"FLOPs: {flops}")
                    print(f"Params: {params}")
            else:
                if flag is False:  
                    for i in range(length):
                        for view, angle_y in enumerate(
                                np.linspace(-self.args.angle_y_abs, self.args.angle_y_abs,
                                            self.args.sample_views)):
                            cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y,
                                                                           np.pi / 2 + self.args.angle_p,
                                                                           cam_pivot, radius=cam_radius,
                                                                           device=device)
                            camera_params_view = torch.cat(
                                [cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)],
                                dim=1)
                            img_origin = generator.synthesis(w_origin[[i]], camera_params_view)["image"]
                            img_origin = self.tensor_to_image(img_origin)
                            img_origin.save(os.path.join(result_dir, f"unlearn_after_{i}_{view}.png"))
                else: 
                    for view, angle_y in enumerate(
                            np.linspace(-self.args.angle_y_abs, self.args.angle_y_abs, self.args.sample_views)):
                        cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y,
                                                                       np.pi / 2 + self.args.angle_p,
                                                                       cam_pivot,
                                                                       radius=cam_radius, device=device)
                        camera_params_view = torch.cat(
                            [cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)],
                            dim=1)
                        img_u = generator.synthesis(w_u, camera_params_view)["image"]
                        img_u = self.tensor_to_image(img_u)
                        img_u.save(os.path.join(result_dir, f"unlearn_after_0_{view}.png"))
            generator.train()

        snapshot_data = dict() 
        snapshot_data["G_ema"] = copy.deepcopy(generator).eval().requires_grad_(False).cpu()
        current_time = datetime.datetime.now()
        print("current time:", current_time)
        with open(os.path.join(ckpt_dir, f"last.pkl"), "wb") as f:
            pickle.dump(snapshot_data, f)


    def add_gaussian_noise(self,feature_vector):
        epsilon = 0.5  
        delta = 1e-5  
        sensitivity = 0.001/64 * math.sqrt(2 * math.log(1.25/delta))/epsilon 
        sigma = (sensitivity/epsilon * math.sqrt(2 * math.log(1.25/delta)))/math.sqrt(5)
        noise = torch.normal(mean=0.0, std=sigma, size=feature_vector.shape, device=feature_vector.device)

        return feature_vector + noise
