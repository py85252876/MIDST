from copy import deepcopy

import numpy as np
import pandas as pd
import torch

from .utils_train import update_ema


class Trainer:
    def __init__(
        self,
        diffusion,
        train_iter,
        lr,
        weight_decay,
        steps,
        device=torch.device("cuda:1"),
    ):
        self.diffusion = diffusion
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()

        self.train_iter = train_iter
        self.steps = steps
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(
            self.diffusion.parameters(), lr=0, weight_decay=weight_decay
        )
        self.device = device
        self.loss_history = pd.DataFrame(columns=["step", "mloss", "gloss", "loss"])
        self.log_every = 100
        self.print_every = 500
        self.ema_every = 1000

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x, out_dict):
        x = x.to(self.device)
        for k in out_dict:
            out_dict[k] = out_dict[k].long().to(self.device)
        self.optimizer.zero_grad()
        loss_multi, loss_gauss = self.diffusion.mixed_loss(x, out_dict)
        loss = loss_multi + loss_gauss
        loss.backward()
        self.optimizer.step()

        return loss_multi, loss_gauss
    
    def _extract_gradient_step(self, x, out_dict):
        x = x.to(self.device)

        num_timesteps = 2000
        num_samples = 30
        t_list = torch.linspace(0, num_timesteps - 1, steps=num_samples, device=self.device).long()
        gradients_l2_list = []
        for k in out_dict:
            out_dict[k] = out_dict[k].long().to(self.device)
        for t in t_list:
            self.optimizer.zero_grad()
            loss_multi, loss_gauss = self.diffusion.updated_loss(x, out_dict,t)
            loss = loss_multi + loss_gauss
            loss.backward(retain_graph=True)
            inner_gradient = []
            for p in self.diffusion._denoise_fn.parameters():
                inner_gradient.append(torch.norm(p.grad).unsqueeze(0))
            temp_gradient_list = torch.cat(inner_gradient)        
            self.optimizer.zero_grad()
            gradients_l2_list.append(temp_gradient_list)
        return torch.stack(gradients_l2_list).mean(dim=0).unsqueeze(0)

    def run_loop(self):
        step = 0
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0

        curr_count = 0
        while step < self.steps:
            x, out_dict = next(self.train_iter)
            out_dict = {"y": out_dict}
            batch_loss_multi, batch_loss_gauss = self._run_step(x, out_dict)

            self._anneal_lr(step)

            curr_count += len(x)
            curr_loss_multi += batch_loss_multi.item() * len(x)
            curr_loss_gauss += batch_loss_gauss.item() * len(x)

            if (step + 1) % self.log_every == 0:
                mloss = np.around(curr_loss_multi / curr_count, 4)
                gloss = np.around(curr_loss_gauss / curr_count, 4)
                if (step + 1) % self.print_every == 0:
                    print(
                        f"Step {(step + 1)}/{self.steps} MLoss: {mloss} GLoss: {gloss} Sum: {mloss + gloss}"
                    )
                self.loss_history.loc[len(self.loss_history)] = [
                    step + 1,
                    mloss,
                    gloss,
                    mloss + gloss,
                ]
                curr_count = 0
                curr_loss_gauss = 0.0
                curr_loss_multi = 0.0

            update_ema(
                self.ema_model.parameters(), self.diffusion._denoise_fn.parameters()
            )

            step += 1
    
    def gradient_loop(self, gradient_settings = None):
        step = 0
        gradient_list = []
        index_list = []
        while step < self.steps:
            x, out_dict = next(self.train_iter)
            index_list.append(out_dict)
            out_dict = {"y": out_dict}
            x = x.to(self.device)
            t_list = torch.linspace(0, gradient_settings["step_range"], steps=gradient_settings["num_step"], dtype=torch.int64).to(self.device)
            gradients_l2_list = []
            for k in out_dict:
                out_dict[k] = out_dict[k].long().to(self.device)
            k = 5
            for t in t_list:
                self.optimizer.zero_grad()
                loss_multi, loss_gauss = self.diffusion.updated_loss(x, out_dict,t)
                loss = loss_multi + loss_gauss
                loss.backward(retain_graph=True)
                inner_gradient = []
                for p in self.diffusion._denoise_fn.parameters():
                    if p.requires_grad and p.grad is not None:
                        
                        grad_flat = p.grad.view(-1)
                        
                        mean_val = grad_flat.mean()
                        std_val  = grad_flat.std()
                        max_val  = grad_flat.max()
                        min_val  = grad_flat.min()
                        l2_norm  = torch.norm(grad_flat, p=2)

                        topk_vals, _ = torch.topk(grad_flat.abs(), k)
                        
                       
                        feature_tensor = torch.cat([
                            mean_val.unsqueeze(0),
                            std_val.unsqueeze(0),
                            max_val.unsqueeze(0),
                            min_val.unsqueeze(0),
                            l2_norm.unsqueeze(0),
                            topk_vals,  # size = k
                        ])
                       
                        feature_tensor = feature_tensor.unsqueeze(0)

                        
                        inner_gradient.append(feature_tensor)

                    else:
                        
                        pass
                
                temp_gradient_list = torch.cat(inner_gradient).flatten().unsqueeze(0)       
                self.optimizer.zero_grad()
                gradients_l2_list.append(temp_gradient_list)
            # print(torch.stack(gradients_l2_list).squeeze(0).mean(dim=0).shape)
            gradient_list.append(torch.stack(gradients_l2_list).squeeze(0).mean(dim=0))
            step += 1
        return gradient_list, index_list
