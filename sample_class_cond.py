import argparse
from pathlib import Path
import numpy as np
import mcubes
import torch
import trimesh

import models_class_cond, models_ae

if __name__ == "__main__":
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--ae', type=str, required=True)
    parser.add_argument('--ae-pth', type=str, required=True)
    parser.add_argument('--dm', type=str, required=True)
    parser.add_argument('--dm-pth', type=str, required=True)
    parser.add_argument('--out-dir', type=str, default='generated_samples')
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ae = models_ae.__dict__[args.ae]()
    ae.eval()
    ae_ckpt = torch.load(args.ae_pth, weights_only=False, map_location='cpu')
    ae.load_state_dict(ae_ckpt['model'] if 'model' in ae_ckpt else ae_ckpt)
    ae.to(device)

    model = models_class_cond.__dict__[args.dm]()
    model.eval()
    dm_ckpt = torch.load(args.dm_pth, weights_only=False, map_location='cpu')
    model.load_state_dict(dm_ckpt['model'] if 'model' in dm_ckpt else dm_ckpt)
    model.to(device)

    density = 128
    gap = 2.0 / density
    coords = np.linspace(-1, 1, density + 1)
    xv, yv, zv = np.meshgrid(coords, coords, coords)
    grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32)).view(3, -1).transpose(0, 1)[None].to(device)

    num_samples = 2

    with torch.no_grad():
        batch_seeds = torch.arange(num_samples, device=device)

        sampled_array = model.sample(batch_seeds=batch_seeds).float()

        for j in range(num_samples):
            latent_sample = sampled_array[j:j+1]

            chunk_size = 512  
            num_points = grid.shape[1]
            logits_list = []

            for start_idx in range(0, num_points, chunk_size):
                end_idx = min(start_idx + chunk_size, num_points)
                grid_chunk = grid[:, start_idx:end_idx]  

                chunk_logits = ae.decode(latent_sample, grid_chunk)
                if isinstance(chunk_logits, dict):
                    chunk_logits = chunk_logits["logits"]

                logits_list.append(chunk_logits.detach())

            logits = torch.cat(logits_list, dim=1)
            
            if isinstance(logits, dict):
                logits = logits['logits']
            logits = logits.detach()

            volume = logits.view(density + 1, density + 1, density + 1).permute(1, 0, 2).cpu().numpy()

            verts, faces = mcubes.marching_cubes(volume, 0.0)

            verts = (verts * gap) - 1.0

            mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            file_name = out_path / f"sample_{j:02d}.obj"
            mesh.export(str(file_name))
            print(f"Exported: {file_name}")