import argparse

import numpy as np
import mcubes
import torch
import trimesh

import models_class_cond, models_ae

from pathlib import Path


def decode_grid_chunked(ae, latent, grid, chunk=50000):
    """Decode a large query grid in chunks to avoid OOM in decoder_cross_attn."""
    outs = []
    for i in range(0, grid.shape[1], chunk):
        outs.append(ae.decode(latent, grid[:, i:i + chunk]).detach())
    return torch.cat(outs, dim=1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--ae', type=str, required=True)          
    parser.add_argument('--ae-pth', type=str, required=True)      
    parser.add_argument('--dm', type=str, required=True)          
    parser.add_argument('--dm-pth', type=str, required=True)      
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--density', type=int, default=128)
    parser.add_argument('--decode_chunk', type=int, default=50000)
    parser.add_argument('--out_dir', type=str, default='generated_obj')
    parser.add_argument('--seed_offset', type=int, default=0)
    args = parser.parse_args()
    print(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda:0')

    ae = models_ae.__dict__[args.ae]()
    ae.eval()
    ae.load_state_dict(
        torch.load(args.ae_pth, map_location='cpu', weights_only=False)['model']
    )
    ae.to(device)

    model = models_class_cond.__dict__[args.dm]()
    model.eval()
    model.load_state_dict(
        torch.load(args.dm_pth, map_location='cpu', weights_only=False)['model']
    )
    model.to(device)

    torch.cuda.empty_cache()

    density = args.density
    gap = 2. / density
    x = y = z = np.linspace(-1, 1, density + 1)
    xv, yv, zv = np.meshgrid(x, y, z)
    grid = torch.from_numpy(
        np.stack([xv, yv, zv]).astype(np.float32)
    ).view(3, -1).transpose(0, 1)[None].to(device)

    with torch.no_grad():
        batch_seeds = torch.arange(args.seed_offset, args.seed_offset + args.num_samples).to(device)
        sampled_array = model.sample(batch_seeds=batch_seeds).float()
        print(sampled_array.shape, sampled_array.mean().item(), sampled_array.std().item())

        for j in range(sampled_array.shape[0]):
            logits = decode_grid_chunked(
                ae, sampled_array[j:j + 1], grid, chunk=args.decode_chunk
            )

            volume = logits.view(density + 1, density + 1, density + 1).permute(1, 0, 2).cpu().numpy()
            verts, faces = mcubes.marching_cubes(volume, 0)
            verts = verts * gap - 1

            m = trimesh.Trimesh(verts, faces)
            out_path = out_dir / f'sample_{j + args.seed_offset:03d}.obj'
            m.export(str(out_path))
            print(f'saved {out_path}')