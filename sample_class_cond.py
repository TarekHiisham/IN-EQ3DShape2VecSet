import argparse
from pathlib import Path
import numpy as np
from skimage import measure 
import torch
import trimesh

import models_ae, models_class_cond

if __name__ == "__main__":
  parser = argparse.ArgumentParser("", add_help=False)
  parser.add_argument("--ae", type=str, required=True)
  parser.add_argument("--ae-pth", type=str, required=True)
  parser.add_argument("--dm", type=str, required=True)
  parser.add_argument("--dm-pth", type=str, required=True)
  parser.add_argument("--num-samples", type=int, default=3)
  parser.add_argument("--out-dir", type=str, default="generated_samples")
  args = parser.parse_args()

  out_path = Path(args.out_dir)
  out_path.mkdir(parents=True, exist_ok=True)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  # AutoEncoder
  ae = models_ae.__dict__[args.ae]()
  ae.eval()
  ae_ckpt = torch.load(args.ae_pth, weights_only=False, map_location="cpu")
  ae.load_state_dict(ae_ckpt["model"] if "model" in ae_ckpt else ae_ckpt)
  ae.to(device)

  # Diffusion Model
  model = models_class_cond.__dict__[args.dm]()
  model.eval()
  dm_ckpt = torch.load(args.dm_pth, weights_only=False, map_location="cpu")
  model.load_state_dict(dm_ckpt["model"] if "model" in dm_ckpt else dm_ckpt)
  model.to(device)

  density = 128
  gap = 2.0 / density
  coords = np.linspace(-1, 1, density + 1)
  xv, yv, zv = np.meshgrid(coords, coords, coords)
  grid = (
      torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32))
      .view(3, -1)
      .transpose(0, 1)[None]
      .to(device)
  )

  num_samples = args.num_samples

  with torch.no_grad():
    batch_seeds = torch.arange(num_samples, device=device)

    # محاولة التوليد بدون cond ثم مع cond إذا كانت الدالة تتطلبها
    try:
      sampled_array = model.sample(batch_seeds=batch_seeds).float()
    except TypeError:
      cond = torch.zeros(num_samples, dtype=torch.long, device=device)
      sampled_array = model.sample(
          cond=cond, batch_seeds=batch_seeds
      ).float()

    for j in range(num_samples):
      latent_sample = sampled_array[j : j + 1]

      chunk_size = 50000
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

      volume = (
          logits.view(density + 1, density + 1, density + 1)
          .permute(1, 0, 2)
          .cpu()
          .numpy()
      )

      if volume.max() > 0.0:
        verts, faces, _, _ = measure.marching_cubes(volume, level=0.0)
        verts = (verts * gap) - 1.0

        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        file_name = out_path / f"sample_{j:02d}.obj"
        mesh.export(str(file_name))
        print(f"Exported: {file_name} (Vertices: {len(verts)})")
      else:
        print(
            f"Sample {j} skipped: No positive logits found (Max:"
            f" {volume.max():.2f})"
        )