"""Graba episodios de un agente ALEATORIO en un GIF: para mirar el entorno y tener una referencia.

Uso:
    python scripts/record_random.py                          # 2 episodios aleatorios de reach
    python scripts/record_random.py --task push --episodes 3
    python scripts/record_random.py --camera overhead        # la vista que usara la vision (fase 4)
    python scripts/record_random.py --camera both            # las dos vistas lado a lado

La politica es aleatoria: este script es para mirar el ENTORNO, no un agente.
Para grabar y medir un modelo entrenado: scripts/evaluate.py.
"""

import argparse
from pathlib import Path

import imageio.v2 as imageio

# el proyecto no es un paquete instalado, asi que anado la raiz al path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from env import ArmEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="reach", choices=["reach", "push"])
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--camera", default="showcase",
                        choices=["showcase", "front", "overhead", "both"])
    parser.add_argument("--out", default="media/random_{task}.gif")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--every", type=int, default=2,
                        help="guarda 1 de cada N fotogramas (2 = GIF la mitad de pesado)")
    parser.add_argument("--size", default="240x320", metavar="ALTOxANCHO",
                        help="resolucion por camara; subela solo para la demo final")
    args = parser.parse_args()

    height, width = (int(v) for v in args.size.split("x"))
    env = ArmEnv(task=args.task, render_mode="rgb_array", render_size=(height, width))

    frames = []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        total_reward, n_steps, done = 0.0, 0, False

        while not done:
            action = env.action_space.sample()          # politica aleatoria
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            n_steps += 1
            done = terminated or truncated

            if n_steps % args.every == 0:
                frames.append(env.render(args.camera))

        print(f"episodio {ep}: reward {total_reward:7.2f}   pasos {n_steps:3d}   "
              f"exito {info['is_success']}")

    env.close()

    out_path = Path(args.out.format(task=args.task))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fps = env.metadata["render_fps"] / args.every
    imageio.mimsave(out_path, frames, fps=fps, loop=0)

    size_mb = out_path.stat().st_size / 1e6
    print(f"\n{len(frames)} fotogramas a {fps:.0f} fps -> {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
