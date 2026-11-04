"""Mide la tasa de éxito de un modelo entrenado y graba un GIF (fases 3 y 5).

Uso:
    python scripts/evaluate.py --task reach --episodes 100
    python scripts/evaluate.py --task push --seed 1 --camera both

Juega en modo determinista (la media de la campana, sin sortear) y carga el .pkl
de VecNormalize en modo evaluación. Sin eso la red ve números en otra escala y
parece que no ha aprendido nada.
"""

import argparse
from pathlib import Path
import sys

import cv2
import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env import ArmEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="reach", choices=["reach", "push"])
    parser.add_argument("--seed", type=int, default=0, help="semilla del modelo a cargar")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--record", type=int, default=3,
                        help="cuantos de los primeros episodios grabar (0 = ninguno)")
    parser.add_argument("--camera", default="showcase",
                        choices=["showcase", "front", "overhead", "both"])
    parser.add_argument("--out", default="media/ppo_{task}.gif")
    return parser.parse_args()


def annotate(frame, lines, color=(255, 255, 255)):
    """Escribe el texto arriba a la izquierda, con un recuadro negro detrás para que se lea siempre."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, text in enumerate(lines):
        x, y = 8, 20 + 20 * i
        (w, h), _ = cv2.getTextSize(text, font, 0.5, 1)
        cv2.rectangle(frame, (x - 3, y - h - 4), (x + w + 3, y + 5), (0, 0, 0), -1)   # -1 = relleno
        cv2.putText(frame, text, (x, y), font, 0.5, color, 1, cv2.LINE_AA)
    return frame


def main():
    args = parse_args()

    # mismos nombres que pone train.py
    run_name = f"{args.task}_s{args.seed}"
    model_path = ROOT / "models" / f"ppo_{run_name}.zip"
    stats_path = ROOT / "models" / f"vecnormalize_{run_name}.pkl"
    def crear_entorno():
        return ArmEnv(task=args.task, render_mode="rgb_array", render_size=(240, 320))  # GIF ligero
    # aquí basta con 1 entorno. La semilla es distinta a la de entrenar para
    # probar con posiciones del cubo que no ha visto
    env = DummyVecEnv (
        [crear_entorno]
    )
    env.seed(1000)
    # cargo las estadísticas del entrenamiento y las congelo: que no se actualicen
    # con estos datos y que la recompensa salga sin normalizar
    env=VecNormalize.load(str(stats_path), env)
    env.training=False
    env.norm_reward=False

    arm = env.venv.envs[0]          # Cogemos el entorno ArmEnv (VecNormalize -> DummyVecEnv -> ArmEnv)

    model=PPO.load(str(model_path))
    successes, ep_rewards, ep_lengths, frames = [], [], [], []
    obs=env.reset() #reseteamos el entorno, solo aquí: el VecEnv se resetea solo al acabar cada episodio
    for ep in range(args.episodes): #Bucle de prueba
        done=False
        total_reward, total_steps=0.0, 0
        recording = ep < args.record
        while not done:
            if recording:
                raw = arm.render(args.camera)     # la foto va antes del step, si no la última ya sería del episodio siguiente
                frames.append(annotate(raw.copy(), [f"{args.task} | episodio {ep + 1}/{args.record}",
                                                    f"paso {total_steps}  reward {total_reward:.2f}"]))
            action,_= model.predict(obs, deterministic=True)
            obs,rewards, done, info=env.step(action) #Devuelve listas y hace el step
            total_steps+=1
            total_reward+=rewards[0]
            done=done[0]

        #Se completa un episodio y añadimos success, recompensa y length
        successes.append(info[0]["is_success"])
        ep_rewards.append(total_reward)
        ep_lengths.append(total_steps)
        if recording:                           # pausa de 0.4 s al final con el resultado, en verde o rojo
            ok = info[0]["is_success"]
            end = annotate(raw, [f"{args.task} | episodio {ep + 1}/{args.record}",
                                 f"{total_steps} pasos  reward {total_reward:.2f}",
                                 "EXITO" if ok else "FALLO"], color=(0, 220, 0) if ok else (255, 60, 60))
            frames.extend([end] * 10)

    # resumen de todos los episodios
    n_ok = int(np.sum(successes))
    print(f"\n{args.task} | modelo {run_name} | {args.episodes} episodios deterministas")
    print(f"  Exito:            {n_ok}/{args.episodes} ({100 * n_ok / args.episodes:.0f} %)")
    print(f"  Recompensa media: {np.mean(ep_rewards):.2f} +- {np.std(ep_rewards):.2f}")
    print(f"  Pasos medios:     {np.mean(ep_lengths):.1f} +- {np.std(ep_lengths):.1f}")

    # guardo el GIF de los episodios grabados (25 fps = la frecuencia de control)
    env.close()
    if frames:
        out_path = ROOT / args.out.format(task=args.task)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(out_path, frames, fps=25, loop=0)
        print(f"  GIF: {len(frames)} fotogramas -> {out_path}")


if __name__ == "__main__":
    main()
