Ver la escena y mover sliders: python -m mujoco.viewer --mjcf assets/arm.xml
Grabar un GIF de episodios random:  python scripts/record_random.py
El GIF con las dos cámaras: python scripts/record_random.py --camera both
Comprobar que el entorno es valido:      python -c "from gymnasium.utils.env_checker import check_env; from env import ArmEnv;check_env(ArmEnv()); print('OK')"