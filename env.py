import gymnasium as gym
import numpy as np
import mujoco                      
from pathlib import Path           

class ArmEnv(gym.Env):
    # CAMBIO: antes esto era el cuerpo de _get_obs. Ahora lee data y devuelve las
    # magnitudes con nombre, para que step() pueda pedir lo que necesite para la
    # recompensa sin trocear el vector de observación con indices magicos.
    def _get_state(self):

        joint_angles = self.data.qpos[self.joint_qpos_idx].copy()
        joint_vels   = self.data.qvel[self.joint_qvel_idx].copy()

        # site_xpos ya da su posición en el mundo, y la cojo en 3D porque la altura importa para tocar el cubo.
        tip_pos = self.data.site_xpos[self.tip_site_id].copy()                 # x, y, z


        # [x, y, z, qw, qx, qy, qz]; me quedo con los 3 primeros.
        cube_pos = self.data.qpos[self.cube_qpos_idx : self.cube_qpos_idx + 3].copy()
        cube_xy = cube_pos[:2]

        goal_xy = self.data.mocap_pos[self.goal_idx][:2].copy()
        cube_tip_diff = cube_pos - tip_pos
        goal_cube_diff = goal_xy - cube_xy

        return {
            "joint_angles":   joint_angles,     # 4 valores (ángulos)
            "joint_vels":     joint_vels,       # 4 valores (velocidades)
            "tip_pos":        tip_pos,          # 3 valores
            "tip_xy":         tip_pos[:2],      # 2 valores (para push, que es en el plano)
            "cube_pos":       cube_pos,         # 3 valores
            "cube_xy":        cube_xy,          # 2 valores
            "goal_xy":        goal_xy,          # 2 valores
            "cube_tip_diff":  cube_tip_diff,    # 3 valores
            "goal_cube_diff": goal_cube_diff,   # 2 valores
        }

    # CAMBIO: _get_obs ahora solo ordena el dict en el vector que ve la red.
    # El orden de esta lista es el que la red aprende: no lo cambies a mitad de
    # un entrenamiento, o el modelo guardado deja de valer.
    def _get_obs(self):
        state = self._get_state()

        obs= np.concatenate([
            state["joint_angles"],     # 4 valores (ángulos)
            state["joint_vels"],       # 4 valores (velocidades)
            state["tip_pos"],          # 3 valores
            state["cube_xy"],          # 2 valores
            state["goal_xy"],          # 2 valores
            state["cube_tip_diff"],    # 3 valores
            state["goal_cube_diff"],   # 2 valores
        ]).astype(np.float32)

        return obs

    def reach_reward_func(self, cube_tip_difference):
        distance = np.linalg.norm(cube_tip_difference)   
        value = - distance
        terminate=False
        if (distance < 0.05):                           
            value+=5
            terminate=True
        return value, terminate

    def push_reward_func(self, cube_xy, goal_xy, tip_xy, action):
        terminated=False
        # CAMBIO: dirección normalizada, así el punto queda SIEMPRE a 4.5 cm del cubo
        goal_dir = (goal_xy - cube_xy) / np.linalg.norm(goal_xy - cube_xy)
        push_point= cube_xy - 0.045 * goal_dir
        value = -1 * (np.linalg.norm(tip_xy -push_point)) - 5 * (np.linalg.norm(cube_xy -goal_xy)) -0.01 * np.sum(action**2)
        if(np.linalg.norm(cube_xy - goal_xy) < 0.03):   # CAMBIO: norma de la diferencia
            value+=5
            terminated=True
        return value, terminated


    def __init__(self, task="reach"):
        # cargar MjModel/MjData, definir espacios
        assert task in ("reach", "push"), f"task desconocida: {task!r}"   # CAMBIO: un typo aquí entrenaría otra cosa en silencio
        self.task = task
        self.frame_skip = 20                               # 20 x 0.002 s = 25 Hz de control
        self.max_steps = 150                               # CAMBIO: 150 x 0.04 s = 6 s de episodio

        # cargar el modelo
        xml_path = Path(__file__).parent / "assets" / "arm.xml"
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        joint_names = ["yaw", "shoulder", "elbow", "wrist"]
        self.joint_qpos_idx = [self.model.joint(name).qposadr[0] for name in joint_names] #Cogemos los ids apropiados de cada hinge que queremos( no tienen por que estar en posicion consecutiva)
        self.joint_qvel_idx = [self.model.joint(name).dofadr[0] for name in joint_names]
        self.cube_qpos_idx = self.model.joint("cube").qposadr[0]
        self.tip_site_id = self.model.site("tip").id       
        self.goal_idx = self.model.body("goal").mocapid[0]

        # pongo el brazo en la postura "home" y llamo a mj_forward para que
        # site_xpos esté calculado; si no, _get_obs() leería posiciones sin actualizar.
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        mujoco.mj_forward(self.model, self.data)

        # el tamaño de la observación lo mide _get_obs(), así no hay que
        # acordarse de cambiar N cada vez que toques la observación (ahora son 20).
        obs_dim = self._get_obs().size
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

        self.cube_qvel_idx = self.model.joint("cube").dofadr[0]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)    
        # poner el brazo en la postura inicial, colocar el cubo y el objetivo al azar
        cube_xy=self.np_random.uniform([0.20, -0.10], [0.35, 0.10])
        #la fórmula del objetivo
        def draw_goal():
            goal_angle = self.np_random.uniform(-np.pi, np.pi)      # dirección
            goal_radius = self.np_random.uniform(0.08, 0.15)        # distancia al cubo
            offset = goal_radius * np.array([np.cos(goal_angle), np.sin(goal_angle)])
            return cube_xy + offset

        goal_xy = draw_goal()                        # np.array([a, b]), con corchetes
        goal_distance = np.linalg.norm(goal_xy)      # la distancia sale de goal_xy
        iters=0
        # paréntesis en el or, para que el tope de intentos valga en los dos casos
        while (goal_distance >= 0.40 or goal_distance <= 0.15) and iters<5:
            goal_xy = draw_goal()                    # actualiza goal_xy
            goal_distance = np.linalg.norm(goal_xy)
            iters+=1

        # muevo el objetivo a lo largo de su radio hasta meterlo en el anillo alcanzable.
        if goal_distance >= 0.40 or goal_distance <= 0.15:
            goal_xy = goal_xy / goal_distance * np.clip(goal_distance, 0.16, 0.39)

        mujoco.mj_resetDataKeyframe(self.model,self.data, self.model.key("home").id)   #reiniciar el modelo
        self.data.qpos[self.cube_qpos_idx : self.cube_qpos_idx+3]=cube_xy[0], cube_xy[1], 0.025 #escribir pos del cubo
        self.data.qvel[self.cube_qvel_idx : self.cube_qvel_idx+6]=0,0,0,0,0,0 #poner velocidad a 0
        self.data.mocap_pos[self.goal_idx]=goal_xy[0], goal_xy[1], 0.001
        mujoco.mj_forward(self.model, self.data)
        self.step_counter=0
        obs=self._get_obs()
        info={}                                     
        return obs, info

    def step(self, action):
        terminated = truncated = False
        # 1. convertir la acción en objetivos para los servos
        
        # CAMBIO: quitado el 0.05 de delante del clip. El 0.05 escala el INCREMENTO
        # (ya está dentro); multiplicando el resultado encogías el objetivo acumulado
        # un 95% en cada paso, así que el brazo nunca se alejaba de ctrl = 0.
        self.data.ctrl[:]=np.clip(self.data.ctrl + 0.05 * action, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
        # 2. hacer frame_skip pasos de física
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        # 3. calcular observación, recompensa y si ha terminado
        obs = self._get_obs()
        state = self._get_state()
        if self.task == "reach":
            reward, success = self.reach_reward_func(state["cube_tip_diff"])
        else:
            reward, success = self.push_reward_func(state["cube_xy"], state["goal_xy"], state["tip_xy"], action)
        self.step_counter+=1                    # CAMBIO: se había pegado al final de la línea anterior
        if success:                             # CAMBIO: ya es un booleano
            terminated=True
        elif(self.step_counter>=self.max_steps):    # CAMBIO: el 150 ahora está en __init__
            truncated=True
        info={"is_success" : terminated}
        return obs, float(reward), terminated, truncated, info   # CAMBIO: float de Python, no np.float64
