import gymnasium as gym
import numpy as np
import mujoco                      
from pathlib import Path           

class ArmEnv(gym.Env):
    # Gymnasium mira aquí qué sabe hacer render(). render_fps = 1/(0.002*20) = 25,
    # que es la frecuencia a la que decide el agente
    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    # leo todo lo que me interesa de data y lo devuelvo con nombre, así step() coge
    # lo que necesita para la recompensa sin andar con índices del vector de obs
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

    # junta el dict en el vector que ve la red. Ojo, la red aprende este orden:
    # si lo cambio, los modelos ya entrenados dejan de valer
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
        # punto de empuje: justo detrás del cubo, en el lado contrario al objetivo.
        # normalizo la dirección para que quede siempre a 4.5 cm del cubo
        goal_dir = (goal_xy - cube_xy) / np.linalg.norm(goal_xy - cube_xy)
        push_point= cube_xy - 0.045 * goal_dir
        value = -1 * (np.linalg.norm(tip_xy -push_point)) - 5 * (np.linalg.norm(cube_xy -goal_xy)) -0.01 * np.sum(action**2)
        if(np.linalg.norm(cube_xy - goal_xy) < 0.03):   # cubo a menos de 3 cm del objetivo
            value+=5
            terminated=True
        return value, terminated


    def __init__(self, task="reach", render_mode=None, camera="showcase", render_size=(480, 640)):
        # cargar MjModel/MjData, definir espacios
        assert task in ("reach", "push"), f"task desconocida: {task!r}"   # si me equivoco al escribirla, que falle y no entrene otra cosa
        self.task = task
        # el Renderer no lo creo aquí sino la primera vez que llamo a render(),
        # así al entrenar con 8 entornos no se abren 8 contextos de OpenGL
        self.render_mode = render_mode
        self.camera = camera
        self.render_size = render_size
        self._renderer = None
        self.frame_skip = 20                               # 20 x 0.002 s = 25 Hz de control
        self.max_steps = 150                               # 150 x 0.04 s = 6 s de episodio

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
        # 1. convertir la acción en objetivos para los servos. La acción es un incremento:
        # muevo cada objetivo como mucho 0.05 rad por paso y lo dejo dentro de su rango
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
        self.step_counter+=1
        if success:                             # terminated: ha llegado
            terminated=True
        elif(self.step_counter>=self.max_steps):    # truncated: se acabó el tiempo, no es que haya fallado
            truncated=True
        info={"is_success" : terminated}
        return obs, float(reward), terminated, truncated, info   # float normal, check_env no quiere np.float64

    # devuelve una foto de la escena (alto, ancho, 3) en uint8. El Renderer se crea
    # una sola vez porque crearlo en cada fotograma es lentísimo
    def render(self, camera=None):
        # "both": las dos vistas lado a lado
        if camera == "both":
            return np.hstack([self.render("showcase"), self.render("overhead")])
        if self._renderer is None:
            height, width = self.render_size
            self._renderer = mujoco.Renderer(self.model, height, width)
        self._renderer.update_scene(self.data, camera=camera or self.camera)
        return self._renderer.render()

    # Gymnasium llama a close() al terminar, aquí suelto el contexto gráfico
    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
