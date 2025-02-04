#!/usr/bin/env python
import argparse
import numpy as np
import torch
import ml_casadi.torch as mc
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

class NormalizedMLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.model = mc.nn.MultiLayerPerceptron(
            input_size, hidden_size, output_size, num_layers, 'Tanh'
        )
        self.register_buffer('x_mean', torch.zeros(input_size))
        self.register_buffer('x_std', torch.ones(input_size))
        self.register_buffer('y_mean', torch.zeros(output_size))
        self.register_buffer('y_std', torch.ones(output_size))
        
    def forward(self, x):
        if x.shape[-1] != self.x_mean.shape[0]:
            raise ValueError(f"Input dim mismatch: expected {self.x_mean.shape[0]}, got {x.shape[-1]}")
        x_norm = (x - self.x_mean) / self.x_std
        y_norm = self.model(x_norm)
        return y_norm * self.y_std + self.y_mean

_map_res = 0.1
_static_ground_map = np.zeros((10, 10))
_org_to_map_org = np.array([-0.5, -0.5])

def get_ground_effect_features(state, u):
    features_17 = np.concatenate([state, u])
    position = features_17[0:3]
    orientation = features_17[3:7]
    map_pos = (position[:2] - _org_to_map_org) / _map_res
    x_idx = int(np.clip(map_pos[0], 0, _static_ground_map.shape[0]-1))
    y_idx = int(np.clip(map_pos[1], 0, _static_ground_map.shape[1]-1))
    patch = np.zeros((3, 3))
    for i in range(-1, 2):
        for j in range(-1, 2):
            if (0 <= x_idx+i < _static_ground_map.shape[0]) and (0 <= y_idx+j < _static_ground_map.shape[1]):
                patch[i+1, j+1] = _static_ground_map[x_idx+i, y_idx+j]
    return np.concatenate([patch.flatten(), orientation])

def nominal_dynamics(state, u, dt):
    p = state[0:3]
    q = state[3:7]
    v = state[7:10]
    r = state[10:13]
    p_next = p + dt * v
    v_next = v + dt * u
    return np.concatenate([p_next, q, v_next, r])

def augmented_dynamics(state, u, model, dt):
    state_nom = nominal_dynamics(state, u, dt)
    u_full = np.zeros(4)
    u_full[:3] = u
    base_input = np.concatenate([state, u_full, get_ground_effect_features(state, u_full)])  # 30-dim
    pad_width = 69 - base_input.shape[0]
    if pad_width < 0:
        raise ValueError("Base input dimension is larger than 69.")
    nn_input = np.pad(base_input, (0, pad_width), mode='constant')
    nn_input_t = torch.tensor(nn_input, dtype=torch.float32).unsqueeze(0)
    correction = model(nn_input_t).detach().cpu().numpy().flatten()
    state_nom[7:10] += dt * correction[:3]
    return state_nom

def compute_linearization(state, u, model, dt):
    # u is the baseline control (here chosen as zeros)
    u_full = np.zeros(4)
    u_full[:3] = u
    base_input = np.concatenate([state, u_full, get_ground_effect_features(state, u_full)])
    pad_width = 69 - base_input.shape[0]
    if pad_width < 0:
        raise ValueError("Base input dimension is larger than 69.")
    nn_input0 = np.pad(base_input, (0, pad_width), mode='constant')
    nn_input0_t = torch.tensor(nn_input0, dtype=torch.float32, requires_grad=True).unsqueeze(0)
    base_output = model(nn_input0_t)  # shape (1, output_dim)
    output_dim = base_output.shape[1]
    jacobian_list = []
    for i in range(output_dim):
        grad_output = torch.zeros_like(base_output)
        grad_output[0, i] = 1.0
        grad = torch.autograd.grad(base_output, nn_input0_t, grad_outputs=grad_output, retain_graph=True)[0]
        jacobian_list.append(grad.squeeze(0))
    jacobian = torch.stack(jacobian_list).detach().cpu().numpy()  # shape (output_dim, input_dim)
    base_output_np = base_output.detach().cpu().numpy().flatten()
    return nn_input0, base_output_np, jacobian

def taylor_approx_dynamics_with_linearization(state, u, model, dt, nn_input0, base_output, jacobian):
    u_full_candidate = np.zeros(4)
    u_full_candidate[:3] = u
    candidate_input = np.concatenate([state, u_full_candidate, get_ground_effect_features(state, u_full_candidate)])
    pad_width = 69 - candidate_input.shape[0]
    if pad_width < 0:
        raise ValueError("Candidate input dimension is larger than 69.")
    candidate_input = np.pad(candidate_input, (0, pad_width), mode='constant')
    delta = candidate_input - nn_input0
    correction = base_output + jacobian.dot(delta)
    state_nom = nominal_dynamics(state, u, dt)
    state_nom[7:10] += dt * correction[:3]
    return state_nom


def mpc_controller_random_shooting(state, target_traj, horizon, dt, model, mode, num_samples=500):
    best_cost = float('inf')
    best_u0 = None
    if mode == 'taylor':
        baseline_u = np.zeros(3)  # You could also use the last applied control if available.
        nn_input0, base_output, jacobian = compute_linearization(state, baseline_u, model, dt)
    for i in range(num_samples):
        u_seq = np.random.uniform(-1, 1, (horizon, 4))
        cost = 0.0
        x_sim = state.copy()
        for t in range(horizon):
            u_t = u_seq[t]
            u_t_eff = u_t[:3]
            if mode == 'naive':
                x_sim = nominal_dynamics(x_sim, u_t_eff, dt)
            elif mode == 'raw':
                x_sim = augmented_dynamics(x_sim, u_t_eff, model, dt)
            elif mode == 'taylor':
                x_sim = taylor_approx_dynamics_with_linearization(x_sim, u_t_eff, model, dt, nn_input0, base_output, jacobian)
            else:
                raise ValueError("Unknown mode")
            desired_pos = target_traj[t]
            cost += np.sum((x_sim[0:3] - desired_pos) ** 2)
        if cost < best_cost:
            best_cost = cost
            best_u0 = u_seq[0]
    return best_u0

def true_dynamics(state, u, dt):
    c_d = 0.1
    p = state[0:3]
    q = state[3:7]
    v = state[7:10]
    r = state[10:13]
    p_next = p + dt * v
    v_next = v + dt * (u - c_d * v)
    return np.concatenate([p_next, q, v_next, r])

def simulate(mode, model, total_time=10.0, dt=0.05, horizon=10):
    num_steps = int(total_time / dt)
    # Initial state: origin, identity quaternion, zero velocity and rates.
    x = np.concatenate([np.zeros(3), np.array([1, 0, 0, 0]), np.zeros(3), np.zeros(3)])
    traj = [x.copy()]
    mpc_time_total = 0.0
    for i in tqdm(range(num_steps), desc=f"Simulating {mode} controller"):
        t_current = i * dt
        R = 5.0
        omega = 0.2
        target_traj = []
        for j in range(horizon):
            t_future = t_current + j * dt
            target_traj.append(np.array([R * np.cos(omega * t_future),
                                          R * np.sin(omega * t_future),
                                          1.0]))
        start_time = time.time()
        u_opt = mpc_controller_random_shooting(x, target_traj, horizon, dt, model, mode)
        mpc_time_total += time.time() - start_time
        x = true_dynamics(x, u_opt[:3], dt)
        traj.append(x.copy())
    avg_mpc_time = mpc_time_total / num_steps
    return np.array(traj), avg_mpc_time

def load_trained_model(model_path):
    checkpoint = torch.load(model_path, map_location="cpu")
    input_size = checkpoint["input_size"]
    hidden_size = checkpoint["hidden_size"]
    output_size = checkpoint["output_size"]
    hidden_layers = checkpoint["hidden_layers"]
    model = NormalizedMLP(input_size, hidden_size, output_size, hidden_layers)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

def compute_rmse(traj, dt):
    num_steps = traj.shape[0]
    R = 5.0
    omega = 0.2
    errors = []
    for i in range(num_steps):
        t = i * dt
        desired = np.array([R * np.cos(omega * t), R * np.sin(omega * t), 1.0])
        error = traj[i, 0:3] - desired
        errors.append(np.sum(error ** 2))
    rmse = np.sqrt(np.mean(errors))
    return rmse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="all",
                        choices=["naive", "raw", "taylor", "all"],
                        help="Which MPC controller to test")
    parser.add_argument("--model_path", type=str,
                        default="/home/rajanagarwal/neural-mpc/ros_dd_mpc/results/model_fitting/9137159/new/drag__motor_noise__noisy__no_payload.pt",
                        help="Path to the trained model checkpoint")
    parser.add_argument("--total_time", type=float, default=10.0,
                        help="Total simulation time")
    parser.add_argument("--dt", type=float, default=0.05,
                        help="Time step for simulation")
    parser.add_argument("--horizon", type=int, default=10,
                        help="MPC prediction horizon")
    args = parser.parse_args()

    model = load_trained_model(args.model_path)

    if args.mode == "all":
        modes = ["naive", "taylor"]
    else:
        modes = [args.mode]

    results = {}
    avg_times = {}
    rmses = {}

    for m in modes:
        print(f"\nSimulating mode: {m}")
        traj, avg_mpc_time = simulate(m, model, total_time=args.total_time, dt=args.dt, horizon=args.horizon)
        rmse = compute_rmse(traj, args.dt)
        results[m] = traj
        avg_times[m] = avg_mpc_time
        rmses[m] = rmse
        plt.figure()
        plt.plot(traj[:, 0], traj[:, 1], label=m)
        plt.xlabel("X position")
        plt.ylabel("Y position")
        plt.title(f"Trajectory for {m} MPC")
        plt.legend()
        plt.savefig(f"trajectory_{m}.png")
        plt.close()

    plt.figure()
    for m in modes:
        plt.plot(results[m][:, 0], results[m][:, 1], label=m)
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.title("Comparison of Trajectories")
    plt.legend()
    plt.savefig("trajectory_comparison.png")
    plt.show()

    print("RMSE and Average MPC Time (per step):")
    for m in modes:
        print(f"Mode: {m} - RMSE: {rmses[m]:.4f}, Avg MPC Time: {avg_times[m]*1000:.2f} ms")

if __name__ == "__main__":
    main()
