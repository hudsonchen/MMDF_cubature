
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import jax
import jax.numpy as jnp
from mmd_flow.distributions import Distribution
from mmd_flow.kernels import gaussian_kernel
from mmd_flow.mmd import mmd_fixed_target

plt.rcParams['axes.grid'] = True
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.labelsize'] = 20
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath, amsfonts}')
plt.tight_layout()

plt.rc('font', size=20)
plt.rc('lines', linewidth=2)
plt.rc('legend', fontsize=18, frameon=False)
plt.rc('xtick', labelsize=14, direction='in')
plt.rc('ytick', labelsize=14, direction='in')
plt.rc('figure', figsize=(6, 4))

def evaluate(args, trajectory, rate, rng_key):
    # Save the trajectory
    eval_freq = rate
    jnp.save(f'{args.save_path}/Ys.npy', trajectory[::eval_freq, :, :])

    T = trajectory.shape[0]
    Y = trajectory[0, :, :]
    kernel = gaussian_kernel(args.bandwidth)
    distribution = Distribution(kernel=kernel, means=jnp.array([0.0]), covariances=jnp.array([[1.0]]), weights=None)
    mmd_divergence = mmd_fixed_target(args, kernel, distribution)
    mmd_distance = jnp.sqrt(jax.vmap(mmd_divergence)(trajectory[::eval_freq, :, :], jax.random.split(rng_key, trajectory[::eval_freq, :, :].shape[0])))

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(mmd_distance, label='mmd')
    axs[0].set_yscale('log')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('MMD distance')
    plt.savefig(f'{args.save_path}/distance.png')
    return 

def save_animation_2d(args, trajectory, distribution, rate, save_path):
    num_timesteps = trajectory.shape[0]
    num_frames = max(num_timesteps // rate, 1)

    def update(frame):
        _animate_scatter.set_offsets(trajectory[frame * rate, :, :])
        return (_animate_scatter,)

    # create initial plot
    animate_fig, animate_ax = plt.subplots()
    # animate_fig.patch.set_alpha(0.)
    # plt.axis('off')
    # animate_ax.scatter(trajectory.Ys[:, 0], trajectory.Ys[:, 1], label='source')
    animate_ax.set_xlim(-5, 5)
    animate_ax.set_ylim(-5, 5)
    x_range = (-5, 5)
    y_range = (-5, 5)
    resolution = 100
    x_vals = jnp.linspace(x_range[0], x_range[1], resolution)
    y_vals = jnp.linspace(y_range[0], y_range[1], resolution)
    X, Y = jnp.meshgrid(x_vals, y_vals)
    grid = jnp.stack([X.ravel(), Y.ravel()], axis=1)
    logpdf = distribution.logpdf(grid).reshape(resolution, resolution)
    contour = animate_ax.contourf(X, Y, logpdf, levels=20, cmap='viridis')
    plt.colorbar(contour, ax=animate_ax, label="Log PDF")

    _animate_scatter = animate_ax.scatter(trajectory[0, :, 1], trajectory[0, :, 0], label='source')

    ani_kale = FuncAnimation(
        animate_fig,
        update,
        frames=num_frames,
        # init_func=init,
        blit=True,
        interval=50,
    )
    ani_kale.save(f'{save_path}/animation.mp4',
                   writer='ffmpeg', fps=20)
    return    
