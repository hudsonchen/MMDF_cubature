
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


def save_animation_2d(args, trajectory, kernel, distribution, rate, rng_key, save_path):
    T = trajectory.shape[0]
    Y = trajectory[0, :, :]
    mmd_divergence = mmd_fixed_target(args, kernel, distribution)
    mmd_distance = jnp.sqrt(jax.vmap(mmd_divergence)(trajectory[::rate, :, :], jax.random.split(rng_key, trajectory[::rate, :, :].shape[0])))

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(mmd_distance, label='mmd')
    axs[0].set_yscale('log')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('MMD distance')
    plt.savefig(f'{args.save_path}/distance.png')
    jnp.save(f'{args.save_path}/Ys.npy', trajectory[::rate, :, :])

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
    logpdf = jnp.log(distribution.pdf(grid).reshape(resolution, resolution))
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


def evaluate_integral(args, distribution, trajectory):
    true_value = distribution.integral()
    iid_samples = distribution.sample(args.particle_num, jax.random.PRNGKey(0))
    iid_emprical = jnp.mean(distribution.integrand(iid_samples))
    mmd_flow_samples = trajectory[-1, :, :]
    mmd_flow_empirical = jnp.mean(distribution.integrand(mmd_flow_samples))
    print(f'True value: {true_value}')
    print(f'IID empirical: {iid_emprical}')
    print(f'MMD flow empirical: {mmd_flow_empirical}')
    mmd_flow_err = jnp.abs(true_value - mmd_flow_empirical)
    iid_err = jnp.abs(true_value - iid_emprical)
    jnp.save(f'{args.save_path}/mmd_flow_err.npy', mmd_flow_err)
    jnp.save(f'{args.save_path}/iid_err.npy', iid_err)
    return


def exact_integral(args, distribution, rate, trajectory):
    # Verify that \int \nabla_{x_i} k(x_i, y) d \mu(y) = \sum_{j=1}^N \nabla_{x_i} k(x_i, y_j)
    # Only works for d dimensional unit Gaussian distribution
    ell = args.bandwidth
    dim = distribution.means.shape[1]
    diff_exact_list = []

    # This is the integral with exact approximation
    for i in range(0, trajectory.shape[0], rate):
        x = trajectory[i, 0, :]
        # Compute the closed-form solution using the provided formula
        factor1 = (1/2 / (1/2 + ell**-2))**(dim/2)
        factor2 = - (2 * ell**-2) / (1 + 2 * ell**-2)
        exp_term = jnp.exp(- (ell**-2) / (1 + 2 * ell**-2) * jnp.linalg.norm(x)**2)
        closed_form = factor1 * factor2 * exp_term * x

        # Compute the integral using numerical integration
        y = trajectory[i, :, :]
        part_1 = jnp.exp(- ell**-2 * jnp.linalg.norm(x[None, :] - y))
        part_2 = 2 * ell**-2 * (x[None, :] - y)
        empirical_form = part_1 * part_2

        diff_exact = jnp.linalg.norm(closed_form - empirical_form)
        diff_exact_list.append(diff_exact)

    # This is the integral that does not
    diff_inexact_list = []
    for i in range(0, trajectory.shape[0], rate):
        closed_form = 0
        empirical = trajectory[i, :, :].mean(0)
        diff_inexact = jnp.linalg.norm(closed_form - empirical)
        diff_inexact_list.append(diff_inexact)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(diff_exact_list, label='Exact')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel(r'$|I - \hat{I}|$')
    axs[0].plot(diff_inexact_list, label='Inexact')
    axs[0].legend()
    plt.savefig(f'{args.save_path}/integral_error_exact.png')
    return
