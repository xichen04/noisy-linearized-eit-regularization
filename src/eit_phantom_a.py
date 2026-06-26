import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# Setting
NX = 40
NY = 40
N_ELECTRODES = 16

NOISE_LEVELS = [0.00, 0.01, 0.03, 0.05, 0.10]
TIKHONOV_ALPHAS = np.logspace(-8, -2, 40)
TV_ALPHAS = np.logspace(-10, -2, 40)

RUN_TIKHONOV = True
RUN_TV = True
SHOW_PLOTS = True

# Mesh
def create_square_mesh(nx=40, ny=40):
    xs = np.linspace(0.0, 1.0, nx + 1)
    ys = np.linspace(0.0, 1.0, ny + 1)
    nodes = np.array([[x, y] for y in ys for x in xs])

    def node_id(i, j):
        return j * (nx + 1) + i

    triangles = []
    for j in range(ny):
        for i in range(nx):
            n00 = node_id(i, j)
            n10 = node_id(i + 1, j)
            n01 = node_id(i, j + 1)
            n11 = node_id(i + 1, j + 1)
            triangles.append([n00, n10, n11])
            triangles.append([n00, n11, n01])

    return nodes, np.array(triangles, dtype=int)


def sigma0(x, y):
    return 1.0


def sigma_phantom_A(x, y, kappa=0.5, center=(0.5, 0.5), radius=0.2):
    """Piecewise-constant circular inclusion: sigma_A = 1 + kappa inside disk."""
    cx, cy = center
    if (x - cx) ** 2 + (y - cy) ** 2 < radius ** 2:
        return 1.0 + kappa
    return 1.0

# FEM model
def triangle_gradients_areas_centroids(nodes, triangles):
    grads_all = []
    areas = []
    centroids = []

    for tri in triangles:
        pts = nodes[tri]
        x1, y1 = pts[0]
        x2, y2 = pts[1]
        x3, y3 = pts[2]
        area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        b = np.array([y2 - y3, y3 - y1, y1 - y2])
        c = np.array([x3 - x2, x1 - x3, x2 - x1])
        grads = np.vstack([b, c]).T / (2.0 * area)
        grads_all.append(grads)
        areas.append(area)
        centroids.append(pts.mean(axis=0))

    return np.array(grads_all), np.array(areas), np.array(centroids)


def assemble_stiffness(nodes, triangles, sigma_func):
    n_nodes = len(nodes)
    rows = []
    cols = []
    data = []
    grads_all, areas, centroids = triangle_gradients_areas_centroids(nodes, triangles)

    for n, tri in enumerate(triangles):
        x, y = centroids[n]
        sigma_val = sigma_func(x, y)
        Ke = sigma_val * areas[n] * (grads_all[n] @ grads_all[n].T)
        for a in range(3):
            for b in range(3):
                rows.append(tri[a])
                cols.append(tri[b])
                data.append(Ke[a, b])

    return sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))


def build_electrodes(nodes, nx, ny, n_electrodes=16):
    """Split the square boundary into equal counterclockwise electrodes."""

    def node_id(i, j):
        return j * (nx + 1) + i

    boundary_edges = []

    for i in range(nx):
        boundary_edges.append((node_id(i, 0), node_id(i + 1, 0)))
    for j in range(ny):
        boundary_edges.append((node_id(nx, j), node_id(nx, j + 1)))
    for i in range(nx, 0, -1):
        boundary_edges.append((node_id(i, ny), node_id(i - 1, ny)))
    for j in range(ny, 0, -1):
        boundary_edges.append((node_id(0, j), node_id(0, j - 1)))

    total_edges = len(boundary_edges)
    if total_edges % n_electrodes != 0:
        raise ValueError("Boundary edges must divide evenly into electrodes.")

    edges_per_electrode = total_edges // n_electrodes
    electrodes = []
    for e in range(n_electrodes):
        start = e * edges_per_electrode
        end = (e + 1) * edges_per_electrode
        electrodes.append(boundary_edges[start:end])

    return electrodes


def electrode_length(nodes, electrode):
    return sum(np.linalg.norm(nodes[j] - nodes[i]) for i, j in electrode)


def assemble_neumann_load(nodes, electrodes, k):
    """Adjacent drive: +1 on electrode k and -1 on electrode k+1."""
    n_nodes = len(nodes)
    rhs = np.zeros(n_nodes)
    n_electrodes = len(electrodes)
    inject = k
    withdraw = (k + 1) % n_electrodes
    lengths = [electrode_length(nodes, e) for e in electrodes]

    for e_idx, electrode in enumerate(electrodes):
        if e_idx == inject:
            g_value = 1.0 / lengths[e_idx]
        elif e_idx == withdraw:
            g_value = -1.0 / lengths[e_idx]
        else:
            g_value = 0.0

        for i, j in electrode:
            edge_len = np.linalg.norm(nodes[j] - nodes[i])
            rhs[i] += g_value * edge_len / 2.0
            rhs[j] += g_value * edge_len / 2.0

    return rhs


def make_grounded_solver(K):
    """Return a reusable solver for K u = rhs with sum(u)=0 grounding."""
    n = K.shape[0]
    ones = np.ones((n, 1))
    A = sp.bmat([[K, ones], [ones.T, None]], format="csc")
    solve = spla.factorized(A)

    def solve_rhs(rhs):
        b_aug = np.concatenate([rhs, [0.0]])
        sol = solve(b_aug)
        return sol[:n]

    return solve_rhs


def compute_electrode_averages(nodes, u, electrodes):
    averages = []
    for electrode in electrodes:
        numerator = 0.0
        denominator = 0.0
        for i, j in electrode:
            edge_len = np.linalg.norm(nodes[j] - nodes[i])
            numerator += edge_len * (u[i] + u[j]) / 2.0
            denominator += edge_len
        averages.append(numerator / denominator)
    return np.array(averages)


def adjacent_voltage_differences(U):
    return U - np.roll(U, -1)


def compute_all_solutions_and_data(nodes, triangles, electrodes, sigma_func):
    K = assemble_stiffness(nodes, triangles, sigma_func)
    solve = make_grounded_solver(K)

    all_u = []
    all_V = []
    for k in range(len(electrodes)):
        rhs = assemble_neumann_load(nodes, electrodes, k)
        u = solve(rhs)
        U = compute_electrode_averages(nodes, u, electrodes)
        V = adjacent_voltage_differences(U)
        all_u.append(u)
        all_V.append(V)

    return np.array(all_u), np.array(all_V).reshape(-1)

# Sensitivity matrix
def build_sensitivity_matrix(nodes, triangles, ref_solutions):
    """
    Build J in R^{256 x Ntriangles} using reference sigma0 solutions.

    J[(k,l), n] = - area_n * grad(u_k)^T grad(u_l) on triangle n.
    """
    n_patterns = ref_solutions.shape[0]
    n_triangles = len(triangles)
    grads_all, areas, centroids = triangle_gradients_areas_centroids(nodes, triangles)
    J = np.zeros((n_patterns * n_patterns, n_triangles))

    for n, tri in enumerate(triangles):
        grads = grads_all[n]
        grad_u = np.zeros((n_patterns, 2))
        for k in range(n_patterns):
            grad_u[k] = ref_solutions[k, tri] @ grads

        block = -areas[n] * (grad_u @ grad_u.T)
        J[:, n] = block.reshape(-1)

    return J, centroids


def compute_true_z_A_on_triangles(centroids, kappa=0.5, center=(0.5, 0.5), radius=0.2):
    z_true_A = np.zeros(len(centroids))
    cx, cy = center
    for i, (x, y) in enumerate(centroids):
        if (x - cx) ** 2 + (y - cy) ** 2 < radius ** 2:
            z_true_A[i] = kappa
    return z_true_A

# Noise and Tikhonov
def add_noise(data, noise_level, seed=0):
    """noise_level=0.01 means 1 percent relative noise."""
    data = np.asarray(data)
    if noise_level == 0:
        return data.copy()
    rng = np.random.default_rng(seed)
    eta = rng.standard_normal(data.shape)
    return data + noise_level * np.linalg.norm(data) / np.linalg.norm(eta) * eta


def tikhonov_dual(J, data, alpha):
    """
    Solve min_z 0.5 ||Jz-data||^2 + alpha/2 ||z||^2 by the dual system.

    This solves a 256 x 256 system instead of a 3200 x 3200 system.
    """
    m = J.shape[0]
    A = J @ J.T + alpha * np.eye(m)
    w = np.linalg.solve(A, data)
    return J.T @ w


def relative_error(z_rec, z_true):
    return np.linalg.norm(z_rec - z_true) / np.linalg.norm(z_true)


def choose_best_alpha_tikhonov(J, data, z_true, alpha_grid):
    best_alpha = None
    best_z = None
    best_error = np.inf
    errors = []

    for alpha in alpha_grid:
        z_rec = tikhonov_dual(J, data, alpha)
        err = relative_error(z_rec, z_true)
        errors.append(err)
        if err < best_error:
            best_error = err
            best_alpha = alpha
            best_z = z_rec

    return best_alpha, best_z, best_error, np.array(errors)

# TV
def build_triangle_adjacency(triangles):
    edge_to_triangle = {}
    graph_edges = []

    for t_idx, tri in enumerate(triangles):
        local_edges = [
            tuple(sorted((tri[0], tri[1]))),
            tuple(sorted((tri[1], tri[2]))),
            tuple(sorted((tri[2], tri[0]))),
        ]
        for edge in local_edges:
            if edge in edge_to_triangle:
                graph_edges.append((edge_to_triangle[edge], t_idx))
            else:
                edge_to_triangle[edge] = t_idx

    return np.array(graph_edges, dtype=int)


def build_difference_matrix(n_triangles, graph_edges):
    rows = []
    cols = []
    data = []
    for r, (i, j) in enumerate(graph_edges):
        rows.extend([r, r])
        cols.extend([i, j])
        data.extend([1.0, -1.0])
    return sp.csr_matrix((data, (rows, cols)), shape=(len(graph_edges), n_triangles))


def tv_pd_ipm(
    J,
    data,
    D,
    alpha,
    beta0=1e-3,
    beta_min=1e-8,
    beta_factor=0.2,
    max_outer=8,
    max_inner=25,
    tol=1e-6,
    cg_tol=1e-5,
    cg_maxiter=400,
    damping=1e-8,
    dual_bound=1.0 - 1e-10,
    z0=None,
    y0=None,
    verbose=False,
):
    """
    Matrix-free Borsic-style primal-dual IPM for
        min_z 0.5 ||Jz-data||^2 + alpha * sum(abs(Dz)).
    """
    D = D.tocsr()
    data = np.asarray(data, dtype=float).reshape(-1)
    J = np.asarray(J, dtype=float)

    n = J.shape[1]
    m = D.shape[0]
    z = np.zeros(n) if z0 is None else np.asarray(z0, dtype=float).reshape(-1).copy()
    y = np.zeros(m) if y0 is None else np.asarray(y0, dtype=float).reshape(-1).copy()

    if data.size != J.shape[0]:
        raise ValueError(f"data length {data.size} does not match J rows {J.shape[0]}.")
    if D.shape[1] != n:
        raise ValueError(f"D columns {D.shape[1]} do not match J columns {n}.")

    J_diag = np.sum(J * J, axis=0)

    def compute_residuals(z_vec, y_vec, beta_value):
        Dz = D @ z_vec
        E = np.sqrt(Dz * Dz + beta_value)
        r1 = J.T @ (J @ z_vec - data) + alpha * (D.T @ y_vec)
        r2 = Dz - E * y_vec
        total = np.hypot(np.linalg.norm(r1), np.linalg.norm(r2))
        return Dz, E, r1, r2, total

    history = []
    beta = beta0

    for outer in range(max_outer):
        for inner in range(max_inner):
            Dz, E, r1, r2, res = compute_residuals(z, y, beta)
            history.append(
                {
                    "outer": outer,
                    "inner": inner,
                    "beta": beta,
                    "residual": res,
                    "stationarity": np.linalg.norm(r1),
                    "complementarity": np.linalg.norm(r2),
                    "dual_max": np.max(np.abs(y)) if y.size else 0.0,
                    "cg_info": None,
                    "step": None,
                }
            )

            if verbose:
                print(
                    f"outer={outer}, inner={inner}, beta={beta:.1e}, "
                    f"res={res:.2e}, dual_max={history[-1]['dual_max']:.6f}",
                    flush=True,
                )

            if res < tol:
                break

            K = 1.0 - y * Dz / E
            W = np.maximum(K / E, 1e-10)
            rhs = -r1 - alpha * (D.T @ (r2 / E))

            def H_matvec(v):
                return J.T @ (J @ v) + alpha * (D.T @ (W * (D @ v))) + damping * v

            H_op = spla.LinearOperator((n, n), matvec=H_matvec, dtype=float)
            D_diag = np.asarray(D.power(2).T @ W).ravel()
            M_diag = np.maximum(J_diag + alpha * D_diag + damping, 1e-12)
            M_op = spla.LinearOperator((n, n), matvec=lambda v: v / M_diag, dtype=float)

            dz, info = spla.cg(H_op, rhs, M=M_op, rtol=cg_tol, atol=0.0, maxiter=cg_maxiter)
            history[-1]["cg_info"] = int(info)
            if info != 0 and verbose:
                print(f"CG warning: info={info}", flush=True)
            if not np.all(np.isfinite(dz)):
                raise FloatingPointError("CG returned a non-finite Newton step.")

            dy = (r2 + K * (D @ dz)) / E

            step = 1.0
            accepted = False
            old_res = res
            for _ in range(24):
                z_new = z + step * dz
                y_new = y + step * dy
                if np.all(np.isfinite(z_new)) and np.all(np.isfinite(y_new)) and np.max(np.abs(y_new)) < dual_bound:
                    _, _, _, _, new_res = compute_residuals(z_new, y_new, beta)
                    if np.isfinite(new_res) and (new_res <= old_res or step <= 1e-4):
                        accepted = True
                        break
                step *= 0.5

            if not accepted:
                step = 1e-4
                z_new = z + step * dz
                y_new = np.clip(y + step * dy, -dual_bound, dual_bound)

            history[-1]["step"] = step
            z, y = z_new, y_new

            if np.linalg.norm(step * dz) <= tol * (1.0 + np.linalg.norm(z)):
                break

        if beta <= beta_min:
            break
        beta = max(beta * beta_factor, beta_min)

    return z, y, history


def run_tv_reconstruction_A(J, Delta_V_A, D, z_true_A, noise_levels, tv_alphas):
    if J.shape[0] != Delta_V_A.size:
        raise ValueError("J rows must match Delta_V_A length.")
    if J.shape[1] != z_true_A.size:
        raise ValueError("J columns must match z_true_A length.")
    if D.shape[1] != z_true_A.size:
        raise ValueError("D columns must match z_true_A length.")

    tv_results_A = {}
    for noise in noise_levels:
        print(f"\nStarting TV reconstruction for noise {100 * noise:.0f}%", flush=True)
        data_noisy = add_noise(Delta_V_A, noise, seed=123)

        best_error = np.inf
        best_alpha = None
        best_z = None
        best_y = None
        best_history = None
        
        alpha_errors = []


        # Warm-start across alpha values for the same noise level.
        z_start = None
        y_start = None
        for k, alpha in enumerate(tv_alphas):
            print(f"  alpha {k + 1}/{len(tv_alphas)} = {alpha:.2e}", flush=True)
            z_tv, y_tv, history = tv_pd_ipm(
                J,
                data_noisy,
                D,
                alpha=alpha,
                beta0=1e-3,
                beta_min=1e-8,
                beta_factor=0.2,
                max_outer=8,
                max_inner=25,
                tol=1e-6,
                cg_tol=1e-5,
                cg_maxiter=400,
                z0=z_start,
                y0=y_start,
                verbose=False,
            )
            z_start, y_start = z_tv, y_tv

            err = relative_error(z_tv, z_true_A)
            alpha_errors.append(err)
            print(f"    error={err:.4f}, final residual={history[-1]['residual']:.2e}", flush=True)
            
            if err < best_error:
                best_error = err
                best_alpha = alpha
                best_z = z_tv.copy()
                best_y = y_tv.copy()
                best_history = history

        tv_results_A[noise] = {
            "z": best_z,
            "y": best_y,
            "alpha": best_alpha,
            "error": best_error,
            "history": best_history,
            "alpha_errors": np.array(alpha_errors),
        }

        print(
            f"Best TV noise {100 * noise:.0f}%: "
            f"alpha={best_alpha:.2e}, "
            f"relative error={best_error:.4f}, "
            f"final residual={best_history[-1]['residual']:.2e}",
            flush=True,
        )

    return tv_results_A

# Plot
# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_conductivity(nodes, triangles, sigma_func, title):
    sigma_values = np.array([sigma_func(x, y) for x, y in nodes])

    plt.figure(figsize=(6, 5))
    plt.tripcolor(
        nodes[:, 0],
        nodes[:, 1],
        triangles,
        sigma_values,
        shading="gouraud",
    )
    plt.colorbar(label=r"$\sigma(x)$")
    plt.gca().set_aspect("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_triangle_values(
    nodes,
    triangles,
    values,
    title,
    vmin=None,
    vmax=None,
    label=r"$\Delta\sigma$",
):
    plt.figure(figsize=(6, 5))
    plt.tripcolor(
        nodes[:, 0],
        nodes[:, 1],
        triangles,
        facecolors=values,
        shading="flat",
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(label=label)
    plt.gca().set_aspect("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_error_comparison_A(
    noise_levels,
    tikhonov_results_A,
    tv_results_A,
    title="Phantom A: Tikhonov vs TV",
):
    noise_percent = 100 * np.array(noise_levels)

    tikh_errors = [tikhonov_results_A[n]["error"] for n in noise_levels]
    tv_errors = [tv_results_A[n]["error"] for n in noise_levels]

    plt.figure(figsize=(6, 4))

    plt.plot(
        noise_percent,
        tikh_errors,
        marker="o",
        linewidth=2,
        label="Tikhonov",
    )

    plt.plot(
        noise_percent,
        tv_errors,
        marker="s",
        linewidth=2,
        label="TV",
    )

    plt.xlabel("Noise level (%)")
    plt.ylabel("Relative error")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_tikhonov_alpha_errors(
    alpha_grid,
    tikhonov_results,
    title="Tikhonov alpha selection",
):
    plt.figure(figsize=(6, 4))

    for noise, r in tikhonov_results.items():
        plt.semilogx(
            alpha_grid,
            r["alpha_errors"],
            marker="o",
            linewidth=2,
            label=f"{100 * noise:.0f}% noise",
        )

    plt.xlabel(r"$\alpha$")
    plt.ylabel("Relative error")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_tv_alpha_errors(
    alpha_grid,
    tv_results,
    title="TV alpha selection",
):
    plt.figure(figsize=(6, 4))

    for noise, r in tv_results.items():
        plt.semilogx(
            alpha_grid,
            r["alpha_errors"],
            marker="s",
            linewidth=2,
            label=f"{100 * noise:.0f}% noise",
        )

    plt.xlabel(r"$\alpha$")
    plt.ylabel("Relative error")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_tv_history(tv_results_A, noise_levels):
    for noise in noise_levels:
        history = tv_results_A[noise]["history"]

        residuals = [h["residual"] for h in history]
        stationarity = [h["stationarity"] for h in history]
        complementarity = [h["complementarity"] for h in history]

        plt.figure(figsize=(6, 4))

        plt.semilogy(
            residuals,
            linewidth=2,
            label="total residual",
        )
        plt.semilogy(
            stationarity,
            linewidth=2,
            label="stationarity",
        )
        plt.semilogy(
            complementarity,
            linewidth=2,
            label="complementarity",
        )

        plt.xlabel("Newton iteration")
        plt.ylabel("Residual")
        plt.title(f"TV PD-IPM convergence, noise={100 * noise:.0f}%")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def compute_relative_residual(J, z_rec, data):
    return np.linalg.norm(J @ z_rec - data) / np.linalg.norm(data)


def print_error_comparison_table(noise_levels, tikhonov_results, tv_results):
    print("\nError comparison table for Phantom A")
    print("-" * 72)
    print(
        f"{'Noise':>8} | "
        f"{'Tik alpha':>12} | "
        f"{'Tik error':>10} | "
        f"{'TV alpha':>12} | "
        f"{'TV error':>10}"
    )
    print("-" * 72)

    for noise in noise_levels:
        tik = tikhonov_results[noise]
        tv = tv_results[noise]

        print(
            f"{100 * noise:7.0f}% | "
            f"{tik['alpha']:12.2e} | "
            f"{tik['error']:10.4f} | "
            f"{tv['alpha']:12.2e} | "
            f"{tv['error']:10.4f}"
        )

    print("-" * 72)


def print_error_residual_comparison_table(
    noise_levels,
    J,
    tikhonov_results,
    tv_results,
):
    print("\nError and data-fit comparison table for Phantom A")
    print("-" * 104)
    print(
        f"{'Noise':>8} | "
        f"{'Tik alpha':>12} | "
        f"{'Tik error':>10} | "
        f"{'Tik residual':>12} | "
        f"{'TV alpha':>12} | "
        f"{'TV error':>10} | "
        f"{'TV residual':>12}"
    )
    print("-" * 104)

    for noise in noise_levels:
        tik = tikhonov_results[noise]
        tv = tv_results[noise]

        data = tik["data"]

        tik_residual = compute_relative_residual(
            J,
            tik["z_rec"],
            data,
        )

        tv_residual = compute_relative_residual(
            J,
            tv["z"],
            data,
        )

        print(
            f"{100 * noise:7.0f}% | "
            f"{tik['alpha']:12.2e} | "
            f"{tik['error']:10.4f} | "
            f"{tik_residual:12.4e} | "
            f"{tv['alpha']:12.2e} | "
            f"{tv['error']:10.4f} | "
            f"{tv_residual:12.4e}"
        )

    print("-" * 104)
# main
def main():
    print("Building mesh and electrodes...")
    nodes, triangles = create_square_mesh(NX, NY)
    electrodes = build_electrodes(nodes, NX, NY, n_electrodes=N_ELECTRODES)

    print("Computing reference and Phantom A forward data...")
    u_ref, V_ref = compute_all_solutions_and_data(nodes, triangles, electrodes, sigma0)
    _, V_A = compute_all_solutions_and_data(nodes, triangles, electrodes, sigma_phantom_A)
    Delta_V_A = V_A - V_ref

    print("V_ref shape:", V_ref.shape)
    print("V_A shape:", V_A.shape)
    print("Delta_V_A shape:", Delta_V_A.shape)
    print("||Delta_V_A||_2:", np.linalg.norm(Delta_V_A))

    if SHOW_PLOTS:
        plot_conductivity(nodes, triangles, sigma_phantom_A, "True Phantom A conductivity")

    print("Building sensitivity matrix J...")
    J, centroids = build_sensitivity_matrix(nodes, triangles, u_ref)
    z_true_A = compute_true_z_A_on_triangles(centroids)

    print("J shape:", J.shape)
    print("Number of unknowns:", len(triangles))

    assert J.shape[0] == Delta_V_A.shape[0]
    assert J.shape[1] == z_true_A.shape[0]

    pred = J @ z_true_A
    rel_lin_error = np.linalg.norm(pred - Delta_V_A) / np.linalg.norm(Delta_V_A)
    rel_lin_error_flipped = np.linalg.norm(-pred - Delta_V_A) / np.linalg.norm(Delta_V_A)

    print("||J z_true_A - Delta_V_A|| / ||Delta_V_A|| =", rel_lin_error)
    print("||-J z_true_A - Delta_V_A|| / ||Delta_V_A|| =", rel_lin_error_flipped)

    
    tikhonov_results_A = {}

    if RUN_TIKHONOV:
        print("\nRunning Tikhonov reconstructions...")

        for noise in NOISE_LEVELS:
            print(f"\nTikhonov reconstruction for noise {100 * noise:.0f}%")

            data_noisy = add_noise(Delta_V_A, noise, seed=123)

            best_alpha, z_rec, best_err, errors = choose_best_alpha_tikhonov(
                J,
                data_noisy,
                z_true_A,
                TIKHONOV_ALPHAS,
            )

            tikhonov_results_A[noise] = {
                "alpha": best_alpha,
                "z_rec": z_rec,
                "error": best_err,
                "alpha_errors": errors,
                "data": data_noisy,
            }

            print(f"  best alpha = {best_alpha:.2e}")
            print(f"  relative error = {best_err:.4f}")

        if SHOW_PLOTS:
            vmin = min(
                z_true_A.min(),
                min(r["z_rec"].min() for r in tikhonov_results_A.values()),
            )
            vmax = max(
                z_true_A.max(),
                max(r["z_rec"].max() for r in tikhonov_results_A.values()),
            )

            plot_triangle_values(
                nodes,
                triangles,
                z_true_A,
                "True Phantom A: sigma_A - 1",
                vmin,
                vmax,
            )

            for noise in NOISE_LEVELS:
                r = tikhonov_results_A[noise]
                plot_triangle_values(
                    nodes,
                    triangles,
                    r["z_rec"],
                    f"Tikhonov Phantom A, noise={100 * noise:.0f}%, "
                    f"alpha={r['alpha']:.1e}, error={r['error']:.3f}",
                    vmin,
                    vmax,
                )

    
    tv_results_A = {}

    if RUN_TV:
        print("\nBuilding TV difference matrix D...")

        graph_edges = build_triangle_adjacency(triangles)
        D = build_difference_matrix(len(triangles), graph_edges)

        print("D shape:", D.shape)
        print("\nRunning TV reconstructions...")

        tv_results_A = run_tv_reconstruction_A(
            J,
            Delta_V_A,
            D,
            z_true_A,
            NOISE_LEVELS,
            TV_ALPHAS,
        )

        if SHOW_PLOTS:
            vmin = min(
                z_true_A.min(),
                min(tv_results_A[n]["z"].min() for n in NOISE_LEVELS),
            )
            vmax = max(
                z_true_A.max(),
                max(tv_results_A[n]["z"].max() for n in NOISE_LEVELS),
            )

            plot_triangle_values(
                nodes,
                triangles,
                z_true_A,
                "True Phantom A: sigma_A - 1",
                vmin,
                vmax,
            )

            for noise in NOISE_LEVELS:
                r = tv_results_A[noise]
                plot_triangle_values(
                    nodes,
                    triangles,
                    r["z"],
                    f"TV Phantom A, noise={100 * noise:.0f}%, "
                    f"alpha={r['alpha']:.1e}, error={r['error']:.3f}",
                    vmin,
                    vmax,
                    label=r"reconstructed $\Delta\sigma$",
                )

            plot_tv_history(tv_results_A, NOISE_LEVELS)

    
    if RUN_TIKHONOV and RUN_TV:
        print_error_comparison_table(
            NOISE_LEVELS,
            tikhonov_results_A,
            tv_results_A,
        )

        print_error_residual_comparison_table(
            NOISE_LEVELS,
            J,
            tikhonov_results_A,
            tv_results_A,
        )

    # -------------------------------------------------------------------------
    # Final comparison plots
    # -------------------------------------------------------------------------
    if RUN_TIKHONOV and RUN_TV and SHOW_PLOTS:
        plot_error_comparison_A(
            NOISE_LEVELS,
            tikhonov_results_A,
            tv_results_A,
            title="Phantom A: Tikhonov vs TV",
        )

        plot_tikhonov_alpha_errors(
            TIKHONOV_ALPHAS,
            tikhonov_results_A,
            title="Tikhonov alpha selection for Phantom A",
        )

        plot_tv_alpha_errors(
            TV_ALPHAS,
            tv_results_A,
            title="TV alpha selection for Phantom A",
        )

    return {
        "nodes": nodes,
        "triangles": triangles,
        "electrodes": electrodes,
        "V_ref": V_ref,
        "V_A": V_A,
        "Delta_V_A": Delta_V_A,
        "J": J,
        "z_true_A": z_true_A,
        "tikhonov_results_A": tikhonov_results_A,
        "tv_results_A": tv_results_A,
    }
if __name__ == "__main__":
    results = main()
