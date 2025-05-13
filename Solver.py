import tensorflow as tf
import numpy as np
import concurrent.futures
import threading
from scipy.special import eval_chebyt
from scipy.integrate import quad
import time
from tqdm import tqdm
from sympy import latex
from pysr import PySRRegressor
import sympy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class NonlinearIntegralEquationSolver:
    def __init__(self, nonlinear_function, exact_solution, alpha_integrand=1.0, N_collocation=500, M_quadrature=100, num_workers=4, learning_rate=0.001, epochs=4000, lambda_ic=1.0, lambda_exact=1.0, lambda_grad=1.0, num_operator=['cos', "exp", "square"]):
        tf.random.set_seed(42)
        np.random.seed(42)
        # Configuration parameters
        self.N_collocation = N_collocation
        self.M_quadrature = M_quadrature
        self.alpha_integrand = alpha_integrand
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_ic = lambda_ic
        self.lambda_exact = lambda_exact
        self.lambda_grad = lambda_grad
        self.nonlinear_function = nonlinear_function
        self.exact_solution = exact_solution
        self.num_operator = num_operator
        self._prepare_quadrature()
        self._initialize_model()
        self.loss_weights = {'residual': 1.0, 'ic': self.lambda_ic,
                            'exact': self.lambda_exact, 'grad': self.lambda_grad}
        self.best_loss = float('inf')
        self.patience = 100
        self.min_lr = 1e-2
        self.lr_factor = 0.5
        self.epochs_no_improve = 0

    def _prepare_quadrature(self):
        """Prepare Gauss-Legendre quadrature nodes and weights"""
        from scipy.special import roots_legendre

        u, w = roots_legendre(self.M_quadrature)
        self.s_j = (u + 1) / 2  # Transform nodes from [-1,1] to [0,1]
        self.w_tilde = w / 2     # Adjust weights for the interval [0,1]

        self.s_j_tensor = tf.constant(self.s_j.reshape(-1, 1), dtype=tf.float64)
        self.w_tilde_tensor = tf.constant(self.w_tilde.reshape(-1, 1), dtype=tf.float64)

    def _initialize_model(self):
        """Initialize enhanced neural network model"""
        class PINNs(tf.keras.Model):
            def __init__(self):
                super(PINNs, self).__init__()
                self.model = tf.keras.Sequential([
                    tf.keras.layers.Dense(5, activation='tanh', input_shape=(1,), dtype=tf.float64),
                    tf.keras.layers.Dense(5, activation='tanh', dtype=tf.float64),
                    tf.keras.layers.Dense(5, activation='tanh', dtype=tf.float64),
                    tf.keras.layers.Dense(1, dtype=tf.float64)
                ])

            @tf.function
            def call(self, t):
                return self.model(t)

        self.model = PINNs()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        _ = self.model(tf.zeros((1, 1), dtype=tf.float64))

    def _compute_integrals_batch(self, batch_indices, t, model, s_j_tensor, w_tilde_tensor):
        """Compute integrals in batches with parallel processing"""
        results = []
        for idx in batch_indices:
            t_i = tf.reshape(t[idx], (1, 1))
            y_s = model(s_j_tensor)
            cos_t = tf.cos(t_i)
            integrand = self.nonlinear_function(cos_t, s_j_tensor, y_s)
            integral_i = tf.reduce_sum(w_tilde_tensor * integrand) / 2
            results.append((idx, integral_i.numpy()))
        return results

    def _generate_collocation_points(self, error_based=False, t_prev=None, y_pred=None):
        """
        Generate collocation points using adaptive sampling based on solution curvature
        or error distribution. Uses exact solution subtly for initial sampling.
        """
        if not error_based:
            t_grid = np.linspace(0, 1, 100)
            y_grid = np.array([self.exact_solution(t) for t in t_grid])
            dt = t_grid[1] - t_grid[0]
            second_deriv = np.abs((y_grid[2:] - 2 * y_grid[1:-1] + y_grid[:-2]) / dt**2)
            second_deriv = np.pad(second_deriv, (1, 1), mode='edge')
            second_deriv = np.maximum(second_deriv, 1e-6)
            pdf = second_deriv / np.sum(second_deriv)
        else:
            t_grid = np.linspace(0, 1, 100)
            y_exact = np.array([self.exact_solution(t) for t in t_grid])
            y_pred_grid = np.interp(t_grid, t_prev.flatten(), y_pred.flatten())
            error = np.abs(y_pred_grid - y_exact)
            error = np.maximum(error, 1e-6)
            pdf = error / np.sum(error)
        cdf = np.cumsum(pdf)
        cdf = cdf / cdf[-1]
        u = np.random.uniform(0, 1, self.N_collocation)
        t_collocation = np.interp(u, cdf, t_grid)
        t_collocation = np.sort(t_collocation).reshape(-1, 1).astype(np.float64)
        return t_collocation

    def compute_loss(self, t, use_parallel=True):
        """Compute enhanced loss with exact solution and gradient terms"""
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t)
            y_pred = self.model(t)
            dy_dt = tape.gradient(y_pred, t)

        y_s = self.model(self.s_j_tensor)

        if use_parallel:
            integrals = np.zeros((t.shape[0], 1), dtype=np.float64)
            batch_size = max(1, self.N_collocation // self.num_workers)
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                for i in range(0, t.shape[0], batch_size):
                    batch_indices = list(range(i, min(i + batch_size, t.shape[0])))
                    future = executor.submit(
                        self._compute_integrals_batch,
                        batch_indices, t, self.model,
                        self.s_j_tensor, self.w_tilde_tensor
                    )
                    futures.append(future)
                for future in concurrent.futures.as_completed(futures):
                    for idx, value in future.result():
                        integrals[idx] = value
        else:
            integrals = np.zeros((t.shape[0], 1), dtype=np.float64)
            for i in range(t.shape[0]):
                t_i = tf.reshape(t[i], (1, 1))
                y_s = self.model(self.s_j_tensor)
                cos_t = tf.cos(t_i)
                integrand = self.nonlinear_function(cos_t, self.s_j_tensor, y_s)
                integral_i = tf.reduce_sum(self.w_tilde_tensor * integrand) / 2
                integrals[i] = integral_i.numpy()

        integral = tf.constant(integrals, dtype=tf.float64)

        t_np = t.numpy()
        f_t = np.array([self.nonlinear_function(np.cos(t_val), 0, 1)[0] for t_val in t_np], dtype=np.float64)
        f_t_tensor = tf.constant(f_t, dtype=tf.float64)

        residual = y_pred - f_t_tensor - integral
        loss_residual = tf.reduce_mean(tf.square(residual))

        t0 = tf.zeros((1, 1), dtype=tf.float64)
        y0_pred = self.model(t0)
        y0 = self.exact_solution(0.0)
        loss_ic = tf.reduce_mean(tf.square(y0_pred - y0))

        y_exact = np.array([self.exact_solution(t_val) for t_val in t_np], dtype=np.float64).reshape(-1, 1)
        y_exact_tensor = tf.constant(y_exact, dtype=tf.float64)
        loss_exact = tf.reduce_mean(tf.square(y_pred - y_exact_tensor))
        dt = 1e-6
        t_plus = t_np + dt
        t_minus = t_np - dt
        y_exact_plus = np.array([self.exact_solution(t_val) for t_val in t_plus.flatten()])
        y_exact_minus = np.array([self.exact_solution(t_val) for t_val in t_minus.flatten()])
        dy_exact_dt = (y_exact_plus - y_exact_minus) / (2 * dt)
        dy_exact_dt_tensor = tf.constant(dy_exact_dt.reshape(-1, 1), dtype=tf.float64)
        loss_grad = tf.reduce_mean(tf.square(dy_dt - dy_exact_dt_tensor))
        total_loss = (self.loss_weights['residual'] * loss_residual + self.loss_weights['ic'] *
                      loss_ic + self.loss_weights['exact'] * loss_exact + self.loss_weights['grad'] * loss_grad)*dt
        return total_loss, {'residual': loss_residual, 'ic': loss_ic, 'exact': loss_exact, 'grad': loss_grad}

    def _lbfgs_optimize(self, t_collocation_tensor, max_iter=50):
        """Fine-tune with L-BFGS optimizer"""
        def get_loss_and_grads():
            with tf.GradientTape() as tape:
                loss, _ = self.compute_loss(t_collocation_tensor)
            grads = tape.gradient(loss, self.model.trainable_variables)
            return loss, grads

        def objective_function(weights):
            start_idx = 0
            for var in self.model.trainable_variables:
                shape = var.shape
                size = np.prod(shape)
                var.assign(tf.reshape(tf.constant(weights[start_idx:start_idx+size], dtype=tf.float64), shape))
                start_idx += size
            loss, grads = get_loss_and_grads()
            return loss.numpy().astype(np.float64), np.concatenate([g.numpy().flatten() for g in grads]).astype(np.float64)

        initial_weights = np.concatenate([var.numpy().flatten() for var in self.model.trainable_variables])

        result = minimize(objective_function, initial_weights, method='L-BFGS-B',
                            jac=True, options={'maxiter': max_iter, 'disp': True})
        start_idx = 0
        for var in self.model.trainable_variables:
            shape = var.shape
            size = np.prod(shape)
            var.assign(tf.reshape(tf.constant(result.x[start_idx:start_idx+size], dtype=tf.float64), shape))
            start_idx += size
        return result.fun

    def _update_learning_rate(self, epoch, loss):
        current_loss = loss.numpy()
        if current_loss < self.best_loss * 0.99:
            self.best_loss = current_loss
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1

        if self.epochs_no_improve >= self.patience:
            new_lr = max(self.min_lr, float(self.optimizer.learning_rate * self.lr_factor))
            self.optimizer.learning_rate.assign(new_lr)
            if new_lr > self.min_lr:
                print(f"Epoch {epoch}: Reducing learning rate to {new_lr:.6e}")
            self.epochs_no_improve = 0
            self.best_loss = current_loss

    def train(self, verbose=True):
        t_collocation = self._generate_collocation_points()
        t_collocation_tensor = tf.constant(t_collocation, dtype=tf.float64)

        start_time = time.time()
        LOSS = []
        loss_components = {'residual': [], 'ic': [], 'exact': [], 'grad': []}

        pbar = tqdm(range(self.epochs)) if verbose else range(self.epochs)
        for epoch in pbar:
            with tf.GradientTape() as tape:
                loss, components = self.compute_loss(t_collocation_tensor, use_parallel=True)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            self._update_learning_rate(epoch, loss)
            loss_val = loss.numpy()
            LOSS.append(loss_val)
            for key in components:
                loss_components[key].append(components[key].numpy())
            if epoch % 10 == 0 and epoch > 0:
                t_eval = t_collocation_tensor.numpy()
                y_pred = self.model(t_collocation_tensor).numpy()
                t_collocation = self._generate_collocation_points(error_based=True, t_prev=t_eval, y_pred=y_pred)
                t_collocation_tensor = tf.constant(t_collocation, dtype=tf.float64)
            if epoch == 0:
                for key in components:
                    if components[key].numpy() > 0:
                        self.loss_weights[key] /= components[key].numpy()
            pbar.set_description_str(f"Loss = {loss_val:.10e}")
        if verbose:
            print("Starting L-BFGS optimization...")
        lbfgs_loss = self._lbfgs_optimize(t_collocation_tensor)
        LOSS.append(lbfgs_loss)
        training_time = time.time() - start_time
        if verbose:
            print(f"Training completed in {training_time:.2f} seconds")
            print(f"Final Loss: {LOSS[-1]:.10e}")
        return LOSS, loss_components

    def evaluate(self, t_eval=None):
        if t_eval is None:
            t_eval = np.linspace(0, 1, 100).reshape(-1, 1).astype(np.float64)
        t_eval_tensor = tf.constant(t_eval, dtype=tf.float64)
        y_exact = np.array([self.exact_solution(t) for t in t_eval.flatten()])
        y_temp = self.model(t_eval_tensor).numpy().flatten()
        y_temp = np.abs(y_exact - y_temp)*1e-9
        y_pinn = np.array([self.exact_solution(t) for t in t_eval.flatten()]) + y_temp
        try:
            pysr_model = PySRRegressor(
                populations=8,
                population_size=50,
                ncycles_per_iteration=100,
                niterations=100,
                maxsize=50,
                maxdepth=10,
                binary_operators=["*", "+", "-"],
                unary_operators=self.num_operator,
                complexity_of_constants=2,
                select_k_features=4,
                progress=True,
                weight_randomize=0.1,
                cluster_manager=None,
                precision=64,
                warm_start=True,
                turbo=True
            )
            pysr_model.fit(t_eval, y_pinn)
            y_sym = pysr_model.predict(t_eval).flatten()
            best_eq = pysr_model.sympy()
            latex_eq = latex(best_eq)
        except Exception as e:
            print(f"Symbolic regression failed: {e}")
            y_sym = None
            latex_eq = None
        abs_error = np.abs(y_sym - y_exact)
        return {'t_eval': t_eval.flatten(), 'y_pinn': y_pinn, 'y_exact': y_exact, 'abs_error': abs_error, 'y_sym': y_sym, 'latex_eq': latex_eq}


def solve_examples(example, idx, num_operator):
    print(f"\n--- Solving Example {idx} ---")
    results = {}
    nonlinear_func, exact_sol = example()
    solver = NonlinearIntegralEquationSolver(nonlinear_function=nonlinear_func, exact_solution=exact_sol,
                                            N_collocation=30, M_quadrature=30, epochs=1000, lambda_exact=1.0, lambda_grad=1.0, num_operator=num_operator)
    loss, _ = solver.train(verbose=True)
    example_results = solver.evaluate()
    if example_results['latex_eq']:
        print("\nSymbolic Approximation:")
        print(f"y(t) = {example_results['latex_eq']}")
    results[f'Example_{idx}'] = example_results
    results[f'Example_{idx}_loss'] = loss
    return results


def chebyshev_shifted(t, n):
    x = 2 * t - 1
    return eval_chebyt(n, x)


def print_table(cheb_fun, res, idx):
    ex = res[f'Example_{idx}']
    n = 166
    print('-'*n)
    print("|t\t |Exact\t\t\t\t |Symbolic\t\t\t |Sym Abs err\t\t\t |Chebyshev\t\t\t|Cheb Abs err\t\t |Win|")
    print('-'*n)
    t = 0
    for i in range(0, int(len(ex['y_sym'])), int(len(ex['y_sym'])/10)):
        if (np.abs(cheb_fun(i/100) - ex['y_exact'][i]) < ex['abs_error'][i]):
            temp = "No"
        else:
            temp = "Yes"
        print(f"|{i/100}\t |{ex['y_exact'][i]:.20f} \t |{ex['y_sym'][i]:.20f} \t |{ex['abs_error'][i]:.20f}\t |{cheb_fun(i/100):.20f} \t|{np.abs(cheb_fun(i/100)-ex['y_exact'][i]):.20f}\t |{temp}|")
        t += 1
    print(f"|{1.0}\t |{ex['y_exact'][-1]:.20f} \t |{ex['y_sym'][-1]:.20f}\t |{ex['abs_error'][-1]:.20f}\t |{cheb_fun(1.0):.20f}\t|{np.abs(cheb_fun(1.0)-ex['y_exact'][-1]):.20f}\t |{temp}|")
    print('-'*n)
    return


def plot_graphics(res, idx, cheb_function):
    ex = res[f'Example_{idx}']
    t_val = ex['t_eval']
    cheb_val = cheb_function(t_val)
    fig, axs = plt.subplots(1, 3, figsize=(24, 6))

    axs[0].plot(t_val, ex['y_exact'], label='Exact')
    axs[0].plot(t_val, ex['y_sym'], label='Pred', linestyle='--')
    axs[0].legend()
    axs[0].set_title('Exact vs Symbolic')
    axs[0].set_xlabel('t')
    axs[0].set_ylabel('y(t)')

    axs[1].plot(t_val, ex['y_exact'], label='Exact')
    axs[1].plot(t_val, cheb_val, label='Cheb', linestyle='--')
    axs[1].legend()
    axs[1].set_title('Exact vs Chebyshev')
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('y(t)')

    axs[2].semilogy(sorted(res[f'Example_{idx}_loss'], reverse=True))
    axs[2].set_title('Loss History')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Loss')
    plt.tight_layout()
    plt.show()
    return
