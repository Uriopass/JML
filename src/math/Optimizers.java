package math;

public class Optimizers {
	/**
	 * Appique la descente de gradient et remet le gradient à zero
	 */
	public static void SGD(Matrix w, Matrix w_grad, double lr) {
		for (int l = 0; l < w.height; l++) {
			for (int m = 0; m < w.width; m++) {
				// w -= lr * grad
				w.v[l][m] -= lr * w_grad.v[l][m];
				w_grad.v[l][m] = 0;
			}
		}
	}

	public static void SGD(Vector v, Vector v_grad, double lr) {
		for (int m = 0; m < v.length; m++) {
			v.v[m] -= lr * v_grad.v[m];
			v_grad.v[m] = 0;
		}
	}

	/**
	 * Applique RMSProp et remets le gradient à zero
	 */
	public static void RMSProp(Matrix w, Matrix w_grad, Matrix acc, double gamma, double lr, double eps) {
		for (int l = 0; l < acc.height; l++) {
			for (int m = 0; m < acc.width; m++) {
				// acc[t+1] = gamma * acc[t] + (1 - gamma) * grad^2
				acc.v[l][m] = gamma * acc.v[l][m] + (1 - gamma) * w_grad.v[l][m] * w_grad.v[l][m];

				// grad *= - lr / sqrt(epsilon + acc[t+1])
				w_grad.v[l][m] *= -lr / (Math.sqrt(eps + acc.v[l][m]));

				// w += grad
				w.v[l][m] += w_grad.v[l][m];
				w_grad.v[l][m] = 0;
			}
		}
	}

	public static void RMSProp(Vector b, Vector b_grad, Vector acc, double gamma, double lr, double eps) {
		for (int l = 0; l < acc.length; l++) {
			acc.v[l] = gamma * acc.v[l] + (1 - gamma) * b_grad.v[l] * b_grad.v[l];
			b_grad.v[l] *= -lr / (Math.sqrt(eps + acc.v[l]));
			b.v[l] += b_grad.v[l];
			b_grad.v[l] = 0;
		}
	}

	public static void RMSProp(RMSMatrix w, double gamma, double lr, double eps) {
		RMSProp(w, w.grad, w.acc, gamma, lr, eps);
	}

	public static void RMSProp(RMSVector v, double gamma, double lr, double eps) {
		RMSProp(v, v.grad, v.acc, gamma, lr, eps);
	}
}
