package math;

public class Optimizers {
	
	/**
	 * Apply RMSProp and set gradient to zero
	 */
	public static void RMSProp(Matrix w, Matrix w_grad, Matrix acc, double gamma, double lr, double eps) {
		for (int l = 0; l < acc.height; l++) {
			for (int m = 0; m < acc.width; m++) {
				acc.v[l][m] = gamma * acc.v[l][m]
						+ (1 - gamma) * w_grad.v[l][m] * w_grad.v[l][m];
				w_grad.v[l][m] *= -lr / (Math.sqrt(eps + acc.v[l][m]));
				w.v[l][m] += w_grad.v[l][m];
				w_grad.v[l][m] = 0;
			}
		}
	}
	
	public static void RMSProp(FeatureMatrix w, FeatureMatrix w_grad, FeatureMatrix acc, double gamma, double lr, double eps) {
		for(int f = 0 ; f < acc.features ; f++) {
			RMSProp(w.v[f], w_grad.v[f], acc.v[f], gamma, lr, eps);
		}
	}
	
	public static void RMSProp(Vector b, Vector b_grad, Vector acc, double gamma, double lr, double eps) {
		for (int l = 0; l < acc.length; l++) {
			acc.v[l] = gamma * acc.v[l]
					+ (1 - gamma) * b_grad.v[l] * b_grad.v[l];
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
