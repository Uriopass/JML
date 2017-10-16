package math;

public abstract class Activations {
	public static Vector softmax(Vector activations) {
		double max = activations.max();

		float expSum = 0;
		for (int i = 0; i < activations.length; i++) {
			activations.v[i] = (float) Math.exp(activations.v[i] - max);
			expSum += activations.v[i];
		}
		for (int i = 0; i < activations.length; i++) {
			activations.v[i] /= expSum;
		}
		return activations;
	}

	public static Matrix softmax(Matrix activations, int axis) {
		if (axis == 0) {
			for (int j = 0; j < activations.height; j++) {
				double max = activations.get_row(j).max();

				float expSum = 0;
				for (int i = 0; i < activations.width; i++) {
					activations.v[j][i] = (float) Math.exp(activations.v[j][i] - max);
					expSum += activations.v[j][i];
				}
				for (int i = 0; i < activations.width; i++) {
					activations.v[j][i] /= expSum;
				}
			}
			return activations;
		}
		if (axis == 1) {
			for (int j = 0; j < activations.width; j++) {
				double max = activations.get_column(j).max();

				float expSum = 0;
				for (int i = 0; i < activations.height; i++) {
					activations.v[i][j] = (float) Math.exp(activations.v[i][j] - max);
					expSum += activations.v[i][j];
				}
				for (int i = 0; i < activations.height; i++) {
					activations.v[i][j] /= expSum;
				}
			}
			return activations;
		}
		return null;
	}
	
	public static Vector sigmoid(Vector x) {
		for (int i = 0; i < x.length; i++) {
			x.v[i] = (float) (1 / (1 + Math.exp(-x.v[i])));
		}
		return x;
	}

	public static Matrix sigmoid(Matrix x) {
		for (int j = 0; j < x.height; j++) {
			for (int i = 0; i < x.width; i++) {
				x.v[j][i] = (float) (1 / (1 + Math.exp(-x.v[j][i])));
			}
		}
		return x;
	}
	
	public static double sigmoid_backward(double x) {
		return x*(1-x);
	}

	public static Vector ReLU(Vector x) {
		for (int i = 0; i < x.length; i++) {
			x.v[i] = Math.max(0, x.v[i]);
		}
		return x;
	}

	public static Matrix ReLU(Matrix x) {
		for (int j = 0; j < x.height; j++) {
			for (int i = 0; i < x.width; i++) {
				x.v[j][i] = Math.max(x.v[j][i], 0);
			}
		}
		return x;
	}
	
	public static double ReLU_backward(double x) {
		return x > 0 ? 1 : 0;
	}
	
	public static Vector TanH(Vector x) {
		for (int i = 0; i < x.length; i++) {
			x.v[i] = Math.tanh(x.v[i]);
		}
		return x;
	}

	public static Matrix TanH(Matrix x) {
		for (int j = 0; j < x.height; j++) {
			for (int i = 0; i < x.width; i++) {
				x.v[j][i] = Math.tanh(x.v[j][i]);
			}
		}
		return x;
	}
	
	public static double TanH_backward(double x) {
		return 1-x*x;
	}
}
