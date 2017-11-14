package math;

/**
 * Implémentation de références pour diverses activation.
 */
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
		if (axis == Matrix.AXIS_WIDTH) {
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
		if (axis == Matrix.AXIS_HEIGHT) {
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

	public static double sigmoid(double x) {
		return 1 / (1 + Math.exp(-x));
	}

	public static double sigmoid_backward(double x) {
		return x * (1 - x);
	}

	public static double ReLU(double x) {
		return Math.max(0, x);
	}

	public static double ReLU_backward(double x) {
		return x > 0 ? 1 : 0;
	}

	public static double TanH(double x) {
		return Math.tanh(x);
	}

	public static double TanH_backward(double x) {
		return 1 - x * x;
	}
}
