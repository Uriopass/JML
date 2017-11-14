package layers.activations;

/**
 * Activation Sigmoid
 * - sigmoid(x)  = 1 / (1 + e^(-x))
 * - sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
 */
public class SigmoidActivation extends ActivationLayer {
	public SigmoidActivation() {
		needs_cache_after = true;
	}

	@Override
	public double activation_forward(double in) {
		return 1 / (1 + Math.exp(-in));
	}

	@Override
	public double activation_backward() {
		double in = get_after();
		return in * (1 - in);
	}

	@Override
	public String toString() {
		return "SigmoidActivation()";
	}
}
