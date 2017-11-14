package layers.activations;

/**
 * Activation Tangeante Hyperbolique (TanH)
 * - TanH(x)  = (e^x - e^(-x))/(e^x + e^(-x))
 * - TanH'(x) = 1 - TanH(x)^2
 */
public class TanhActivation extends ActivationLayer {
	public TanhActivation() {
		needs_cache_after = true;
	}

	@Override
	public double activation_forward(double in) {
		return Math.tanh(in);
	}

	@Override
	public double activation_backward() {
		double in = get_after();
		return 1 - in * in;
	}

	@Override
	public String toString() {
		return "TanhActivation()";
	}
}
