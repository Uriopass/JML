package layers.activations;

/**
 * Activation ReLU
 * - ReLU(x)  = max(0, x)
 * - ReLU'(x) = 1 si x > 0 et 0 sinon
 */
public class ReLUActivation extends ActivationLayer {
	public ReLUActivation() {
		needs_cache_after = true;
	}

	@Override
	public double activation_forward(double in) {
		return Math.max(0, in);
	}

	@Override
	public double activation_backward() {
		// get_after() vaut soit 0 soit un nombre positif si x était positif, la dérivée vaut donc bien signe(ReLU(x))
		return Math.signum(get_after());
	}

	@Override
	public String toString() {
		return "ReLUActivation()";
	}
}
