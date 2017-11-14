package layers.activations;

/**
 * Activation Swish
 * - swish(x)  = x * sigmoid(x) = x / (1 + e^(-x))
 * - swish'(x) = swish(x) + (1-swish(x)) / (1 + e^(-x))
 */
public class SwishActivation extends ActivationLayer {

	public SwishActivation() {
		needs_cache_before = true;
		needs_cache_after = true;
	}

	@Override
	public double activation_forward(double in) {
		return in / (1 + Math.exp(-in));
	}

	@Override
	public double activation_backward() {
		double after = get_after();
		double before = get_before();
		return after + (1 - after) / (1 + Math.exp(-before));
	}

	@Override
	public String toString() {
		return "SwishActivation()";
	}
}
