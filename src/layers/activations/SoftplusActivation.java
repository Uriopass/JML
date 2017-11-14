package layers.activations;

/**
 * Activation SoftPlus
 * - softplus(x)  = ln(1 + e^x)
 * - softplus'(x) = sigmoid(x)
 */
public class SoftplusActivation extends ActivationLayer {

	public SoftplusActivation() {
		needs_cache_before = true;
	}

	@Override
	public double activation_forward(double in) {
		return Math.log(1 + Math.exp(in));
	}

	@Override
	public double activation_backward() {
		return 1 / (1 + Math.exp(-get_before()));
	}

	@Override
	public String toString() {
		return "SoftplusActivation()";
	}
}
