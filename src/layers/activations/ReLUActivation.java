package layers.activations;

public class ReLUActivation extends ActivationLayer {
	@Override
	public double activation_forward(double in) {
		return Math.max(0, in);
	}

	@Override
	public double activation_backward() {
		return Math.signum(get_after());
	}
	
	@Override
	public String toString() {
		return "ReLUActivation()";
	}
}
