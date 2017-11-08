package layers.activations;

public class SigmoidActivation extends ActivationLayer {
	public SigmoidActivation() {
		needs_cache_after = true;
	}
	
	@Override
	public double activation_forward(double in) {
		return 1/(1+Math.exp(-in));
	}
	
	@Override
	public double activation_backward() {
		double in = get_after();
		return in*(1-in);
	}
	
	@Override
	public String toString() {
		return "SigmoidActivation()";
	}
}
