package layers.activations;

public class TanhActivation extends ActivationLayer {
	
	@Override
	public double activation_forward(double in) {
		return 2 / (1 + Math.exp(-2*in)) - 1;
	}
	
	@Override
	public double activation_backward() {
		double in = get_after();
		return 1 - in*in;
	}
	
	@Override
	public String toString() {
		return "TanhActivation()";
	}
}
