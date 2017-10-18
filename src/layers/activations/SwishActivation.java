package layers.activations;

public class SwishActivation extends ActivationLayer {
	
	public SwishActivation() {
		needs_cache_before = true;
	}
	
	@Override
	public double activation_forward(double in) {
		return in / (1+Math.exp(-in));
	}
	
	@Override
	public double activation_backward() {
		double aft = get_after();
		double be4 = get_before();
		return aft + (1-aft)/(1+Math.exp(-be4));
	}
	
	@Override
	public String toString() {
		return "SwishActivation()";
	}
}
