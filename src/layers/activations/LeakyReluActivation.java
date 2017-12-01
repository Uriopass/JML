package layers.activations;

public class LeakyReluActivation extends ActivationLayer {
	double leak;
	public LeakyReluActivation(double leak) {
		needs_cache_after = true;
		this.leak = leak;
	}

	@Override
	public double activation_forward(double in) {
		if(in > 0)
			return in;
		return in * leak;
	}

	@Override
	public double activation_backward() {
		// get_after() vaut soit 0 soit un nombre positif si x était positif, la dérivée vaut donc bien signe(ReLU(x))
		double a = get_after();
		if(a > 0)
			return 1;
		if(a == 0)
			return 0;
		return leak;
	}

	@Override
	public String toString() {
		return "ReLUActivation()";
	}
}
