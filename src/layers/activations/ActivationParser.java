package layers.activations;

public class ActivationParser {
	public static ActivationLayer getActivationByName(String name) {
		switch (name.toLowerCase()) {
		case "tanh":
			return new TanhActivation();
		case "sigmoid":
		case "sig":
			return new SigmoidActivation();
		case "relu":
			return new ReLUActivation();
		case "swish":
			return new SwishActivation();
		case "softplus":
			return new SoftplusActivation();
		default:
			return null;
		}
	}
}
