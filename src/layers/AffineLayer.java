package layers;

import math.Matrix;
import math.Vector;

public class AffineLayer extends Layer {
	public Matrix weight;
	public Vector bias;
	private Matrix cache;
	private Matrix w_grad;
	private Matrix w_acceleration;
	private Vector b_grad;
	private Vector b_acceleration;

	public double regularization;
	public double learning_rate;
	public double learning_rate_decay;	
	public double gamma;
	public double epsilon;
	
	public boolean calculate_dout;

	public int fan_in, fan_out;
	
	public AffineLayer(int fan_in, int fan_out, boolean init, Parameters p) {
		this.fan_in = fan_in;
		this.fan_out = fan_out;
		weight = new Matrix(fan_in, fan_out);
		w_acceleration = new Matrix(fan_in, fan_out);
		bias = new Vector(fan_out);
		b_acceleration = new Vector(fan_out);
		
		if(init) {
			double bound = 4 * Math.sqrt(6f / (fan_in + fan_out));
			// System.out.println("mult:" + bound);
			for (int i = 0; i < weight.height; i++) {
				for (int j = 0; j < weight.width; j++) {
					weight.v[i][j] = RandomGenerator.uniform(-bound, bound);
				}
			}
			for (int i = 0; i < bias.length; i++) {
				bias.v[i] = 0;
			}
		}
		
		if(p == null) {
			p = new Parameters();
		}
		
		this.regularization = p.getAsDouble("reg", 0);
		this.learning_rate = p.getAsDouble("lr", 0.01);
		this.learning_rate_decay = p.getAsDouble("lrdecay", 1);
		this.gamma = p.getAsDouble("gamma", 0.9);
		this.epsilon = p.getAsDouble("epsilon", 1e-8);
		this.calculate_dout = p.getAsString("dout", "true").equalsIgnoreCase("true");
	}
	
	@Override
	public Matrix forward(Matrix in, boolean training) {
		if(training)
			cache = new Matrix(in);
		Matrix next = weight.parralel_mult(in);
		for (int i = 0; i < bias.length; i++) {
			for (int j = 0; j < next.width; j++) {
				next.v[i][j] += bias.v[i];
			}
		}
		return next;
	}
	
	@Override
	public Matrix backward(Matrix dout) {
		w_grad = dout.parralel_mult(cache.T()).scaleInPlace(1.0 / cache.width)
				.addInPlace(weight.scale(regularization));
		b_grad = dout.sum(1).scaleInPlace(1.0 / cache.width);
		
		if(!calculate_dout)
			return null;
		return weight.T().parralel_mult(dout);
	}
	
	@Override
	public void apply_gradient() {
		
		for (int l = 0; l < w_acceleration.height; l++) {
			for (int m = 0; m < w_acceleration.width; m++) {
				w_acceleration.v[l][m] = gamma * w_acceleration.v[l][m]
						+ (1 - gamma) * w_grad.v[l][m] * w_grad.v[l][m];
				w_grad.v[l][m] *= -learning_rate / (Math.sqrt(epsilon + w_acceleration.v[l][m]));
				weight.v[l][m] += w_grad.v[l][m];
			}
		}
		for (int l = 0; l < b_acceleration.length; l++) {
			b_acceleration.v[l] = gamma * b_acceleration.v[l]
					+ (1 - gamma) * b_grad.v[l] * b_grad.v[l];
			b_grad.v[l] *= -learning_rate / (Math.sqrt(epsilon + b_acceleration.v[l]));
			bias.v[l] += b_grad.v[l];
		}
	}
	
	@Override
	public String toString() {
		return "AffineLayer"+weight.shape();
	}
}
