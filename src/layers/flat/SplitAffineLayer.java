package layers.flat;

import layers.FlatLayer;
import layers.Parameters;
import math.Initialisations;
import math.Matrix;
import math.Optimizers;
import math.Vector;

public class SplitAffineLayer implements FlatLayer {
	public Vector weight;
	public Vector bias;

	private Matrix cache;
	private Vector w_grad;
	private Vector w_acceleration;
	private Vector b_grad;
	private Vector b_acceleration;

	public double regularization;
	public double learning_rate;
	public double learning_rate_decay;
	public double gamma;
	public double epsilon;

	public boolean calculate_dout;

	public int fan_in, fan_out, per_out;

	public SplitAffineLayer(int fan_in, int fan_out, boolean init, Parameters p) {
		if (fan_in % fan_out != 0) {
			throw new RuntimeException("fan_in (" + fan_in + ") must be a multiple of fan_out (" + fan_out + ")");
		}
		this.fan_in = fan_in;
		this.fan_out = fan_out;
		this.per_out = fan_in / fan_out;

		weight = new Vector(fan_in);
		w_grad = new Vector(fan_in);
		w_acceleration = new Vector(fan_in);
		bias = new Vector(fan_out);
		b_grad = new Vector(fan_out);
		b_acceleration = new Vector(fan_out);

		if (init) {
			Initialisations.he_uniform(weight, per_out, p.getAsDouble("init_multiplier", 1));

			for (int i = 0; i < bias.length; i++) {
				bias.v[i] = 0;
			}
		}

		if (p == null) {
			p = new Parameters();
		}

		this.regularization = p.getAsDouble("reg", 0);
		this.learning_rate = p.getAsDouble("lr", 0.001);
		this.learning_rate_decay = p.getAsDouble("lrdecay", 1);
		this.gamma = p.getAsDouble("gamma", 0.9);
		this.epsilon = p.getAsDouble("epsilon", 1e-8);
		this.calculate_dout = p.getAsString("dout", "true").equalsIgnoreCase("true");
	}

	public void end_of_epoch() {
		learning_rate *= learning_rate_decay;
	}

	@Override
	public Matrix forward(Matrix in, boolean training) {
		if (training)
			cache = new Matrix(in);
		
		Matrix next = new Matrix(in.width, fan_out);
		in.scale(weight, Matrix.AXIS_WIDTH);
		for(int k = 0 ; k < fan_out ; k++) {
			for(int i = 0 ; i < in.width ; i++) {
				double sum = 0;
				for(int j = 0 ; j < per_out ; j++) {
					sum += in.v[j+k*per_out][i];
				}
				next.v[k][i] = sum;
			}
		}
		
		next.add(bias, Matrix.AXIS_WIDTH);
		return next;
	}

	@Override
	public Matrix backward(Matrix dout) {
		b_grad.add(dout.sum(Matrix.AXIS_WIDTH).scale(1.0 / cache.width));
		
		Matrix expanded_dout = new Matrix(cache.width, cache.height);
		
		for(int i = 0 ; i < expanded_dout.height ; i++) {
			for(int j = 0 ; j < expanded_dout.width ; j++) {
				expanded_dout.v[i][j] = dout.v[i/per_out][j];
			}
		}
		w_grad.add(Matrix.hadamart(expanded_dout, cache).sum(Matrix.AXIS_WIDTH).scale(1.0 / cache.width));
		if (!calculate_dout)
			return null;
		return expanded_dout.scale(weight, Matrix.AXIS_WIDTH);
	}

	@Override
	public void apply_gradient() {
		Optimizers.RMSProp(weight, w_grad, w_acceleration, gamma, learning_rate, epsilon);
		Optimizers.RMSProp(bias, b_grad, b_acceleration, gamma, learning_rate, epsilon);
		/*
		weight.add(w_grad.scale(-learning_rate));
		bias.add(b_grad.scale(-learning_rate));
		*/
		w_grad.fill(0);
		b_grad.fill(0);
	}

	@Override
	public String toString() {
		return "AffineLayer(" + fan_in + ", " + fan_out + ", lr=" + learning_rate + ", reg=" + regularization
				+ ", lrdecay=" + learning_rate_decay + ((calculate_dout) ? "" : ", dout=false") + ")";
	}
}
