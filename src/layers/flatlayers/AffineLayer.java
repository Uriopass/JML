package layers.flatlayers;

import layers.FlatLayer;
import layers.Parameters;
import math.Initialisations;
import math.Matrix;
import math.Optimizers;
import math.Vector;

public class AffineLayer implements FlatLayer {
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
		w_grad = new Matrix(fan_in, fan_out);
		w_acceleration = new Matrix(fan_in, fan_out);
		bias = new Vector(fan_out);
		b_grad = new Vector(fan_out);
		b_acceleration = new Vector(fan_out);
		
		if(init) {
			Initialisations.he_uniform(weight, fan_in, p.getAsDouble("init_multiplier", 1));
			
			for (int i = 0; i < bias.length; i++) {
				bias.v[i] = 0;
			}
		}
		
		if(p == null) {
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
		if(training)
			cache = new Matrix(in);
		Matrix next = weight.parralel_mult(in);
		next.add(bias, Matrix.AXIS_WIDTH);
		return next;
	}
	
	@Override
	public Matrix backward(Matrix dout) {
		w_grad.add(dout.parralel_mult(cache.T()).scale(1.0 / cache.width)
				.add(Matrix.scale(weight, regularization)));
		b_grad.add(dout.sum(1).scale(1.0 / cache.width));
		
		if(!calculate_dout)
			return null;
		return weight.T().parralel_mult(dout);
	}
	
	@Override
	public void apply_gradient() {
		Optimizers.RMSProp(weight, w_grad, w_acceleration, gamma, learning_rate, epsilon);
		Optimizers.RMSProp(bias, b_grad, b_acceleration, gamma, learning_rate, epsilon);
		/*
		weight.add(w_grad.scale(-learning_rate));
		bias.add(b_grad.scale(-learning_rate));
		*/
		learning_rate *= learning_rate_decay;
		w_grad.fill(0);
		b_grad.fill(0);
	}
	
	@Override
	public String toString() {
		return "AffineLayer("+fan_in+", "+fan_out+", lr="+learning_rate+", reg="+regularization+", lrdecay="+learning_rate_decay+((calculate_dout)?"":", dout=false")+")";
	}
}
