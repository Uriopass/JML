package layers.flat;

import layers.FlatLayer;
import layers.Parameters;
import math.Initialisations;
import math.Matrix;
import math.Optimizers;
import math.RMSMatrix;
import math.RMSVector;

public class AffineLayer implements FlatLayer {
	public RMSMatrix weight;
	public RMSVector bias;
	
	private Matrix cache;

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
		weight = new RMSMatrix(fan_in, fan_out);
		bias = new RMSVector(fan_out);
		
		if(init) {
			Initialisations.xavier(weight, fan_in, fan_out);
			
			bias.fill(0);
		}
		
		if(p == null) {
			p = new Parameters();
		}
		
		this.regularization = p.getAsDouble("reg", 0);
		this.learning_rate = p.getAsDouble("lr", 0.001);
		this.learning_rate_decay = p.getAsDouble("lrdecay", 1);
		this.gamma = p.getAsDouble("gamma", 0.1);
		this.epsilon = p.getAsDouble("epsilon", 1e-6);
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
		weight.grad.add(dout.parralel_mult(cache.T()).scale(1.0 / cache.width)
				.add(Matrix.scale(weight, regularization)));
		bias.grad.add(dout.sum(Matrix.AXIS_WIDTH).scale(1.0 / cache.width));
		
		if(!calculate_dout)
			return null;
		return weight.T().parralel_mult(dout);
	}
	
	@Override
	public void apply_gradient() {
		Optimizers.RMSProp(weight, gamma, learning_rate, epsilon);
		Optimizers.RMSProp(bias, gamma, learning_rate, epsilon);
	}
	
	@Override
	public String toString() {
		return "AffineLayer("+fan_in+", "+fan_out+", lr="+learning_rate+", reg="+regularization+", lrdecay="+learning_rate_decay+((calculate_dout)?"":", dout=false")+")";
	}
}
