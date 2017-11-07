package layers.flat;

import layers.FlatLayer;
import layers.Parameters;
import math.Matrix;
import math.Optimizers;
import math.RMSVector;
import math.Vector;

public class BatchnormLayer implements FlatLayer {
	
	final static double epsilon = 1e-4;
	final static double rms_gamma = 0.9;
	RMSVector gamma, beta;
	
	Vector running_mean, running_var;
	
	private Matrix xmu, carre;
	
	private Vector var, sqrtvar, invvar, mu;
	private Matrix va2;

	public double learning_rate;
	public double learning_rate_decay;
	public double momentum;
	
	public int fan_in;
	
	public BatchnormLayer(int fan_in, Parameters param) {
		this.fan_in = fan_in;
		gamma = new RMSVector(fan_in);
		beta = new RMSVector(fan_in);
		running_mean = new Vector(fan_in);
		running_var = new Vector(fan_in);
		
		for(int i = 0 ; i < gamma.length ; i++) {
			gamma.v[i] = 1;
			running_var.v[i] = 1;
		}
		learning_rate = param.getAsDouble("lr", 0.001);
		learning_rate_decay = param.getAsDouble("lrdecay", 1);
		momentum = param.getAsDouble("momentum", 0.9);
	}
	
	@Override
	public Matrix forward(Matrix in, boolean training) {
		if(training) {
			// Step 1
			mu = in.sum(Matrix.AXIS_WIDTH).scale(-1.0 / in.width);
			
			// Step 2
			in.add(mu, Matrix.AXIS_WIDTH);
			xmu = new Matrix(in);
			
			// Step 3
			carre = Matrix.hadamart(xmu, xmu); // square
			// Step 4
			var = carre.sum(Matrix.AXIS_WIDTH).scale(1.0 / in.width); // average
	
			// Step 5
			sqrtvar = new Vector(var).add(epsilon).power(0.5);
			
			// Step 6
			invvar = new Vector(sqrtvar).inverse();
			
			// Step 7
			in.scale(invvar, Matrix.AXIS_WIDTH);
			va2 = new Matrix(in);
			// Step 8
			in.scale(gamma, Matrix.AXIS_WIDTH);
			// Step 9
			in.add(beta, Matrix.AXIS_WIDTH);
			running_mean = running_mean.scale(momentum).add(new Vector(mu ).scale(1 - momentum));
			running_var  = running_var .scale(momentum).add(new Vector(var).scale(1 - momentum));
		} else {
			in.add(running_mean, Matrix.AXIS_WIDTH);
			in.scale(new Vector(running_var).power(0.5).add(epsilon).inverse(), Matrix.AXIS_WIDTH);
			in.scale(gamma, Matrix.AXIS_WIDTH);
			in.add(beta, Matrix.AXIS_WIDTH);
		}
		return in;
	}

	@Override
	public Matrix backward(Matrix dout) {
		int N = dout.width;
		// Step 9
		Matrix dva3 = dout;
		beta.grad.add(dout.sum(Matrix.AXIS_WIDTH).scale(1.0 / N));
		// Step 8
		gamma.grad.add(va2.hadamart(dva3).sum(Matrix.AXIS_WIDTH).scale(1.0 / N));
		Matrix dva2 = dva3.scale(gamma, Matrix.AXIS_WIDTH);
		// Step 7
		Vector dinvvar = new Matrix(xmu).hadamart(dva2).sum(Matrix.AXIS_WIDTH);
		Matrix dxmu = dva2.scale(invvar, Matrix.AXIS_WIDTH);
		// Step 6
		Vector dsqrtvar = sqrtvar.scale(sqrtvar).inverse().scale(-1).scale(dinvvar);
		// Step 5
		Vector dvar = var.add(epsilon).power(0.5).inverse().scale(dsqrtvar).scale(0.5);
		// Step 4
		Matrix dcarre = new Matrix(carre.width, carre.height);
		dcarre.fill(1).scale(1.0 / dout.width).scale(dvar, Matrix.AXIS_WIDTH);
		// Step 3
		dxmu.add(xmu.hadamart(dcarre).scale(2));
		// Step 2
		Vector dmu = dxmu.sum(Matrix.AXIS_WIDTH).scale(-1.0 / N);
		// Step 1
		dxmu.add(dmu, Matrix.AXIS_WIDTH);
		return dxmu;
	}
	
	public void end_of_epoch() {
		learning_rate *= learning_rate_decay;
	}

	@Override
	public void apply_gradient() {
		Optimizers.RMSProp(gamma, rms_gamma, learning_rate, epsilon);
		Optimizers.RMSProp(beta, rms_gamma, learning_rate, epsilon);
	}
	
	@Override
	public String toString() {
		return "BatchnormLayer("+fan_in+")";
	}

}