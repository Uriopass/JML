package layers.reccurent;

import layers.FlatLayer;
import layers.Parameters;
import layers.activations.ActivationLayer;
import layers.activations.ActivationParser;
import math.Initialisations;
import math.Matrix;
import math.TrainableMatrix;
import math.TrainableVector;
// Deprecated, UPDATE WITH TRAINABLE MATRICES
@Deprecated
public class ReccurentAffineLayer implements FlatLayer {
	public TrainableMatrix weight_out, weight_state;
	public TrainableVector bias_out, bias_state;
	
	public Matrix state;
	
	public ActivationLayer act;
	
	private Matrix cache_state, cache_concat;

	public double regularization;
	public double learning_rate;
	public double learning_rate_decay;	
	public double gamma;
	public double epsilon;
	
	public boolean calculate_dout;

	public int fan_in, fan_out;
	
	public ReccurentAffineLayer(int fan_in, int fan_out, boolean init, int state_size, String state_act, Parameters p) {
		this.fan_in = fan_in;
		this.fan_out = fan_out;
		state = new Matrix(1, state_size);
		weight_state = new TrainableMatrix(fan_in+state.height, state.height);
		weight_out = new TrainableMatrix(state.height, fan_out);
		bias_out = new TrainableVector(fan_out);
		bias_state = new TrainableVector(state.height);
		
		act = ActivationParser.get_activation_by_name(state_act);
		
		if(init) {
			Initialisations.he_uniform(weight_out, fan_in, p.get_as_double("init_multiplier", 1));
			Initialisations.he_uniform(weight_state, fan_in+state.height, p.get_as_double("init_multiplier", 1));
			
			//bias.fill(0);
		}
		
		if(p == null) {
			p = new Parameters();
		}
		
		this.regularization = p.get_as_double("reg", 0);
		this.learning_rate = p.get_as_double("lr", 0.001);
		this.learning_rate_decay = p.get_as_double("lrdecay", 1);
		this.gamma = p.get_as_double("gamma", 0.9);
		this.epsilon = p.get_as_double("epsilon", 1e-8);
		this.calculate_dout = p.get_or_default("dout", "true").equalsIgnoreCase("true");
	}
	
	public void end_of_epoch() {
		learning_rate *= learning_rate_decay;
	}
	
	@Override
	public Matrix forward(Matrix in, boolean training) {
		assert in.width == 1;
		Matrix in_c = new Matrix(1, in.height+state.height);
		in_c.set_column(0, in.get_column(0).append(state.get_column(0)));
		
		if(training)
			cache_concat = new Matrix(in_c);
		state = act.forward(weight_state.parralel_mult(in_c).add(bias_state, Matrix.AXIS_WIDTH), training);
		
		if(training)
			cache_state = new Matrix(state);
		
		Matrix next = weight_out.parralel_mult(state).add(bias_out, Matrix.AXIS_WIDTH);
		return next;
	}
	
	@Override
	public Matrix backward(Matrix dout, boolean train) {
		weight_out.grad.add(dout.parralel_mult(cache_state.T()).scale(1.0 / cache_state.width)
				.add(Matrix.scale(weight_out, regularization)));
		bias_out.grad.add(dout.sum(Matrix.AXIS_WIDTH).scale(1.0 / cache_state.width));
		
		dout = weight_out.T().parralel_mult(dout);
		dout = act.backward(dout, train);
		
		weight_state.grad.add(dout.parralel_mult(cache_concat.T()).scale(1.0 / cache_concat.width)
				.add(Matrix.scale(weight_state, regularization)));
		bias_state.grad.add(dout.sum(Matrix.AXIS_WIDTH).scale(1.0 / cache_state.width));
		
		if(!calculate_dout)
			return null;
		Matrix in_grad = new Matrix(1, fan_in);
		Matrix in_c_grad = weight_state.T().parralel_mult(dout);
		for(int i = 0 ; i < fan_in ; i++) {
			in_grad.v[i] = in_c_grad.v[i];
		}
		return in_grad;
	}
	
	@Override
	public String toString() {
		return "AffineLayer("+fan_in+", "+fan_out+", lr="+learning_rate+", reg="+regularization+", lrdecay="+learning_rate_decay+((calculate_dout)?"":", dout=false")+")";
	}
}
