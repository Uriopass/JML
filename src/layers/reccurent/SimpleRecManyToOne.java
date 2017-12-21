package layers.reccurent;

import java.util.ArrayList;
import java.util.Collection;

import layers.Parameters;
import layers.TrainableMatrices;
import layers.TrainableVectors;
import layers.activations.ActivationLayer;
import layers.activations.ActivationParser;
import layers.flat.AffineLayer;
import math.Matrix;
import math.TrainableMatrix;
import math.TrainableVector;
public class SimpleRecManyToOne implements TrainableMatrices, TrainableVectors {
	public Matrix state;
	
	public ActivationLayer state_act;
	public AffineLayer state_aff;
	public AffineLayer out_aff;

	public int fan_in, fan_out;
	public ArrayList<Matrix> caches;
	
	public int state_size;
	
	public SimpleRecManyToOne(int fan_in, int fan_out, boolean init, int state_size, String state_act, Parameters p) {
		this.fan_in = fan_in;
		this.fan_out = fan_out;
		this.state_size = state_size;
		state = new Matrix(1, state_size);
		this.state_act = ActivationParser.get_activation_by_name(state_act);
		state_aff = new AffineLayer(fan_in+state_size, state_size, init, p);
		out_aff = new AffineLayer(state_size, fan_out, init, p);
		caches = new ArrayList<>();
	}
	
	public void initRec() {
		caches.clear();
		state.fill(0);
	}
	
	public void tick(Matrix in, boolean training) {
		Matrix in_c = new Matrix(1, in.height+state.height);
		in_c.set_column(0, in.get_column(0).append(state.get_column(0)));
		if(training)
			caches.add(in_c);
		state = state_act.forward(state_aff.forward(in_c, false), training);
	}
	
	public Matrix get_out(boolean training) {
		return out_aff.forward(state, training);
	}
	
	public void backwardAll(Matrix dout, boolean train) {
		Matrix dhid = out_aff.backward(dout, train);
		
		for(int i = caches.size()-1 ; i >= 0 ; i--) {
			Matrix dact = state_act.backward(dhid, train);
			state_aff.setCache(caches.get(i));
			Matrix dcach = state_aff.backward(dact, train);
			
			for(int j = 0 ; j < state_size ; j++) {
				dhid.v[j][0] = dcach.v[fan_in+j][0];
			}
		}
	}

	@Override
	public Collection<TrainableVector> get_trainable_vectors() {
		ArrayList<TrainableVector> ok = new ArrayList<>();
		ok.addAll(state_aff.get_trainable_vectors());
		ok.addAll(out_aff.get_trainable_vectors());
		return ok;
	}

	@Override
	public Collection<TrainableMatrix> get_trainable_matrices() {
		ArrayList<TrainableMatrix> ok = new ArrayList<>();
		ok.addAll(state_aff.get_trainable_matrices());
		ok.addAll(out_aff.get_trainable_matrices());
		return ok;
	}
}
