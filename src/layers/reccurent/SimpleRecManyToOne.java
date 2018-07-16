package layers.reccurent;

import java.util.ArrayList;
import java.util.Collection;

import layers.FlatLayer;
import layers.Parameters;
import layers.TrainableMatrices;
import layers.TrainableVectors;
import layers.activations.ActivationParser;
import layers.flat.AffineLayer;
import math.Matrix;
import math.TrainableMatrix;
import math.TrainableVector;
public class SimpleRecManyToOne implements TrainableMatrices, TrainableVectors {
	public Matrix state;
	
	public FlatLayer state_act;
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
		Matrix in_c = new Matrix(in.width, in.height+state.height);
		
		for(int i = 0 ; i < in_c.width ; i++) {
			for(int j = 0 ; j < in.height ; j++) {
				in_c.v[j][i] = in.v[j][i];
			}
			for(int j = 0 ; j < state.height ; j++) {
				in_c.v[j+in.height][i] = state.v[j][i];
			}
		}
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
			//System.out.print(i+" ");
			//caches.get(i).T().print_values();
			//dact.T().print_values();
			Matrix dcach = state_aff.backward(dact, train);
			
			for(int w = 0 ; w < dcach.width ; w++) {
				for(int j = 0 ; j < state_size ; j++) {
					dhid.v[j][w] = dcach.v[fan_in+j][w];
				}
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
