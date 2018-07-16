package layers.reccurent;

import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Collection;

import layers.Parameters;
import layers.activations.SigmoidActivation;
import layers.activations.TanhActivation;
import layers.flat.DenseLayer;
import math.Initialisations;
import math.Matrix;
import math.TrainableMatrix;
import math.TrainableVector;

public class GRUCell extends RNNCell {

	public DenseLayer ru_layer, c_layer;
	/**
	 * @param state_size
	 * @param input_size
	 * @param p
	 */
	public GRUCell(int state_size, int input_size, Parameters p) {
		super(state_size, input_size);
		ru_layer = new DenseLayer(state_size+input_size, state_size*2, 0, "sigmoid", false, p);
		c_layer  = new DenseLayer(state_size+input_size, state_size, 0, "tanh", false, p);
		Initialisations.fill(c_layer.al.vectors.get("b"), 0);
		Initialisations.fill(ru_layer.al.vectors.get("b"), -1);
	}
	Matrix h_pr = new Matrix(0, 0);
	
	@Override
	public Matrix step(Matrix input, Matrix state, boolean training, RNNCellCache cache) {
		Matrix h_p = state;
		Matrix x = input;
		
		int W = input.width;
		
		Matrix x_h_p = new Matrix(W, input_size + state_size);
		
		Matrix.copy(x, x_h_p, 0, W, 0, input_size, 0, 0);
		Matrix.copy(h_p, x_h_p, 0, W, 0, state_size, 0, input_size);
		
		Matrix ru = ru_layer.forward(x_h_p, false);

		Matrix r = ru;
		Matrix u = ru.height_cut(state_size, state_size*2);
		r.height = state_size;
		
		if(h_pr.width != W) 
			h_pr = new Matrix(W, state_size);
		Matrix.copy(h_p, h_pr, 0, W, 0, state_size, 0, 0);
		h_pr.hadamart(r);
		
		Matrix x_h_pr = input.height_concat(h_pr);
		 
		Matrix c = c_layer.forward(x_h_pr, false); 
		
		Matrix h = new Matrix(input.width, state.height);
		for(int i = 0 ; i < h.height ; i++) {
			for(int j = 0 ; j < h.width ; j++) {
				h.v[i][j] = (1-u.v[i][j])*c.v[i][j] + h_p.v[i][j] * u.v[i][j];
			}
		}
		
		if(training) {
			cache.remember("h_p", state);
			cache.remember("c", c);
			
			cache.remember("r", r);
			cache.remember("u", u);
			
			cache.remember("c_cache", x_h_pr);
			cache.remember("ru_cache", x_h_p);
		}
		return h;
	}

	@Override
	public Matrix backward(Matrix dh, RNNCellCache cache) {
		
		Matrix r = cache.get_m("r");
		Matrix u = cache.get_m("u");
		
		Matrix c = cache.get_m("c");
		Matrix h_p = cache.get_m("h_p");
		int W = r.width;
		
		Matrix dc = new Matrix(u).scale(-1).add(1).hadamart(dh);
		
		
		c_layer.al.setCache(cache.get_m("c_cache"));
		TanhActivation act = (TanhActivation) c_layer.act;
		act.cache = c;
		Matrix dxhpr = c_layer.backward(dc, true);
		
		Matrix du = new Matrix(c).scale(-1).add(h_p).hadamart(dh);
		
		Matrix dhpr = dxhpr.height_cut(input_size, input_size+state_size);
		Matrix dr = new Matrix(dhpr).hadamart(h_p);
		
		
		Matrix dru = new Matrix(W, state_size*2);
		Matrix.copy(du, dru, 0, W, 0, state_size, 0, state_size);
		Matrix.copy(dr, dru, 0, W, 0, state_size, 0, 0);
		
		ru_layer.al.setCache(cache.get_m("ru_cache"));
		r.height = 2*state_size;
		
		SigmoidActivation act2 = (SigmoidActivation) ru_layer.act;
		
		act2.cache = r;
		
		Matrix dxhp = ru_layer.backward(dru, true);
		r.height = state_size;
		
		Matrix dhp = new Matrix(W, state_size);
		Matrix.copy(dxhp, dhp, 0, W, input_size, input_size+state_size, 0, 0);

		dhp.add(r.hadamart(dhpr));
		dhp.add(u.hadamart(dh));
		
		return dhp;
	}


	public void write_to_file(String name) {
		ru_layer.write_to_file(name+"/ru");
		c_layer.write_to_file(name+"/c");
	}
	
	public void load_from_file(String name) throws FileNotFoundException {
		ru_layer.load_from_file(name+"/ru");
		c_layer.load_from_file(name+"/c");
	}
	
	@Override
	public Collection<TrainableMatrix> get_trainable_matrices() {
		ArrayList<TrainableMatrix> ok = new ArrayList<>();
		ok.addAll(ru_layer.get_trainable_matrices());
		ok.addAll(c_layer.get_trainable_matrices());
		return ok;
	}

	@Override
	public Collection<TrainableVector> get_trainable_vectors() {
		ArrayList<TrainableVector> ok = new ArrayList<>();
		ok.addAll(ru_layer.get_trainable_vectors());
		ok.addAll(c_layer.get_trainable_vectors());
		return ok;
	}
}
