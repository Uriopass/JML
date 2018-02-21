package optimizers;

import java.util.HashMap;
import java.util.Map.Entry;

import layers.Parameters;
import layers.TrainableMatrices;
import layers.TrainableVectors;
import math.Matrix;
import math.TrainableMatrix;
import math.TrainableVector;
import math.Vector;

public class RMSOptimizer extends Optimizer {
	HashMap<TrainableMatrix, Matrix> acc_mats;
	HashMap<TrainableVector, Vector> acc_vec;
	
	public double learning_rate;
	public double learning_rate_decay;
	public double clip_low_bound;
	public double clip_high_bound;
	
	
	
	
	public RMSOptimizer(Parameters p) {
		super(p);
		acc_mats = new HashMap<>();
		acc_vec = new HashMap<>();
		learning_rate = p.get_as_double("lr", 0.001);
		learning_rate_decay = p.get_as_double("lrdecay", 1);
		clip_low_bound = p.get_as_double("clip_l", Double.NEGATIVE_INFINITY);
		clip_high_bound = p.get_as_double("clip_h", Double.POSITIVE_INFINITY);
	}

	@Override
	public void init_mat(TrainableMatrices layer) {
		for(TrainableMatrix tm : layer.get_trainable_matrices()) {
			acc_mats.put(tm, new Matrix(tm.width, tm.height));
		}
	}

	@Override
	public void init_vec(TrainableVectors layer) {
		for(TrainableVector tv : layer.get_trainable_vectors()) {
			acc_vec.put(tv, new Vector(tv.length));
		}
	}
	
	public Iterable<TrainableMatrix> get_mats() {
		return acc_mats.keySet();
	}
	
	public Iterable<TrainableVector> get_vecs() {
		return acc_vec.keySet();
	}
	
	@Override
	public void optimize() {
		double gamma = p.get_as_double("gamma", 0.9);
		double eps = p.get_as_double("eps", 1e-6);
		
		for(Entry<TrainableMatrix, Matrix> entry : acc_mats.entrySet()) {
			Matrix acc = entry.getValue();
			TrainableMatrix w = entry.getKey();
			Matrix w_grad = w.grad;
			for (int l = 0; l < acc.height; l++) {
				for (int m = 0; m < acc.width; m++) {
					// acc[t+1] = gamma * acc[t] + (1 - gamma) * grad^2
					acc.v[l][m] = gamma * acc.v[l][m] + (1 - gamma) * w_grad.v[l][m] * w_grad.v[l][m];
	
					// grad *= - lr / sqrt(epsilon + acc[t+1])
					w_grad.v[l][m] *= -learning_rate / (Math.sqrt(eps + acc.v[l][m]));
	
					// w += grad
					w.v[l][m] += w_grad.v[l][m];
					w.v[l][m] = Math.min(clip_high_bound, Math.max(clip_low_bound, w.v[l][m]));
					
					
					w_grad.v[l][m] = 0;
				}
			}
		}
		for(Entry<TrainableVector, Vector> entry : acc_vec.entrySet()) {
			Vector acc = entry.getValue();
			TrainableVector b = entry.getKey();
			Vector b_grad = b.grad;
			for (int l = 0; l < acc.length; l++) {
				acc.v[l] = gamma * acc.v[l] + (1 - gamma) * b_grad.v[l] * b_grad.v[l];
				b_grad.v[l] *= -learning_rate / (Math.sqrt(eps + acc.v[l]));
				b.v[l] += b_grad.v[l];
				b_grad.v[l] = 0;
			}
		}
	}

	@Override
	public void end_of_epoch() {
		learning_rate *= learning_rate_decay;
	}
}
