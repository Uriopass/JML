package optimizers;

import java.util.ArrayList;

import layers.Parameters;
import layers.TrainableMatrices;
import layers.TrainableVectors;
import math.Matrix;
import math.TrainableMatrix;
import math.TrainableVector;
import math.Vector;

public class SGDOptimizer extends Optimizer {
	ArrayList<TrainableMatrix> mats;
	ArrayList<TrainableVector> vec;
	
	public double learning_rate;
	public double learning_rate_decay;
	
	public SGDOptimizer(Parameters p) {
		super(p);
		mats = new ArrayList<>();
		vec = new ArrayList<>();

		learning_rate = p.get_as_double("lr", 0.001);
		learning_rate_decay = p.get_as_double("lrdecay", 1);
	}

	@Override
	public void init_mat(TrainableMatrices layer) {
		mats.addAll(layer.get_trainable_matrices());
	}

	@Override
	public void init_vec(TrainableVectors layer) {
		vec.addAll(layer.get_trainable_vectors());
	}
	
	@Override
	public void optimize() {
		for(TrainableMatrix m : mats) {
			Matrix m_grad = m.grad;
			for (int i = 0; i < m.height; i++) {
				for (int j = 0; j < m.width; j++) {
					// w -= lr * grad
					m.v[i][j] -= learning_rate * m_grad.v[i][j];
					m_grad.v[i][j] = 0;
				}
			}
		}
		
		for(TrainableVector v : vec) {
			Vector v_grad = v.grad;
			for (int i = 0; i < v.length; i++) {
				v.v[i] -= learning_rate * v_grad.v[i];
				v_grad.v[i] = 0;
			}
		}
	}
	

	@Override
	public void end_of_epoch() {
		learning_rate *= learning_rate_decay;
	}
}
