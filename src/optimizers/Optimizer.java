package optimizers;

import layers.Parameters;
import layers.TrainableMatrices;
import layers.TrainableVectors;

public abstract class Optimizer {
	Parameters p;
	public Optimizer(Parameters p) {
		this.p = p;
	}

	public abstract void init_mat(TrainableMatrices layer);
	public abstract void init_vec(TrainableVectors layer);
	public abstract void optimize();

	public abstract void end_of_epoch();
}
