package layers.reccurent;

import layers.TrainableMatrices;
import layers.TrainableVectors;
import math.Matrix;

public abstract class RNNCell implements TrainableMatrices, TrainableVectors {
	public int state_size, output_size, input_size;

	public RNNCell(int state_size, int input_size) {
		this.state_size = state_size;
		this.input_size = input_size;
	}

	public abstract Matrix step(Matrix input, Matrix state, boolean training, RNNCellCache cache);

	public abstract Matrix backward(Matrix dstate, RNNCellCache cache);
}
