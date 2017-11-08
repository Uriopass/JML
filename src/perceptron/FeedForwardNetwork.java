package perceptron;

import layers.losses.Loss;
import math.Matrix;

public abstract class FeedForwardNetwork {
	public double last_average_loss = 0;
	public final int mini_batch = 32;
	
	public abstract Matrix forward(Matrix data);
	public abstract void print_architecture();
	public abstract Loss getLoss();
}