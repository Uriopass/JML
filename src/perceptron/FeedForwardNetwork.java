package perceptron;

import layers.losses.Loss;
import math.Matrix;
/**
 * Classe g�n�rique de r�seau s�quentiel
 */
public abstract class FeedForwardNetwork {
	public double last_average_loss = 0;
	public int mini_batch;

	public abstract Matrix forward(Matrix data);

	public abstract void print_architecture();

	public abstract Loss get_loss_layer();
}