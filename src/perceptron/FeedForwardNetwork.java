package perceptron;

import java.util.ArrayList;

import layers.Layer;
import math.Matrix;

public abstract class FeedForwardNetwork {
	public ArrayList<Layer> layers;

	public double last_average_loss = 0;
	public final int mini_batch = 64;
	
	public FeedForwardNetwork() {
		layers = new ArrayList<Layer>();
	}
	
	public void add(Layer l) {
		layers.add(l);
	}
	
	public abstract Matrix forward(Matrix data);
	
	public void print_architecture() {
		for(Layer l : layers) {
			System.out.println("# - "+l);
		}
	}
	
}