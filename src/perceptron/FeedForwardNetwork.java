package perceptron;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Scanner;

import layers.Layer;
import layers.Parameters;
import layers.activations.TanhActivation;
import layers.flatlayers.AffineLayer;
import layers.flatlayers.BatchnormLayer;
import layers.flatlayers.SoftmaxCrossEntropy;
import math.Matrix;
import math.Vector;

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