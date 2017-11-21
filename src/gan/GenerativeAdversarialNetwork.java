package gan;

import java.util.ArrayList;

import layers.FlatLayer;
import layers.Parameters;
import layers.flat.ConstantLayer;
import layers.flat.DenseLayer;
import layers.flat.GaussianNoise;
import layers.losses.EntropyLoss;
import layers.losses.Loss;
import math.Matrix;

public class GenerativeAdversarialNetwork {
	int latent_space;
	int input_size;
	ArrayList<FlatLayer> generator;
	ArrayList<FlatLayer> discriminator;
	
	public GenerativeAdversarialNetwork(int input_size, int latent_space_size) {
		this.latent_space = latent_space_size;
		this.input_size = input_size;
		
		generator = new ArrayList<>();
		discriminator = new ArrayList<>();
		
		Parameters p = new Parameters("lr=0.001", "reg=0", "dout=false");
		
		generator.add(new ConstantLayer(1, latent_space_size, 0));
		generator.add(new GaussianNoise(1));
		generator.add(new DenseLayer(latent_space_size, 200, 0, "tanh", true, p));
		p.set("dout", "true");
		generator.add(new DenseLayer(200, 400, 0, "tanh", true, p));
		generator.add(new DenseLayer(400, 600, 0, "tanh", true, p));
		generator.add(new DenseLayer(600, input_size, 0, "sigmoid", true, p));
		
		discriminator.add(new DenseLayer(784, 500, 0, "tanh", true, p));
		discriminator.add(new DenseLayer(500, 100, 0, "tanh", true, p));
		discriminator.add(new DenseLayer(100, 1, 0, "sigmoid", false, p));
		discriminator.add(new EntropyLoss());
	}
	
	public Vector train_gan(int samples, Matrix real_data, int start, int end) {
		
		// Generate images
		((ConstantLayer) generator.get(0)).width = samples+(end-start);
		Matrix forward = null;
		for(FlatLayer l : generator) {
			forward = l.forward(forward, true);
		}
		
		// Add real images
		for(int i = start ; i < end ; i++) {
			forward.set_column(i-start, real_data.get_column(i));
		}
		
		// Discriminate
		for(FlatLayer l : discriminator) {
			forward = l.forward(forward, true);
		}
		
		// Feed discriminator references
		Loss l = ((Loss)discriminator.get(discriminator.size()-1));
		Matrix refs_d = new Matrix(samples, 1);
		for(int i = 0 ; i < end-start ; i++) {
			refs_d.v[0][i+samples] = 1;
		}
		l.feed_ref(refs_d);
		
		// Train discriminator
		Matrix backward = l.backward(forward);
		
		for(int i = discriminator.size()-1 ; i >= 0 ; i--) {
			backward = discriminator.get(i).backward(backward);
			discriminator.get(i).apply_gradient();
		}
		
		
		
		for(int i = generator.size()-1 ; i >= 0 ; i--) {
			backward = generator.get(i).backward(backward);
			generator.get(i).apply_gradient();
		}
	}
	
	public Matrix test_generator(int samples) {
		((ConstantLayer) generator.get(0)).width = samples;
		Matrix forward = null;
		for(FlatLayer l : generator) {
			forward = l.forward(forward, true);
		}
		return forward;
	}
}
