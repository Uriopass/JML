package gan;

import layers.Parameters;
import layers.activations.ReLUActivation;
import layers.flat.AffineLayer;
import layers.flat.DropoutLayer;
import layers.losses.Loss;
import layers.losses.SigmoidBinaryEntropyLoss;
import math.Matrix;
import math.RandomGenerator;
import optimizers.RMSOptimizer;
import perceptron.FlatSequential;

public class GenerativeAdversarialNetwork {
	int latent_space;
	int input_size;
	public FlatSequential generator, discriminator;
	
	public Matrix coordinates;

	public GenerativeAdversarialNetwork(int input_size, int latent_space_size) {
		this.latent_space = latent_space_size;
		this.input_size = input_size;

		Parameters p = new Parameters("lr=0.001", "reg=0.00001",  "dout=false");

		generator = new FlatSequential(new RMSOptimizer(p));
		discriminator = new FlatSequential(new RMSOptimizer(p));

		generator.add(new AffineLayer(latent_space_size, 32, true, p));
		generator.add(new ReLUActivation());
		p.set("dout", "true");
		
		generator.add(new AffineLayer(32, 32, true, p));
		generator.add(new ReLUActivation());
		
		generator.add(new AffineLayer(32, 2, true, p));
		
		discriminator.add(new AffineLayer(2, 32, true, p));
		discriminator.add(new ReLUActivation());
		discriminator.add(new DropoutLayer(0.3));
		
		discriminator.add(new AffineLayer(32, 32, true, p));
		discriminator.add(new ReLUActivation());
		discriminator.add(new DropoutLayer(0.3));
		
		discriminator.add(new AffineLayer(32, 32, true, p));
		discriminator.add(new ReLUActivation());
		discriminator.add(new DropoutLayer(0.3));
		
		discriminator.add(new AffineLayer(32, 1, true, p));
		discriminator.add(new SigmoidBinaryEntropyLoss());
		
		coordinates = RandomGenerator.normal(0, 1, 100, latent_space_size);	
	}
	
	int counter = 0;
	public double train_discriminator(int samples, Matrix real_data, int start, int end) {
		// Generate images
		Matrix base = RandomGenerator.normal(0, 1, samples, latent_space);
		Matrix generated = generator.forward(base, false);

		Matrix fw = new Matrix(samples + (end-start), this.input_size);
		
		// Add real images
		for (int i = 0; i < end - start; i++) {
			fw.set_column(i, real_data.get_column(i + start));
		}
		for (int i = 0; i < samples; i++) {
			fw.set_column(i+(end-start), generated.get_column(i));
		}
		
		//((DenseLayer)discriminator.get_layers().get(0)).al.matrices.get("w").visualize("fig/okok"+(counter++), 28, 10, 3, true, false);
		//fw.T().visualize("fig/okok"+(counter++), 28, fw.width, 1, true, true);
		fw = discriminator.forward(fw, true);
		// Feed discriminator references
		Loss l = discriminator.get_loss_layer();
		Matrix refs_d = new Matrix(samples + (end - start), 1);
		for (int i = 0; i < end - start; i++) {
			refs_d.v[0][i] = 1;
		}
		l.feed_ref(refs_d);

		// Train discriminator
		discriminator.backward(fw, true);
		return discriminator.get_loss_layer().loss;
	}

	public double train_generator(int samples) {
		// Regenerate images
		Matrix base = RandomGenerator.normal(0, 1, samples, latent_space);
		Matrix fw = generator.forward(base, true);

		// Discriminate
		fw = discriminator.forward(fw, true);
		Loss l = discriminator.get_loss_layer();
		// Pretends it's genuine
		Matrix refs_d = new Matrix(samples, 1).fill(1);
		l.feed_ref(refs_d);

		// Find dout from image
		Matrix back = discriminator.backward(fw, false);
		// Learn to better generate
		generator.backward(back, true);
		return discriminator.get_loss_layer().loss;
	}
	public Matrix test_generator() {
		return generator.forward(coordinates, false);
	}
}
