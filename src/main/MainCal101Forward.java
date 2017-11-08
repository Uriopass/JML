package main;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.Locale;

import datareaders.Cal101Reader;
import datareaders.MnistReader;
import image.ImageConverter;
import layers.Parameters;
import layers.activations.SoftmaxActivation;
import layers.flat.DenseLayer;
import layers.flat.SplitAffineLayer;
import layers.losses.EntropyLoss;
import layers.losses.SoftmaxCrossEntropy;
import math.Matrix;
import math.RandomGenerator;
import math.Vector;
import perceptron.MultiLayerPerceptron;

public class MainCal101Forward {

	public static MultiLayerPerceptron model;

	// Nombre d'epoque max
	public final static int EPOCHMAX = 5;

	public static final int N_t = 4100;

	public static int T_t = 2000;

	public static int N;
	public static int T;

	public static Matrix trainData, testData;
	public static int[] trainRefs, testRefs;

	public static long seed = System.currentTimeMillis();

	public static void load_data() {
		N = N_t - (N_t % model.mini_batch);
		T = T_t - (T_t % model.mini_batch);

		System.out.println("# Loading the database !");
		trainData = new Matrix(N, 784);
		trainRefs = new int[N];
		int i = 0;
		for(double[] v : Cal101Reader.get_train_data().transpose().v) {
			trainData.set_column(i, new Vector(v));
			trainRefs[i] = Cal101Reader.get_train_refs()[i];
			i++;
			if(i >= N)
				break;
		}
		
		System.out.println("# Database loaded !");
		System.out.println("# Train set " + N + " images");

		/* Donnees de test */
		System.out.println("# Build test");
		testData = new Matrix(T, 784);
		testRefs = new int[T];
		i = 0;
		for(double[] v : Cal101Reader.get_test_data().transpose().v) {
			testData.set_column(i, new Vector(v));
			testRefs[i] = Cal101Reader.get_test_refs()[i];
			i++;
			if(i >= T)
				break;
		}
		System.out.println("# Test set " + T + " images");
	}
	
	public static void main(String[] args) {
		long time = System.currentTimeMillis();
		RandomGenerator.init(seed);
		model = new MultiLayerPerceptron();
		load_data();
		Parameters p = new Parameters("reg=0", "lr=0.005");
		model.add(new DenseLayer(trainData.height, 500, 0, "tanh", true, p));
		model.add(new DenseLayer(500, 101, 0, "tanh", true, p));
		model.add(new SoftmaxCrossEntropy());
		System.out.println("# Model created with following architecture : ");
		model.print_architecture();
	
		System.out.println("# Seed : " + seed);
		System.out.println("# Processors : " + Runtime.getRuntime().availableProcessors());

		double[] trainAccuracy = new double[EPOCHMAX + 1];
		double[] testAccuracy = new double[EPOCHMAX + 1];
		
		DecimalFormatSymbols otherSymbols = new DecimalFormatSymbols(Locale.getDefault());
		otherSymbols.setDecimalSeparator('.');
		otherSymbols.setGroupingSeparator(',');
		DecimalFormat df = new DecimalFormat("#0.00", otherSymbols);
		
		System.out.println("# Initialization took " + (System.currentTimeMillis() - time) + " ms");

		for (int i = 1; i <= EPOCHMAX; i++) {
			long t = System.currentTimeMillis();
			model.epoch(trainData, trainRefs);
			
			double rms = (System.currentTimeMillis() - t) / 1000.;
			t = System.currentTimeMillis();
			testAccuracy[i] = (100. * model.correct_count(testData, testRefs)) / T;
			double test_forward_t = (System.currentTimeMillis() - t) / 1000.;
			trainAccuracy[i] = (100. * model.last_correct_count) / N;
			
			System.out.print(i + ((i >= 10) ? " " : "  "));
			System.out.print("Top 1 accuracy (train, test) : " + df.format(trainAccuracy[i]) + "% "
					+ df.format(testAccuracy[i]) + "% ");
			System.out.print("loss " + model.last_average_loss + "\t");
			System.out.print("epoch time " + df.format(rms) + "s ");
			System.out.print("test time " + df.format(test_forward_t) + "s");
			System.out.println(" ETA " + df.format((EPOCHMAX - i) * (rms)) + "s");
			//model.write_weights("temp");
			System.out.println();
		}
		
		((DenseLayer)model.layers.get(0)).al.weight.visualize("test", 28, 25, 20, true);

		for (double f : trainAccuracy) {
			System.out.print(df.format(f) + ";");
		}
		System.out.println();
		for (double f : testAccuracy) {
			System.out.print(df.format(f) + ";");
		}
		System.out.println();
		// System.out.print("MLPerceptron On the test set : ");
		// System.out.println((100f * model.correct_count(testData, testRefs)) / T);
	}
}