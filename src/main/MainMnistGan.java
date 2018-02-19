package main;

import java.io.File;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.List;
import java.util.Locale;

import datareaders.MnistReader;
import gan.GenerativeAdversarialNetwork;
import image.ImageConverter;
import layers.Parameters;
import layers.flat.DenseLayer;
import layers.losses.SoftmaxCrossEntropy;
import math.Matrix;
import math.RandomGenerator;
import math.Vector;
import perceptron.MLPMetrics;
import perceptron.FlatSequential;

public class MainMnistGan {
	// Chemin vers les donn�es
	static String path = "";
	static String train_labelDB = path + "train-labels.idx1-ubyte";
	static String train_imageDB = path + "train-images.idx3-ubyte";

	public static GenerativeAdversarialNetwork model;

	// Nombre d'epoque max
	public final static int EPOCHMAX = 30;

	// Nombre de donn�es d'entrainements
	public static final int N = 16000;

	// Matrices de donn�es
	public static Matrix train_data;
	
	// Seed utilis� pour la reproducibilit�
	public static long seed = System.currentTimeMillis();

	public static void load_data() {
		System.out.println("# Loading the database !");
		/* Lecteur d'image */

		if (!new File(train_imageDB).exists())
			throw new RuntimeException(train_imageDB + " not found");

		List<int[][]> train_images = MnistReader.getImages(train_imageDB);
		System.out.println("# Database loaded !");

		/* Taille des images et donc de l'espace de representation */
		int SIZEW = 28 * 28;

		/* Creation des donnees */
		train_data = new Matrix(N, SIZEW);

		final int TOTAL = train_images.size();

		/* Donnees d'apprentissage */
		for (int l = 0; l < N; l++) {
			double[] image = ImageConverter.image2VecteurReel(train_images.get(l));
			if (l < N) {
				train_data.set_column(l, new Vector(image));
			} 
		}

		System.out.println("# Train set built with " + N+ " images");
	}

	public static void main(String[] args) {
		// On initialise le g�n�rateur al�atoire
		long time = System.currentTimeMillis();
		seed = 1510437982659L;
		RandomGenerator.init(seed);
		System.out.println("# Seed : " + seed);

		model = new GenerativeAdversarialNetwork(784, 8);
		load_data();
		
		System.out.println("# Processors : " + Runtime.getRuntime().availableProcessors());

		// Permet d'afficher les nombres avec une pr�cision d�finie � l'avance
		DecimalFormatSymbols otherSymbols = new DecimalFormatSymbols(Locale.getDefault());
		otherSymbols.setDecimalSeparator('.');
		otherSymbols.setGroupingSeparator(',');
		DecimalFormat df2 = new DecimalFormat("#0.00", otherSymbols);
		DecimalFormat df5 = new DecimalFormat("#0.00000", otherSymbols);

		System.out.println("# Initialization took " + (System.currentTimeMillis() - time) + " ms");

		int mini_batch = 32;
		int counterlala = 0;
		
		for (int i = 1; i <= EPOCHMAX; i++) {
			long t = System.currentTimeMillis();

			int tenth = train_data.width / (mini_batch * 10);
			System.out.print("[");
			Vector total_loss = new Vector(2);
			double l1=0, l2=0;
			for (int k = 0; k < train_data.width / mini_batch; k++) {
				if(k%tenth == 0) {
					System.out.print("=");
					System.out.println(l1 + " " + l2);
					model.test_generator().T().visualize("fig2/test_gan"+(counterlala++), 28, 10, 10, true, true, false);
				}

				l1 = model.train_discriminator(32, train_data, k*mini_batch, (k+1)*mini_batch);
				l2 = model.train_generator(64);
				total_loss.v[0] += l1;
				total_loss.v[1] += l2;
			}
			total_loss.scale(1. / (train_data.width / mini_batch));
			System.out.print("] ");
			
			System.out.println(total_loss);

			// Temps que cela a pris pour effectuer l'�poque
			double epoch_time = (System.currentTimeMillis() - t) / 1000.;

			t = System.currentTimeMillis(); 
			
			System.out.print(i + ((i >= 10) ? " " : "  "));
			System.out.print("epoch time " + df2.format(epoch_time) + "s ");

			// Temps avant la fin de l'entra�nement
			System.out.println(" ETA " + df2.format((EPOCHMAX - i) * (epoch_time)) + "s");
		}
	}
}
