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
import perceptron.MultiLayerPerceptron;

public class MainMnistGan {
	// Chemin vers les données
	static String path = "";
	static String train_labelDB = path + "train-labels.idx1-ubyte";
	static String train_imageDB = path + "train-images.idx3-ubyte";

	public static GenerativeAdversarialNetwork model;

	// Nombre d'epoque max
	public final static int EPOCHMAX = 20;

	// Nombre de données d'entrainements
	public static final int N = 8000;

	// Matrices de données
	public static Matrix train_data;
	
	// Seed utilisé pour la reproducibilité
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
		// On initialise le générateur aléatoire
		long time = System.currentTimeMillis();
		seed = 1510437982659L;
		RandomGenerator.init(seed);
		System.out.println("# Seed : " + seed);

		// On crée notre modèle vide avec un mini_batch de 40
		model = new GenerativeAdversarialNetwork(784, 30);
		load_data();
		
		System.out.println("# Processors : " + Runtime.getRuntime().availableProcessors());

		// Permet d'afficher les nombres avec une précision définie à l'avance
		DecimalFormatSymbols otherSymbols = new DecimalFormatSymbols(Locale.getDefault());
		otherSymbols.setDecimalSeparator('.');
		otherSymbols.setGroupingSeparator(',');
		DecimalFormat df2 = new DecimalFormat("#0.00", otherSymbols);
		DecimalFormat df5 = new DecimalFormat("#0.00000", otherSymbols);

		System.out.println("# Initialization took " + (System.currentTimeMillis() - time) + " ms");

		int mini_batch = 20;
		
		for (int i = 1; i <= EPOCHMAX; i++) {
			long t = System.currentTimeMillis();

			int tenth = train_data.width / (mini_batch * 10);
			System.out.print("[");
			for (int k = 0; k < train_data.width / mini_batch; k++) {
				if(k%tenth == 0) {
					System.out.print("=");
				}
			
				model.train_gan(20, train_data, k*mini_batch, (k+1)*mini_batch);
			}
			System.out.print("] ");

			// Temps que cela a pris pour effectuer l'époque
			double epoch_time = (System.currentTimeMillis() - t) / 1000.;

			t = System.currentTimeMillis();
			
			System.out.print(i + ((i >= 10) ? " " : "  "));
			System.out.print("epoch time " + df2.format(epoch_time) + "s ");

			// Temps avant la fin de l'entraînement
			System.out.println(" ETA " + df2.format((EPOCHMAX - i) * (epoch_time)) + "s");
			
			model.test_generator(10).T().visualize("test_gan", 28, 10, 1, true, true);
		}
	}
}
