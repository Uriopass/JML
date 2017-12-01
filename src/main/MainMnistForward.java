package main;

import java.io.File;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.List;
import java.util.Locale;

import datareaders.MnistReader;
import image.ImageConverter;
import layers.Parameters;
import layers.flat.DenseLayer;
import layers.losses.SoftmaxCrossEntropy;
import math.Matrix;
import math.RandomGenerator;
import math.Vector;
import optimizers.RMSOptimizer;
import perceptron.MLPMetrics;
import perceptron.FlatSequential;

public class MainMnistForward {

	// Chemin vers les données
	static String path = "";
	static String train_labelDB = path + "train-labels.idx1-ubyte";
	static String train_imageDB = path + "train-images.idx3-ubyte";

	static String test_labelDB = path + "t10k-labels.idx1-ubyte";
	static String test_imageDB = path + "t10k-images.idx3-ubyte";

	public static FlatSequential model;

	// Nombre d'epoque max
	public final static int EPOCHMAX = 20;

	// Nombre de données d'entrainements
	public static final int N = 20000;
	// Nombre de données de validation
	public static final int V = 10000;

	public static int out_size = 10;

	// Nombre de données de test
	public static int T = 10000;

	// Matrices de données
	public static Matrix train_data, validation_data, test_data;
	// Tableaux de références
	public static Matrix train_refs, validation_refs, test_refs;

	// Seed utilisé pour la reproducibilité
	public static long seed = System.currentTimeMillis();

	public static void load_data() {
		System.out.println("# Loading the database !");
		/* Lecteur d'image */

		if (!new File(train_imageDB).exists())
			throw new RuntimeException(train_imageDB + " not found");
		if (!new File(test_imageDB).exists())
			throw new RuntimeException(test_imageDB + " not found");
		if (!new File(train_labelDB).exists())
			throw new RuntimeException(train_labelDB + " not found");
		if (!new File(test_imageDB).exists())
			throw new RuntimeException(test_labelDB + " not found");

		List<int[][]> train_images = MnistReader.getImages(train_imageDB);
		int[] all_train_refs = MnistReader.getLabels(train_labelDB);

		List<int[][]> test_images = MnistReader.getImages(test_imageDB);
		int[] all_test_refs = MnistReader.getLabels(test_labelDB);
		System.out.println("# Database loaded !");

		/* Taille des images et donc de l'espace de representation */
		int SIZEW = 28 * 28;

		/* Creation des donnees */
		train_data = new Matrix(N, SIZEW);
		train_refs = new Matrix(N, out_size);

		validation_data = new Matrix(V, SIZEW);
		validation_refs = new Matrix(V, out_size);

		final int TOTAL = train_images.size();
		if (N + V > TOTAL) {
			throw new RuntimeException("N+V (" + (N + V) + ") > Total (" + TOTAL + ")");
		}

		/* Donnees d'apprentissage */
		for (int l = 0; l < N + V; l++) {
			int label = all_train_refs[l];
			double[] image = ImageConverter.image2VecteurReel(train_images.get(l));
			if (l < N) {
				train_data.set_column(l, new Vector(image));
				train_refs.set_column(l, Vector.one_hot(out_size, label));
			} else {
				validation_data.set_column(l-N, new Vector(image));
				validation_refs.set_column(l-N, Vector.one_hot(out_size, label));
			}
		}

		System.out.println("# Train/Validation set built with " + N + "/" + V + " images");

		/* Donnees de test */
		System.out.println("# Build test");

		test_data = new Matrix(T, SIZEW);
		test_refs = new Matrix(T, out_size);
		for (int i = 0; i < T; i++) {
			test_data.set_column(i, new Vector(ImageConverter.image2VecteurReel(test_images.get(i))));
			int label = all_test_refs[i];
			test_refs.set_column(i, Vector.one_hot(out_size, label));
		}
		System.out.println("# Test set built with " + T + " images");
	}

	public static void main(String[] args) {
		// On initialise le générateur aléatoire
		long time = System.currentTimeMillis();
		seed = 1510437982659L;
		RandomGenerator.init(seed);
		System.out.println("# Seed : " + seed);

		// Paramètres du modele
		Parameters p = new Parameters("reg=0.00005", "lr=0.01");
		// On crée notre modèle vide avec un mini_batch de 40
		model = new FlatSequential(40, new RMSOptimizer(p));
		load_data();

		// Modèle classique à 4 couches (entrée + cachée + cachée + sortie) avec 1000 et 100 neurones intermédiaires et des activations en sigmoide

		// Dout est inutile pour la première couche
		p.set("dout", "false");
		model.add(new DenseLayer(784, 500, 0, "tanh", true, p));
		p.set("dout", "true");
		model.add(new DenseLayer(500, 100, 0, "tanh", true, p));
		model.add(new DenseLayer(100, 10, 0, "none", false, p));
		// Fonction de coût entropie croisée avec softmax
		model.add(new SoftmaxCrossEntropy());

		System.out.println("# Model created with following architecture : ");
		model.print_architecture();

		System.out.println("# Processors : " + Runtime.getRuntime().availableProcessors());

		// Permet d'afficher les nombres avec une précision définie à l'avance
		DecimalFormatSymbols otherSymbols = new DecimalFormatSymbols(Locale.getDefault());
		otherSymbols.setDecimalSeparator('.');
		otherSymbols.setGroupingSeparator(',');
		DecimalFormat df2 = new DecimalFormat("#0.00", otherSymbols);
		DecimalFormat df5 = new DecimalFormat("#0.00000", otherSymbols);

		// Permet d'enregistrer toutes les données intéréssantes à écrire à la fin
		MLPMetrics metrics = new MLPMetrics();

		/*metrics.add_time_series(model.correct_count(train_data, train_refs) / (double) N,
				model.correct_count(validation_data, validation_refs) / (double) V,
				model.get_loss(train_data, train_refs));
		 */

		System.out.println("# Initialization took " + (System.currentTimeMillis() - time) + " ms");

		for (int i = 1; i <= EPOCHMAX; i++) {
			long t = System.currentTimeMillis();
			// On lance l'époque
			model.train_on_batch(train_data, train_refs);

			// Temps que cela a pris pour effectuer l'époque
			double epoch_time = (System.currentTimeMillis() - t) / 1000.;

			t = System.currentTimeMillis();
			double validation_accuracy = (100. * model.correct_count(validation_data, validation_refs)) / V;

			// Temps que cela a pris de regarder le nombre de données de validation correct
			double validation_forward_t = (System.currentTimeMillis() - t) / 1000.;

			double train_accuracy = (100. * model.last_correct_count) / N;

			metrics.add_time_series(train_accuracy, validation_accuracy, model.last_average_loss);

			// Exemple d'affichage : 
			// [==========] - 3  Top 1 accuracy (train, test) : 93.92% 89.23% loss 1.23 epoch time 3.21s test time 1.01s ETA 32.12s

			System.out.print(i + ((i >= 10) ? " " : "  "));
			System.out.print("Top 1 accuracy (train, val) : " + df2.format(train_accuracy) + "% "
					+ df2.format(validation_accuracy) + "% ");
			System.out.print("loss " + df5.format(model.last_average_loss) + " ");
			System.out.print("epoch time " + df2.format(epoch_time) + "s ");
			System.out.print("forward time " + df2.format(validation_forward_t) + "s");

			// Temps avant la fin de l'entraînement
			System.out.println(" ETA " + df2.format((EPOCHMAX - i) * (epoch_time)) + "s");
		}

		// Ecrit les données intéréssantes, comme la matrice de confusion etc.
		/*
		metrics.measure_and_write("./out_mnist/train", model, train_data, train_refs, true);
		metrics.measure_and_write("./out_mnist/test", model, test_data, test_refs, true);
		metrics.write_time_series_csv("./out_mnist/time_series.csv");
		 */
		// Valeur sur les données de test
		System.out.println(
				"Value at final test  : " + df2.format((100. * model.correct_count(test_data, test_refs)) / T) + "%");
	}
}