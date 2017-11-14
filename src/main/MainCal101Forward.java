package main;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.Locale;

import datareaders.Cal101Reader;
import layers.Parameters;
import layers.activations.SigmoidActivation;
import layers.flat.AffineLayer;
import layers.losses.SoftmaxCrossEntropy;
import math.Matrix;
import math.RandomGenerator;
import math.Vector;
import perceptron.MLPMetrics;
import perceptron.MultiLayerPerceptron;

public class MainCal101Forward {

	// Modèle à utiliser
	public static MultiLayerPerceptron model;

	// Nombre d'epoque max
	public final static int EPOCHMAX = 30;

	// Nombre de données d'entrainements
	public static final int N = 3100;

	// Nombre de données de validation
	public static final int V = 1000;

	// Nombre de données de test
	public static int T = 2000;

	// Matrices de données
	public static Matrix train_data, test_data, validation_data;
	// Tableaus de labels
	public static int[] train_refs, test_refs, validation_refs;

	// Seed à utiliser pour la reproducibilité
	public static long seed = System.currentTimeMillis();

	public static void load_data() {
		// On calcule le nombre de données divisible par le mini_batch

		System.out.println("# Loading the database !");
		Cal101Reader.load_data();
		System.out.println("# Database loaded !");
		// On charge les données d'entrainements et de validation
		System.out.println("# Build train & validation");
		train_data = new Matrix(N, 784);
		train_refs = new int[N];
		validation_data = new Matrix(N, 784);
		validation_refs = new int[N];
		int i = 0;
		for (double[] v : Cal101Reader.get_train_data().transpose().v) {
			if(i < N) {
				train_data.set_column(i, new Vector(v));
				train_refs[i] = Cal101Reader.get_train_refs()[i];
			} else if(i < N+V) {
				validation_data.set_column(i-N, new Vector(v));
				validation_refs[i-N] = Cal101Reader.get_train_refs()[i];
			} else {
				break;
			}
			i++;
		}

		System.out.println("# Train/Validation set built with " + N + "/" + V + " images");
		
		// On charge les données de test
		System.out.println("# Build test");
		test_data = new Matrix(T, 784);
		test_refs = new int[T];
		i = 0;
		for (double[] v : Cal101Reader.get_test_data().transpose().v) {
			test_data.set_column(i, new Vector(v));
			test_refs[i] = Cal101Reader.get_test_refs()[i];
			i++;
			if (i >= T)
				break;
		}
		System.out.println("# Test set built with " + T + " images");
	}

	public static void main(String[] args) {
		seed = 1510445586196L;
		// On initialise le générateur aléatoire
		long time = System.currentTimeMillis();
		RandomGenerator.init(seed);
		System.out.println("# Seed : " + seed);
		
		// On crée notre modèle vide avec un mini_batch de 100
		model = new MultiLayerPerceptron(100);
		load_data();
		
		Parameters p = new Parameters("reg=0.0001", "lr=0.01");
		
		// Modèle classique à 4 couches (entrée + cachée + cachée + sortie) avec 1000/300 neurones intermédiaires et des activations en sigmoide
		model.add(new AffineLayer(784, 1300, true, p));
		model.add(new SigmoidActivation());
		model.add(new AffineLayer(1300, 500, true, p));
		model.add(new SigmoidActivation());
		model.add(new AffineLayer(500, 101, true, p));
		
		// Fonction de coût entropie croisée avec softmax
		model.add(new SoftmaxCrossEntropy());
		
		System.out.println("# Model created with following architecture : ");
		model.print_architecture();

		System.out.println("# Processors : " + Runtime.getRuntime().availableProcessors());

		// Permet d'afficher simplement les nombres avec une précision définie à l'avance
		DecimalFormatSymbols otherSymbols = new DecimalFormatSymbols(Locale.getDefault());
		otherSymbols.setDecimalSeparator('.');
		otherSymbols.setGroupingSeparator(',');
		DecimalFormat df2 = new DecimalFormat("#0.00", otherSymbols);
		DecimalFormat df5 = new DecimalFormat("#0.00000", otherSymbols);
		
		// Permet d'enregistrer toutes les données intéréssantes à écrire à la fin
		MLPMetrics metrics = new MLPMetrics();
		metrics.add_time_series(model.correct_count(train_data, train_refs)/(double)N, 
								model.correct_count(validation_data, validation_refs)/(double)V, 
								model.get_loss(train_data, train_refs));

		System.out.println("# Initialization took " + (System.currentTimeMillis() - time) + " ms");

		for (int i = 1; i <= EPOCHMAX; i++) {
			long t = System.currentTimeMillis();
			// On lance l'époque
			model.epoch(train_data, train_refs);

			// Temps que cela a pris pour effectuer l'époque
			double epoch_time = (System.currentTimeMillis() - t) / 1000.;
			
			t = System.currentTimeMillis();
			double validation_accuracy = (100. * model.correct_count(validation_data, validation_refs)) / V;
			
			// Temps que cela a pris de regarder le nombre de données de test correct
			double validation_forward_t = (System.currentTimeMillis() - t) / 1000.;
			
			double train_accuracy = (100. * model.last_correct_count) / N;

			metrics.add_time_series(train_accuracy, validation_accuracy, model.last_average_loss);
			
			// Exemple d'affichage : 
			// [==========] - 3  Top 1 accuracy (train, test) : 53.92% 43.23% loss 1.23 epoch time 8.21s test time 1.01s ETA 32.12s
			
			System.out.print(i + ((i >= 10) ? " " : "  "));
			System.out.print("Top 1 accuracy (validation, test) : " + df2.format(train_accuracy) + "% "
					+ df2.format(validation_accuracy) + "% ");
			System.out.print("loss " + df5.format(model.last_average_loss) + " ");
			System.out.print("epoch time " + df2.format(epoch_time) + "s ");
			System.out.print("validation time " + df2.format(validation_forward_t) + "s");
			System.out.println(" ETA " + df2.format((EPOCHMAX - i) * (epoch_time)) + "s");
		}

		metrics.measure_and_write("./out_cal101/train", model, train_data, train_refs, false);
		metrics.measure_and_write("./out_cal101/test", model, test_data, test_refs, false);
		metrics.write_time_series_csv("./out_cal101/accuracy.csv");
		System.out.println("Value at final test  : "+df2.format((100. * model.correct_count(test_data, test_refs)) / T)+"%");
	}
}