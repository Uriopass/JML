package main;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.Locale;

import datareaders.Cal101Reader;
import layers.Parameters;
import layers.flat.DenseLayer;
import layers.losses.SoftmaxCrossEntropy;
import math.Matrix;
import math.RandomGenerator;
import math.Vector;
import optimizers.RMSOptimizer;
import perceptron.FlatSequential;

public class MainCal101Forward {

	// Mod�le � utiliser
	public static FlatSequential model;

	// Nombre d'epoque max
	public final static int EPOCHMAX = 50;

	// Nombre de donn�es d'entrainements
	public static final int N = 4100;

	// Nombre de donn�es de validation
	public static final int V = 0;

	// Nombre de donn�es de test
	public static int T = 2307;
	
	public static int out_size = 101;

	// Matrices de donn�es
	public static Matrix train_data, test_data, validation_data;
	// Tableaus de labels
	public static Matrix train_refs, test_refs, validation_refs;

	// Seed � utiliser pour la reproducibilit�
	public static long seed = System.currentTimeMillis();

	public static void load_data() {
		// On calcule le nombre de donn�es divisible par le mini_batch

		System.out.println("# Loading the database !");
		Cal101Reader.load_data();
		System.out.println("# Database loaded !");
		// On charge les donn�es d'entrainements et de validation
		System.out.println("# Build train & validation");
		train_data = new Matrix(N*3, 784);
		train_refs = new Matrix(N*3, out_size);
		validation_data = new Matrix(V, 784);
		validation_refs = new Matrix(train_refs.width, train_refs.height);
		int i = 0;
		for (double[] v : Cal101Reader.get_train_data().transpose().v) {
			if(i < N) {
				train_data.set_column(i, new Vector(v));
				train_refs.set_column(i, Vector.one_hot(out_size, Cal101Reader.get_train_refs()[i]));
			} else if(i < N+V) {
				validation_data.set_column(i-N, new Vector(v));
				validation_refs.set_column(i, Vector.one_hot(out_size, Cal101Reader.get_train_refs()[i]));
			} else {
				break;
			}
			i++;
		}
		for(int k = 0 ; k < train_data.width-N ; k++) {
			double mix = 0.9;
			int a = RandomGenerator.uniform_int(N), b = RandomGenerator.uniform_int(N);
			Vector a_v = train_data.get_column(a);
			Vector b_v = train_data.get_column(b);
			Vector a_r = train_refs.get_column(a);
			Vector b_r = train_refs.get_column(b);
			
			train_data.set_column(k+N, a_v.scale(mix).add(b_v.scale(1-mix)));
			train_refs.set_column(k+N, a_r.scale(mix).add(b_r.scale(1-mix)));
		}

		System.out.println("# Train/Validation set built with " + train_data.width + "/" + validation_data.width + " images");
		
		// On charge les donn�es de test
		System.out.println("# Build test");
		test_data = new Matrix(T, 784);
		test_refs = new Matrix(T, out_size);
		i = 0;
		for (double[] v : Cal101Reader.get_test_data().transpose().v) {
			test_data.set_column(i, new Vector(v));
			test_refs.set_column(i, Vector.one_hot(out_size, Cal101Reader.get_test_refs()[i]));
			i++;
			if (i >= T)
				break;
		}
		System.out.println("# Test set built with " + T + " images");
	}

	public static void main(String[] args) {
		// seed = 1510445586196L;
		// On initialise le g�n�rateur al�atoire
		long time = System.currentTimeMillis();
		RandomGenerator.init(seed);
		System.out.println("# Seed : " + seed);

		Parameters p = new Parameters("reg=0.000001", "lr=0.001", "lrdecay=0.99");
		
		// On cr�e notre mod�le vide avec un mini_batch de 100
		model = new FlatSequential(100, new RMSOptimizer(p));
		load_data();
		
		
		// Mod�le classique � 4 couches (entr�e + cach�e + cach�e + sortie) avec 1000/300 neurones interm�diaires et des activations en sigmoide
		p.set("dout", "false");
		model.add(new DenseLayer(784, 1000, 0.3, "tanh", true, p));
		p.set("dout", "true");
		model.add(new DenseLayer(1000, 1000, 0.3, "tanh", true, p));
		model.add(new DenseLayer(1000, 102, 0, "none", false, p));
		// Fonction de co�t entropie crois�e avec softmax
		model.add(new SoftmaxCrossEntropy());
		
		System.out.println("# Model created with following architecture : ");
		model.print_architecture();

		System.out.println("# Processors : " + Runtime.getRuntime().availableProcessors());

		// Permet d'afficher simplement les nombres avec une pr�cision d�finie � l'avance
		DecimalFormatSymbols otherSymbols = new DecimalFormatSymbols(Locale.getDefault());
		otherSymbols.setDecimalSeparator('.');
		otherSymbols.setGroupingSeparator(',');
		DecimalFormat df2 = new DecimalFormat("#0.00", otherSymbols);
		DecimalFormat df5 = new DecimalFormat("#0.00000", otherSymbols);
		
		// Permet d'enregistrer toutes les donn�es int�r�ssantes � �crire � la fin
		/*MLPMetrics metrics = new MLPMetrics();
		metrics.add_time_series(model.correct_count(train_data, train_refs)/(double)N, 
								model.correct_count(validation_data, validation_refs)/(double)V, 
								model.get_loss(train_data, train_refs));
*/
		System.out.println("# Initialization took " + (System.currentTimeMillis() - time) + " ms");

		for (int i = 1; i <= EPOCHMAX; i++) {
			long t = System.currentTimeMillis();
			// On lance l'�poque
			model.train_on_batch(train_data, train_refs);

			// Temps que cela a pris pour effectuer l'�poque
			double epoch_time = (System.currentTimeMillis() - t) / 1000.;
			
			t = System.currentTimeMillis();
			double validation_accuracy = (100. * model.correct_count(test_data, test_refs)) / T;
			
			// Temps que cela a pris de regarder le nombre de donn�es de test correct
			double validation_forward_t = (System.currentTimeMillis() - t) / 1000.;
			
			double train_accuracy = (100. * model.last_correct_count) / train_data.width;

			//metrics.add_time_series(train_accuracy, validation_accuracy, model.last_average_loss);
			
			// Exemple d'affichage : 
			// [==========] - 3  Top 1 accuracy (train, test) : 53.92% 43.23% loss 1.23 epoch time 8.21s test time 1.01s ETA 32.12s
			
			System.out.print(i + ((i >= 10) ? " " : "  "));
			System.out.print("Top 1 accuracy (train, test) : " + df2.format(train_accuracy) + "% "
					+ df2.format(validation_accuracy) + "% ");
			System.out.print("loss " + df5.format(model.last_average_loss) + " ");
			System.out.print("epoch time " + df2.format(epoch_time) + "s ");
			System.out.print("validation time " + df2.format(validation_forward_t) + "s");
			System.out.println(" ETA " + df2.format((EPOCHMAX - i) * (epoch_time)) + "s");
		}
/*
		metrics.measure_and_write("./out_cal101/train", model, train_data, train_refs, false);
		metrics.measure_and_write("./out_cal101/test", model, test_data, test_refs, false);
		metrics.write_time_series_csv("./out_cal101/accuracy.csv");

*/
		System.out.println("Value at final test  : "+df2.format((100. * model.correct_count(test_data, test_refs)) / T)+"%");
	}
}