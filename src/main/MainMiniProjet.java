package main;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.Locale;
import java.util.Scanner;

import layers.Parameters;
import layers.flat.DenseLayer;
import layers.losses.SoftmaxCrossEntropy;
import math.Matrix;
import math.RandomGenerator;
import math.Vector;
import optimizers.RMSOptimizer;
import perceptron.FlatSequential;

public class MainMiniProjet {
	// Modèle à utiliser
		public static FlatSequential model;

		// Nombre d'epoque max
		public final static int EPOCHMAX = 10;

		// Nombre de données d'entrainements
		public static final int N = 35000;

		// Nombre de données de validation
		public static final int V = 5000;
		

		// Nombre de données de validation
		public static final int real_V = 10000;

		// Nombre de données de test
		public static int T = 10000;

		public static int out_size = 10;
		public static int feat_size = 256;

		// Matrices de données
		public static Matrix train_data, test_data, validation_data, real_validation_data;
		
		// Tableaus de labels
		public static Matrix train_refs, validation_refs;

		// Seed à utiliser pour la reproducibilité
		public static long seed = System.currentTimeMillis();
		static class VLabel {
			Vector data;
			Vector label_one_hot;
		}

		static Scanner trainF;
		static Scanner trainFlabel;
		
		public static VLabel nextTrain() {
			if(trainF == null) {
				try {
					trainF = new Scanner(new File("cifar10_mp/cifar10_train.data"));
					trainFlabel = new Scanner(new File("cifar10_mp/cifar10_train.solution"));
				} catch (FileNotFoundException e) {
					e.printStackTrace();
				}
			}
			
			VLabel vl = new VLabel();
			String[] line = trainF.nextLine().split(" ");
			String[] line2 = trainFlabel.nextLine().split(" ");
			vl.data = new Vector(feat_size);
			vl.label_one_hot = new Vector(out_size);
			int i = 0;
			for(String s : line) {
				vl.data.v[i++] = Double.parseDouble(s);
			}
			i = 0;
			for(String s : line2) {
				vl.label_one_hot.v[i++] = Double.parseDouble(s);
			}
			return vl;
		}
		
		
		public static void write_prediction() {
			Matrix real_validation_refs = model.forward(real_validation_data, false);
			Matrix test_refs = model.forward(test_data, false);
			Matrix train_pre_refs = model.forward(train_data, false);
			Vector predicted_real_validation = real_validation_refs.argmax(Matrix.AXIS_HEIGHT);
			Vector predicted_test = test_refs.argmax(Matrix.AXIS_HEIGHT);
			Vector predicted_train = train_pre_refs.argmax(Matrix.AXIS_HEIGHT);
			

			
			PrintWriter out_train = null;
			try {
				out_train = new PrintWriter(new File("cifar10_mp/cifar10_train.predict"));
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			}
			//System.out.println(predicted_train);
			for(int i = 0 ; i < N ; i++) {
				StringBuilder sb = new StringBuilder();
				for(int k = 0 ; k < out_size ; k++) {
					if(k == predicted_train.v[i]) {
						sb.append("1 ");
					} else {
						sb.append("0 ");
					}
				}

				out_train.println(sb.toString());
			}
			out_train.close();
			
			PrintWriter out_validation = null;
			try {
				out_validation = new PrintWriter(new File("cifar10_mp/cifar10_valid.predict"));
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			}
			for(int i = 0 ; i < real_V ; i++) {
				
				StringBuilder sb = new StringBuilder();
				for(int k = 0 ; k < out_size ; k++) {
					if(k == predicted_real_validation.v[i]) {
						sb.append("1 ");
					} else {
						sb.append("0 ");
					}
				}
				out_validation.println(sb.toString());
			}
			out_validation.close();
			
			PrintWriter out_test = null;
			try {
				out_test = new PrintWriter(new File("cifar10_mp/cifar10_test.predict"));
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			}
			for(int i = 0 ; i < T ; i++) {
				StringBuilder sb = new StringBuilder();
				for(int k = 0 ; k < out_size ; k++) {
					if(k == predicted_test.v[i]) {
						sb.append("1 ");
					} else {
						sb.append("0 ");
					}
				}
				out_test.println(sb.toString());
			}
			out_test.close();
			
		}
		
		public static void load_data() {
			// On charge les données d'entrainements et de validation
			System.out.println("# Build train & validation");
			train_data = new Matrix(N, feat_size);
			train_refs = new Matrix(N, out_size);
			validation_data = new Matrix(V, feat_size);
			validation_refs = new Matrix(V, out_size);
			Vector means = new Vector(feat_size), sigmas = new Vector(feat_size);
			for (int i = 0 ; i < N+V ; i++) {
				VLabel vl = nextTrain();
				means.add(vl.data);
				if(i < N) {
					train_data.set_column(i, vl.data);
					train_refs.set_column(i, vl.label_one_hot);
				} else if(i < N+V) {
					validation_data.set_column(i-N, vl.data);
					validation_refs.set_column(i-N, vl.label_one_hot);
				} else {
					break;
				}
			}
			means.scale(1. / (N+V));
			
			for(int i = 0 ; i < N+V ; i++) {
				Vector diff_sq = null;
				if(i < N) {
					diff_sq = train_data.get_column(i).scale(-1).add(means).power(2);
				} else if(i < N+V) {
					diff_sq = validation_data.get_column(i-N).scale(-1).add(means).power(2);
				} else {
					break;
				}
				sigmas.add(diff_sq);
			}
			sigmas.scale(1. / (N+V)).power(0.5);
			
			System.out.println("means "+means);
			System.out.println("sigmas"+sigmas);

			System.out.println("# Train/Validation set built with " + train_data.width + "/" + validation_data.width + " images");
			Scanner testF=null, valF=null;
			try {
				testF = new Scanner(new File("cifar10_mp/cifar10_test.data"));
				valF  = new Scanner(new File("cifar10_mp/cifar10_valid.data"));
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			}
			test_data = new Matrix(T, feat_size);
			real_validation_data = new Matrix(real_V, feat_size);
			
			for(int d_c = 0 ; d_c < T ; d_c++) {
				String[] line = testF.nextLine().split(" ");	
				double[] d = new double[feat_size];
				int i = 0;
				for(String s : line) {
					d[i++] = Double.parseDouble(s);
				}
				test_data.set_column(d_c, new Vector(d));
			}
			for(int d_c = 0 ; d_c < real_V ; d_c++) {
				String[] line = valF.nextLine().split(" ");	
				double[] d = new double[feat_size];
				int i = 0;
				for(String s : line) {
					d[i++] = Double.parseDouble(s);
				}
				real_validation_data.set_column(d_c, new Vector(d));
			}
			means.scale(-1);
			sigmas.inverse();
			train_data.add(means, Matrix.AXIS_WIDTH).scale(sigmas, Matrix.AXIS_WIDTH);
			validation_data.add(means, Matrix.AXIS_WIDTH).scale(sigmas, Matrix.AXIS_WIDTH);
			real_validation_data.add(means, Matrix.AXIS_WIDTH).scale(sigmas, Matrix.AXIS_WIDTH);
			test_data.add(means, Matrix.AXIS_WIDTH).scale(sigmas, Matrix.AXIS_WIDTH);
		}

		public static void main(String[] args) {
			// seed = 1510445586196L;
			// On initialise le générateur aléatoire
			long time = System.currentTimeMillis();
			RandomGenerator.init(seed);
			System.out.println("# Seed : " + seed);

			Parameters p = new Parameters("reg=0.00000", "lr=0.001", "lrdecay=0.9");
			
			// On crée notre modèle vide avec un mini_batch de 100
			model = new FlatSequential(64, new RMSOptimizer(p));
			load_data();
			
			
			// Modèle classique à 4 couches (entrée + cachée + cachée + sortie) avec 1000/300 neurones intermédiaires et des activations en sigmoide
			p.set("dout", "false");
			model.add(new DenseLayer(feat_size, 256, 0.5, "swish", true, p));
			p.set("dout", "true");
			model.add(new DenseLayer(256, 256, 0.5, "swish", true, p));
			model.add(new DenseLayer(256, out_size, 0, "none", false, p));
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
			/*MLPMetrics metrics = new MLPMetrics();
			metrics.add_time_series(model.correct_count(train_data, train_refs)/(double)N, 
									model.correct_count(validation_data, validation_refs)/(double)V, 
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
				
				// Temps que cela a pris de regarder le nombre de données de test correct
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
			write_prediction();
	/*
			metrics.measure_and_write("./out_cal101/train", model, train_data, train_refs, false);
			metrics.measure_and_write("./out_cal101/test", model, test_data, test_refs, false);
			metrics.write_time_series_csv("./out_cal101/accuracy.csv");

	*/
			//System.out.println("Value at final test  : "+df2.format((100. * model.correct_count(test_data, test_refs)) / T)+"%");
		}
}
