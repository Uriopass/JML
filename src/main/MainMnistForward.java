package main;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.Locale;

import datareaders.SimpleMnistReader;
import layers.Parameters;
import layers.activations.ReLUActivation;
import layers.flat.AffineLayer;
import layers.flat.DenseLayer;
import layers.losses.SoftmaxCrossEntropy;
import math.Matrix;
import math.RandomGenerator;
import math.Vector;
import optimizers.Optimizer;
import optimizers.RMSOptimizer;
import perceptron.FlatSequential;
import perceptron.MLPMetrics;

public class MainMnistForward {
	public static FlatSequential model;

	// Nombre d'epoque max
	public final static int EPOCHMAX = 6;
	
	// Matrices de données
	public static Matrix train_data, validation_data, test_data;
	// Tableaux de références
	public static Matrix train_refs, validation_refs, test_refs;

	// Seed utilisé pour la reproducibilité
	public static long seed = System.currentTimeMillis();

	public static void load_data() {
		train_data = SimpleMnistReader.getTrainData();
		train_refs = SimpleMnistReader.getTrainRefs();
		
		test_data = SimpleMnistReader.getTestData();
		test_refs = SimpleMnistReader.getTestRefs();
		
		validation_data = SimpleMnistReader.getValidationData();
		validation_refs = SimpleMnistReader.getValidationRefs();
	}


	public static void draw_loss(Optimizer opt) {
		int nb_params = 0;
		
		for(Matrix m : opt.get_mats()) {
			nb_params += m.width*m.height;
		}
		for(Vector v : opt.get_vecs()) {
			nb_params += v.length;
		}
		System.out.println("# "+nb_params +" parameters");

		Vector params = new Vector(nb_params);
		
		
		Vector axe1 = new Vector(nb_params);
		Vector axe2 = new Vector(nb_params);//Vector.random_gaussian_vector(nb_params).set_len(Math.sqrt(norm));
		int counter = 0;
		for(Matrix m : opt.get_mats()) {
			double m_norm = m.norm();
			Vector m_gaus = Vector.random_gaussian_vector(m.height*m.width).set_len(m_norm);
			Vector m_gaus2 = Vector.random_gaussian_vector(m.height*m.width).set_len(m_norm);
			for(int i = 0 ; i < m.height ; i++) {
				for(int j = 0 ; j < m.width ; j++) {
					params.v[counter] = m.v[i][j];
					axe1.v[counter] = m_gaus.v[i*m.width+j];
					axe2.v[counter] = m_gaus2.v[i*m.width+j];
					counter++;
				}
			}
		}
		for(Vector v : opt.get_vecs()) {
			for(int i = 0 ; i < v.length; i++) {
				params.v[counter++] = v.v[i];
			}
		}
		
		int size = 12;
		double[][] losses = new double[size+1][size+1];
		for(int u = 0 ; u <= size ; u++) {
			for(int v = 0 ; v <= size ; v++) {
				double inter1 = ((2.0*u)/size)-1.0;
				double inter2 = ((2.0*v)/size)-1.0;
				Vector newP = new Vector(params);
				for(int k = 0 ; k < nb_params ; k++) {
					newP.v[k] += axe1.v[k]*inter1 + axe2.v[k]*inter2; 
				}
				counter = 0;
				for(Matrix m : opt.get_mats()) {
					for(int i = 0 ; i < m.height ; i++) {
						for(int j = 0 ; j < m.width ; j++) {
							m.v[i][j] = newP.v[counter++];
						}
					}
				}
				for(Vector v2 : opt.get_vecs()) {
					for(int i = 0 ; i < v2.length; i++) {
						v2.v[i] = newP.v[counter++];
					}
				}
				
				double loss = model.get_loss(test_data, test_refs);
				losses[u][v] = loss;
				System.out.println("For "+inter1 + " "+inter2+" "+loss);
			}
		}
		for(int i = 0 ; i <= size ; i++) {
			for(int j = 0 ; j <= size ; j++) {
				System.out.print(losses[i][j]+";");
			}
			System.out.println();
		}
	}
	
	public static void draw_loss_1D(Optimizer m_1, Optimizer m_2) {
		int nb_params = 0;
		
		for(Matrix m : m_1.get_mats()) {
			nb_params += m.width*m.height;
		}
		for(Vector v : m_1.get_vecs()) {
			nb_params += v.length;
		}
		System.out.println("# "+nb_params +" parameters");

		Vector params = new Vector(nb_params);
		Vector params2 = new Vector(nb_params);
		
		int counter = 0;
		for(Matrix m : m_1.get_mats()) {
			for(int i = 0 ; i < m.height ; i++) {
				for(int j = 0 ; j < m.width ; j++) {
					params.v[counter++] = m.v[i][j];
				}
			}
		}
		for(Vector v : m_1.get_vecs()) {
			for(int i = 0 ; i < v.length; i++) {
				params.v[counter++] = v.v[i];
			}
		}
		counter = 0;
		
		for(Matrix m : m_2.get_mats()) {
			for(int i = 0 ; i < m.height ; i++) {
				for(int j = 0 ; j < m.width ; j++) {
					params2.v[counter++] = m.v[i][j];
				}
			}
		}
		for(Vector v : m_2.get_vecs()) {
			for(int i = 0 ; i < v.length; i++) {
				params2.v[counter++] = v.v[i];
			}
		}
		
		
		Vector axe = Vector.sub(params2, params);
		counter=0;
		for(Matrix m : m_1.get_mats()) {
			for(int i = 0 ; i < m.height ; i++) {
				for(int j = 0 ; j < m.width ; j++) {
					m.v[i][j] = params2.v[counter++];
				}
			}
		}
		for(Vector v2 : m_1.get_vecs()) {
			for(int i = 0 ; i < v2.length; i++) {
				v2.v[i] = params2.v[counter++];
			}
		}
		System.out.println("Loss at model2 ? "+model.get_loss(train_data, train_refs));
		int size = 100;
		double[] losses = new double[size+1];
		for(int u = 0 ; u <= size ; u++) {
			double inter1 = ((2.0*u)/size)-0.5;
			Vector newP = new Vector(params);
			
			for(int k = 0 ; k < nb_params ; k++) {
				newP.v[k] += axe.v[k]*inter1; 
			}
			
			counter = 0;
			for(Matrix m : m_1.get_mats()) {
				for(int i = 0 ; i < m.height ; i++) {
					for(int j = 0 ; j < m.width ; j++) {
						m.v[i][j] = newP.v[counter++];
					}
				}
			}
			for(Vector v2 : m_1.get_vecs()) {
				for(int i = 0 ; i < v2.length; i++) {
					v2.v[i] = newP.v[counter++];
				}
			}
			
			double loss = model.get_loss(train_data, train_refs);
			losses[u] = loss;
			m_1.get_mats().iterator().next().visualize("losses/loss"+u, 28, 16, 8, true, false, false);
			System.out.println("For "+inter1 + " "+loss);
		}
		for(int i = 0 ; i <= size ; i++) {
			System.out.println(losses[i]+";");
		}
	}
	
	public static void main(String[] args) {
		// On initialise le générateur aléatoire
		long time = System.currentTimeMillis();
		seed = 1510437982659L;
		RandomGenerator.init(seed);
		System.out.println("# Seed : " + seed);

		// Paramètres du modele
		Parameters p = new Parameters("reg=0.00005", "lr=0.001");
		// On crée notre modèle vide avec un mini_batch de 40
		Optimizer opt = new RMSOptimizer(p);
		model = new FlatSequential(400, opt);
		load_data();

		// Modèle classique à 4 couches (entrée + cachée + cachée + sortie) avec 1000 et 100 neurones intermédiaires et des activations en sigmoide

		// Dout est inutile pour la première couche
		p.set("dout", "false");
		model.add(new AffineLayer(784, 128, true, p));
		p.set("dout", "true");
		model.add(new ReLUActivation());
		model.add(new AffineLayer(128, 128, true, p));
		//model.add(new BatchnormLayer(128, p));
		model.add(new ReLUActivation());
		model.add(new AffineLayer(128, 128, true, p));
		//model.add(new BatchnormLayer(128, p));
		model.add(new ReLUActivation());
		model.add(new AffineLayer(128, 128, true, p));
		//model.add(new BatchnormLayer(128, p));
		model.add(new ReLUActivation());
		model.add(new DenseLayer(128, 10, 0, "none", false, p));
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

		System.out.println("# Initialization took " + (System.currentTimeMillis() - time) + " ms");

		for (int i = 1; i <= EPOCHMAX; i++) {
			long t = System.currentTimeMillis();
			// On lance l'époque
			model.train_on_batch(train_data, train_refs);

			// Temps que cela a pris pour effectuer l'époque
			double epoch_time = (System.currentTimeMillis() - t) / 1000.;

			t = System.currentTimeMillis();
			double validation_accuracy = (100. * model.correct_count(validation_data, validation_refs)) / SimpleMnistReader.V;

			// Temps que cela a pris de regarder le nombre de données de validation correct
			double validation_forward_t = (System.currentTimeMillis() - t) / 1000.;

			double train_accuracy = (100. * model.last_correct_count) / SimpleMnistReader.N;

			metrics.add_time_series(train_accuracy, validation_accuracy, model.last_average_loss);

			// Exemple d'affichage : 
			// [==========] - 3  Top 1 accuracy (train, test) : 93.92% 89.23% loss 1.23 epoch time 3.21s test time 1.01s ETA 32.12s

			System.out.print(i + ((i >= 10) ? " " : "  "));
			System.out.print("Top 1 accuracy (train, val) : " + df2.format(train_accuracy) + "% "
					+ df2.format(validation_accuracy) + "% ");
			System.out.print("loss " + df5.format(model.last_average_loss) + " ");
			System.out.print("Acc at test "+100.*model.correct_count(test_data, test_refs)/SimpleMnistReader.T+" ");
			System.out.print("epoch time " + df2.format(epoch_time) + "s ");
			System.out.print("forward time " + df2.format(validation_forward_t) + "s");

			// Temps avant la fin de l'entraînement
			System.out.println(" ETA " + df2.format((EPOCHMAX - i) * (epoch_time)) + "s");
		}
		draw_loss(opt);
		
		System.out.println("Value at final test  : " + df2.format((100. * model.correct_count(test_data, test_refs)) / SimpleMnistReader.T) + "%");
	}
}