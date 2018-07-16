package perceptron;

import java.util.ArrayList;
import java.util.Collections;

import layers.FlatLayer;
import layers.Layer;
import layers.TrainableMatrices;
import layers.TrainableVectors;
import layers.flat.DenseLayer;
import layers.losses.Loss;
import math.Matrix;
import math.RandomGenerator;
import math.Vector;
import optimizers.Optimizer;

/**
 * Classe principale de perceptron multi couches
 */
public class FlatSequential extends FeedForwardNetwork {

	// Couches
	private ArrayList<FlatLayer> layers;
	public int last_correct_count = 0;
	
	public Optimizer opt;

	public void add(FlatLayer l) {
		layers.add(l);
		if(l instanceof TrainableMatrices) {
			opt.init_mat((TrainableMatrices)l);
		}
		if(l instanceof TrainableVectors) {
			opt.init_vec((TrainableVectors)l);
		}
		if(l instanceof DenseLayer) {
			opt.init_mat(((DenseLayer) l).al);
			opt.init_vec(((DenseLayer) l).al);
		}
	}
	
	public ArrayList<FlatLayer> get_layers() {
		return layers;
	}
	
	/**
	 * @param batch_size Taille du batch � utiliser
	 */
	public FlatSequential(int batch_size, Optimizer opt) {
		layers = new ArrayList<FlatLayer>();
		this.mini_batch = batch_size;
		this.opt = opt;
	}
	
	public FlatSequential(Optimizer opt) {
		layers = new ArrayList<FlatLayer>();
		this.opt = opt;
	}

	/**
	 * Passe un ensemble de donn�e � travers le r�seau, sans modifier l'ensemble de d�part
	 */
	@Override
	public Matrix forward(Matrix data, boolean train) {
		Matrix next = null;
		int i = 0;

		for (FlatLayer l : layers) {
//			long t = System.currentTimeMillis();
			if(i == 0) {
				next = l.forward(data, train);
			} else {
				next = l.forward(next, train);
			}
			i++;
		}
		return next;
	}

	/**
	 * Calcule les d�riv�es partielles couches � couches, en appliquant le gradient � chaque fois.
	 * N�cessite d'avoir fait une passe de forward_train avant.
	 * @return 
	 */
	public Matrix backward(Matrix dout, boolean train) {
		for (int j = layers.size() - 1; j >= 0; j--) {
			dout = layers.get(j).backward(dout, train);
		}
		if(train) {
			opt.optimize();
		}
		return dout;
	}

	public void train_on_batch(Matrix data, Matrix refs) {
		train_on_batch_extended(data, refs, 0, data.width/mini_batch, true);
	}

	public void train_on_batch_extended(Matrix data, Matrix refs, int start, int end, boolean verbose) {
		
		if(data.width % mini_batch != 0) {
			System.err.println("Le nombre de donnees ne sont pas divisble par mini_batch, "+(data.width%mini_batch)+" donn�es seront ignor�es.");
		}
		
		last_average_loss = 0;
		last_correct_count = 0;
		
		// On m�lange les donn�es, en m�langeant une liste d'indice
		ArrayList<Integer> ints = new ArrayList<Integer>();

		for (int i = 0; i < data.width; i++) {
			ints.add(i);
		}
		Collections.shuffle(ints, RandomGenerator.r);

		if(verbose) {
			System.out.print("[");
		}
		int tenth = data.width / (mini_batch * 10);

		
		for (int i = start; i < end; i++) {
			// Mini batch � utiliser
			Matrix batch = new Matrix(mini_batch, data.height);
			// Labels du mini batch
			Matrix refs_v = new Matrix(mini_batch, refs.height);
			// Barre de progression
			if(verbose) {
				if (tenth != 0) {
					if (i % tenth == 0)
						System.out.print("=");
				}
			}
			
			// On g�n�re le mini batch
			for (int j = 0; j < mini_batch; j++) {
				int indice = ints.get(i * mini_batch + j);
				batch.set_column(j, data.get_column(indice));
				refs_v.set_column(j, refs.get_column(indice));
			}

			// Propagation avant
			batch = forward(batch, true);
			Vector predicted = batch.argmax(Matrix.AXIS_HEIGHT);
			Vector right = refs_v.argmax(Matrix.AXIS_HEIGHT);
			last_correct_count += predicted.add(right.scale(-1)).count_zeros();
			
			Matrix dout = batch;
			Loss l = get_loss_layer();
			// On donne les r�f�rences � la fonction de co�t (en transformant les labels en matrice de v�rit� sous la forme d'une liste de one-hot vector)
			l.feed_ref(refs_v);

			// Propagation arri�re
			backward(dout, true);

			last_average_loss += l.loss;
		}
		last_average_loss /= data.width / mini_batch;

		opt.end_of_epoch();
		if(verbose) {
			System.out.print("] ");
		}
	}
	
	public double get_loss(Matrix data, int[] refs, int out_size) {
		return get_loss(data, Loss.from_int_refs(refs, out_size));
	}
	
	public double get_loss(Matrix data, Matrix refs) {
		Matrix end = forward(data, false);
		get_loss_layer().feed_ref(refs);
		get_loss_layer().backward(end, false);
		return get_loss_layer().loss;
	}
	
	/**
	 * Calcule la matrice de confusion
	 * @param data donn�es
	 * @param refs r�f�rences
	 * @return en hauteur les pr�dictions, en largeur la v�rit�.
	 * Ainsi confusion.v[0][2] contient le nombre de fois que le r�seau a reconnu le 0� label alors que la v�rit� �tait le 2� label
	 */
	public Matrix confusion_matrix(Matrix data, int[] refs) {
		Matrix end = forward(data, false);
		Matrix confusion_matrix = new Matrix(end.height, end.height);

		for (int i = 0; i < refs.length; i++) {
			int correct = refs[i];
			int predicted = end.get_column(i).argmax();
			confusion_matrix.v[predicted][correct] += 1;
		}

		return confusion_matrix;
	}

	/**
	 * Renvoie les donn�es o� le r�seau �tait le moins s�r de sa r�ponse
	 * @param data donn�es
	 * @param refs r�f�rences
	 * @param k nombre de pire donn�es � renvoyer
	 * @return k_max (1 - forward(data)[correct]) 
	 */
	public Matrix k_wrongest_data(Matrix data, int[] refs, int k) {
		Matrix end = forward(data, false);

		Matrix wrongest = new Matrix(k, data.height);

		Vector wrongV = new Vector(data.width);
		for (int i = 0; i < data.width; i++) {
			wrongV.v[i] = 1 - end.get_column(i).v[refs[i]];
		}

		for (int i = 0; i < k; i++) {
			int wrongest_i = wrongV.argmax();
			wrongest.set_column(i, data.get_column(wrongest_i));
			wrongV.v[wrongest_i] = 0;
		}

		return wrongest;
	}
	/**
	 * Renvoie la moyenne de chaque donn�e en fonction de chaque classe pr�dite
	 * @param data donn�es
	 * 
	 */
	public Matrix average_data_by_class(Matrix data) {
		Matrix end = forward(data, false);
		Matrix average = new Matrix(end.height, data.height);
		Vector nb_k = new Vector(end.height);

		for (int i = 0; i < end.width; i++) {
			Vector v = end.get_column(i);
			int argmax = v.argmax();
			Vector to_add = data.get_column(i);
			nb_k.v[argmax] += 1;
			average.set_column(argmax, average.get_column(argmax).add(to_add));
		}
		average.scale(nb_k.inverse(), Matrix.AXIS_HEIGHT);
		return average;
	}

	/**
	 * Compte le nombre de donn�es classifi�es correctement
	 */
	public int correct_count(Matrix data, Matrix refs) {
		Matrix end = forward(data, false);
		Vector predicted = end.argmax(Matrix.AXIS_HEIGHT);
		return predicted.add(refs.argmax(Matrix.AXIS_HEIGHT).scale(-1)).count_zeros();
	}

	@Override
	public Loss get_loss_layer() {
		return (Loss) layers.get(layers.size() - 1);
	}

	@Override
	public void print_architecture() {
		for (Layer l : layers) {
			System.out.println("# - " + l);
		}
	}
}
