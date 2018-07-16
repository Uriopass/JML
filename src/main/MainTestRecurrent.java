package main;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Scanner;

import datareaders.MnistReader;
import image.ImageConverter;
import layers.Parameters;
import layers.flat.DenseLayer;
import layers.losses.SoftmaxCrossEntropy;
import layers.reccurent.GRUCell;
import layers.reccurent.RNNCellCache;
import math.Matrix;
import math.RandomGenerator;
import math.Vector;
import optimizers.RMSOptimizer;

public class MainTestRecurrent {
	static long seed;
	static int EPOCHMAX = 20;
	
	static HashMap<Character, Integer> dico = new HashMap<>();
	static {
		for(char i = 0 ; i < 26 ; i++) {
			dico.put((char) ((char)('a')+i), (int)i);
		}
		dico.put('à', 26);
		dico.put('é', 27);
		dico.put('è', 28);
		dico.put('ê', 29);
		dico.put('ç', 30);
		dico.put(' ', 31);
		dico.put('-', 32);
		dico.put('>', 33);
	}
	
	public static ArrayList<Matrix> train_data, train_refs;
	public static ArrayList<Matrix> validation_data, validation_refs;
	public static ArrayList<Matrix> test_data, test_refs;
	
	
	public static void load_data() {
		Scanner sc = null;
		try {
			sc = new Scanner(new File("parsed_nouns.txt"));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		String line;
		
		train_data = new ArrayList<>();
		train_refs = new ArrayList<>();
		
		while(sc.hasNextLine()) {
			line = sc.nextLine();
			String[] yes = line.split(";");
			String mot = yes[0];
			int x = Integer.parseInt(yes[1]);
			Matrix alright = new Matrix(mot.length()+1, dico.size());
			int counter = 0;
			for(char c : mot.toCharArray()) {
				alright.set_column(counter, Vector.one_hot(dico.size(), dico.get(c)));
				counter++;
			}
			alright.set_column(counter, Vector.one_hot(dico.size(), dico.get('>')));
			train_data.add(alright);
			train_refs.add(new Matrix(1, 2).set_column(0, Vector.one_hot(2, x)));
		}
		
		sc.close();
	}

	// Nombre de données d'entrainements
	public static final int N = 20000;
	// Nombre de données de validation
	public static final int V = 10000;
	// Nombre de données de test
	public static int T = 10000;
	
	
	public static void load_mnist_data() {
		
		 String path = "";
		 String train_labelDB = path + "train-labels.idx1-ubyte";
		 String train_imageDB = path + "train-images.idx3-ubyte";

		 String test_labelDB = path + "t10k-labels.idx1-ubyte";
		 String test_imageDB = path + "t10k-images.idx3-ubyte";
		
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

		/* Creation des donnees */
		train_data = new ArrayList<>();
		train_refs = new ArrayList<>();

		validation_data = new ArrayList<>();
		validation_refs = new ArrayList<>();
		final int TOTAL = train_images.size();
		if (N + V > TOTAL) {
			throw new RuntimeException("N+V (" + (N + V) + ") > Total (" + TOTAL + ")");
		}

		/* Donnees d'apprentissage */
		for (int l = 0; l < N + V; l++) {
			int label = all_train_refs[l];
			double[] image = ImageConverter.image2VecteurReel(train_images.get(l));
			Matrix m = new Vector(image).to_row_matrix();
			if (l < N) {
				train_data.add(m);
				train_refs.add(Vector.one_hot(10, label).to_column_matrix());
			} else {
				validation_data.add(m);
				validation_refs.add(Vector.one_hot(10, label).to_column_matrix());
			}
		}

		System.out.println("# Train/Validation set built with " + N + "/" + V + " images");

		/* Donnees de test */
		System.out.println("# Build test");

		test_data = new ArrayList<>();
		test_refs = new ArrayList<>();
		for (int i = 0; i < T; i++) {
			test_data.add(new Vector(ImageConverter.image2VecteurReel(test_images.get(i))).to_row_matrix());
			int label = all_test_refs[i];
			test_refs.add(Vector.one_hot(10, label).to_column_matrix());
		}
		System.out.println("# Test set built with " + T + " images");
	}
	
	public static String matrix_to_name(Matrix d) {
		String s = "";
		
		for(int i = 0 ; i < d.width ; i++) {
			Vector v = d.get_column(i);
			int x = v.argmax();
			for(Character c : dico.keySet()) {
				if(dico.get(c) == x) {
					s += c;
					break;
				}
			}
		}
		
		return s;
	}
	
	public static String out_to_genre(Vector v) {
		if(v.argmax() == 0)
			return "feminin";
		return "masculin";
	}
	
	public static void main(String[] args) throws IOException {
		System.in.read();
		load_mnist_data();
		// On initialise le générateur aléatoire
		long time = System.currentTimeMillis();
		seed = System.currentTimeMillis();
		RandomGenerator.init(seed);
		System.out.println("# Seed : " + seed);

		// Paramètres du modele
		Parameters p = new Parameters("reg=0", "lr=0.001");

		int units = 50;
		
		GRUCell model = new GRUCell(units, 1, p);
		SoftmaxCrossEntropy loss = new SoftmaxCrossEntropy();
		DenseLayer out_l = new DenseLayer(units, 10, 0, "none", false, p);
		RMSOptimizer rmsopt = new RMSOptimizer(p);
		
		rmsopt.init_mat(model);
		rmsopt.init_vec(model);
		
		rmsopt.init_mat(out_l.al);
		rmsopt.init_vec(out_l.al);
		
		System.out.println("# Processors : " + Runtime.getRuntime().availableProcessors());

		System.out.println("# Initialization took " + (System.currentTimeMillis() - time) + " ms");

		for (int epoch = 1; epoch <= 1; epoch++) {
			double lossAverage = 0;
			//int correct = 0;
			long haha = System.currentTimeMillis();
			for(int k = 0 ; k < train_data.size()/32-1 ; k++) {
				Matrix batch = new Matrix(32, 784);
				for(int j = 0 ; j < 32 ; j++)
					batch.set_column(j, train_data.get(k*32+j).get_row(0));
				ArrayList<RNNCellCache> caches = new ArrayList<>();
				Matrix state = new Matrix(32, units);
				
				for(int l = 0 ; l < batch.height ; l++) {
					RNNCellCache c = new RNNCellCache();
					state = model.step(batch.get_row(l).to_row_matrix(), state, true, c);
					caches.add(c);
				}
				Matrix out = out_l.forward(state, true);
				loss.feed_ref(train_refs.get(k));
				Matrix real_out = loss.forward(out, true);
				
				//System.out.println(real_out.get_column(0));
				
				Matrix dout = loss.backward(new Matrix(real_out), true);
				dout = out_l.backward(dout, true);
				
				while(caches.size() > 0) {
					dout = model.backward(dout, caches.get(caches.size()-1));
					caches.remove(caches.size()-1);
				}
				lossAverage += loss.loss;
				
				System.out.println(lossAverage + " " + (k+1)*32 + " " + train_data.size());
				lossAverage = 0;
				rmsopt.optimize();
			}
			System.out.println(System.currentTimeMillis()-haha);
			System.out.println("EPOCH END");
			System.out.println( "--");
		}
		
	}
}