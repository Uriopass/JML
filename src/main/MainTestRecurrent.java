package main;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Scanner;

import layers.Parameters;
import layers.losses.SoftmaxCrossEntropy;
import layers.reccurent.SimpleRecManyToOne;
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
	
	public static ArrayList<Matrix> data, refs;
	
	public static void load_data() {
		Scanner sc = null;
		try {
			sc = new Scanner(new File("parsed_nouns.txt"));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		String line;
		
		data = new ArrayList<>();
		refs = new ArrayList<>();
		
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
			data.add(alright);
			refs.add(new Matrix(1, 2).set_column(0, Vector.one_hot(2, x)));
		}
		
		sc.close();
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
	
	public static void main(String[] args) {
		load_data();
		// On initialise le générateur aléatoire
		long time = System.currentTimeMillis();
		seed = System.currentTimeMillis();
		RandomGenerator.init(seed);
		System.out.println("# Seed : " + seed);

		// Paramètres du modele
		Parameters p = new Parameters("reg=0", "lr=0.001");

		SimpleRecManyToOne model = new SimpleRecManyToOne(dico.size(), 2, true, 1, "tanh", p);
		SoftmaxCrossEntropy loss = new SoftmaxCrossEntropy();
		RMSOptimizer rmsopt = new RMSOptimizer(p);
		rmsopt.init_mat(model);
		rmsopt.init_vec(model);
		
		System.out.println("# Processors : " + Runtime.getRuntime().availableProcessors());

		System.out.println("# Initialization took " + (System.currentTimeMillis() - time) + " ms");

		for (int epoch = 1; epoch <= 1; epoch++) {
			//long t = System.currentTimeMillis();
			model.initRec();
			double lossAverage = 0;
			int correct = 0;
			long haha = System.currentTimeMillis();
			for(int k = 0 ; k < data.size() ; k++) {
				Matrix d = data.get(k);
				model.initRec();
				for(int l = 0 ; l < d.width ; l++) {
					model.tick(d.get_column(l).to_column_matrix(), true);
				}
				Matrix out = model.get_out(true);
				loss.feed_ref(refs.get(k));
				Matrix real_out = loss.forward(out, true);
				Matrix dout = loss.backward(new Matrix(real_out), true);
				model.backwardAll(dout, true);
				rmsopt.optimize();
				lossAverage += loss.loss;
				if(real_out.get_column(0).argmax() == refs.get(k).get_column(0).argmax()) {
					correct++;
				}
				
				if((k+1)%1000 == 0) {
					System.out.println(correct + " " + lossAverage/1000 + " " + (k+1) + " " + data.size() +" "+
							model.state_aff.get_weight().sum());
					correct = 0;
					lossAverage = 0;
				}
			}
			System.out.println(System.currentTimeMillis()-haha);
			System.out.println("EPOCH END");
			model.state_aff.get_weight().print_values();
			System.out.println( "--");
			model.out_aff.get_weight().print_values();
			System.out.println(model.out_aff.get_bias());
			for(int i = 0 ; i < 5 ; i++) {
				int indice = RandomGenerator.uniform_int(data.size());
				System.out.println("example of state for "+matrix_to_name(data.get(indice)));
				Matrix d = data.get(indice);
				
				model.initRec();
				model.state.T().print_values();
				for(int l = 0 ; l < d.width ; l++) {
					model.tick(d.get_column(l).to_column_matrix(), false);
					System.out.println(matrix_to_name(d.get_column(l).to_column_matrix())+" " + model.state.get_column(0));
				}
				Matrix out = model.get_out(false);
				out.T().print_values();
				refs.get(indice).T().print_values();
			
				System.out.println(out_to_genre(out.get_column(0)));
			}
		}
		
	}
}