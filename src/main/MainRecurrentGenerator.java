package main;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map.Entry;
import java.util.Scanner;

import layers.Parameters;
import layers.flat.AffineLayer;
import layers.flat.DenseLayer;
import layers.losses.SoftmaxCrossEntropy;
import layers.reccurent.GRUCell;
import layers.reccurent.RNNCellCache;
import math.Activations;
import math.Matrix;
import math.RandomGenerator;
import math.Vector;
import optimizers.Optimizer;
import optimizers.SGDOptimizer;

public class MainRecurrentGenerator {
	static long seed;
	static int EPOCHMAX = 3;

	static HashMap<Character, Integer> dico = new HashMap<>();
	static HashMap<Integer, Character> reverse_dico = new HashMap<>();
	static {
		for(char i = 0 ; i < 26 ; i++) {
			dico.put((char) ((char)('a')+i), (int)i);
		}
		for(char i = 0 ; i < 10 ; i++) {
			dico.put((char) ((char)('0')+i), 26+i);
		}
		dico.put('à', dico.size());
		dico.put('é', dico.size());
		dico.put('è', dico.size());
		dico.put('ê', dico.size());
		dico.put('ç', dico.size());
		dico.put(' ', dico.size());
		dico.put('-', dico.size());
		dico.put('>', dico.size());
		dico.put('<', dico.size());
		
		for(Entry<Character, Integer> a : dico.entrySet()) {
			reverse_dico.put(a.getValue(), a.getKey());
		}
	}
	
	public static String matrix_to_name(Matrix d) {
		String s = "";
		
		for(int i = 0 ; i < d.width ; i++) {
			Vector v = d.get_column(i);
			int x = v.argmax();
			char c = reverse_dico.get(x);
			if(c != '>' && c != '<') {
				s += reverse_dico.get(x);
			}
		}
		
		return s;
	}
	
	public static double get_loss(GRUCell c, AffineLayer l, SoftmaxCrossEntropy loss, Matrix input, Vector ref) {
		Matrix state = new Matrix(1, c.state_size);
		Matrix step = c.step(input, state, false, null);
		step = c.step(input, step, false, null);
		
		step = l.forward(step, false);
		step = loss.forward(step, false);
		
		loss.feed_ref(ref.to_column_matrix());
		loss.backward(step, true);
		return loss.loss;
	}
	
	public static void test_GRU() {
		RandomGenerator.init(10);
		Vector super_v = Vector.random_gaussian_vector(1);
		Vector super_v2 = Vector.random_gaussian_vector(2);
		GRUCell test = new GRUCell(1, 1, null);
		{
			RNNCellCache cc2 = new RNNCellCache();
			RNNCellCache cc3 = new RNNCellCache();
			
			AffineLayer al = new AffineLayer(1, 2, true, null);
			SoftmaxCrossEntropy loss = new SoftmaxCrossEntropy();
			
			Matrix state = new Matrix(1, 1);
			Matrix new_s = test.step(super_v.to_column_matrix(), state, true, cc3);
			new_s = test.step(super_v.to_column_matrix(), new_s, true, cc2);
			
			Matrix out = al.forward(new_s, true);
			Matrix real_out = loss.forward(out, true);
			
			loss.feed_ref(super_v2.to_column_matrix());
			Matrix dout = loss.backward(real_out, true);
			
			dout = al.backward(dout, true);
			dout = test.backward(dout, cc2);
			dout = test.backward(dout, cc3);
		}
		
		RandomGenerator.init(10);
		super_v = Vector.random_gaussian_vector(1);
		super_v2 = Vector.random_gaussian_vector(2);
		GRUCell test2 = new GRUCell(1, 1, null);
		AffineLayer al = new AffineLayer(1, 2, true, null);
		
		
		
		SoftmaxCrossEntropy loss = new SoftmaxCrossEntropy();
		double fx = get_loss(test2, al, loss, super_v.to_column_matrix(), super_v2);
		
		test2.ru_layer.al.get_weight().v[1][1] += 0.0000001;
		
		double fx2 = get_loss(test2, al, loss, super_v.to_column_matrix(), super_v2);
		
		double v = (fx2 - fx)/0.0000001;
		/*
		Matrix haha = new Matrix(test.ru_layer.al.get_weight());
		for(int i = 0 ; i < haha.height ; i++) {
			for(int j = 0 ; j < haha.width ; j++) {
				double fx = get_loss(test2, al, loss, super_v.to_column_matrix(), super_v2);
				
				test2.ru_layer.al.get_weight().v[i][j] += 0.0000001;
				
				double fx2 = get_loss(test2, al, loss, super_v.to_column_matrix(), super_v2);
				
				haha.v[i][j] = (fx2 - fx)/0.0000001;
			}
		}
		*/
		
		//new Matrix(test.ru_layer.al.get_weight().grad).scale(1000).print_values();
		System.out.println("-");
		System.out.println(test.ru_layer.al.get_weight().grad.v[1][1] / v); /*
		haha.scale(-1).add(test.ru_layer.al.get_weight().grad).scale(1000).print_values();*/
	}

	public static ArrayList<Matrix> train_data;
	public static ArrayList<Matrix> train_refs;

	public static void load_noun_data() {
		Scanner sc = null;
		try {
			sc = new Scanner(new File("parsed_nouns_2.txt"));
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
			Matrix alright = new Matrix(mot.length()+2, dico.size());
			alright.set_column(0, Vector.one_hot(dico.size(), dico.get('<')));
			int counter = 1;
			for(char c : mot.toCharArray()) {
				c = Character.toLowerCase(c);
				if(!dico.containsKey(c)) {
					System.out.println(c);
					System.out.println(mot);
					System.exit(0);
				}
				alright.set_column(counter, Vector.one_hot(dico.size(), dico.get(c)));
				counter++;
			}
			alright.set_column(counter, Vector.one_hot(dico.size(), dico.get('>')));
			train_data.add(alright);
			train_refs.add(new Matrix(1, 2).set_column(0, Vector.one_hot(2, x)));
		}
		
		sc.close();
	}
	
	public static int sample(Vector prob) {
		int i = 0;
		double p = 0;
		double r = RandomGenerator.uniform(0, 1);
		while(p < r) {
			p += prob.v[i];
			i++;
		}
		return i-1;
	}
	
	public static void sample(GRUCell model, DenseLayer out_l) {
		Matrix state = new Matrix(1, model.state_size);
		Matrix d = Vector.one_hot(dico.size(), dico.get('<')).to_column_matrix();
	
		int c = 0;
		int end = dico.get('>');
		int p = 0;
		
		while(p != end) {
			state = model.step(d, state, false, null);
			p = sample(Activations.softmax(out_l.forward(state, false).get_column(0)));
			
			
			
			d = Vector.one_hot(dico.size(), p).to_column_matrix();
			
			System.out.print(reverse_dico.get(p));
			c++;
			if(c > 30)
				break;
		}
		System.out.println();
	}
	
	public static void main(String[] args) throws IOException {
		load_noun_data();
		
		long time = System.currentTimeMillis();
		seed = System.currentTimeMillis();
		RandomGenerator.init(seed);
		System.out.println("# Seed : " + seed);

		Parameters p = new Parameters("reg=0.0", "lr=0.01");

		int units = 2;
		
		GRUCell model = new GRUCell(units, dico.size(), p);
		SoftmaxCrossEntropy loss = new SoftmaxCrossEntropy();
		DenseLayer out_l = new DenseLayer(units, dico.size(), 0, "none", false, p);
		Optimizer opt = new SGDOptimizer(p);
		
		opt.init_mat(model);
		opt.init_vec(model);
		
		opt.init_mat(out_l.al);
		opt.init_vec(out_l.al);
		
		
		System.out.println("# Processors : " + Runtime.getRuntime().availableProcessors());
		System.out.println("# Initialization took " + (System.currentTimeMillis() - time) + " ms");
		
		int total = 0;
		int TROLL = 0;
		
		ArrayList<RNNCellCache> caches = new ArrayList<>();
		int lossN = 0;
		for (int epoch = 1; epoch <= EPOCHMAX; epoch++) {
			double lossAverage = 0;
			long haha = System.currentTimeMillis();
			
			for(int k = 0 ; k < train_data.size() ; k++) {
				Matrix d = train_data.get(k);
				
				Matrix state = new Matrix(1, units);
				
				caches.clear();
				
				for(int l = 0 ; l < d.width-1 ; l++) {
					RNNCellCache c = new RNNCellCache();
					state = model.step(d.get_column(l).to_column_matrix(), state, true, c);
					caches.add(c);
				}
				
				Matrix out = out_l.forward(state, true);
				Matrix real_out = loss.forward(out, true);
				
				loss.feed_ref(train_refs.get(k));
				Matrix dout = loss.backward(new Matrix(real_out), true);
				
				dout.scale(1/32.0);
				
				if(Double.isNaN(loss.loss)) {
					System.out.println(real_out.get_column(0));
					System.out.println(dout.get_column(0));
					System.out.println(loss.loss);
					System.exit(0);
				}
				
				dout = out_l.backward(dout, true);
				for(int i = caches.size()-1 ; i >= 0 ; i--) {
					dout = model.backward(dout, caches.get(i));
				}
				lossAverage += loss.loss;
				total++;
				if(total%32 == 0) {
					opt.optimize();
				}
				lossN += 1;
				
				if((k+1)%1000==0) {
					Matrix m = new Matrix(model.c_layer.al.matrices.get("w"));
					double a = m.hadamart(m).sum();
					

					Matrix m2 = new Matrix(model.ru_layer.al.matrices.get("w"));
					double b = m2.hadamart(m2).sum();
					

					Matrix m3 = new Matrix(out_l.al.matrices.get("w"));
					double c = m3.hadamart(m3).sum();
					System.out.println((lossAverage / (lossN)) + " " + k + " " + train_data.size());
					System.out.println((a+b+c)*p.get_as_double("reg", 0)+" - "+a+" "+b+" "+c);
					lossAverage = 0;
					lossN = 0;
					model.write_to_file("grumodel");
					out_l.write_to_file("grumodel/out_al");
					
					TROLL++;
					File f = new File("out_pos_"+TROLL+".txt");
					if(!f.exists())
						f.createNewFile();
					
					PrintWriter pw = new PrintWriter(f);
					int i = 0;
					for(Matrix d3 : train_data) {
						Matrix special_state = new Matrix(1, units);
						for(int l = 0 ; l < d3.width-1 ; l++) {
							special_state = model.step(d3.get_column(l).to_column_matrix(), special_state, false, null);
						}
						System.out.println(special_state.get_column(0));
						pw.write(matrix_to_name(d3)+" "+train_refs.get(i).get_column(0).argmax()+" "+special_state.v[0][0] +" " + special_state.v[1][0]+"\n");
						i++;
						if(i>300)
							break;
					}
					pw.close();
				}
			}
			System.out.println(System.currentTimeMillis()-haha);
			System.out.println("EPOCH END");
			System.out.println( "--");
		}
	}
}