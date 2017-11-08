package datareaders;

import caltech.CalTech101;
import math.Matrix;
import math.Vector;

public class Cal101Reader {
	static Matrix train_data;
	static int[] train_refs;

	static Matrix test_data;
	static int[] test_refs;
	
	static boolean loaded = false;
	
	public static Matrix get_train_data() {
		if(!loaded)
			loadData();
		return train_data;
	}
	
	public static int[] get_train_refs() {
		if(!loaded)
			loadData();
		return train_refs;
	}
	
	public static Matrix get_test_data() {
		if(!loaded)
			loadData();
		return test_data;
	}
	
	public static int[] get_test_refs() {
		if(!loaded)
			loadData();
		return test_refs;
	}
	
	public static Vector flip(Vector v) {
		Vector v2 = new Vector(v.length);
		for(int i = 0 ; i < v2.length ; i++) {
			v2.v[i] = v.v[(i%28)*28+i/28];
		}
		return v2;
	}
	
	public static void loadData() {
		loaded = true;
		CalTech101 CT = null;
		try {
			CT = new CalTech101("caltech101.mat");
		} catch (Exception e) {
			e.printStackTrace();
		}
		train_data = new Matrix(4100, 784);
		train_refs = new int[4100];
		
		test_data = new Matrix(2307, 784);
		test_refs = new int[2307];

		for(int i=0; i<4100; i++) {
			train_refs[i] = CT.getTrainLabel(i) - 1;
			train_data.set_column(i, flip(new Vector(CT.getTrainImage(i))));
		}

		for(int i=0; i<2307; i++) {
			test_refs[i] = CT.getTestLabel(i) - 1;
			test_data.set_column(i, flip(new Vector(CT.getTestImage(i))));
		}
	}
}
