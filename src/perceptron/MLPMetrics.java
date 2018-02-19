package perceptron;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;

import math.Matrix;
import math.Vector;

public class MLPMetrics {
	public ArrayList<Double> train_acc, validation_acc, loss;
	public Matrix confusion;
	public Matrix k_wrongest;
	public Matrix average_classes;

	public Matrix confusion_validation;
	public Matrix k_wrongest_validation;
	public Matrix average_classes_validation;
	
	public MLPMetrics() {
		train_acc = new ArrayList<Double>();
		validation_acc = new ArrayList<Double>();
		loss = new ArrayList<Double>();
	}

	public void add_time_series(double train, double validation, double loss) {
		train_acc.add(train);
		validation_acc.add(validation);
		this.loss.add(loss);
	}

	public void measure_and_write(String base_path, FlatSequential model, Matrix data, int[] refs, boolean confusion_as_csv) {
		Matrix confusion = model.confusion_matrix(data, refs);
		Matrix k_wrongest = model.k_wrongest_data(data, refs, 10);
		Matrix average_classes = model.average_data_by_class(data);
		
		k_wrongest.T().visualize(base_path + "_k_wrongest", 28, k_wrongest.width, 1, true, true, false);
		average_classes.T().visualize(base_path + "_average_classes", 28, average_classes.width, 1, true, true, false);
		if(confusion_as_csv) {
			File d = new File(base_path+ "_confusion.csv");
			if (!d.exists()) {
				try {
					d.createNewFile();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
			try {
				PrintWriter pw = new PrintWriter(d);
				StringBuilder sb = new StringBuilder();
				
				sb.append("/;");
				for(int i = 0 ; i < confusion.width ; i++) {
					sb.append(i+";");
				}
				sb.append("Correct\n");
				for(int j = 0 ; j < confusion.height ; j++) {
					sb.append(j+";");
					for(int k = 0 ; k < confusion.width ; k++) {
						sb.append((int)confusion.v[j][k]+";");
					}
					sb.append('\n');
				}
				sb.append("Prediction;");
				pw.write(sb.toString());
				pw.close();
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			}
		} else {
			Vector v = new Vector(confusion.v[0]);
			for (int i = 1; i < confusion.width; i++) {
				v.append(new Vector(confusion.v[i]));
			}
			for(int i = 0 ; i < v.length ; i++) {
				v.v[i] = Math.log(1 + v.v[i]);
			}
			v.to_row_matrix().visualize(base_path + "confusion", confusion.width, 1, 1, true, true, false);
		}
	}
	
	public void write_time_series_csv(String base_path) {
		File d = new File(base_path);
		if (!d.exists()) {
			try {
				d.createNewFile();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		try {
			PrintWriter pw = new PrintWriter(d);
			StringBuilder sb = new StringBuilder();
			sb.append("train accuracy;");
			for (double val : train_acc) {
				sb.append(val + ";");
			}
			sb.append('\n');
			sb.append("validation accuracy;");
			for (double val : validation_acc) {
				sb.append(val + ";");
			}
			sb.append('\n');
			sb.append("Loss;");
			for (double val : loss) {
				sb.append(val + ";");
			}
			sb.append('\n');
			pw.write(sb.toString().replace('.', ','));
			pw.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
}
