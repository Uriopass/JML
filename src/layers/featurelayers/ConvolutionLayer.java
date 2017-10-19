package layers.featurelayers;

import layers.FeatureLayer;
import layers.Parameters;
import math.FeatureMatrix;
import math.Initialisations;
import math.Matrix;
import math.Optimizers;
import math.Vector;

public class ConvolutionLayer implements FeatureLayer {
	public int width_out, height_out, features_out;
	public int width_in, height_in, features_in;
	public FeatureMatrix[] weights;
	public FeatureMatrix[] acc;
	public Vector biases;
	public int pad, stride, conv_size;

	public FeatureMatrix[] w_grad;
	public Vector b_grad;
	public double learning_rate, gamma, eps;
	public FeatureMatrix cache;
	public boolean calculate_dout = true;

	public ConvolutionLayer(int width_in, int height_in, int features_in, int features_out, int conv_size, int stride,
			int padding, Parameters p) {
		width_out = width_in - conv_size + 2 * padding;
		height_out = height_in - conv_size + 2 * padding;
		if ((width_out % stride) != 0 || (height_out % stride) != 0) {
			throw new RuntimeException("Invalid convolution dimensions");
		}
		width_out = width_out / stride + 1;
		height_out = height_out / stride + 1;
		this.width_in = width_in;
		this.height_in = height_in;
		this.features_in = features_in;
		this.features_out = features_out;
		pad = padding;
		this.stride = stride;
		this.conv_size = conv_size;
		weights = new FeatureMatrix[features_out];
		w_grad = new FeatureMatrix[features_out];
		acc = new FeatureMatrix[features_out];
		biases = new Vector(features_out);
		b_grad = new Vector(features_out);
		for (int f = 0; f < features_out; f++) {
			weights[f] = new FeatureMatrix(features_in, conv_size, conv_size);
			w_grad[f] = new FeatureMatrix(features_in, conv_size, conv_size);
			acc[f] = new FeatureMatrix(features_in, conv_size, conv_size);
			for (int i = 0; i < weights[f].features; i++) {
				Initialisations.he_uniform(weights[f].v[i], conv_size * conv_size, 1);
			}
		}
		learning_rate = p.getAsDouble("lr", 0.01);
		gamma = p.getAsDouble("gamma", 0.9);
		eps = p.getAsDouble("eps", 1e-8);
		calculate_dout = p.getAsString("dout", "true").equalsIgnoreCase("true");
	}

	@Override
	public FeatureMatrix forward(FeatureMatrix in, boolean training) {
		in.zero_pad(pad);
		if (training)
			cache = in;
		FeatureMatrix out = new FeatureMatrix(features_out, width_out, height_out);
		for (int f_out = 0; f_out < out.features; f_out++) {
			FeatureMatrix conv_w = weights[f_out];
			for (int x_out = 0; x_out < out.width; x_out++) {
				for (int y_out = 0; y_out < out.height; y_out++) {
					double sum = 0;
					for (int f_in = 0; f_in < conv_w.features; f_in++) {
						for (int off_x = 0; off_x < conv_w.width; off_x++) {
							for (int off_y = 0; off_y < conv_w.height; off_y++) {
								sum += in.v[f_in].v[y_out * stride + off_y][x_out * stride + off_x];
							}
						}
					}
					sum += biases.v[f_out];
					out.v[f_out].v[y_out][x_out] = sum;
				}
			}
		}
		return out;
	}

	@Override
	public FeatureMatrix backward(FeatureMatrix dout) {
		for (int i = 0; i < dout.features; i++) {
			b_grad.v[i] += dout.v[i].sum();
		}

		for (int f_out = 0; f_out < features_out; f_out++) {
			for (int f_in = 0; f_in < features_in; f_in++) {
				for (int x = 0; x < conv_size; x++) {
					for (int y = 0; y < conv_size; y++) {
						double sum = 0;
						for (int offx = x; offx < x + dout.width * stride; offx += stride) {
							for (int offy = y; offy < y + dout.height * stride; offy += stride) {
								sum += dout.v[f_out].v[(offy - y) / stride][(offx - x) / stride]
										* cache.v[f_in].v[y][x];
							}
						}
						w_grad[f_out].v[f_in].v[y][x] += sum;
					}
				}
			}
		}

		FeatureMatrix dx = new FeatureMatrix(features_in, width_in, height_in);
		//System.out.println("hmm... "+dx.height*dx.width*features_out*dout.width*dout.height*features_in*features_in*conv_size);

		if(calculate_dout) {
			Thread[] threads;
			threads = new Thread[features_out];
			
			for (int f_out = 0; f_out < features_out; f_out++) {
				final Matrix doutv = dout.v[f_out];
				final FeatureMatrix weightsfout = weights[f_out];
				threads[f_out] = new Thread(new Runnable() {
					public void run() {
						for (int c = 0; c < features_in; c++) {
							for (int y = 0; y < height_in; y++) {
								for (int x = 0; x < width_in; x++) {
									double megasum = 0;
									boolean ok1 = true, ok2 = true;
									for (int y_out = 0; y_out < height_out && (ok1 || ok2); y_out++) {
										for (int x_out = 0; x_out < width_out && (ok1 || ok2); x_out++) {
											double sum = 0;
											int dec2 = y + pad - y_out * stride;
											ok1 = (dec2) >= 0;
											if ((dec2) < conv_size && ok1) {
												for (int a = 0; a < features_in; a++) {
													for (int b = 0; b < conv_size; b++) {
														sum += weightsfout.v[a].v[dec2][b];
													}
												}
											}
											int dec = x + pad - x_out * stride;
											ok2 = (dec) >= 0;
											if ((dec) < conv_size && ok2) {
												for (int a = 0; a < features_in; a++) {
													for (int b = 0; b < conv_size; b++) {
														if (b != dec2)
															sum += weightsfout.v[a].v[b][dec];
													}
												}
											}
	
											megasum += doutv.v[y_out][x_out] * sum;
										}
									}
									dx.v[c].v[y][x] += megasum;
								}
							}
						}
					}
				});
				threads[f_out].start();
			}
			for(int i = 0 ; i < features_out ; i++) {
				try {
					threads[i].join();
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
		}
		return dx;
	}

	@Override
	public void apply_gradient() {
		for (int i = 0; i < weights.length; i++) {
			Optimizers.RMSProp(weights[i], w_grad[i], acc[i], gamma, learning_rate, eps);
		}
	}

	@Override
	public String toString() {
		return "ConvolutionLayer(("+conv_size+", "+conv_size+"), "+features_out+", "+stride+", "+pad+")";
	}
}
