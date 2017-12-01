package main;

import RL.EpsilonFunction;
import RL.QLearn;
import RL.environments.Environment;
import RL.environments.GridWorld;
import layers.Parameters;
import layers.flat.DenseLayer;
import layers.losses.QuadraticLoss;
import math.RandomGenerator;
import optimizers.RMSOptimizer;
import perceptron.FlatSequential;

public class MainQLearning {
	
	
	public static void main(String[] args) {
		/*System.out.println("Appuyez sur ENTER pour d�marrer : ");
		try {
			System.in.read();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}*/

		for(int i = 1 ; i < 10 ; i++) {
			Parameters p = new Parameters("reg=0.00001");
			Environment env = new GridWorld(8, 8);
			FlatSequential model = new FlatSequential(64, new RMSOptimizer(p));
			int hidden = 48;
			model.add(new DenseLayer(env.state_size, hidden, 0, "tanh", false, p));		
			// model.add(new GaussianNoise(0.05));
			model.add(new DenseLayer(hidden, env.action_size, 0, "none", false, p));
			model.add(new QuadraticLoss());
			
			RandomGenerator.init(System.currentTimeMillis());
			QLearn learner = new QLearn(model, env, 0.1, EpsilonFunction.constant);
			learner.learn(1000, 100);
			System.out.println(learner.cumulated_reward/1000);
		}
	}
}
