package RL;

import java.util.ArrayList;

import RL.environments.Environment;
import layers.losses.Loss;
import math.Matrix;
import math.RandomGenerator;
import math.Vector;
import perceptron.MultiLayerPerceptron;

public class QLearn {
	MultiLayerPerceptron net;
	Environment env;
	
	public double epsilon_greed = 0.1;
	EpsilonFunction eps_strategy = EpsilonFunction.constant;
	
	public double discount = 0.9;
	public int mini_batch_size = 64;
	public double loss = 0;
	public double cumulated_reward = 0;

	ArrayList<Experience> experiences;
	
	public QLearn(MultiLayerPerceptron net, Environment env, double epsilon, EpsilonFunction epsilon_strategy) {
		this.net = net;
		this.env = env;
		this.epsilon_greed = epsilon;
		this.eps_strategy = epsilon_strategy;
		experiences = new ArrayList<Experience>();
	}
	
	
	public void experience_replay() {
		int mini_batch = Math.min(experiences.size(), mini_batch_size);
		
		Matrix batch = new Matrix(mini_batch, env.state_size);
		Matrix batch_next = new Matrix(mini_batch, env.state_size);
		Vector refs = new Vector(mini_batch);
		
		int[] sample = RandomGenerator.sample(experiences.size(), mini_batch);
		
		
		
		for(int i = 0 ; i < mini_batch ; i++) {
			Experience e = experiences.get(sample[i]);
			batch.set_column(i, e.s);
			batch_next.set_column(i, e.next_s);
		}
		
		Matrix batch_next_result = net.forward(batch_next);
		
		for (int i = 0; i < mini_batch; i++) {
			Experience e = experiences.get(sample[i]);
			double yj;
			if(e.is_terminal) {
				yj = e.r; // r
			} else {
				yj = e.r + discount*batch_next_result.get_column(i).max(); // r + discount*max_Q(s+1)
			}
			refs.v[i] = yj;
		
		}
		
		Matrix result = net.forward_train(batch);
		Matrix dout = new Matrix(result);
		for(int i = 0 ; i < mini_batch ; i++) {
			result.v[experiences.get(sample[i]).a][i] = refs.v[i];
		}
		Loss ql = net.getLoss();
		ql.feed_ref(result);
		net.backward_train(dout);
		loss += ql.loss;
	}
	
	public void learn(int epochs, int print_every) {
		double local_reward = 0;
		for(int episode = 1 ; episode < epochs ; episode ++) {
			double eps = epsilon_greed * eps_strategy.epsilon(episode, epochs);
			env.init();
			if(episode%print_every == 0) {
				System.out.println("Average reward over "+print_every+" : " + (local_reward/print_every));
				System.out.println("Average loss   over "+print_every+" : " + (loss/print_every));
				
				local_reward = 0;
				loss = 0;
			}
			//System.out.println("--");
			//env.print_state();
			for(int t = 1 ; ; t++) {
				Vector previous = env.get_state();
				int a;
				if(RandomGenerator.uniform(0.0, 1.0) < eps) {
					a = RandomGenerator.uniform_int(0, env.action_size);
				} else {
					a = net.forward(env.state.to_column_matrix()).get_column(0).argmax();
				}
				double reward = env.apply_action(a);
				this.cumulated_reward += reward;
				local_reward += reward;
				Vector after = env.get_state();
				Experience e = new Experience();
				e.s = previous;
				e.r = reward;
				e.a = a;
				e.next_s = after;
				e.is_terminal = env.is_terminal_state();
				experiences.add(e);
				experience_replay();
				if(e.is_terminal)
					break;
			}
			/*
			if(episode%print_every == 0) {
				env.init();
				while(!env.is_terminal_state()) {
					env.print_state();
					Vector a = net.forward(env.state.to_column_matrix()).get_column(0);
					System.out.println(a);
					env.apply_action(a.argmax());
				}
				env.print_state();
			}
			*/
			//System.out.println("->");
			//env.print_state();
		}
	}
}
