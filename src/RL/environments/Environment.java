package RL.environments;

import math.Vector;

public abstract class Environment {
	public Integer action_size;
	public Integer state_size;
	
	public Vector state;
	
	public abstract void init();
	public abstract double apply_action(int action); // Returns reward
	public abstract boolean is_terminal_state();
	public Vector get_state() {
		return new Vector(state);
	}
}
