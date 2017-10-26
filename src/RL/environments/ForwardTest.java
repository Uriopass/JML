package RL.environments;

import math.Vector;

public class ForwardTest extends Environment {

	public int x;
	
	public ForwardTest(int distance) {
		this.action_size = 2;
		this.state_size = distance+2;
		this.state = new Vector(state_size);
		x = 1;
		state.v[1] = 1;
	}
	
	@Override
	public void init() {
		state.fill(0);
		state.v[1] = 1;
		x = 1;
	}

	@Override
	public double apply_action(int action) {
		state.v[x] = 0;
		switch(action) {
		case 0:
			x -= 1;
			state.v[x] = 1;
			if(x == 0)
				return 1;
			return 0;
		case 1:
			x += 1;
			state.v[x] = 1;
			if(x == state.length-1)
				return 1000;
			return 0;
		}
		return 0;
	}

	@Override
	public boolean is_terminal_state() {
		return x == 0 || x == state.length-1;
	}

}
