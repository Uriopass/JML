package layers;

import java.util.HashMap;

/**
 * Cette classe correspond à une liste de paramètres, qui est donc simplement une surcouche d'une hashmap avec un parser pratique.
 */
public class Parameters {
	public HashMap<String, String> values;

	/**
	 * Paramètres pour initialiser les paramètres, par exemple Parameters p = new Parameters("lr=0.1", "reg=3");
	 */
	public Parameters(String... strings) {
		values = new HashMap<String, String>();
		for (String s : strings) {
			String[] keyval = s.split("=");
			if (keyval.length != 2) {
				throw new RuntimeException("Error parsing " + s);
			}
			values.put(keyval[0], keyval[1]);
		}
	}

	public String get(String key) {
		return values.get(key);
	}

	public String set(String key, String value) {
		return values.put(key, value);
	}

	public String get_or_default(String key, String default_value) {
		if (values.containsKey(key))
			return values.get(key);
		return default_value;
	}

	public double get_as_double(String key, double default_value) {
		if (values.containsKey(key))
			return Double.parseDouble(values.get(key));
		return default_value;
	}
}
