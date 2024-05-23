import json
import random


def load_json_data(filepath):
    """Load JSON data from a file."""
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data


def sample_incidents(data, n=50):
    """Sample n incidents from the data."""
    return random.sample(data, n)
