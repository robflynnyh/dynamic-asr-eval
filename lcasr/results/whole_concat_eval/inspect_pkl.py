import pickle
with open('earnings22-test-whole-concat-epoch-1-lr-9em5_1.pkl', 'rb') as f:
    data = pickle.load(f)
    print("Baseline keys:", data['baseline'].keys())
    print("Baseline content:", data['baseline'])
