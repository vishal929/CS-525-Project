# file for testing our model
from Data_Preparation import data_util
import train

def test(model, test_set):
    # same concern with steps in train function, need batch_size
    test_loss, test_acc = model.evaluate(test_set, verbose=2, steps=None)
    return test_loss, test_acc

test_set = data_util.tf_dataset('test')
model = train.trained_model
test_results = test(model, test_set)
# can also use model.predict() on a fraction of the test set to demonstrate some results