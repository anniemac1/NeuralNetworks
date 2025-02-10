from neural import NeuralNet

# each row is an (input, output) tuple
xor_data = [
    # input output corresponding example
    ([0.0, 0.0], [0.0]),  # [0, 0] => 0
    ([0.0, 1.0], [1.0]),  # [0, 1] => 1
    ([1.0, 0.0], [1.0]),  # [1, 1] => 1
    ([1.0, 1.0], [0.0])  # [1, 0] => 0
]

# TODO by students
# Make sure you read the pdf and add the "lab notes" in github
# one "lab note" for each task that you complete.

nn = NeuralNet(2, 10, 1)
nn.train(xor_data, learning_rate=0.5, print_interval=100, iters=2000)

for i in nn.test_with_expected(xor_data):
    print(f"input: {i[0]}, desired: {i[1]}, actual: {i[2]}")

'''
Task 1: 

- Chaning Learning Rate:

Control: Learning rate is 0.5:
Error after 1000 iterations: 0.010556989366710737

Decreasing learning rate to 0.25: The error converges slowly 
Error after 1000 iterations: 0.016569588980528283

Increasing learning rate to 1: The error converges quicker
Error after 1000 iterations: 0.00171507716526054


- Changing print interval to 50: 

Error after 50 iterations: 0.5581518963099978
Error after 100 iterations: 0.5215646836025305
Error after 150 iterations: 0.46548149005566936
Error after 200 iterations: 0.37332463318769005


- Changing number of epochs to 2000:

Error after 1700 iterations: 0.002903302911385395
Error after 1800 iterations: 0.002685638626737212
Error after 1900 iterations: 0.002496875602687413
Error after 2000 iterations: 0.002331732622419867


- Changing number of hidden nodes to 10: Error falls quickly 

Error after 1000 iterations: 0.00465598225795001

'''

nn_task2 = NeuralNet(2, 1, 1)
nn_task2.train(xor_data, iters=1000)

for i in nn_task2.test_with_expected(xor_data):
    print(f"input: {i[0]}, desired: {i[1]}, actual: {i[2]}")

'''
Task 2: 

Errors are essentially random, oscillating around 0.5 and 0.3. 
It indicates that a single perceptron can not be programmed to predict XOR functions. 

'''

with open('./wine.data', 'r') as file:
    content = file.readlines()

num_columns = len(content[0].split(","))

min_arr = [10000] * num_columns
max_arr = [-100000] * num_columns

for line in content:
    values = line.split(",")
    for i in range(len(values)):
        min_arr[i] = min(min_arr[i], float(values[i]))
        max_arr[i] = max(max_arr[i], float(values[i]))

with open('./wine_normalized.data', 'w') as wine_normalized_file:
    for line in content:
        values = line.split(",")
        for i in range(len(values)):
            norm_val = ((float(values[i]) - min_arr[i]) / (max_arr[i] - min_arr[i]))
            wine_normalized_file.write(str(norm_val))
            wine_normalized_file.write(",")
        wine_normalized_file.write("\n")

wine_nn = NeuralNet(num_columns - 1, 5, 1)

training_data = []

with open('./wine_normalized.data', 'r') as wine_normalized_file:
    content = wine_normalized_file.readlines()

    for line in content:
        values = line.split(",")
        values = (values[0:len(values) - 1])
        values = [float(val) for val in values]
        y_output = [values[0]]
        x_input = values[1:]
        data_point = (x_input, y_output)
        training_data += [data_point]

    print("Start training")
    wine_nn.train(training_data)
    print("End training")

    '''
    Error after 100 iterations: 0.49911247767986994
    Error after 200 iterations: 0.2730044337159427
    Error after 300 iterations: 0.17432068651583268
    Error after 400 iterations: 0.1243399052984436
    Error after 500 iterations: 0.09533650803949796
    Error after 600 iterations: 0.07670620517688388
    Error after 700 iterations: 0.06384481502920164
    Error after 800 iterations: 0.0544842220291106
    Error after 900 iterations: 0.04739202626714649
    Error after 1000 iterations: 0.04184606603874662
    '''