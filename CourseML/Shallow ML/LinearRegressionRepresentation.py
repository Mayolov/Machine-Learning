import numpy as np
import matplotlib.pyplot as plt
"""
    x_trai: training Examples(size in this program)
    y_train: training example targets(Price in this program)
    x_i,y_i: ith training Example 
    m = number of training examples
    b = bias parameter
    w= weight parameter
"""
def computer_model_output(x,w,b):
    """
    Compute the output of the linear model
    x: input data(ndarray)
    w,b: weights and bias(scalar model parameters)
    y: output of the linear model(ndarray target values)
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w*x[i] + b
        print(f"{f_wb[i]} = {w*x[i]} + {b}")
    return f_wb

x_train = np.array([1.0,2.0])
y_train = np.array([300.0,500.0])

print(f"x_train = {x_train}, y_train = {y_train}")

print(f"x_train.shape = {x_train.shape}")
m = x_train.shape[0]
# m=len(x_train) also works
print(f" number of training examples 'm' = {m}")

i = 0
x_i = x_train[i]
y_i = y_train[i]
print(f"(x^{i}) = {i}, (y^{i}) = {i}")

plt.scatter(x_train, y_train,marker='x',c='r')

plt.title('Housing Prices')

plt.ylabel('price(in 1000s of dollars)')

plt.xlabel('Size (1000sqft)')
plt.show()

w = 150
b = 150
print(f"w = {w}, b = {b}")

tmp_f_wb = computer_model_output(x_train,w,b)

plt.plot(x_train,tmp_f_wb,c='b',label="Our Prediction")

plt.scatter(x_train,y_train,marker = 'x' ,c='r',label="Actual Values")

plt.title('Housing Prices')

plt.ylabel('price(in 1000s of dollars)')

plt.xlabel('Size (1000sqft)')
plt.legend()
plt.show()

w = 200
b = 100
x_i = 1.2
cost_1200sqft = w*x_i + b
print(f"cost_1200sqft = {cost_1200sqft} thousand of dollars")