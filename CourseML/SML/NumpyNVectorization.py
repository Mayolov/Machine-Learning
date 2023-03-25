import numpy as np
import time 

# NumPy routines which allocate memory and fill arrays with value
a = np.zeros(4);                   print(f"np,zeros(4) : {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.zeros((4,));                print(f"np,zeros(4) : {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.random.random_sample((4,)); print(f"np,zeros(4) : {a}, a={a}, a shape = {a.shape}, a data type = {a.dtype}")

# NumPy routines which allocate memory and fill arrays with value but do not accept shape as input argument
a = np.arange(4.);                 print(f"np,arange(4) : {a}, a = {a.shape}, a data type = {a.dtype}")
a = np.random.rand((4)); print(f"np.random.rand(4) : {a}, a = {a.shape}, a data type = {a.dtype}")

# NumPy routines which allocate memory and fill with user specified values
a = np.array([5,4,3,2]);  print(f"np.array([5,4,3,2]):  a = {a},     a shape = {a.shape}, a data type = {a.dtype}")
a = np.array([5.,4,3,2]); print(f"np.array([5.,4,3,2]): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

#vector indexing operations on 1-D vectors
a = np.arange(10)
print(a)

#accessing elements of 1-D vectors
print(f"a[2].shape = {a[2].shape} a[2] = {a[2]} Acessing an element returns a scalar")
# acces last of
print(f"a[-1].shape = {a[-1].shape}")


# example of calling an index from out of range of the vector and producing error 
try:
    c = a[10]
except Exception as e:
    print("There error is: {e}")
    
# Slicing creates an array of indics using a set of three values (start, stop, step)

#vector slicing operations
a = np.arange(10)
print(f"a = {a}")

# access 5 consecutive elements 
c = a[2:7:1]; print(f"a[2:7:1] = {c}")

# access 3 consecutive elements 
c = a[2:7:2]; print(f"a[2:7:2] = {c}")

# access all elements index 3 and above
c = a[3::]; print(f"a[3::] = {c}")

# access all elements below index 3
c = a[:3]; print(f"a[:3] = {c}")

# access all els
c = a[:]; print("a[:] = {c}")

# some ways to use numpy for single vector operations

a = np.array([1,2,3,4])
print(f"a = {a}")

#negate els of a
b = -a; print(f"b = = -a : {b}")

# sum all elements of a, returns a a scalar

b = np.sum(a); print(f"b = np.sum(a) : {b}")

b = np.mean(a); print(f"b = np.mean(a) : {b}")

b = a**2; print(f"b = a**2 : {b}")

# Vector Vector element wise operations
a = np.array([1,2,3,4])
b = np.array([-1,-2,3,4])

print(f" Binary operators work element wise : {a+b}")

# vectors have to be the same size or this happens
c = np.array([1,2])

try:
    d = a+c
except Exception as e:
    print(f"Exception is {e}")

#vectors can be scaled

a = np.array([1,2,3,4])
#multiply by a scalar

b = 5*a; print(f"b = 5*a : {b}")

# Dot product
# its faster to use  np.dot() than a for loop
# test 1-D
a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])
print(f"my_dot(a, b) = {np.dot(a, b)}")
# test 1-D
a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])
c = np.dot(a, b)
print(f"NumPy 1-D np.dot(a, b) = {c}, np.dot(a, b).shape = {c.shape} ") 
c = np.dot(b, a)
print(f"NumPy 1-D np.dot(b, a) = {c}, np.dot(a, b).shape = {c.shape} ")

# matrix creation 2D arrays made by the tuple in the params
a = np.zeros((1,5)); print(f"a shape = {a.shape}, a data type = {a.dtype}")

a = np.zeros((2,1)); print(f"a shape = {a.shape}, a data type = {a.dtype}")

a = np.random.random_sample((1,1)); print(f"a shape = {a.shape}, a data type = {a.dtype}")

# NumPy routines which allocate memory and fill with user specified values
a = np.array([[5], [4], [3]]);   print(f" a shape = {a.shape}, np.array: a = {a}")
a = np.array([[5],   # One can also
              [4],   # separate values
              [3]]); #into separate rows
print(f" a shape = {a.shape}, np.array: a = {a}")

# indexing a matrix

a = np.arange(6).reshape(-1,2); print(f"a shape = {a.shape}, \na = {a}")

#accessing elements 
print(f"\na[2,0].shape = {a[2,0].shape}, a[2,0] = {a[2,0]}, type(a[2,0]) = {type(a[2])}")

# this creates a 1d vector of 6 els tehn using reshape changed ontp a 2Dvector 
a = np.arange(6).reshape(3,2)

# Slicing Matrices (start: stop: step)
# vector 2D slicing operations
a = np.arange(20).reshape(-1,10)
print(f"\na shape = {a.shape}, a = {a}")

# Acessing 5 consecutive elements(start: stop: step)
print(f"a[0,2:7:1] = {a[1,2:7:1]}, a[0,2:7:1].shape = {a[0,2:7:1].shape}, a 1-D array")

#access 5 consecutive elements (start:stop:step) in two rows
print("a[:, 2:7:1] = \n", a[:, 2:7:1]," a[:, 2:7:1].shape = ", a[:, 2:7:1].shape, "a 2-D array")

# all els
print("a[:,:] = \n", a[:,:], ",  a[:,:].shape =", a[:,:].shape)

# access all elements in one row (very common usage)
print("a[1,:] = ", a[1,:], ",  a[1,:].shape =", a[1,:].shape, "a 1-D array")

# same as
print("a[1]   = ", a[1],   ",  a[1].shape   =", a[1].shape, "a 1-D array")