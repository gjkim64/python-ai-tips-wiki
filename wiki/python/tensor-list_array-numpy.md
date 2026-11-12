# tensor flow attributes or methods?

# prob is a tensor

print (prob.shape) # returns torch.Size object
print (prob.dtype) # returns torch data type
print (prob.size()) # returns torch.Size object

loss = torch.tensor(0.2341, requires_grad=True)

loss_value = loss.item() # only cpu tensor
print (loss_value)

# python objects or list / array

arr.shape
arr.size
arr.dtype

# numpy array

same as above

arra = np.array ( ... ) 
arr.shape
arr.size
arr.dtype


There is no .value or .value() method or attribute either for tensor or python objects

There is .values() function on tensor


tensor.values()	Only on sparse tensors	
Returns a tensor of non-zero values for sparse tensors; 
raises a RuntimeError or AttributeError on standard dense tensors.
