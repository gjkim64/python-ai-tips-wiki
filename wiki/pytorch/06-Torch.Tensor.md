## dimension

A standard PyTorch torch.Tensor cannot have the shape ([99], [1, 2, 3, 4, 5, 6]) because all rows in a standard 2D tensor must have the exact same length.

## Tensor and its value are different

x = torch.rand(1) --> get a tensor with some random value between 0 and 1 (e.g. 0.5555)

x.item() --> get actual value of 0.5555


## you can index tensor like array but ...

But do not use it in a for loop to iterate and do something --> extremely slow

If possible use vector operation, e.g. instead of for loop to sum, just use tensor.sum, tensor. add, ...

## tensor.view 

torch.Tensor.view() is a PyTorch method that returns a new tensor with a different shape while sharing the exact same underlying memory as the original tensor. 
Because it does not copy data—it only changes how PyTorch reads the tensor's metadata (shape and strides)—it is extremely fast and memory-efficient.

