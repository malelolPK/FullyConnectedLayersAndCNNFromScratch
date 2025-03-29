import torch
import torch.nn as nn
# I have tested this with the Michigan Online course:
# EECS 498.008 / 598.008 - Deep Learning for Computer Vision, Winter 2022.
# So I know it works fine, but it's so unoptimized that you wouldn't want to run it.

# IMPORTANT: The comments in """" are partially from the course,
# but the actual code was written by me.
class Conv():

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        Simple impelentaction of the forward of conv

        Input:
        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - conv_param: A dictionary with the following keys:
          - 'stride': The number of pixels between adjacent receptive fields
            in the horizontal and vertical directions.
          - 'pad': The number of pixels that is used to zero-pad the input.

        Returns a tuple of:
        - out: Output data of shape (N, F, H', W') where H' and W' are given by
          H' = 1 + (H + 2 * pad - HH) / stride
          W' = 1 + (W + 2 * pad - WW) / stride
        - cache: (x, w, b, conv_param)
        """
        N, C, H, W = x.shape
        F, C, HH, WW = w.shape
        H_out = int(1 + (H + 2 * conv_param["pad"] - HH) / conv_param["stride"])
        W_out = int(1 + (W + 2 * conv_param["pad"] - WW) / conv_param["stride"])
        out = torch.zeros(size=(N, F, H_out, W_out), dtype=torch.float64, device='cuda')
        paded_x = x.clone()
        paded_x = torch.nn.functional.pad(paded_x, pad=(conv_param["pad"],conv_param["pad"],conv_param["pad"],conv_param["pad"]))

        for n in range(N):
            for f in range(F):
                for i in range(0, H, conv_param["stride"]):
                    h_out_idx = i // conv_param["stride"]
                    for j in range(0, W, conv_param["stride"]):
                        w_out_idx = j // conv_param["stride"]
                        for c in range(C):
                            out[n, f, h_out_idx, w_out_idx] += (paded_x[n, c, i:(i + HH), j:(j + WW)] * w[f, c]).sum()
                out[n, f] += b[f]

        cache = (x, w, b, conv_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a convolutional layer.
          Inputs:
        - dout: Upstream derivatives.
        - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

        Returns a tuple of:
        - dx: Gradient with respect to x
        - dw: Gradient with respect to w
        - db: Gradient with respect to b
        dout = to pewnie dloss

        db = dloss
        dx = i=1 do n sum(dloss * FULL kij)
        dw = dout(dloss) razy Iteracja po filtrach
        czyli jak mamy ten x to leicmy po fragmentach wejścia x i mnożymy te fragmenty z dloss

        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
         F ilość filtry
        -dout:  (4, 2, 5, 5)
        -dout: (N, F, HH, WW)
        conv_param: {'stride': 1, 'pad': 1}
        """
        x, w, b ,cons_param = cache
        F, C, HH, WW = w.shape
        N, _, H, W = x.shape
        _, _, h_out, w_out = dout.shape

        stride, pad = cons_param['stride'], cons_param['pad']

        x_padded = torch.nn.functional.pad(x, (pad, pad, pad, pad))
        dx_padded = torch.zeros_like(x_padded,  dtype=torch.float64, device='cuda')


        dx = torch.zeros_like(x_padded,  dtype=torch.float64, device='cuda')
        dw = torch.zeros_like(w,  dtype=torch.float64, device='cuda')
        db = torch.zeros_like(b,  dtype=torch.float64, device='cuda')

        db = dout.sum(dim=(0, 2, 3)) # 0 po 4 przykładach i 3 i 4  to że po HH i WW czyli wymiarach i mamy

        for n in range(N): # iteracja po przykładach
            for f in range(F): # iteracja po filtrach
                for i in range(h_out): # iteracja po wysokości wyjścia
                    for j in range(w_out): # iteracja po szerokości wyjścia
                        h_start = i * stride
                        h_end = h_start + HH
                        w_start = j * stride
                        w_end = w_start + WW

                        x_window = x_padded[n, :, h_start:h_end, w_start:w_end]
                        dw[f] += x_window * dout[n, f, i, j]


                        dx_padded[n, :, h_start:h_end, w_start:w_end] += w[f] * dout[n, f, i, j]

        dx = dx_padded[:, :, pad:pad+H, pad:pad+W]

        return dx, dw, db


class MaxPool():

    @staticmethod
    def forward(x, pool_param):
        """
        A naive implementation of the forward pass for a max-pooling layer.

        Inputs:
        - x: Input data, of shape (N, C, H, W)
        - pool_param: dictionary with the following keys:
          - 'pool_height': The height of each pooling region
          - 'pool_width': The width of each pooling region
          - 'stride': The distance between adjacent pooling regions
        No padding is necessary here.

        Returns a tuple of:
        - out: Output of shape (N, C, H', W') where H' and W' are given by
          H' = 1 + (H - pool_height) / stride
          W' = 1 + (W - pool_width) / stride
        - cache: (x, pool_param)
        """

        N, C, H, W = x.shape
        pool_height, pool_width, stride = pool_param["pool_height"], pool_param["pool_width"], pool_param["stride"]
        HH = int(1 + (H - pool_height) / stride)
        WW = int(1 + (W - pool_width) / stride)
        out = torch.zeros(size=(N, C, HH, WW),  dtype=torch.float64, device='cuda')

        for n in range(N):
            for c in range(C):
                for i in range(HH):
                    for j in range(WW):
                        h_start = i * stride
                        h_end = h_start + pool_height
                        w_start = j * stride
                        w_end = w_start + pool_width
                        value = x[n, c, h_start:h_end, w_start:w_end]
                        out[n, c, i, j] = torch.max(value)

        cache = (x, pool_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a max-pooling layer.
        Inputs:
        - dout: Upstream derivatives
        - cache: A tuple of (x, pool_param) as in the forward pass.
        Returns:
        - dx: Gradient with respect to x
        dout (3,2,4,4)
        x (3.2.8.8)
        """

        x, pool_param = cache
        N, C, H, W = x.shape
        _, _, HH, WW = dout.shape
        pool_height, pool_width, stride = pool_param["pool_height"], pool_param["pool_width"], pool_param["stride"]

        dx = torch.zeros_like(x)

        for n in range(N):
            for c in range(C):
                for h in range(0, H - pool_height + 1, stride):
                    for w in range(0, W - pool_width + 1, stride):
                        # Wybierz okno do przetworzenia
                        window = x[n, c, h:h + pool_height, w:w + pool_width]

                        # Znajdź pozycję maksymalnej wartości w oknie
                        # tworzy tablcę bool gdzie 1 jest gdy jest równe max a reszta 0 zatem jak pomnożę tę maskę przez dout to dostaniemy to co chcemy
                        mask = window == window.max()

                        # Oblicz indeksy w przestrzeni wyjściowej
                        h_out = h // stride
                        w_out = w // stride

                        # Przypisz gradient tylko do pozycji maksymalnej wartości
                        dx[n, c, h:h + pool_height, w:w + pool_width] += mask * dout[n, c, h_out, w_out]

        return dx


class ThreeLayerConvNet():
    """
    A three-layer convolutional network with the following architecture:
    conv - relu - 2x2 max pool - linear - relu - linear - softmax
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=32,
                 filter_size=7,
                 hidden_dim=100,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 dtype=torch.float,
                 device='cpu'):
        """
        Initialize a new network.
        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in convolutional layer
        - hidden_dim: Number of units to use in fully-connected hidden layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: A torch data type object; all computations will be performed
          using this datatype. float is faster but less accurate, so you
          should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        input_size = 8192
        self.params["W1"] = torch.normal(0.0, weight_scale, size=(num_filters, input_dims[0], filter_size, filter_size), dtype=dtype, device=device)
        self.params["b1"] = torch.normal(0.0, weight_scale, size=(num_filters,), dtype=dtype, device=device)
        self.params["W2"] = torch.normal(0.0, weight_scale, size=(input_size, hidden_dim), dtype=dtype, device=device)
        self.params["b2"] = torch.normal(0.0, weight_scale, size=(hidden_dim, ), dtype=dtype, device=device)
        self.params["W3"] = torch.normal(0.0, weight_scale, size=(hidden_dim, num_classes), dtype=dtype, device=device)
        self.params["b3"] = torch.normal(0.0, weight_scale, size=(num_classes,), dtype=dtype, device=device)


    def loss(self, X, y=None):

        X = X.to(self.dtype)
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        out_con1, cache = Conv_ReLU_Pool.forward(X, self.params['W1'], self.params['b1'], conv_param, pool_param)
        N = X.shape[0]
        flatten_con1 = torch.flatten(out_con1).view(N, -1) # ważne bo dostajemy w kolumnowym a potrzebujemy w wierszowym aby było (N, d) nie (d, N)
        z1, z1_cache = Linear.forward(flatten_con1, self.params["W2"], self.params["b2"])
        h1, h1_cache = ReLU.forward(z1)
        scores, z2_cache = Linear.forward(h1, self.params["W3"], self.params["b3"])

        if y is None:
            return scores

        loss, grads = 0.0, {}

        loss, dloss = softmax_loss(scores, y)
        l2_reg = self.reg * (torch.sum(self.params["W1"]**2) + torch.sum(self.params["W2"]**2) + torch.sum(self.params["W3"]**2))
        loss += l2_reg


        dz2, dW3, db3 = Linear.backward(dloss, (h1, self.params["W3"], self.params["b3"]))

        dW3 += self.reg * 2 * self.params["W3"]
        grads["W3"] = dW3
        grads["b3"] = db3

        dz2 = ReLU.backward(dz2, h1)

        dz1, dW2, db2 = Linear.backward(dz2, (flatten_con1, self.params["W2"], self.params["b2"]))

        dW2 += self.reg * 2 * self.params["W2"]
        grads["W2"] = dW2
        grads["b2"] = db2

        N, C, H, W = out_con1.shape
        dz1 = dz1.view(N, C, H, W)

        dx, dW1, db1 = Conv_ReLU_Pool.backward(dz1, cache)

        dW1 += self.reg * 2 * self.params["W1"]
        grads["W1"] = dW1
        grads["b1"] = db1


        return loss, grads


class DeepConvNet(nn.Module):
    """
    A convolutional neural network with an arbitrary number of convolutional
    layers in VGG-Net style. All convolution layers will use kernel size 3 and
    padding 1 to preserve the feature map size, and all pooling layers will be
    max pooling layers with 2x2 receptive fields and a stride of 2 to halve the
    size of the feature map.

    kernel size: 3x3
    padding: P = 1

    pooling: max 2x2
    stride: S = 2


    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """
    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=[8, 8, 8, 8, 8],
                 max_pools=[0, 1, 2, 3, 4],
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 weight_initializer=None,
                 dtype=torch.float,
                 device='cpu'):
        super().__init__()
        """
        Initialize a new network.

        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: List of length (L - 1) giving the number of
          convolutional filters to use in each macro layer.
        - max_pools: List of integers giving the indices of the macro
          layers that should have max pooling (zero-indexed).
        - batchnorm: Whether to include batch normalization in each macro layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights, or the string "kaiming" to use Kaiming
          initialization instead
        - reg: Scalar giving L2 regularization strength. L2 regularization
          should only be applied to convolutional and fully-connected weight
          matrices; it should not be applied to biases or to batchnorm scale
          and shifts.
        - dtype: A torch data type object; all computations will be performed
          using this datatype. float is faster but less accurate, so you should
          use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.num_layers = len(num_filters)+1
        self.max_pools = max_pools
        self.reg = reg
        self.dtype = dtype

        if device == 'cuda':
            device = 'cuda:0'

        filter_size = 3
        depth_input = input_dims[0]
        H, W = input_dims[1], input_dims[2]
        for l in range(0, self.num_layers-1):
            if weight_scale == 'kaiming':

                self.params["W" + str(l + 1)] = nn.Parameter(kaiming_initializer(depth_input, num_filters[l], K = filter_size, dtype=dtype, device=device))

                self.params[f"b{l + 1}"] = nn.Parameter(torch.zeros(num_filters[l], dtype=dtype, device=device))
            else:
                self.params["W" + str(l + 1)] = nn.Parameter(torch.normal(0.0, weight_scale,
                                                 size=(num_filters[l], depth_input, filter_size, filter_size),
                                                 dtype=dtype,
                                                 device=device))
                self.params[f"b{l+1}"] = nn.Parameter(torch.zeros(num_filters[l], dtype=dtype, device=device))
            depth_input = num_filters[l]

        final_conv_size = 128
        if weight_scale == 'kaiming':
            self.params["W" + str(self.num_layers)] = nn.Parameter(kaiming_initializer(final_conv_size, num_classes, K=None, dtype=dtype, device=device))
            self.params[f"b{self.num_layers}"] = nn.Parameter(torch.zeros(num_classes, dtype=dtype, device=device))
        else:
            self.params["W" + str(self.num_layers)] = nn.Parameter(torch.normal(0.0, weight_scale, size=(final_conv_size, num_classes), dtype=dtype, device=device))
            self.params[f"b{self.num_layers}"] = nn.Parameter(torch.zeros(num_classes, dtype=dtype, device=device))

    def parameters(self):
        return self.params

    def loss(self, X, y=None):
        X = X.to(self.dtype)
        mode = 'test' if y is None else 'train'
        filter_size = 3
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}


        scores = None

        out_con = {}
        cache_out_con = {}

        if 1 in self.max_pools:
            out_con["C" + str(1)], cache_out_con["C" + str(1)] = Conv_ReLU_Pool.forward(X, self.params['W' + str(1)], self.params['b' + str(1)], conv_param,pool_param)
        else:
            out_con["C" + str(1)], cache_out_con["C" + str(1)] = Conv_ReLU.forward(X, self.params['W' + str(1)],
                                                                                        self.params['b' + str(1)],
                                                                                        conv_param)
        for l in range(2, self.num_layers):
            if l in self.max_pools:
                out_con["C" + str(l)], cache_out_con["C" + str(l)]  = Conv_ReLU_Pool.forward(out_con["C" + str(l-1)], self.params['W' + str(l)], self.params['b' + str(l)], conv_param, pool_param)
            else:

                out_con["C" + str(l)], cache_out_con["C" + str(l)] = Conv_ReLU.forward(out_con["C" + str(l-1)], self.params['W' + str(l)],
                                                                                        self.params['b' + str(l)], conv_param)
        N = X.shape[0] # n przykładów
        flatten_con = torch.flatten(out_con["C" + str(self.num_layers-1)]).view(N, -1)
        scores, z_cache = Linear.forward(flatten_con, self.params['W' + str(self.num_layers)], self.params['b' + str(self.num_layers)])

        if y is None:
            return scores

        loss, grads = 0, {}

        loss, dloss = softmax_loss(scores, y)
        l2_reg = sum(torch.sum(self.params[f"W{i}"] ** 2) for i in range(1, self.num_layers + 1))
        loss += self.reg * l2_reg

        # Backpropagate  fully connected layer
        dprev, dW, db = Linear.backward(dloss, z_cache)
        grads[f"W{self.num_layers}"] = dW + self.reg * 2 * self.params[f"W{self.num_layers}"]
        grads[f"b{self.num_layers}"] = db  # Bias gradients do NOT include regularization

        # Backpropagate  convolutional layers
        N, C, H, W = out_con[f"C{self.num_layers-1}"].shape
        dprev = dprev.view(N, C, H, W)
        for l in range(self.num_layers - 1, 0, -1):
            cache = cache_out_con[f"C{l}"]
            if l  in self.max_pools:
                dprev, dW, db = Conv_ReLU_Pool.backward(dprev, cache)
            else:
                dprev, dW, db = Conv_ReLU.backward(dprev, cache)

            # Add L2 regularization to weight gradients
            grads[f"W{l}"] = dW + self.reg * 2 * self.params[f"W{l}"]
            grads[f"b{l}"] = db

        return loss, grads


def kaiming_initializer(Din, Dout, K=None, relu=True, device='cpu', dtype=torch.float32):
    """
    Implement Kaiming initialization for linear and convolution layers.

    Inputs:
    - Din (int): Number of input dimensions
    - Dout (int): Number of output dimensions
    - K: If K is None, then initialize weights for a linear layer with
      Din input dimensions and Dout output dimensions. Otherwise if K is
      a nonnegative integer then initialize the weights for a convolution
      layer with Din input channels, Dout output channels, and a kernel size
      of KxK.
    - relu: If ReLU=True, then initialize weights with a gain of 2 to
      account for a ReLU nonlinearity (Kaiming initializaiton); otherwise
      initialize weights with a gain of 1 (Xavier initialization).
    - device, dtype: The device and datatype for the output tensor.

    Returns:
    - weight:
      linear layer (Din, Dout);
      convolution layer (Dout, Din, K, K).
    """
    gain = 2. if relu else 1.
    weight = None
    if K is None: # for linear layer
        kaiming = torch.sqrt(torch.as_tensor(gain / Din, device=device, dtype=dtype))
        weight = torch.normal(0, kaiming, size=(Din, Dout), device=device, dtype=dtype)

    else: # for conv leyer
        fan_in = Din * K * K
        kaiming = torch.sqrt(torch.as_tensor(gain / fan_in, device=device, dtype=dtype))
        weight = torch.normal(0, kaiming, size=(Dout, Din, K, K), device=device, dtype=dtype)
    return weight


class Linear():

    @staticmethod
    def forward(x, w, b):
        """
        Inputs:
        - x: A tensor containing input data, of shape (N, d_1, ..., d_k) like (32, 3, 28, 28)
        - w: A tensor of weights, of shape (D, M), where D is d_1 * ... * d_k
        - b: A tensor of biases, of shape (M,)
        Returns a tuple of:
        - out: output, of shape (N, M)
        - cache: (x, w, b)
        """
        out = torch.matmul(x.reshape(x.shape[0], -1), w) + b
        cache = (x, w, b)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        - dout: Upstream derivative, of shape (N, M)
        - cache: Tuple of:
          - x: Input data, of shape (N, d_1, ... d_k)
          - w: Weights, of shape (D, M)
          - b: Biases, of shape (M,)
        Returns a tuple of:
        - dx: Gradient with respect to x, of shape
          (N, d1, ..., d_k)
        - dw: Gradient with respect to w, of shape (D, M)
        - db: Gradient with respect to b, of shape (M,)
        """
        x, w, b = cache
        dx = torch.matmul(dout, w.T).reshape(x.shape)
        dw = torch.matmul(x.reshape(x.shape[0], -1).T, dout)
        db = torch.sum(dout, dim=0)
        return dx, dw, db


class ReLU():

    @staticmethod
    def forward(x):
        """
        Input:
        - x: Input; a tensor of any shape
        Returns a tuple of:
        - out: Output, a tensor of the same shape as x
        - cache: x
        """
        x_relu = x.clone()
        x_relu[x <= 0] = 0
        out = x_relu
        cache = x
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Input:
        - dout: Upstream derivatives, of any shape
        - cache: Input x, of same shape as dout
        Returns:
        - dx: Gradient with respect to x
        """
        dx, x = None, cache

        dx = dout.clone()
        dx[x <= 0] = 0

        return dx

def softmax_loss(X, y):
    '''
    Computes the softmax loss (cross-entropy) and its gradient.
    Input:
        - X: Unnormalized scores/logits of shape (N, C), where N is the number of samples and C is the number of classes.
        - y: True class labels, a 1D tensor of length N.
    Return:
        - loss: Average cross-entropy loss.
        - dloss: Gradient of the loss with respect to input logits X.
    '''
    exp_scores = torch.exp(X - torch.max(X, dim=1, keepdim=True)[0])
    probs = exp_scores / torch.sum(exp_scores, dim=1, keepdim=True)

    N = X.shape[0]
    eps = 1e-15
    y_pred = torch.clamp(y, eps, 1.0)
    correct_class_probs = y_pred[y, range(N)]
    loss = -torch.sum(torch.log(correct_class_probs)) / N
    dloss = probs.clone()
    dloss[torch.arange(N), y] -= 1
    dloss /= N
    return loss, dloss


class Conv_ReLU():

    @staticmethod
    def forward(x, w, b, conv_param):
        z, conv_cache = Conv.forward(x=x, w=w, b=b, conv_param=conv_param)
        z_relu, relu_cache = ReLU.forward(x=z)
        caches = (conv_cache, relu_cache)
        return z_relu, caches

    @staticmethod
    def backward(dz_relu, caches):
        conv_cache, relu_cache = caches
        dz = ReLU.backward(dz_relu, relu_cache)
        dx, dw, db = Conv.backward(dz, conv_cache)
        return dx, dw, db

class Conv_ReLU_Pool():
    @staticmethod
    def forward(x, w, b, conv_param, pool_param):
        z, conv_cache = Conv.forward(x=x, w=w, b=b, conv_param=conv_param)
        z_relu, relu_cache = ReLU.forward(z)
        z_pool, pool_cache = MaxPool.forward(x=z_relu, pool_param=pool_param)
        caches = (conv_cache, relu_cache, pool_cache)
        return z_pool, caches

    @staticmethod
    def backward(dz_relu, caches):
        conv_cache, relu_cache, pool_cache = caches
        dz_pool = MaxPool.backward(dz_relu, pool_cache)
        dz = ReLU.backward(dz_pool, relu_cache)
        dx, dw, db = Conv.backward(dz, conv_cache)
        return dx, dw, db