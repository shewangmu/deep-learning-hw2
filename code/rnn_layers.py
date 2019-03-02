"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""
import numpy as np


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.
    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.
    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)
    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """

    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    y = x@Wx + prev_h@Wh + b
    next_h = np.tanh(y)
    cache = (x, prev_h, Wx, Wh, b, next_h)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.
    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state
    - cache: Cache object from the forward pass
    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """

    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    x, prev_h, Wx, Wh, b, next_h = cache
    dy = (1 - next_h**2)*dnext_h
    dx = dy@Wx.T
    dprev_h = dy@Wh.T
    dWx = x.T@dy
    dWh = prev_h.T@dy
    db = np.sum(dy, axis=0)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.
    Inputs:
    - x: Input data for the entire timeseries, of shape (T, N, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)
    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (T, N, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    T, N, D = x.shape
    _, H = h0.shape
    h = np.zeros((T, N, H))
    
    ht = h0
    for i in range(T):
        ht, cache_temp = rnn_step_forward(x[i], ht, Wx, Wh, b)
        h[i] = ht
    cache = (x, h0, Wx, Wh, b, h)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.
    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (T, N, H)
    Returns a tuple of:
    - dx: Gradient of inputs, of shape (T, N, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    x, h0, Wx, Wh, b, h = cache
    T, N, D = x.shape
    T, N, H = dh.shape
    dx = np.zeros((T, N, D))
    dh0 = np.zeros((N, H))
    dWh = np.zeros((H, H))
    dWx = np.zeros((D, H))
    db = np.zeros((H, ))
    for i in range(T):
        if i == T-1:
            cache_temp = (x[0], h0, Wx, Wh, b, h[0])
            dx[0], dh0, dWx_temp, dWh_temp, db_temp = rnn_step_backward(dh[0], cache_temp)
        else:
            cache_temp = (x[-i-1], h[-i-2], Wx, Wh, b, h[-i-1])
            dx[-i-1], dh_temp, dWx_temp, dWh_temp, db_temp = rnn_step_backward(dh[-i-1], cache_temp)
            dh[-i-2] += dh_temp
        dWh += dWh_temp
        dWx += dWx_temp
        db += db_temp
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db

def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.
    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.
    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)
    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    cache = None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    # For Wx of shape D x 4H, you may assume they are the sequence of parameters#
    # for forget gate, input gate, concurrent input, output gate. Wh and b also #
    # follow the same order.                                                    #
    #############################################################################
    N, H = prev_h.shape
    ft = sigmoid(x@Wx[:,:H] + prev_h@Wh[:,:H] + b[:H])
    it = sigmoid(x@Wx[:,H:2*H] + prev_h@Wh[:,H:2*H] + b[H:2*H])
    Ct_tilt = np.tanh(x@Wx[:,2*H:3*H] + prev_h@Wh[:,2*H:3*H] + b[2*H:3*H])
    next_c = ft*prev_c + it*Ct_tilt
    ot = sigmoid(x@Wx[:,3*H:4*H] + prev_h@Wh[:,3*H:4*H] + b[3*H:4*H])
    next_h = ot*np.tanh(next_c)    
    
    cache = (x, ft, it, Ct_tilt, next_c, ot, next_h, Wx, Wh, prev_h, prev_c)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.
    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass
    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh, dc, dWx, dWh, db, dprev_h, dprev_c = None, None, None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    _, H = dnext_h.shape
    x, ft, it, Ct_tilt, next_c, ot, next_h, Wx, Wh, prev_h, prev_c = cache
    dnext_c += dnext_h * ot * (1 - np.tanh(next_c)**2)
    
    N, D = x.shape
    wfh, wih, wch, woh = Wh[:,:H], Wh[:, H:2*H], Wh[:,2*H:3*H], Wh[:,3*H:]
    wfx, wix, wcx, wox = Wx[:,:H], Wx[:, H:2*H], Wx[:,2*H:3*H], Wx[:,3*H:]
    
    dot_fc = dnext_h * np.tanh(next_c) * ot * (1-ot)
    dctilt_fc = dnext_c * it * (1 - Ct_tilt**2)
    dft_fc = dnext_c * prev_c * ft * (1-ft)
    di_fc = dnext_c * Ct_tilt * it * (1-it)
    
    dprev_h = dot_fc@woh.T + dctilt_fc@wch.T + dft_fc@wfh.T + di_fc@wih.T
    dx = dot_fc@wox.T + dctilt_fc@wcx.T + dft_fc@wfx.T + di_fc@wix.T
    
    dprev_c = dnext_c * ft
    
    dWx = np.zeros((D, 4*H))
    dWh = np.zeros((H, 4*H))
    dWx[:, :H] = x.T @ dft_fc
    dWx[:, H:2*H] = x.T @ di_fc
    dWx[:, 2*H:3*H] = x.T @ dctilt_fc
    dWx[:, 3*H:] = x.T @ dot_fc
    
    dWh[:, :H] = prev_h.T @ dft_fc
    dWh[:, H:2*H] = prev_h.T @ di_fc
    dWh[:, 2*H:3*H] = prev_h.T @ dctilt_fc
    dWh[:, 3*H:] = prev_h.T @ dot_fc
    
    db = np.zeros(4*H,)
    db[:H] = np.sum(dft_fc, axis=0)
    db[H:2*H] = np.sum(di_fc, axis=0)
    db[2*H:3*H] = np.sum(dctilt_fc, axis=0)
    db[3*H:] = np.sum(dot_fc, axis=0)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.
    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.
    Inputs:
    - x: Input data of shape (T, N, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)
    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (T, N, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    T, N, D = x.shape
    _, H = h0.shape
    prev_h = h0
    prev_c = np.zeros_like(h0)
    
    h = np.zeros((T, N, H))
    c = np.zeros_like(h)
    ft = np.zeros_like(h)
    Ct_tilt = np.zeros_like(h)
    ot = np.zeros_like(h)
    it = np.zeros_like(h)
    
    
    for i in range(T):
        next_h, next_c, cache_temp = lstm_step_forward(x[i], prev_h, prev_c, Wx, Wh, b)
        h[i] = next_h
        c[i] = next_c
        ft[i] = cache_temp[1]
        Ct_tilt[i] = cache_temp[3]
        ot[i] = cache_temp[5]
        it[i] = cache_temp[2]
        
        prev_h = h[i]
        prev_c = c[i]
    
    cache = (x, ft, it, Ct_tilt, c, ot, h, Wx, Wh, h0)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]
    Inputs:
    - dh: Upstream gradients of hidden states, of shape (T, N, H)
    - cache: Values from the forward pass
    Returns a tuple of:
    - dx: Gradient of input data of shape (T, N, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    x, ft, it, Ct_tilt, c, ot, h, Wx, Wh, h0 = cache
    c0 = np.zeros_like(h0)
    
    T, N, H = dh.shape
    T, N, D = x.shape
    dx = np.zeros_like(x)
    dh0 = np.zeros_like(h0)
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    db = np.zeros((4*H, ))
    dprev_c = np.zeros_like(dh[0])
    
    
    for t in range(T):
        if t == T-1:
            dnext_c = dprev_c
            cache_temp = (x[0], ft[0], it[0], Ct_tilt[0], c[0], ot[0], h[0], Wx, Wh, h0, c0)
            dx[0], dh0, dc0, dWx_temp, dWh_temp, db_temp = lstm_step_backward(dh[0], dnext_c, cache_temp)
        else:            
            dnext_c = dprev_c
            cache_temp = (x[-t-1], ft[-t-1], it[-t-1], Ct_tilt[-t-1], c[-t-1], ot[-t-1], h[-t-1], Wx, Wh, h[-t-2], c[-t-2])
            dx[-t-1], dprev_h, dprev_c, dWx_temp, dWh_temp, db_temp = lstm_step_backward(dh[-t-1], dnext_c, cache_temp)
            dh[-t-2] += dprev_h
        dWx += dWx_temp
        dWh += dWh_temp
        db += db_temp
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    to a vector of dimension D.
    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x must be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.
    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    out = W[x, :]
    cache = x, W
    return out, cache

def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.
    HINT: Look up the function np.add.at
    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass
    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    x, W = cache
    dW = np.zeros_like(W)
    np.add.at(dW, x, dout)
    return dW



def temporal_fc_forward(x, w, b):
    """
    Forward pass for a temporal fully-connected layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.
    Inputs:
    - x: Input data of shape (T, N, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)
    Returns a tuple of:
    - out: Output data of shape (T, N, M)
    - cache: Values needed for the backward pass
    """
    out = x@w + b
    cache = (x, w)
    
    return out, cache

def temporal_fc_backward(dout, cache):
    """
    Backward pass for temporal fully-connected layer.
    Input:
    - dout: Upstream gradients of shape (T, N, M)
    - cache: Values from forward pass
    Returns a tuple of:
    - dx: Gradient of input, of shape (T, N, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w = cache
    T,N,M = dout.shape
    dx = dout@w.T
    dw = np.zeros_like(w)
    db = np.zeros((M, ))
    for t in range(T):
        dw += x[t].T@dout[t]
        db += np.sum(dout[t], axis=0)
    return dx, dw, db

def temporal_softmax_loss(x, y, mask):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.
    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.
    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.
    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """
    N, T, V = x.shape
    loss = 0
    exp = np.exp(x)
    dx = np.zeros_like(x)
    for n in range(N):
        for t in range(T):
            if mask[n,t]:
                l = y[n][t]
                loss += np.log(exp[n][t][l]/np.sum(exp[n][t]))
                for v in range(V):
                    dx[n][t][v] = -exp[n][t][v]/np.sum(exp[n][t])
                dx[n][t][l] = 1 - exp[n][t][l]/np.sum(exp[n][t])
    loss = -loss/N    
    dx = -dx/N
    return loss, dx


