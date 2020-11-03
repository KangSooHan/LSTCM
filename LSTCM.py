# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Module for constructing RNN Cells."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from tensorflow.contrib.compiler import jit
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras import layers as keras_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl  # pylint: disable=unused-import
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables  # pylint: disable=unused-import
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest



_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h"))
class LSTMStateTuple(_LSTMStateTuple):
  """Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.

  Stores two elements: `(c, h)`, in that order. Where `c` is the hidden state
  and `h` is the output.

  Only used when `state_is_tuple=True`.
  """
  __slots__ = ()

  @property
  def dtype(self):
    (c, h) = self
    if c.dtype != h.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s" %
                      (str(c.dtype), str(h.dtype)))
    return c.dtype

class LSTCMCell(rnn_cell_impl.LayerRNNCell):
  def __init__(self,
               num_units,
               forget_bias=1.0,
               input_size=None,
               activation=math_ops.tanh,
               kernel_initializer=None,
               bias_initializer=None,
               name=None,
               dtype=None,
               reuse=None):



    super(LSTCMCell, self).__init__(_reuse=reuse, name=name, dtype=dtype)

    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)

    self._num_units = num_units
    self._activation = activation
    self._forget_bias = forget_bias
    self._reuse = reuse
    self.input_spec = input_spec.InputSpec(ndim=2)
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer


  @property
  def state_size(self):
    return LSTMStateTuple(self._num_units, self._num_units)


  @property
  def output_size(self):
    return self._num_units


  def build(self, inputs_shape):
    if tensor_shape.dimension_value(inputs_shape[1]) is None:
      raise ValueError(
          "Expected inputs.shape[-1] to be known, saw shape: %s" % inputs_shape)

    input_depth = tensor_shape.dimension_value(inputs_shape[1])
    # pylint: disable=protected-access

    self._kernel_w = self.add_variable(
        "%s_w" % rnn_cell_impl._WEIGHTS_VARIABLE_NAME,
        shape=[input_depth, self._num_units*4],
        initializer=self._kernel_initializer)

    self._kernel_u = self.add_variable(
        "%s_u" % rnn_cell_impl._WEIGHTS_VARIABLE_NAME,
        shape=[1,self._num_units*4],
        initializer=init_ops.random_uniform_initializer(
            minval=-1.0, maxval=1.0, dtype=self.dtype))

    self._bias = self.add_variable(
        rnn_cell_impl._BIAS_VARIABLE_NAME,
        shape=[self._num_units*4],
        initializer=(self._bias_initializer
                     if self._bias_initializer is not None else
                     init_ops.zeros_initializer(dtype=self.dtype)))

    self.built = True

  def call(self, inputs, state):
    sigmoid = math_ops.sigmoid
    one = constant_op.constant(1, dtype=dtypes.int32)
    add = math_ops.add
    multiply = math_ops.multiply
    c, h = state

    #batch_size = tensor_shape.dimension_value(
    #    inputs.shape[0]) or array_ops.shape(inputs)[0]

    #ss = array_ops.tile(self.ss, [batch_size, 1])

    gate_inputs = math_ops.matmul(inputs, self._kernel_w)
    gate_inputs += multiply(gen_array_ops.tile(h, [1, 4]),self._kernel_u)
    gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    f, j,i,o = array_ops.split(
        value=gate_inputs, num_or_size_splits=4, axis=one)

    
    forget_bias_tensor = constant_op.constant(1, dtype=f.dtype)
    # Note that using `add` and `multiply` instead of `+` and `*` gives a
    # performance improvement. So using those at the cost of readability.

    new_c = add(
            multiply(math_ops.tanh(c), sigmoid(f+1)),
            multiply(sigmoid(i), math_ops.tanh(j)))

    new_h = multiply(new_c , sigmoid(o)) 

    new_state = LSTMStateTuple(new_c, new_h)

    return new_h, new_state


