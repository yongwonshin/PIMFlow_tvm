# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Patterns supported PIM."""
import tvm
from tvm import relay
from tvm.relay import transform
from ...dataflow_pattern import is_tuple, wildcard, is_op, is_constant
from enum import Enum

class Act(Enum):
  RELU = 0
  SWISH = 1
  CLIP = 2

def make_conv2d_pattern(with_bias=True):
  data = wildcard()
  weight = wildcard()
  bias = wildcard()
  pim_start = is_op("sin")(data)
  conv = is_op("nn.conv2d")(pim_start, weight)
  if with_bias:
    conv_out = is_op("add")(conv, bias)
  else:
    conv_out = conv

  return is_op("cos")(conv_out)

def make_conv2d_bias_act_pattern(with_bias=True, act=Act.RELU, p=6):
  data = wildcard()
  weight = wildcard()
  bias = wildcard()
  pim_start = is_op("sin")(data)
  conv = is_op("nn.conv2d")(pim_start, weight)
  if with_bias:
    conv_out = is_op("add")(conv, bias)
  else:
    conv_out = conv

  if act == Act.RELU:
    act_out = is_op("nn.relu")(conv_out)
  elif act == Act.SWISH:
    sigmoid = is_op("sigmoid")(conv_out)
    act_out = is_op("multiply")(conv_out, sigmoid)
  elif act == Act.CLIP:
    raise Exception("Not implemented")

  return is_op("cos")(act_out)

def make_memory_optimized_node(t):
  data = wildcard()
  sin_out = is_op("sin")(data)
  opt_out = is_op(t)(sin_out)
  return is_op("cos")(opt_out)

def make_memory_optimized_node_const(t):
  data = wildcard()
  const = is_constant()
  sin_out = is_op("sin")(data)
  opt_out = is_op(t)(sin_out, const)
  return is_op("cos")(opt_out)

def make_memory_optimized_node_concat():
  input1 = wildcard()
  input2 = wildcard()
  data = is_tuple([input1, input2])
  concat_out = is_op("concatenate")(data)
  return is_op("cos")(concat_out)

def make_layout_transform():
  data = wildcard()
  return is_op("layout_transform")(data)

def partition_for_pim(mod):
  """Partition the input module into PIM-supported subgraphs."""
  conv2d_pat = ("pim.conv2d", make_conv2d_pattern(with_bias=False))
  conv2d_pat_bias = ("pim.conv2d_bias", make_conv2d_pattern(with_bias=True))
  conv2d_relu_pat = ("pim.conv2d_relu", make_conv2d_bias_act_pattern(with_bias=False, act=Act.RELU))
  conv2d_bias_relu_pat = ("pim.conv2d_bias_relu", make_conv2d_bias_act_pattern(with_bias=True, act=Act.RELU))
  conv2d_swish_pat = ("pim.conv2d_swish", make_conv2d_bias_act_pattern(with_bias=False, act=Act.SWISH))
  conv2d_bias_swish_pat = ("pim.conv2d_bias_swish", make_conv2d_bias_act_pattern(with_bias=True, act=Act.SWISH))
  memory_optimized_slice_pat = ("pim.memory_optimized_slice", make_memory_optimized_node("strided_slice"))
  memory_optimized_pad_pat = ("pim.memory_optimized_pad", make_memory_optimized_node_const("nn.pad"))
  memory_optimized_concat_pat = ("pim.memory_optimized_concat", make_memory_optimized_node_concat())
  memory_layout_transform_pat = ("pim.layout_transform", make_layout_transform())
  pim_patterns = [
    conv2d_pat,
    conv2d_pat_bias,
    conv2d_relu_pat,
    conv2d_bias_relu_pat,
    conv2d_swish_pat,
    conv2d_bias_swish_pat,
    memory_optimized_slice_pat,
    memory_optimized_pad_pat,
    memory_optimized_concat_pat,
    memory_layout_transform_pat,
  ]
  mod = transform.MergeComposite(pim_patterns)(mod)
  mod = transform.AnnotateTarget(["pim"], include_non_call_ops=False)(mod)
  # mod = transform.MergeCompilerRegions()(mod)
  mod = transform.PartitionGraph(bind_constants=False)(mod)
  return mod
