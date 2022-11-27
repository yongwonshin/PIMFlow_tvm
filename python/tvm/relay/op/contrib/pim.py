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
from ...dataflow_pattern import wildcard, is_op, is_constant
from enum import Enum

class Act(Enum):
  NONE = 0
  RELU = 1
  SWISH = 2
  CLIP = 3
  SIGMOID = 4

def make_conv2d_pattern(with_bias=True, act=Act.NONE):
  data = wildcard()
  weight = wildcard()
  bias = wildcard()
  conv = is_op("nn.conv2d")(data, weight).has_attr({"pim": True})
  if with_bias:
    conv_out = is_op("add")(conv, bias)
  else:
    conv_out = conv

  if act == Act.NONE:
    act_out = conv_out
  if act == Act.RELU:
    act_out = is_op("nn.relu")(conv_out)
  elif act == Act.SWISH:
    sigmoid = is_op("sigmoid")(conv_out)
    act_out = is_op("multiply")(conv_out, sigmoid)
  elif act == Act.CLIP:
    act_out = is_op("clip")(conv_out)
  elif act == Act.SIGMOID:
    act_out = is_op("sigmoid")(conv_out)

  return act_out

def make_conv2d_fc_pattern(with_bias=True, act=Act.NONE):
  data = wildcard()
  weight = wildcard()
  bias = wildcard()

  fc = is_op("nn.conv2d")(data, weight).has_attr({"pim_fc": True})
  if with_bias:
    fc_out = is_op("add")(fc, bias)
  else:
    fc_out = fc

  if act == Act.NONE:
    act_out = fc_out
  if act == Act.RELU:
    act_out = is_op("nn.relu")(fc_out)
  elif act == Act.SWISH:
    sigmoid = is_op("sigmoid")(fc_out)
    act_out = is_op("multiply")(fc_out, sigmoid)
  elif act == Act.CLIP:
    act_out = is_op("clip")(fc_out)
  elif act == Act.SIGMOID:
    act_out = is_op("sigmoid")(fc_out)

  return act_out

def make_nn_dense_pattern(with_bias=False):
  data = wildcard()
  weight = is_constant()
  bias = is_constant()
  if with_bias:
    fc = is_op("nn.dense")(data, weight, bias).has_attr({"pim": True})
  else:
    fc = is_op("nn.dense")(data, weight).has_attr({"pim": True})

  return fc

def partition_for_pim(mod):
  """Partition the input module into PIM-supported subgraphs."""
  conv2d_pat = ("pim.conv2d", make_conv2d_pattern(with_bias=False))
  conv2d_bias_pat = ("pim.conv2d_bias", make_conv2d_pattern(with_bias=True))
  conv2d_relu_pat = ("pim.conv2d_relu", make_conv2d_pattern(with_bias=False, act=Act.RELU))
  conv2d_bias_relu_pat = ("pim.conv2d_bias_relu", make_conv2d_pattern(with_bias=True, act=Act.RELU))
  conv2d_clip_pat = ("pim.conv2d_clip", make_conv2d_pattern(with_bias=False, act=Act.CLIP))
  conv2d_bias_clip_pat = ("pim.conv2d_bias_clip", make_conv2d_pattern(with_bias=True, act=Act.CLIP))
  conv2d_swish_pat = ("pim.conv2d_swish", make_conv2d_pattern(with_bias=False, act=Act.SWISH))
  conv2d_bias_swish_pat = ("pim.conv2d_bias_swish", make_conv2d_pattern(with_bias=True, act=Act.SWISH))
  conv2d_sigmoid_pat = ("pim.conv2d_sigmoid", make_conv2d_pattern(with_bias=False, act=Act.SIGMOID))
  conv2d_bias_sigmoid_pat = ("pim.conv2d_bias_sigmoid", make_conv2d_pattern(with_bias=True, act=Act.SIGMOID))

  conv2d_fc_pat = ("pim.conv2d_fc", make_conv2d_fc_pattern(with_bias=False))
  conv2d_fc_bias_pat = ("pim.conv2d_fc_bias", make_conv2d_fc_pattern(with_bias=True))
  conv2d_fc_relu_pat = ("pim.conv2d_fc_relu", make_conv2d_fc_pattern(with_bias=False, act=Act.RELU))
  conv2d_fc_bias_relu_pat = ("pim.conv2d_fc_bias_relu", make_conv2d_fc_pattern(with_bias=True, act=Act.RELU))
  conv2d_fc_clip_pat = ("pim.conv2d_fc_clip", make_conv2d_fc_pattern(with_bias=False, act=Act.CLIP))
  conv2d_fc_bias_clip_pat = ("pim.conv2d_fc_bias_clip", make_conv2d_fc_pattern(with_bias=True, act=Act.CLIP))
  conv2d_fc_swish_pat = ("pim.conv2d_fc_swish", make_conv2d_fc_pattern(with_bias=False, act=Act.SWISH))
  conv2d_fc_bias_swish_pat = ("pim.conv2d_fc_bias_swish", make_conv2d_fc_pattern(with_bias=True, act=Act.SWISH))
  conv2d_fc_sigmoid_pat = ("pim.conv2d_fc_sigmoid", make_conv2d_fc_pattern(with_bias=False, act=Act.SIGMOID))
  conv2d_fc_bias_sigmoid_pat = ("pim.conv2d_fc_bias_sigmoid", make_conv2d_fc_pattern(with_bias=True, act=Act.SIGMOID))

  nn_dense_pat = ("pim.nn_dense", make_nn_dense_pattern(with_bias=False))
  nn_dense_bias_pat = ("pim.nn_dense_bias", make_nn_dense_pattern(with_bias=True))

  pim_patterns = [
    # conv patterns
    conv2d_bias_relu_pat,
    conv2d_bias_clip_pat,
    conv2d_bias_swish_pat,
    conv2d_bias_sigmoid_pat,
    conv2d_relu_pat,
    conv2d_clip_pat,
    conv2d_swish_pat,
    conv2d_sigmoid_pat,
    conv2d_bias_pat,
    conv2d_pat,
    # conv fc patterns
    conv2d_fc_bias_relu_pat,
    conv2d_fc_bias_clip_pat,
    conv2d_fc_bias_swish_pat,
    conv2d_fc_bias_sigmoid_pat,
    conv2d_fc_relu_pat,
    conv2d_fc_clip_pat,
    conv2d_fc_swish_pat,
    conv2d_fc_sigmoid_pat,
    conv2d_fc_bias_pat,
    conv2d_fc_pat,
    # fc patterns
    nn_dense_bias_pat,
    nn_dense_pat,
  ]
  mod = transform.MergeComposite(pim_patterns)(mod)
  print(mod)
  print("==========[MergeComposite Finished]==========")
  mod = transform.AnnotateTarget(["pim"], include_non_call_ops=False)(mod)
  print(mod)
  print("==========[AnnotateTarget Finished]==========")
  mod = transform.MergeCompilerRegions()(mod)
  mod = transform.PartitionGraph(bind_constants=False)(mod)
  print(mod)
  print("==========[PartitionGraph Finished]==========")
  return mod
