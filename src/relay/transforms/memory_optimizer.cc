/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/relay/transforms/memory_optimizer.cc
 * \brief A pass for memory optimization.
 */

#include "memory_optimizer.h"

#include <tvm/relay/dataflow_matcher.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/logging.h>

#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <iostream>

#include "../op/tensor/transform.h"
#include "pattern_utils.h"

namespace tvm {
namespace relay {

ObjectPtr<Conv2DAttrs> CopyFromConv2dAttr(const Conv2DAttrs *attrs) {
  ObjectPtr<Conv2DAttrs> new_attr = make_object<Conv2DAttrs>();
  new_attr->strides = attrs->strides;
  new_attr->padding = attrs->padding;
  new_attr->dilation = attrs->dilation;
  new_attr->groups = attrs->groups;
  new_attr->channels = attrs->channels;
  new_attr->kernel_size = attrs->kernel_size;
  new_attr->data_layout = attrs->data_layout;
  new_attr->kernel_layout = attrs->kernel_layout;
  new_attr->out_layout = attrs->out_layout;
  new_attr->out_dtype = attrs->out_dtype;
  new_attr->folded_slice = attrs->folded_slice;
  new_attr->conv_id = attrs->conv_id;
  new_attr->h_dim_concat = attrs->h_dim_concat;
  new_attr->pim = attrs->pim;
  new_attr->pim_fc = attrs->pim_fc;
  new_attr->gpu = attrs->gpu;
  new_attr->onnx_node_name = attrs->onnx_node_name;
  return new_attr;
}

class RemovePad : public DFPatternRewrite {
 public:
  RemovePad() {
    x_ = IsWildcard();
    pattern_ = IsOp("nn.pad")({x_, IsWildcard()});
  }

  Expr Callback(const Expr& pre, const Expr& post,
                const Map<DFPattern, Array<Expr>>& node_map) const override {
    const CallNode* call = pre.as<CallNode>();
    ICHECK(call);
    const auto* pad_attr = call->attrs.as<PadAttrs>();
    ICHECK(pad_attr);
    auto p = pad_attr->pad_width;
    auto x = node_map[x_][0];

    for (auto& arr : p) {
      for (auto& e : arr) {
        if (e.as<IntImmNode>()->value != 0) {
          return post;
        }
      }
    }

    return x;
  }

 private:
  DFPattern x_;
};

class PermutePad : public DFPatternRewrite {
 public:
  PermutePad() {
    x_ = IsWildcard();
    pad_width_ = IsWildcard();
    pad_ = IsOp("nn.pad")({x_, pad_width_});
    layout_ = IsOp("layout_transform")({pad_});
    pattern_ = layout_;
  }

  Expr Callback(const Expr& pre, const Expr& post,
                const Map<DFPattern, Array<Expr>>& node_map) const override {
    const CallNode* call = pre.as<CallNode>();
    const auto* layout_attr = call->attrs.as<LayoutTransformAttrs>();
    const CallNode* pre_call = call->args[0].as<CallNode>();
    const auto* pad_attr = pre_call->attrs.as<PadAttrs>();

    if (layout_attr->src_layout == "NCHW" && layout_attr->dst_layout == "NHWC") {
      auto x = node_map[x_][0];
      auto pad_width = node_map[pad_width_][0];
      auto pad = node_map[pad_][0];
      auto layout = node_map[layout_][0];

      auto layout_res = Call(call->op, {x}, call->attrs);
      ObjectPtr<PadAttrs> new_pad_attr = make_object<PadAttrs>();
      new_pad_attr->pad_width = Array<Array<Integer>>{pad_attr->pad_width[0], pad_attr->pad_width[2], pad_attr->pad_width[3], pad_attr->pad_width[1]};
      new_pad_attr->pad_mode = pad_attr->pad_mode;
      return Call(pre_call->op, {layout_res, pad_width}, Attrs(new_pad_attr));
    }
    return post;
  }

 private:
  DFPattern x_;
  DFPattern pad_;
  DFPattern pad_width_;
  DFPattern layout_;
};

class PermuteSlice : public DFPatternRewrite {
 public:
  PermuteSlice() {
    x_ = IsWildcard();
    slice_ = IsOp("strided_slice")({x_});
    layout_ = IsOp("layout_transform")({slice_});
    pattern_ = layout_;
  }

  Expr Callback(const Expr& pre, const Expr& post,
                const Map<DFPattern, Array<Expr>>& node_map) const override {
    const CallNode* call = pre.as<CallNode>();
    const auto* layout_attr = call->attrs.as<LayoutTransformAttrs>();
    const CallNode* pre_call = call->args[0].as<CallNode>();
    const auto* slice_attr = pre_call->attrs.as<StridedSliceAttrs>();

    if (layout_attr->src_layout == "NCHW" && layout_attr->dst_layout == "NHWC") {
      auto x = node_map[x_][0];
      ObjectPtr<StridedSliceAttrs> new_slice_attr = make_object<StridedSliceAttrs>();
      new_slice_attr->begin = slice_attr->begin;
      new_slice_attr->end = slice_attr->end;
      new_slice_attr->strides = slice_attr->strides;
      new_slice_attr->axes = Array<Integer>{IntImm(DataType::Int(64), 1)};
      new_slice_attr->slice_mode = slice_attr->slice_mode;
      auto layout_res = Call(call->op, {x}, call->attrs);
      return Call(pre_call->op, {layout_res}, Attrs(new_slice_attr));
    }
    return post;
  }

 private:
  DFPattern x_;
  DFPattern slice_;
  DFPattern layout_;
};

class FoldSlice : public DFPatternRewrite {
 public:
  FoldSlice() {
    x_ = IsWildcard();
    slice_ = IsOp("strided_slice")({x_});
    weight_ = IsWildcard();
    conv2d_ = IsOp("nn.conv2d")({slice_, weight_});
    pattern_ = conv2d_;
  }

  Expr Callback(const Expr& pre, const Expr& post,
                const Map<DFPattern, Array<Expr>>& node_map) const override {
    const CallNode* call = pre.as<CallNode>();
    const auto* conv2d_attr = call->attrs.as<Conv2DAttrs>();
    const CallNode* pre_call = call->args[0].as<CallNode>();
    const auto* slice_attr = pre_call->attrs.as<StridedSliceAttrs>();

    // we only support NHWC (data_layout) and OHWI (kernel_layout)
    if (conv2d_attr->data_layout != "NHWC" || conv2d_attr->kernel_layout != "OHWI") {
      return post;
    }

    if (slice_attr->begin.value().size() != 1 || slice_attr->end.value().size() != 1 ||
        slice_attr->strides.value()[0] != 1 || slice_attr->axes.value()[0] != 1) {
      return post;
    }

    ObjectPtr<Conv2DAttrs> new_conv2d_attr = CopyFromConv2dAttr(conv2d_attr);
    new_conv2d_attr->folded_slice = Array<Integer>{slice_attr->begin.value()[0], slice_attr->end.value()[0]};

    auto x = node_map[x_][0];
    auto weight = node_map[weight_][0];

    return Call(call->op, {x, weight}, Attrs(new_conv2d_attr), Array<Type>{pre_call->type_args[0], call->type_args[1]});
  }

 private:
  DFPattern x_;
  DFPattern slice_;
  DFPattern weight_;
  DFPattern conv2d_;
};

class FoldPad : public DFPatternRewrite {
 public:
  FoldPad() {
    x_ = IsWildcard();
    pad_width_ = IsWildcard();
    pad_ = IsOp("nn.pad")({x_, pad_width_});
    weight_ = IsWildcard();
    conv2d_ = IsOp("nn.conv2d")({pad_, weight_});
    pattern_ = conv2d_;
  }

  Expr Callback(const Expr& pre, const Expr& post,
                const Map<DFPattern, Array<Expr>>& node_map) const override {
    const CallNode* call = pre.as<CallNode>();
    const auto* conv2d_attr = call->attrs.as<Conv2DAttrs>();
    const CallNode* pre_call = call->args[0].as<CallNode>();
    const auto* pad_attr = pre_call->attrs.as<PadAttrs>();

    // we only support NHWC (data_layout) and OHWI (kernel_layout)
    if (conv2d_attr->data_layout != "NHWC" || conv2d_attr->kernel_layout != "OHWI") {
      return post;
    }

    auto x = node_map[x_][0];
    auto pad_width = node_map[pad_width_][0];
    auto weight = node_map[weight_][0];

    ObjectPtr<Conv2DAttrs> new_conv2d_attr = CopyFromConv2dAttr(conv2d_attr);

    auto pad_res = Call(pre_call->op, {x, pad_width}, pre_call->attrs);
    return Call(call->op, {x, weight}, Attrs(new_conv2d_attr));
  }

 private:
  DFPattern x_;
  DFPattern pad_width_;
  DFPattern pad_;
  DFPattern weight_;
  DFPattern conv2d_;
};

// TODO[ywshin]: runtime implementation is incomplete
class FoldConcat : public DFPatternRewrite {
 public:
  static int conv_id;
  FoldConcat() {
    x1_ = IsWildcard();
    x2_ = IsWildcard();
    weight1_ = IsWildcard();
    weight2_ = IsWildcard();
    conv2d_1_ = IsOp("nn.conv2d")({x1_, weight1_});
    conv2d_2_ = IsOp("nn.conv2d")({x2_, weight2_});
    tuple_ = IsTuple({conv2d_1_, conv2d_2_});
    concat_ = IsOp("concatenate")({tuple_});
    pattern_ = concat_;
  }

  Expr Callback(const Expr& pre, const Expr& post,
                const Map<DFPattern, Array<Expr>>& node_map) const override {
    const CallNode* call = pre.as<CallNode>();
    const TupleNode* tuple = call->args[0].as<TupleNode>();
    const CallNode* conv1_call = tuple->fields[0].as<CallNode>();
    const CallNode* conv2_call = tuple->fields[1].as<CallNode>();
    const auto* concat_attr = call->attrs.as<ConcatenateAttrs>();
    const auto* conv2d_1_attr = conv1_call->attrs.as<Conv2DAttrs>();
    const auto* conv2d_2_attr = conv2_call->attrs.as<Conv2DAttrs>();

    std::cerr << "[YWSHIN] " << call->checked_type() << std::endl;

    // we only support NHWC (data_layout) and OHWI (kernel_layout)
    if (conv2d_1_attr->data_layout != "NHWC" || conv2d_1_attr->kernel_layout != "OHWI" ||
        conv2d_2_attr->data_layout != "NHWC" || conv2d_2_attr->kernel_layout != "OHWI") {
      return post;
    }

    // pass if no folded_slice exists
    if (conv2d_1_attr->folded_slice.size() == 0 || conv2d_2_attr->folded_slice.size() == 0) {
      return post;
    }

    // already processed
    if (concat_attr->conv_id > 0) {
      return post;
    }

    auto x1 = node_map[x1_][0];
    auto x2 = node_map[x2_][0];
    auto weight1 = node_map[weight1_][0];
    auto weight2 = node_map[weight2_][0];

    Type call_type = call->checked_type();
    const auto* imm = call_type.as<TensorTypeNode>()->shape[1].as<IntImmNode>();
    int h_dim = Integer(GetRef<IntImm>(imm));

    ObjectPtr<Conv2DAttrs> new_conv2d_1_attr = CopyFromConv2dAttr(conv2d_1_attr);
    new_conv2d_1_attr->conv_id = conv_id;
    new_conv2d_1_attr->h_dim_concat = h_dim;

    ObjectPtr<Conv2DAttrs> new_conv2d_2_attr = CopyFromConv2dAttr(conv2d_2_attr);
    new_conv2d_2_attr->conv_id = conv_id;
    new_conv2d_2_attr->h_dim_concat = h_dim;

    ObjectPtr<ConcatenateAttrs> new_concat_attr = make_object<ConcatenateAttrs>();
    new_concat_attr->axis = concat_attr->axis;
    new_concat_attr->conv_id = conv_id;

    conv_id++;

    auto conv1_res = Call(conv1_call->op, {x1, weight1}, Attrs(new_conv2d_1_attr));
    auto conv2_res = Call(conv2_call->op, {x2, weight2}, Attrs(new_conv2d_2_attr));
    return Call(call->op, {Tuple({conv1_res, conv2_res})}, Attrs{new_concat_attr});
  }

 private:
  DFPattern x1_;
  DFPattern x2_;
  DFPattern weight1_;
  DFPattern weight2_;
  DFPattern conv2d_1_;
  DFPattern conv2d_2_;
  DFPattern tuple_;
  DFPattern concat_;
};

int FoldConcat::conv_id = 1;

Expr OptimizeMemory(const Expr& expr, const IRModule& mod) {
  // the rewrites will be applied in the given order, and repeated until fixed point
  DFPatternRewriteComposer composer;
  composer.AddRewrite<RemovePad>();
  composer.AddRewrite<PermutePad>();
  composer.AddRewrite<PermuteSlice>();
  composer.AddRewrite<FoldSlice>();
  composer.AddRewrite<FoldConcat>();
  // composer.AddRewrite<FoldPad>(); // TODO[ywshin]
  return RewritePatterns(composer.MakeCallbacks(), expr, mod);
}

namespace transform {

Pass OptimizeMemory() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(OptimizeMemory(f, m));
      };
  return CreateFunctionPass(pass_func, 3, "OptimizeMemory", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.OptimizeMemory").set_body_typed(OptimizeMemory);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
