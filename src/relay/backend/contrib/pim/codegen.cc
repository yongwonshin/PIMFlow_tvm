#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <numeric>
#include <sstream>
#include <vector>
#include <algorithm>
#include <stdexcept>

#include "../../utils.h"
#include "../codegen_c/codegen_c.h"
#include "pim_trace.h"

namespace tvm {
namespace relay {
namespace contrib {

using namespace backend;
using Str2StrMap = std::unordered_map<std::string, std::string>;

static Str2StrMap dtype_map = {{"float16", "half"}, {"float32", "float"}};

enum Act {
  ACT_NONE,
  ACT_RELU,
  ACT_SWISH,
  ACT_CLIP,
  ACT_SIGMOID,
};

Str2StrMap ConvArgs(const CallNode* call, Act act) {
  Str2StrMap args;
  const auto* conv2d_attr = call->attrs.as<Conv2DAttrs>();
  ICHECK(conv2d_attr);

  auto ishape = GetShape(call->args[0]->checked_type());
  auto wshape = GetShape(call->args[1]->checked_type());

  args["N"] = std::to_string(ishape[0]);
  args["H"] = std::to_string(ishape[1]);
  args["W"] = std::to_string(ishape[2]);
  args["C"] = std::to_string(ishape[3]);

  args["K"] = std::to_string(wshape[0]);
  args["R"] = std::to_string(wshape[1]);
  args["S"] = std::to_string(wshape[2]);
  args["C_"] = std::to_string(wshape[3]);

  args["G"] = std::to_string(conv2d_attr->groups);
  args["Ph"] = std::to_string(conv2d_attr->padding[0].as<IntImmNode>()->value);
  args["Pw"] = std::to_string(conv2d_attr->padding[1].as<IntImmNode>()->value);
  // args["Ph1"] = std::to_string(conv2d_attr->padding[2].as<IntImmNode>()->value);
  // args["Pw1"] = std::to_string(conv2d_attr->padding[3].as<IntImmNode>()->value);
  args["Sh"] = std::to_string(conv2d_attr->strides[0].as<IntImmNode>()->value);
  args["Sw"] = std::to_string(conv2d_attr->strides[1].as<IntImmNode>()->value);
  args["Dh"] = std::to_string(conv2d_attr->dilation[0].as<IntImmNode>()->value);
  args["Dw"] = std::to_string(conv2d_attr->dilation[1].as<IntImmNode>()->value);
  args["Act"] = std::to_string((int)act);

  args["ONNX_NODE_NAME"] = conv2d_attr->onnx_node_name.c_str();
  args["N_CHANNEL"] = std::to_string(16);
  args["GW"] = std::to_string(4);

  return args;
}

Str2StrMap FCArgs(const CallNode* call, Act act) {
  Str2StrMap args;
  const auto* dense_attr = call->attrs.as<DenseAttrs>();
  ICHECK(dense_attr);

  auto ishape = GetShape(call->args[0]->checked_type());
  auto wshape = GetShape(call->args[1]->checked_type());

  ICHECK_EQ(ishape[0], 1);
  ICHECK_EQ(ishape[1], wshape[1]);

  args["ROW"] = std::to_string(wshape[0]);
  args["COL"] = std::to_string(wshape[1]);
  args["Act"] = std::to_string((int)act);

  args["ONNX_NODE_NAME"] = dense_attr->onnx_node_name.c_str();
  args["N_CHANNEL"] = std::to_string(16);
  args["GW"] = std::to_string(4);

  return args;
}

Str2StrMap ConvFCArgs(const CallNode* call, Act act) {
  Str2StrMap args;
  const auto* conv2d_attr = call->attrs.as<Conv2DAttrs>();
  ICHECK(conv2d_attr);

  auto ishape = GetShape(call->args[0]->checked_type());
  auto wshape = GetShape(call->args[1]->checked_type());

  ICHECK_EQ(ishape[0], 1);
  ICHECK_EQ(ishape[1], 1);
  ICHECK_EQ(ishape[2], 1);
  ICHECK_EQ(ishape[3], wshape[3]);

  args["ROW"] = std::to_string(wshape[0]);
  args["COL"] = std::to_string(wshape[1]);
  args["Act"] = std::to_string((int)act);

  args["ONNX_NODE_NAME"] = conv2d_attr->onnx_node_name.c_str();
  args["N_CHANNEL"] = std::to_string(16);
  args["GW"] = std::to_string(4);

  return args;
}

inline void CuDNNPrint(std::ostringstream& os, const std::string& stmt, int indent = 2) {
  for (int i = 0; i < indent; ++i) {
    os << " ";
  }
  os << stmt;
}

// TODO[ywshin]: validation by value
inline std::string CuDNNCodeGen(std::string id, const Str2StrMap& attrs,
                    const std::vector<std::string>& func_args) {
  return "";
}

std::vector<std::string> PimCodeGen(std::string id, const Str2StrMap& attrs,
                const std::vector<std::string>& func_args) {
  int n_channel = std::stoi(attrs.at("N_CHANNEL"));
  std::vector<std::string> code;
  std::ostringstream OS;

  int h = std::stoi(attrs.at("H"));
  int w = std::stoi(attrs.at("W"));
  int kh = std::stoi(attrs.at("R"));
  int kw = std::stoi(attrs.at("S"));
  int ph = std::stoi(attrs.at("Ph"));
  int pw = std::stoi(attrs.at("Pw"));
  int stride_ = std::stoi(attrs.at("Sh"));

  int H = (h - kh + 2 * ph) / stride_ + 1;
  int W = (w - kw + 2 * pw) / stride_ + 1;
  int C_o = std::stoi(attrs.at("K"));
  int C_i = std::stoi(attrs.at("C"));

  auto i_c = (((C_i * kh * kw) + 15) / 16) * 16;
  auto o_c = C_o;
  auto row = ((o_c + 15) / 16) * 16;
  int fl, stride, n, col;

  // NOTE: Boundary condition is ignored (e.g., 3x3 kernel with (1, 1) padding)
  if (i_c <= 512) {
    fl = 512 / i_c;
    col = i_c * fl;
    stride = i_c;
    n = ((H * W) + fl - 1) / fl;
    for (int i = 0; i < n; i++) {
      pim::StrideInfo sinfo;
      if (kh > 1 || kw > 1) {
        sinfo.use_stride = true;
        sinfo.num_first_elem = C_i * kh;
        sinfo.stride = C_i * (h - kh);
        sinfo.num_after_elem = C_i * kh;
        sinfo.num_gwrite = 512 / (C_i * kh * kw);
      }
      pim::OutputNewtonTraceV2(OS, id, row, col, stride, sinfo);
    }
    code.push_back(OS.str());
  } else {
    int full = i_c / 512;
    col = 512;
    stride = 512;
    n = (H * W);
    for (int j = 0; j < full; j++) {
      for (int i = 0; i < n; i++) {
        pim::OutputNewtonTraceV2(OS, id, row, col, stride);
      }
    }
    code.push_back(OS.str());
    OS.str("");
    OS.clear();
    int i_c_remain = i_c - (512 * full);
    if (i_c_remain > 0) {
      i_c_remain = ((i_c_remain + 15) / 16) * 16;
      fl = 512 / i_c_remain;
      col = fl * i_c_remain;
      stride = i_c_remain;
      n = (((H * W) + fl - 1) / fl);
      for (int i = 0; i < n; i++) {
        pim::OutputNewtonTraceV2(OS, id, row, col, stride);
      }
    }
    code.push_back(OS.str());
  }
  return code;
}

std::string PimCodeGenFC(std::string id, const Str2StrMap& attrs,
                const std::vector<std::string>& func_args) {
  int n_channel = std::stoi(attrs.at("N_CHANNEL"));
  std::vector<std::string> code;
  std::ostringstream OS;

  int row = std::stoi(attrs.at("ROW"));
  int col = std::stoi(attrs.at("COL"));

  pim::OutputNewtonTrace(OS, id, row, col);

  return OS.str();
}

class Readres {
  std::vector<std::string> head;

public:
  Readres(std::vector<std::string> cmds) {
    head = cmds;
  }

  std::string code() const {
    std::string code;
    for (auto h : head) {
      code += h + "\n";
    }
    return code;
  }

  int n_comp() {
    if (head.size() == 0)
      return 0;
    return head.size() - 1;
  }

  std::string at(int i) {
    return head[i];
  }
};

class GAct {
  std::vector<std::string> head;
  std::vector<Readres> readres;
  std::vector<std::string> buffer;

public:
  GAct(std::string cmd) {
    head.push_back(cmd);
  }
  void add(std::string cmd) {
    if (cmd.find("G_ACT") != std::string::npos) {
      head.push_back(cmd);
    } else {
      buffer.push_back(cmd);
      if (cmd.find("READRES") != std::string::npos) {
        readres.push_back(Readres(buffer));
        buffer.clear();
      }
    }
  }

  Readres& at(int i) {
    return readres[i];
  }

  int n_readres() const {
    return readres.size();
  }

  std::string h() const {
    std::string code;
    for (auto& h : head) {
      code += h + "\n";
    }
    return code;
  }

  std::string code(bool include_head) const {
    std::string code;
    if (include_head) {
      for (auto h : head) {
        code += h + "\n";
      }
    }
    for (auto& r : readres) {
      code += r.code();
    }
    return code;
  }
  void split(int factor) {
    std::vector<Readres> new_readres;
    for (int i = 0, offset = 0; i < n_readres(); i++, offset++) {
      Readres& rr = readres[i];
      int chunk = rr.n_comp() / factor;
      for (int j = 0; j < factor; j++) {
        std::vector<std::string> buf;
        for (int k = 0; k < chunk; k++) {
          if (j*chunk + k + offset >= rr.n_comp()) {
            continue;
          }
          buf.push_back(rr.at(j*chunk + k + offset));
        }
        buf.push_back("READRES");
        new_readres.push_back(Readres(buf));
      }
    }
    readres = new_readres;
  }
};

class GWrite {
  std::string head;
  std::vector<GAct> gacts;

public:
  GWrite(std::string cmd) : head(cmd) { }

  void add(std::string cmd) {
    if (cmd.find("G_ACT0") != std::string::npos) {
      gacts.push_back(GAct(cmd));
    } else {
      auto& gact = gacts.back();
      gact.add(cmd);
    }
  }

  int n_gact() const {
    return gacts.size();
  }

  int n_readres() const {
    int n = 0;
    for (auto& gact : gacts) {
      n += gact.n_readres();
    }
    return n;
  }

  GAct& at(int i) {
    return gacts[i];
  }

  std::string h() const {
    return head + "\n";
  }

  std::string code(bool include_head) const {
    std::string code;
    if (include_head) {
      code += head + "\n";
    }
    for (auto& gact : gacts) {
      code += gact.code(true);
    }
    return code;
  }
};

class Command {
  std::vector<GWrite> gwrites;
  int n_channel;

public:
  Command(int n_channel) : n_channel(n_channel) { }

  void add(std::string cmd) {
    if (cmd.find("GWRITE") != std::string::npos) {
      gwrites.push_back(GWrite(cmd));
    } else {
      auto& gwrite = gwrites.back();
      gwrite.add(cmd);
    }
  }

  void policy_readres_auto(GWrite& gwrite, std::vector<std::string>& code, int n, int offset=0, int n_gwrite=1) {
    // write GWRITE for every channels
    for (int i = 0; i < n; i++) {
      std::string g = gwrite.h();
      int s = g.find("GWRITE_");
      if (s == std::string::npos) {
        // replace regular GWRITE with multiple version (GWRITE_2/GWRITE_4)
        g = g.substr(0, g.find("GWRITE")) + std::string("GWRITE_") + std::to_string(n_gwrite) + g.substr(g.find("GWRITE") + 6, g.length());
      }
      code[i + offset] += g;
    }

    // TODO: pick right gact for validation by value
    GAct& gact = gwrite.at(0);

    // distribute readres
    int parallelism = gwrite.n_gact() * gact.n_readres();

    // exploit finer-grained parallelism at the expense of energy increase.
    while (parallelism <= n / 2) {
      int factor = n / parallelism;
      gact.split(factor);
      parallelism = gwrite.n_gact() * gact.n_readres();
    }

    // # READRES per G_ACT
    int stride = std::min(
      std::max((gwrite.n_gact() * gact.n_readres() + n - 1) / n, 1),
      gact.n_readres());
    int gw_n_readres = gwrite.n_gact() * gact.n_readres();
    for (int j = 0, idx = 0; j < gw_n_readres; j += stride, idx++) {
      code[idx % n + offset] += gact.h();
      for (int k = 0; k < stride; k++) {
        if (j + k >= gw_n_readres) {
          break;
        }
        auto& rr = gact.at(gact.n_readres() - 1);
        if (j + k < gw_n_readres) {
          // TODO: pick right rr for validation by value, considering remainder
          // e.g., ./pim_codegen -oc 32 -ic 3 -h 113 -w 224 -kh 3 -kw 3 -ph 0 -pw 1 -stride 2 -name test12 -gw 4 -n_channel 12
          rr = gact.at(k);
        }
        code[idx % n + offset] += rr.code();
      }
    }
  }
  std::vector<std::string> policy_auto(const Str2StrMap& attrs) {
    std::vector<std::string> best_code(n_channel);
    for (int n_gwrite = 1; n_gwrite <= std::stoi(attrs.at("GW")); n_gwrite *= 2) {
      std::vector<std::string> code(n_channel);
      int stride = n_channel / n_gwrite;
      int chunk = (gwrites.size() + n_gwrite - 1) / n_gwrite;
      for (int i = 0; i < n_gwrite; i++) {
        int offset = i * stride;
        for (int j = 0; j < chunk; j++) {
          if (i * chunk + j >= gwrites.size()) {
            break;
          }
          auto& gwrite = gwrites[i * chunk + j];
          // TODO: for valication by value, (stride) number of gwrites must be passed to the policy_readres_auto
          policy_readres_auto(gwrite, code, stride, offset, n_gwrite);
        }
      }
      std::string::size_type pos = 0;
      int gact_best = 0;
      while (true) {
        pos = best_code[0].find("G_ACT0", pos);
        if (pos == std::string::npos) {
          pos = 0;
          break;
        }
        ++gact_best;
        ++pos;
      }
      int gact_code = 0;
      while (true) {
        pos = code[0].find("G_ACT0", pos);
        if (pos == std::string::npos) {
          pos = 0;
          break;
        }
        ++gact_code;
        ++pos;
      }
      if (best_code[0].size() == 0 || gact_code < gact_best || gact_code == gact_best && code[0].size() < best_code[0].size()) {
        best_code = code;
      }
    }
    return best_code;
  }
};

void PimSchedule(std::string id, const Str2StrMap& attrs,
                 const std::vector<std::string>& func_args, std::string code, bool append=false) {
  int n_channel = std::stoi(attrs.at("N_CHANNEL"));
  int gpu_channel = 32 - n_channel;
  auto mode = std::ios_base::out;
  if (append) {
    mode = std::ios_base::app;
  }

  std::vector<std::string> traces;
  std::string token;
  std::stringstream ss(code);

  Command command(n_channel);

  int idx = 0;
  while (std::getline(ss, token, '\n')) {
    traces.push_back(token);
    command.add(token);
    idx++;
  }

  std::ofstream OS;

  OS.open(id + "-all.pim", mode);
  for (auto trace : traces) {
    OS << trace << "\n";
  }
  OS.flush();
  OS.close();

  std::vector<std::string> cmds = command.policy_auto(attrs);

  for (int i = gpu_channel; i < gpu_channel + n_channel; i++) {
    OS.open(id + "-" + std::to_string(i) + ".pim", mode);
    OS << cmds[i - gpu_channel];
    OS.flush();
    OS.close();
  }
}

std::string ConvOp(std::string id, const Str2StrMap& attrs,
                    const std::vector<std::string>& func_args) {
  std::string kernel_name = attrs.at("ONNX_NODE_NAME");
  std::string cudnn_code = CuDNNCodeGen(kernel_name, attrs, func_args);
  std::vector<std::string> pim_code = PimCodeGen(kernel_name, attrs, func_args);
  for (int i = 0; i < pim_code.size(); i++) {
    PimSchedule(kernel_name, attrs, {}, pim_code[i], i > 0);
  }
  return cudnn_code;
}

std::string FCOp(std::string id, const Str2StrMap& attrs,
                    const std::vector<std::string>& func_args) {
  std::string kernel_name = attrs.at("ONNX_NODE_NAME");
  std::string pim_code = PimCodeGenFC(kernel_name, attrs, func_args);
  PimSchedule(kernel_name, attrs, {}, pim_code);
  return "";
}

class CodegenPim : public MemoizedExprTranslator<std::vector<Output>>, public CodegenCBase {
 public:
  CodegenPim(const std::string& id, const Map<String, ObjectRef>& attrs) {
    this->ext_func_id_ = id;
    this->attrs_ = attrs;
  }

  std::vector<Output> VisitExprDefault_(const Object* op) final {
    LOG(FATAL) << "Pim codegen doesn't support: " << op->GetTypeKey();
    return {};
  }

  std::vector<Output> VisitExpr_(const VarNode* node) final {
    ext_func_args_.push_back(GetRef<Var>(node));
    Output output;
    output.name = node->name_hint();
    return {output};
  }

  std::vector<Output> VisitExpr_(const TupleNode* node) final {
    std::vector<Output> outs;
    for (auto field : node->fields) {
      auto res = VisitExpr(field);
      ICHECK_EQ(res.size(), 1U) << "Do not support tuple nest";
      outs.push_back(res[0]);
    }
    return outs;
  }

  std::vector<Output> VisitExpr_(const TupleGetItemNode* op) final {
    auto res = VisitExpr(op->tuple);
    ICHECK_GT(res.size(), static_cast<size_t>(op->index));

    // Only keep the item we want for the child node.
    // FIXME(@comaniac): The other items should still be requried for the primary outputs.
    return {res[op->index]};
  }

  std::vector<Output> VisitExpr_(const ConstantNode* cn) final {
    Output output;
    // Get const: static_cast<float*>(dnnl_0_consts[0]->data)
    output.name = CreateDataReference(ext_func_id_, const_idx_);
    output.dtype = "float";

    // Generate the global variable for needed ndarrays
    if (const_array_name_.empty()) {
      const_array_name_ = CreateNDArrayPool(ext_func_id_);
      std::string checker = CreateInitChecker(ext_func_id_);
      ext_func_body_.insert(ext_func_body_.begin(), checker);
    }

    // Give the ndarray a unique name to ease the initialization of it at
    // runtime.
    std::string const_var_name = CreateConstVar(ext_func_id_, const_idx_);
    const_vars_.push_back(const_var_name);
    const_idx_++;

    const auto* type_node = cn->checked_type().as<TensorTypeNode>();
    ICHECK(type_node);
    ICHECK_EQ(GetDtypeString(type_node), "float") << "Only float is supported for now.";

    return {output};
  }

  std::vector<Output> VisitExpr_(const CallNode* call) final {
    const auto* func = call->op.as<FunctionNode>();
    ICHECK(func) << "Only composite function is supported for PIM.";
    GenerateBodyOutput ret = GenerateCompositeFunctionCall(func, call);
    ext_func_body_.push_back(ret.decl);
    return ret.outputs;
  }

  std::string JIT(const std::vector<Output>& out) {
    return JitImpl(ext_func_id_, ext_func_args_, buf_decl_, ext_func_body_, const_array_name_, out);
  }

 private:
  std::vector<std::string> GetArgumentNames(const CallNode* call) {
    std::vector<std::string> arg_names;
    for (size_t i = 0; i < call->args.size(); ++i) {
      auto res = VisitExpr(call->args[i]);
      for (const auto& out : res) {
        arg_names.push_back(out.name);
      }
    }
    return arg_names;
  }

  GenerateBodyOutput GenerateCompositeFunctionCall(const FunctionNode* callee,
                                                   const CallNode* caller) {
    const auto pattern_name = callee->GetAttr<runtime::String>(attr::kComposite);
    ICHECK(pattern_name.defined()) << "Only functions with composite attribute are supported.";

    if (pattern_name == "pim.conv2d") {
      const auto* conv_call = GetRootCall(callee->body.as<CallNode>(), 0, std::vector<std::string>{"nn.conv2d"});
      return GenerateBody(conv_call, "pim_conv2d", GetArgumentNames(caller),
                          ConvArgs(conv_call, ACT_NONE));
    } else if (pattern_name == "pim.conv2d_bias") {
      const auto* conv_call = GetRootCall(callee->body.as<CallNode>(), 1, std::vector<std::string>{"nn.conv2d", "add"});
      return GenerateBody(conv_call, "pim_conv2d", GetArgumentNames(caller),
                          ConvArgs(conv_call, ACT_NONE));
    } else if (pattern_name == "pim.conv2d_relu") {
      const auto* conv_call = GetRootCall(callee->body.as<CallNode>(), 1, std::vector<std::string>{"nn.conv2d", "relu"});
      return GenerateBody(conv_call, "pim_conv2d", GetArgumentNames(caller),
                          ConvArgs(conv_call, ACT_RELU));
    } else if (pattern_name == "pim.conv2d_bias_relu") {
      const auto* conv_call = GetRootCall(callee->body.as<CallNode>(), 2, std::vector<std::string>{"nn.conv2d", "add", "nn.relu"});
      return GenerateBody(conv_call, "pim_conv2d", GetArgumentNames(caller),
                          ConvArgs(conv_call, ACT_RELU));
    } else if (pattern_name == "pim.conv2d_clip") {
      const auto* conv_call = GetRootCall(callee->body.as<CallNode>(), 1, std::vector<std::string>{"nn.conv2d", "clip"});
      return GenerateBody(conv_call, "pim_conv2d", GetArgumentNames(caller),
                          ConvArgs(conv_call, ACT_CLIP));
    } else if (pattern_name == "pim.conv2d_bias_clip") {
      const auto* conv_call = GetRootCall(callee->body.as<CallNode>(), 2, std::vector<std::string>{"nn.conv2d", "add", "clip"});
      return GenerateBody(conv_call, "pim_conv2d", GetArgumentNames(caller),
                          ConvArgs(conv_call, ACT_CLIP));
    } else if (pattern_name == "pim.conv2d_swish") {
      const auto* conv_call = GetRootCall(callee->body.as<CallNode>(), 1, std::vector<std::string>{"nn.conv2d", "multiply"});
      return GenerateBody(conv_call, "pim_conv2d", GetArgumentNames(caller),
                          ConvArgs(conv_call, ACT_SWISH));
    } else if (pattern_name == "pim.conv2d_bias_swish") {
      const auto* conv_call = GetRootCall(callee->body.as<CallNode>(), 2, std::vector<std::string>{"nn.conv2d", "add", "multiply"});
      return GenerateBody(conv_call, "pim_conv2d", GetArgumentNames(caller),
                          ConvArgs(conv_call, ACT_SWISH));
    } else if (pattern_name == "pim.conv2d_sigmoid") {
      const auto* conv_call = GetRootCall(callee->body.as<CallNode>(), 1, std::vector<std::string>{"nn.conv2d", "sigmoid"});
      return GenerateBody(conv_call, "pim_conv2d", GetArgumentNames(caller),
                          ConvArgs(conv_call, ACT_SIGMOID));
    } else if (pattern_name == "pim.conv2d_bias_sigmoid") {
      const auto* conv_call = GetRootCall(callee->body.as<CallNode>(), 2, std::vector<std::string>{"nn.conv2d", "add", "sigmoid"});
      return GenerateBody(conv_call, "pim_conv2d", GetArgumentNames(caller),
                          ConvArgs(conv_call, ACT_SIGMOID));
    } else if (pattern_name == "pim.conv2d_fc") {
      const auto* conv_call = GetRootCall(callee->body.as<CallNode>(), 0, std::vector<std::string>{"nn.conv2d"});
      return GenerateBody(conv_call, "pim_conv2d_fc", GetArgumentNames(caller),
                          ConvFCArgs(conv_call, ACT_NONE));
    } else if (pattern_name == "pim.conv2d_fc_bias") {
      const auto* conv_call = GetRootCall(callee->body.as<CallNode>(), 1, std::vector<std::string>{"nn.conv2d", "add"});
      return GenerateBody(conv_call, "pim_conv2d_fc", GetArgumentNames(caller),
                          ConvFCArgs(conv_call, ACT_NONE));
    } else if (pattern_name == "pim.conv2d_fc_relu") {
      const auto* conv_call = GetRootCall(callee->body.as<CallNode>(), 1, std::vector<std::string>{"nn.conv2d", "relu"});
      return GenerateBody(conv_call, "pim_conv2d_fc", GetArgumentNames(caller),
                          ConvFCArgs(conv_call, ACT_RELU));
    } else if (pattern_name == "pim.conv2d_fc_bias_relu") {
      const auto* conv_call = GetRootCall(callee->body.as<CallNode>(), 2, std::vector<std::string>{"nn.conv2d", "add", "nn.relu"});
      return GenerateBody(conv_call, "pim_conv2d_fc", GetArgumentNames(caller),
                          ConvFCArgs(conv_call, ACT_RELU));
    } else if (pattern_name == "pim.conv2d_fc_clip") {
      const auto* conv_call = GetRootCall(callee->body.as<CallNode>(), 1, std::vector<std::string>{"nn.conv2d", "clip"});
      return GenerateBody(conv_call, "pim_conv2d_fc", GetArgumentNames(caller),
                          ConvFCArgs(conv_call, ACT_CLIP));
    } else if (pattern_name == "pim.conv2d_fc_bias_clip") {
      const auto* conv_call = GetRootCall(callee->body.as<CallNode>(), 2, std::vector<std::string>{"nn.conv2d", "add", "clip"});
      return GenerateBody(conv_call, "pim_conv2d_fc", GetArgumentNames(caller),
                          ConvFCArgs(conv_call, ACT_CLIP));
    } else if (pattern_name == "pim.conv2d_fc_swish") {
      const auto* conv_call = GetRootCall(callee->body.as<CallNode>(), 1, std::vector<std::string>{"nn.conv2d", "multiply"});
      return GenerateBody(conv_call, "pim_conv2d_fc", GetArgumentNames(caller),
                          ConvFCArgs(conv_call, ACT_SWISH));
    } else if (pattern_name == "pim.conv2d_fc_bias_swish") {
      const auto* conv_call = GetRootCall(callee->body.as<CallNode>(), 2, std::vector<std::string>{"nn.conv2d", "add", "multiply"});
      return GenerateBody(conv_call, "pim_conv2d_fc", GetArgumentNames(caller),
                          ConvFCArgs(conv_call, ACT_SWISH));
    } else if (pattern_name == "pim.conv2d_fc_sigmoid") {
      const auto* conv_call = GetRootCall(callee->body.as<CallNode>(), 1, std::vector<std::string>{"nn.conv2d", "sigmoid"});
      return GenerateBody(conv_call, "pim_conv2d_fc", GetArgumentNames(caller),
                          ConvFCArgs(conv_call, ACT_SIGMOID));
    } else if (pattern_name == "pim.conv2d_fc_bias_sigmoid") {
      const auto* conv_call = GetRootCall(callee->body.as<CallNode>(), 2, std::vector<std::string>{"nn.conv2d", "add", "sigmoid"});
      return GenerateBody(conv_call, "pim_conv2d_fc", GetArgumentNames(caller),
                          ConvFCArgs(conv_call, ACT_SIGMOID));
    } else if (pattern_name == "pim.nn_dense") {
      const auto* fc_call = GetRootCall(callee->body.as<CallNode>(), 0, std::vector<std::string>{"nn.dense"});
      return GenerateBody(fc_call, "pim_fc", GetArgumentNames(caller),
                          FCArgs(fc_call, ACT_NONE));
    } else if (pattern_name == "pim.nn_dense_bias") {
      const auto* fc_call = GetRootCall(callee->body.as<CallNode>(), 1, std::vector<std::string>{"nn.dense", "add"});
      return GenerateBody(fc_call, "pim_fc", GetArgumentNames(caller),
                          FCArgs(fc_call, ACT_NONE));
    } else if (pattern_name == "pim.layout_transform") {
      const auto* opt_call = GetRootCall(callee->body.as<CallNode>(), 0, std::vector<std::string>{"layout_transform"});
      return GenerateBody(opt_call, "pim_memory_optimized", GetArgumentNames(caller),
                          {});
    } else if (pattern_name == "pim.nn_pad") {
      const auto* opt_call = GetRootCall(callee->body.as<CallNode>(), 0, std::vector<std::string>{"nn.pad"});
      return GenerateBody(opt_call, "pim_memory_optimized", GetArgumentNames(caller),
                          {});
    } else if (pattern_name == "pim.concatenate") {
      const auto* opt_call = GetRootCall(callee->body.as<CallNode>(), 0, std::vector<std::string>{"concatenate"});
      return GenerateBody(opt_call, "pim_memory_optimized", GetArgumentNames(caller),
                          {});
    } else if (pattern_name == "pim.strided_slice") {
      const auto* opt_call = GetRootCall(callee->body.as<CallNode>(), 0, std::vector<std::string>{"strided_slice"});
      return GenerateBody(opt_call, "pim_memory_optimized", GetArgumentNames(caller),
                          {});
    } else if (pattern_name == "pim.conv2d_gpu") {
      const auto* conv_call = GetRootCall(callee->body.as<CallNode>(), 0, std::vector<std::string>{"nn.conv2d"});
      return GenerateBody(conv_call, "pim_gpu", GetArgumentNames(caller),
                          ConvArgs(conv_call, ACT_NONE));
    } else if (pattern_name == "pim.conv2d_bias_gpu") {
      const auto* conv_call = GetRootCall(callee->body.as<CallNode>(), 1, std::vector<std::string>{"nn.conv2d", "add"});
      return GenerateBody(conv_call, "pim_gpu", GetArgumentNames(caller),
                          ConvArgs(conv_call, ACT_NONE));
    } else if (pattern_name == "pim.conv2d_relu_gpu") {
      const auto* conv_call = GetRootCall(callee->body.as<CallNode>(), 1, std::vector<std::string>{"nn.conv2d", "relu"});
      return GenerateBody(conv_call, "pim_gpu", GetArgumentNames(caller),
                          ConvArgs(conv_call, ACT_RELU));
    } else if (pattern_name == "pim.conv2d_bias_relu_gpu") {
      const auto* conv_call = GetRootCall(callee->body.as<CallNode>(), 2, std::vector<std::string>{"nn.conv2d", "add", "nn.relu"});
      return GenerateBody(conv_call, "pim_gpu", GetArgumentNames(caller),
                          ConvArgs(conv_call, ACT_RELU));
    } else if (pattern_name == "pim.conv2d_clip_gpu") {
      const auto* conv_call = GetRootCall(callee->body.as<CallNode>(), 1, std::vector<std::string>{"nn.conv2d", "clip"});
      return GenerateBody(conv_call, "pim_gpu", GetArgumentNames(caller),
                          ConvArgs(conv_call, ACT_CLIP));
    } else if (pattern_name == "pim.conv2d_bias_clip_gpu") {
      const auto* conv_call = GetRootCall(callee->body.as<CallNode>(), 2, std::vector<std::string>{"nn.conv2d", "add", "clip"});
      return GenerateBody(conv_call, "pim_gpu", GetArgumentNames(caller),
                          ConvArgs(conv_call, ACT_CLIP));
    } else if (pattern_name == "pim.conv2d_swish_gpu") {
      const auto* conv_call = GetRootCall(callee->body.as<CallNode>(), 1, std::vector<std::string>{"nn.conv2d", "multiply"});
      return GenerateBody(conv_call, "pim_gpu", GetArgumentNames(caller),
                          ConvArgs(conv_call, ACT_SWISH));
    } else if (pattern_name == "pim.conv2d_bias_swish_gpu") {
      const auto* conv_call = GetRootCall(callee->body.as<CallNode>(), 2, std::vector<std::string>{"nn.conv2d", "add", "multiply"});
      return GenerateBody(conv_call, "pim_gpu", GetArgumentNames(caller),
                          ConvArgs(conv_call, ACT_SWISH));
    } else if (pattern_name == "pim.conv2d_sigmoid_gpu") {
      const auto* conv_call = GetRootCall(callee->body.as<CallNode>(), 1, std::vector<std::string>{"nn.conv2d", "sigmoid"});
      return GenerateBody(conv_call, "pim_gpu", GetArgumentNames(caller),
                          ConvArgs(conv_call, ACT_SIGMOID));
    } else if (pattern_name == "pim.conv2d_bias_sigmoid_gpu") {
      const auto* conv_call = GetRootCall(callee->body.as<CallNode>(), 2, std::vector<std::string>{"nn.conv2d", "add", "sigmoid"});
      return GenerateBody(conv_call, "pim_gpu", GetArgumentNames(caller),
                          ConvArgs(conv_call, ACT_SIGMOID));
    } else if (pattern_name == "pim.nn_dense_gpu") {
      const auto* fc_call = GetRootCall(callee->body.as<CallNode>(), 0, std::vector<std::string>{"nn.dense"});
      return GenerateBody(fc_call, "pim_gpu", GetArgumentNames(caller),
                          FCArgs(fc_call, ACT_NONE));
    } else if (pattern_name == "pim.nn_dense_bias_gpu") {
      const auto* fc_call = GetRootCall(callee->body.as<CallNode>(), 1, std::vector<std::string>{"nn.dense", "add"});
      return GenerateBody(fc_call, "pim_gpu", GetArgumentNames(caller),
                          FCArgs(fc_call, ACT_NONE));
    }

    LOG(FATAL) << "Unknown composite function: " << pattern_name;
    return {};
  }

  GenerateBodyOutput GenerateBody(const CallNode* root_call, const std::string& func_name,
                                  const std::vector<std::string>& func_args,
                                  const Str2StrMap& attribute_args) {
    // Make function call with input buffers when visiting arguements
    ICHECK_GT(func_args.size(), 0);
    std::ostringstream decl_stream;
    decl_stream << "(" << func_args[0];
    for (size_t i = 1; i < func_args.size(); ++i) {
      decl_stream << ", " << func_args[i];
    }
    // Analyze the output buffers
    std::vector<Type> out_types;
    if (root_call->checked_type()->IsInstance<TupleTypeNode>()) {
      auto type_node = root_call->checked_type().as<TupleTypeNode>();
      for (auto field : type_node->fields) {
        ICHECK(field->IsInstance<TensorTypeNode>());
        out_types.push_back(field);
      }
    } else if (root_call->checked_type()->IsInstance<TensorTypeNode>()) {
      ICHECK(root_call->checked_type()->IsInstance<TensorTypeNode>());
      out_types.push_back(root_call->checked_type());
    } else {
      LOG(FATAL) << "Unrecognized type node: " << AsText(root_call->checked_type(), false);
    }
    GenerateBodyOutput ret;
    for (const auto& out_type : out_types) {
      const std::string out = "out" + std::to_string(buf_idx_++);
      decl_stream << ", " << out;
      Output output;
      output.name = out;
      output.dtype = GetDtypeString(out_type.as<TensorTypeNode>());
      output.need_copy = false;
      ret.outputs.push_back(output);
    }
    decl_stream << ");";
    if (func_name == "pim_conv2d") {
      ret.decl = ConvOp(ext_func_id_, attribute_args, func_args);
    } else if (func_name == "pim_conv2d_fc") {
      ret.decl = FCOp(ext_func_id_, attribute_args, func_args);
    } else if (func_name == "pim_fc") {
      ret.decl = FCOp(ext_func_id_, attribute_args, func_args);
    } else if (func_name == "pim_memory_optimized") {
      // do nothing
      ret.decl = "";
    } else if (func_name == "pim_gpu") {
      // do nothing
      ret.decl = "";
    }

    return ret;
  }
  /*! \brief The id of the external pim ext_func. */
  std::string ext_func_id_{""};
  /*! \brief The attrs of the external pim ext_func. */
  Map<String, ObjectRef> attrs_;
  /*!
   * \brief The index to track the output buffer. Each kernel will redirect the
   * output to a buffer that may be consumed by other kernels.
   */
  int buf_idx_{0};
  /*! \brief The index of global constants. */
  int const_idx_{0};
  /*! \brief The arguments used by a wrapped function that calls PIM kernels. */
  Array<Var> ext_func_args_;
  /*! \brief Statement of the function that will be compiled using PIM kernels. */
  std::vector<std::string> ext_func_body_;
  /*! \brief The array declared to store the constant values. */
  std::string const_array_name_;
  /*! \brief The declaration of intermediate buffers. */
  std::vector<std::string> buf_decl_;
  /*! \brief The variable name to constant mapping. */
  Array<String> const_vars_;

  friend class PimModuleCodegen;
};  // class CodegenPim

class PimModuleCodegen : public CSourceModuleCodegenBase {
 public:
  std::pair<std::string, Array<String>> GenCudnnFunc(const Function& func) {
    ICHECK(func.defined()) << "Input error: expect a Relay function.";
    // Record the external symbol for runtime lookup.
    auto sid = GetExtSymbol(func);
    const auto* attrs = func->attrs.as<DictAttrsNode>();
    ICHECK(attrs != nullptr);
    const auto dict = attrs->dict;
    CodegenPim builder(sid, dict);
    auto out = builder.VisitExpr(func->body);
    code_stream_ << builder.JIT(out);
    return {sid, builder.const_vars_};
  }

  runtime::Module CreateCSourceModule(const ObjectRef& ref) override {
    // create header
    code_stream_ << "#include <tvm/runtime/c_runtime_api.h>\n";
    code_stream_ << "#include <tvm/runtime/packed_func.h>\n";
    code_stream_ << "#include <dlpack/dlpack.h>\n";
    code_stream_ << "#include <cudnn.h>\n";
    code_stream_ << "#include <cuda_fp16.h>\n";
    // code_stream_ << "#include <cstdint>\n";
    // code_stream_ << "#include <cstdlib>\n";
    // code_stream_ << "#include <string>\n";
    // code_stream_ << "#include <vector>\n";
    // code_stream_ << "#include <cstring>\n";
    // code_stream_ << "#include <iostream>\n";

    // code_stream_ <<
    //   "static void HandleError(cudaError_t err, const char *file, int line) {\n"
    //   "  if (err != cudaSuccess) {\n"
    //   "    printf(\"%s in %s at line %d\\n\", cudaGetErrorString(err), file, line);\n"
    //   "    exit(EXIT_FAILURE);\n"
    //   "  }\n"
    //   "}\n"
    //   "#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))\n"
    //   "#define checkCUDNN(expression) \\ \n"
    //   "{ \\ \n"
    //   "cudnnStatus_t status(expression); \\ \n"
    //   "if (status != CUDNN_STATUS_SUCCESS) { \\ \n"
    //   "  std::cerr << \"Error on line \" << __LINE__ << \": \" \\ \n"
    //   "            << cudnnGetErrorString(status) << std::endl; \\ \n"
    //   "  std::exit(EXIT_FAILURE); \\ \n"
    //   "} \\ \n"
    //   "}\n";
    // code_stream_ << "typedef float d_type;\n";
    // code_stream_ << "#define DATA_TYPE CUDNN_DATA_FLOAT\n";

    ICHECK(ref->IsInstance<FunctionNode>());
    auto res = GenCudnnFunc(Downcast<Function>(ref));
    std::string code = code_stream_.str();
    String sym = std::get<0>(res);
    Array<String> variables = std::get<1>(res);
    // Create a CSource module
    const auto* pf = runtime::Registry::Get("runtime.CSourceModuleCreate");
    ICHECK(pf != nullptr) << "Cannot find CSource module to create the external runtime module";
    return (*pf)(code, "cu", Array<String>{sym}, variables);
  }

 private:
  /*! \brief The code stream that will be compiled by NVCC */
  std::ostringstream code_stream_;
};  // PimModuleCodegen

/*!
 * \brief The external pim compiler/codegen tool. It takes a Relay
 * expression/module and compile it into a runtime module.
 */
runtime::Module PimCompiler(const ObjectRef& ref) {
  PimModuleCodegen pim;
  return pim.CreateCSourceModule(ref);
}

TVM_REGISTER_GLOBAL("relay.ext.pim").set_body_typed(PimCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
