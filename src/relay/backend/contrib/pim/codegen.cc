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
  ACT_CLIP
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

  return args;
}

inline void CuDNNPrint(std::ostringstream& os, const std::string& stmt, int indent = 2) {
  for (int i = 0; i < indent; ++i) {
    os << " ";
  }
  os << stmt;
}

inline std::string CuDNNCodeGen(std::string id, const Str2StrMap& attrs,
                    const std::vector<std::string>& func_args) {
  std::ostringstream conv_decl;

  // setup
  CuDNNPrint(conv_decl, "cudnnHandle_t cudnn;\n");
  CuDNNPrint(conv_decl, "checkCUDNN(cudnnCreate(&cudnn));\n");

  // input
  // CuDNNPrint(conv_decl, "size_t input_bytes = " +  attrs.at("N") + " * " +  attrs.at("C") + " * " +  attrs.at("H") + " * " +  attrs.at("W") + " * " + "sizeof(d_type);\n");
  // // CuDNNPrint(conv_decl, "d_type *c_input = (d_type*)malloc(input_bytes);\n");
  // CuDNNPrint(conv_decl, "d_type *d_input{nullptr};\n");
  // CuDNNPrint(conv_decl, "cudaMalloc(&d_input, input_bytes);\n");
  // CuDNNPrint(conv_decl, "cudaMemcpy(d_input, " + func_args[0] + ", input_bytes, cudaMemcpyHostToDevice);\n");
  // CuDNNPrint(conv_decl, "HANDLE_ERROR(cudaDeviceSynchronize());\n");

  // kernel
  // CuDNNPrint(conv_decl, "size_t kernel_bytes = " + attrs.at("K") + " * " + attrs.at("C") + " * " + attrs.at("R") + " * " + attrs.at("S") + " * sizeof(d_type);\n");
  // // CuDNNPrint(conv_decl, "d_type *c_kernel = (d_type*)malloc(kernel_bytes);\n");
  // CuDNNPrint(conv_decl, "d_type *d_kernel{nullptr};\n");
  // CuDNNPrint(conv_decl, "cudaMalloc(&d_kernel, kernel_bytes);\n");
  // CuDNNPrint(conv_decl, "cudaMemcpy(d_kernel, " + func_args[1] + ", kernel_bytes, cudaMemcpyHostToDevice);\n");
  // CuDNNPrint(conv_decl, "HANDLE_ERROR(cudaDeviceSynchronize());\n");

  // bias
  // if (func_args.size() > 2) {
  //   CuDNNPrint(conv_decl, "size_t bias_bytes = " + attrs.at("K") + " * sizeof(d_type);\n");
  //   // CuDNNPrint(conv_decl, "d_type *c_bias = (d_type*)malloc(kernel_bytes);\n");
  //   CuDNNPrint(conv_decl, "d_type *d_bias{nullptr};\n");
  //   CuDNNPrint(conv_decl, "cudaMalloc(&d_bias, bias_bytes);\n");
  //   CuDNNPrint(conv_decl, "cudaMemcpy(d_bias, " + func_args[2] + ", bias_bytes, cudaMemcpyHostToDevice);\n");
  //   CuDNNPrint(conv_decl, "HANDLE_ERROR(cudaDeviceSynchronize());\n");
  // } else if (std::stoi(attrs.at("Act")) != ACT_NONE) {
  //   CuDNNPrint(conv_decl, "size_t bias_size = " + attrs.at("C") + ";\n");
  //   CuDNNPrint(conv_decl, "size_t bias_bytes = bias_size * sizeof(d_type);\n");
  //   CuDNNPrint(conv_decl, "d_type *c_bias = (d_type*)calloc(bias_size, sizeof(d_type));\n");
  //   CuDNNPrint(conv_decl, "d_type *d_bias{nullptr};\n");
  //   CuDNNPrint(conv_decl, "cudaMalloc(&d_bias, bias_bytes);\n");
  //   CuDNNPrint(conv_decl, "cudaMemcpy(d_bias, c_bias, bias_bytes, cudaMemcpyHostToDevice);\n");
  //   CuDNNPrint(conv_decl, "HANDLE_ERROR(cudaDeviceSynchronize());\n");
  // }

  // descriptor
  CuDNNPrint(conv_decl, "cudnnTensorDescriptor_t input_descriptor;\n");
  CuDNNPrint(conv_decl, "checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));\n");
  CuDNNPrint(conv_decl, "checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,\n"
                        "/*format=*/CUDNN_TENSOR_NHWC,\n"
                        "/*dataType=*/DATA_TYPE,\n"
                        "/*batch_size=*/" + attrs.at("N") + ",\n"
                        "/*channels=*/" + attrs.at("C") + ",\n"
                        "/*height=*/" + attrs.at("H") + ",\n"
                        "/*width=*/" + attrs.at("W") + "));\n");

  CuDNNPrint(conv_decl, "cudnnFilterDescriptor_t kernel_descriptor;\n");
  CuDNNPrint(conv_decl, "checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));\n");
  CuDNNPrint(conv_decl, "checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,\n"
                        "/*dataType=*/DATA_TYPE,\n"
                        "/*format=*/CUDNN_TENSOR_NHWC,\n"
                        "/*out_channels=*/" + attrs.at("K") + ",\n"
                        "/*in_channels=*/" + attrs.at("C") + ",\n"
                        "/*kernel_height=*/" + attrs.at("R") + ",\n"
                        "/*kernel_width=*/" + attrs.at("S") + "));\n");

  CuDNNPrint(conv_decl, "cudnnConvolutionDescriptor_t convolution_descriptor;\n");
  CuDNNPrint(conv_decl, "checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));\n");
  CuDNNPrint(conv_decl, "checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,\n"
                        "/*pad_height=*/" + attrs.at("Ph") + ",\n"
                        "/*pad_width=*/" + attrs.at("Pw") + ",\n"
                        "/*vertical_stride=*/" + attrs.at("Sh") + ",\n"
                        "/*horizontal_stride=*/" + attrs.at("Sw") + ",\n"
                        "/*dilation_height=*/" + attrs.at("Dh") + ",\n"
                        "/*dilation_width=*/" + attrs.at("Dw") + ",\n"
                        "/*mode=*/CUDNN_CROSS_CORRELATION,\n"
                        "/*conputeType=*/DATA_TYPE));\n");

  CuDNNPrint(conv_decl, "cudnnTensorDescriptor_t bias_descriptor;\n");
  CuDNNPrint(conv_decl, "checkCUDNN(cudnnCreateTensorDescriptor(&bias_descriptor));\n");
  CuDNNPrint(conv_decl, "checkCUDNN(cudnnSetTensor4dDescriptor(bias_descriptor,\n"
                        "/*format=*/CUDNN_TENSOR_NHWC,\n"
                        "/*dataType=*/DATA_TYPE,\n"
                        "/*batch_size=*/1,\n"
                        "/*channels=*/" + attrs.at("K") + ",\n"
                        "/*height=*/1,\n"
                        "/*width=*/1));\n");

  CuDNNPrint(conv_decl, "cudnnActivationDescriptor_t activation_descriptor;\n");
  CuDNNPrint(conv_decl, "checkCUDNN(cudnnCreateActivationDescriptor(&activation_descriptor));\n");
  CuDNNPrint(conv_decl, "checkCUDNN(cudnnSetActivationDescriptor(activation_descriptor,\n"
                        "CUDNN_ACTIVATION_IDENTITY,\n"
                        "CUDNN_NOT_PROPAGATE_NAN,\n"
                        "0));\n");
  if (std::stoi(attrs.at("Act")) == ACT_RELU) {
    CuDNNPrint(conv_decl, "checkCUDNN(cudnnSetActivationDescriptor(activation_descriptor,\n"
                          "CUDNN_ACTIVATION_RELU,\n"
                          "CUDNN_NOT_PROPAGATE_NAN,\n"
                          "0));\n");
  } else if (std::stoi(attrs.at("Act")) == ACT_SWISH) {
    CuDNNPrint(conv_decl, "checkCUDNN(cudnnSetActivationDescriptor(activation_descriptor,\n"
                          "CUDNN_ACTIVATION_SWISH,\n"
                          "CUDNN_NOT_PROPAGATE_NAN,\n"
                          "0));\n");
    CuDNNPrint(conv_decl, "cudnnSetActivationDescriptorSwishBeta(activation_descriptor, 1);\n");
  }
  if (std::stoi(attrs.at("G")) > 1)
    CuDNNPrint(conv_decl, "checkCUDNN(cudnnSetConvolutionGroupCount(convolution_descriptor,"
                          + attrs.at("G") + "));\n");
  CuDNNPrint(conv_decl, "checkCUDNN(cudnnSetConvolutionMathType(convolution_descriptor, CUDNN_TENSOR_OP_MATH));\n");
  CuDNNPrint(conv_decl, "int OUTPUT_BATCH_SIZE = 0, OUTPUT_CHANNELS = 0, OUTPUT_HEIGHT = 0, OUTPUT_WIDTH = 0;\n");
  CuDNNPrint(conv_decl, "checkCUDNN(cudnnGetConvolution2dForwardOutputDim(\n"
                        "convolution_descriptor, input_descriptor, kernel_descriptor,\n"
                        "&OUTPUT_BATCH_SIZE, &OUTPUT_CHANNELS, &OUTPUT_HEIGHT, &OUTPUT_WIDTH));\n");

  // CuDNNPrint(conv_decl, "size_t output_bytes = OUTPUT_BATCH_SIZE * OUTPUT_CHANNELS * OUTPUT_HEIGHT * OUTPUT_WIDTH * sizeof(d_type);\n");
  // CuDNNPrint(conv_decl, "d_type *d_output{nullptr};\n");
  // CuDNNPrint(conv_decl, "cudaMalloc(&d_output, output_bytes);\n");

  CuDNNPrint(conv_decl, "cudnnTensorDescriptor_t output_descriptor;\n");
  CuDNNPrint(conv_decl, "checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));\n");
  CuDNNPrint(conv_decl, "checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,\n"
                        "/*format=*/CUDNN_TENSOR_NHWC,\n"
                        "/*dataType=*/DATA_TYPE,\n"
                        "/*batch_size=*/OUTPUT_BATCH_SIZE,\n"
                        "/*channels=*/OUTPUT_CHANNELS,\n"
                        "/*height=*/OUTPUT_HEIGHT,\n"
                        "/*width=*/OUTPUT_WIDTH));\n");

  CuDNNPrint(conv_decl, "cudnnConvolutionFwdAlgoPerf_t perf;\n");
  CuDNNPrint(conv_decl, "int algo_count = 1;\n");
  CuDNNPrint(conv_decl, "checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(\n"
                      "cudnn,\n"
                      "input_descriptor,\n"
                      "kernel_descriptor,\n"
                      "convolution_descriptor,\n"
                      "output_descriptor,\n"
                      "1,            // requestedAlgoCount\n"
                      "&algo_count,  // returnedAlgoCount\n"
                      "&perf));\n");

  CuDNNPrint(conv_decl, "size_t workspace_bytes = 0;\n");
  CuDNNPrint(conv_decl, "cudnnGetConvolutionForwardWorkspaceSize(\n"
                        "cudnn, input_descriptor, kernel_descriptor, convolution_descriptor,\n"
                        "output_descriptor, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, &workspace_bytes);\n");
  CuDNNPrint(conv_decl, "void *d_workspace{nullptr};\n"
                      "if (workspace_bytes > 0)\n"
                      "  cudaMalloc(&d_workspace, workspace_bytes);\n");

  CuDNNPrint(conv_decl, "HANDLE_ERROR(cudaDeviceSynchronize());\n");
  CuDNNPrint(conv_decl, "const float alpha = 1, beta = 0;\n");
  // CuDNNPrint(conv_decl, "cudnnConvolutionForward(cudnn, &alpha, input_descriptor, " + func_args[0] +
  // ", kernel_descriptor, " + func_args[1] + ", convolution_descriptor, perf.algo, d_workspace, workspace_bytes, &beta, output_descriptor, out0);\n");
  if (func_args.size() == 2 && std::stoi(attrs.at("Act")) == ACT_NONE) {
    // CuDNNPrint(conv_decl, "cudnnConvolutionForward(cudnn, &alpha, input_descriptor, d_input, kernel_descriptor, d_kernel, convolution_descriptor, perf.algo, d_workspace, workspace_bytes, &beta, output_descriptor, d_output);\n");
    CuDNNPrint(conv_decl, "cudnnConvolutionForward(cudnn, &alpha, input_descriptor," + func_args[0] + ", kernel_descriptor," + func_args[1] + ", convolution_descriptor, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, d_workspace, workspace_bytes, &beta, output_descriptor, out0);\n");
  } else {
    // CuDNNPrint(conv_decl, "cudnnConvolutionBiasActivationForward(cudnn, &alpha, input_descriptor, d_input, kernel_descriptor, d_kernel, convolution_descriptor, perf.algo, d_workspace, workspace_bytes, &beta, output_descriptor, d_output, bias_descriptor, d_bias, activation_descriptor, output_descriptor, d_output);\n");
    CuDNNPrint(conv_decl, "cudnnConvolutionBiasActivationForward(cudnn, &alpha, input_descriptor," + func_args[0] + ", kernel_descriptor," + func_args[1] + ", convolution_descriptor, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, d_workspace, workspace_bytes, &beta, output_descriptor, out0, bias_descriptor," + func_args[2] + ", activation_descriptor, output_descriptor, out0);\n");
  }
  CuDNNPrint(conv_decl, "HANDLE_ERROR(cudaDeviceSynchronize());\n");
  // CuDNNPrint(conv_decl, "cudaMemcpy(out0, d_output, output_bytes, cudaMemcpyDeviceToHost);\n");
  // CuDNNPrint(conv_decl, "HANDLE_ERROR(cudaDeviceSynchronize());\n");

  CuDNNPrint(conv_decl, "cudaFree(d_workspace);\n");
  // CuDNNPrint(conv_decl, "cudaFree(d_input);\n");
  // CuDNNPrint(conv_decl, "cudaFree(d_kernel);\n");
  // CuDNNPrint(conv_decl, "cudaFree(d_output);\n");
  CuDNNPrint(conv_decl, "cudnnDestroyTensorDescriptor(input_descriptor);\n");
  CuDNNPrint(conv_decl, "cudnnDestroyTensorDescriptor(output_descriptor);\n");
  CuDNNPrint(conv_decl, "cudnnDestroyFilterDescriptor(kernel_descriptor);\n");
  CuDNNPrint(conv_decl, "cudnnDestroyConvolutionDescriptor(convolution_descriptor);\n");
  CuDNNPrint(conv_decl, "cudnnDestroy(cudnn);\n");
  CuDNNPrint(conv_decl, "HANDLE_ERROR(cudaDeviceSynchronize());\n");
  return conv_decl.str();
}

std::string PimCodeGen(std::string id, const Str2StrMap& attrs,
                const std::vector<std::string>& func_args) {
  std::ostringstream OS;

  int C_o = std::stoi(attrs.at("K"));
  int C_i = std::stoi(attrs.at("C"));
  int H = std::stoi(attrs.at("H"));
  int W = std::stoi(attrs.at("W"));

  auto i_c = ((C_i + 15) / 16) * 16;
  auto o_c = C_o;
  auto row = o_c;
  int fl, stride, n, col;

  if (i_c <= 512) {
    fl = 512 / i_c;
    col = i_c * fl;
    stride = i_c;
    n = ((H * W) + fl - 1) / fl;
    for (int i = 0; i < n; i++) {
      if (i == n - 1) {
        col = ((H * W) - (i * fl)) * i_c;
      }
      pim::OutputNewtonTraceV2(OS, id, row, col, stride);
    }
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
    int i_c_remain = i_c - (512 * full);
    if (i_c_remain > 0) {
      fl = 512 / i_c_remain;
      col = fl * i_c_remain;
      stride = i_c_remain;
      n = (((H * W) + fl - 1) / fl);
      for (int i = 0; i < n; i++) {
        if (i == n - 1) {
          col = ((H * W) - (i * fl)) * i_c_remain;
        }
        pim::OutputNewtonTraceV2(OS, id, row, col, stride);
      }
    }
  }
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
  void policy_basic(GWrite& gwrite, std::vector<std::string>& code) {
    // write GWRITE for every channels
    for (int i = 0; i < std::min(n_channel, gwrite.n_gact()); i++) {
      code[i] += gwrite.h();
    }
    // distrubute G_ACT
    for (int i = 0; i < gwrite.n_gact(); i++) {
      code[i % n_channel] += gwrite.at(i).code(true);
    }
  }
  std::vector<std::string> policy_basic() {
    std::vector<std::string> code(n_channel);

    for (auto& gwrite : gwrites) {
      policy_basic(gwrite, code);
    }

    return code;
  }
  std::vector<std::string> policy_readres() {
    std::vector<std::string> code(n_channel);

    for (auto& gwrite : gwrites) {
      // check for enough parallelism
      if (gwrite.n_gact() <= n_channel / 2 && gwrite.n_readres() > n_channel / 2) {
        // write GWRITE for every channels
        for (int i = 0; i < std::min(n_channel, gwrite.n_readres()); i++) {
          code[i] += gwrite.h();
        }

        // distribute commands
        for (int i = 0, idx = 0; i < gwrite.n_gact(); i++) {
          GAct& gact = gwrite.at(i);
          // distribute readres
          int stride = std::min(
            std::max(gwrite.n_gact() * gact.n_readres() / n_channel, 1),
            gact.n_readres());
          for (int j = 0; j < gact.n_readres(); j += stride, idx++) {
            code[idx % n_channel] += gact.h();
            for (int k = 0; k < stride; k++) {
              code[idx % n_channel] += gact.at(j + k).code();
            }
          }
        }
      } else { // or fallback to basic policy
        policy_basic(gwrite, code);
      }
    }

    return code;
  }
  std::vector<std::string> policy_gwrite() {
    // TODO
    return std::vector<std::string>{};
  }
};

void PimSchedule(std::string id, const Str2StrMap& attrs,
                 const std::vector<std::string>& func_args, std::string code) {
  int n_channel = 16;

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

  // basic
  // auto cmds = command.policy_basic();

  // policy 1
  auto cmds = command.policy_readres();

  // TODO: policy 2

  for (int i = 0; i < n_channel; i++) {
    OS.open(id + "-" + std::to_string(i));
    OS << cmds[i];
    OS.flush();
    OS.close();
  }

  OS.open(id + "-all");
  for (auto trace : traces) {
    OS << trace << "\n";
  }
  OS.flush();
  OS.close();
}

std::string ConvOp(std::string id, const Str2StrMap& attrs,
                    const std::vector<std::string>& func_args) {
  std::string code = CuDNNCodeGen(id, attrs, func_args);
  std::string pim_trace = PimCodeGen(id, attrs, func_args);
  PimSchedule(id, attrs, func_args, pim_trace);
  return code;
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

  // std::vector<Output> VisitExpr_(const ConstantNode* cn) final {
  //   Output output;
  //   // Get const: static_cast<float*>(dnnl_0_consts[0]->data)
  //   output.name = CreateDataReference(ext_func_id_, const_idx_);
  //   output.dtype = "float";

  //   // Generate the global variable for needed ndarrays
  //   if (const_array_name_.empty()) {
  //     const_array_name_ = CreateNDArrayPool(ext_func_id_);
  //     std::string checker = CreateInitChecker(ext_func_id_);
  //     ext_func_body_.insert(ext_func_body_.begin(), checker);
  //   }

  //   // Give the ndarray a unique name to ease the initialization of it at
  //   // runtime.
  //   std::string const_var_name = CreateConstVar(ext_func_id_, const_idx_);
  //   const_vars_.push_back(const_var_name);
  //   const_idx_++;

  //   const auto* type_node = cn->checked_type().as<TensorTypeNode>();
  //   ICHECK(type_node);
  //   ICHECK_EQ(GetDtypeString(type_node), "float") << "Only float is supported for now.";

  //   return {output};
  // }

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
      const auto* conv_call = GetRootCall(callee->body.as<CallNode>(), 1, std::vector<std::string>{"nn.conv2d", "cos"});
      return GenerateBody(conv_call, "pim_conv2d", GetArgumentNames(caller),
                          ConvArgs(conv_call, ACT_NONE));
    } else if (pattern_name == "pim.conv2d_bias") {
      const auto* conv_call = GetRootCall(callee->body.as<CallNode>(), 2, std::vector<std::string>{"nn.conv2d", "add", "cos"});
      return GenerateBody(conv_call, "pim_conv2d", GetArgumentNames(caller),
                          ConvArgs(conv_call, ACT_NONE));
    } else if (pattern_name == "pim.conv2d_relu") {
      const auto* conv_call = GetRootCall(callee->body.as<CallNode>(), 2, std::vector<std::string>{"nn.conv2d", "relu", "cos"});
      return GenerateBody(conv_call, "pim_conv2d", GetArgumentNames(caller),
                          ConvArgs(conv_call, ACT_RELU));
    } else if (pattern_name == "pim.conv2d_bias_relu") {
      const auto* conv_call = GetRootCall(callee->body.as<CallNode>(), 3, std::vector<std::string>{"nn.conv2d", "add", "nn.relu", "cos"});
      return GenerateBody(conv_call, "pim_conv2d", GetArgumentNames(caller),
                          ConvArgs(conv_call, ACT_RELU));
    } else if (pattern_name == "pim.conv2d_swish") {
      const auto* conv_call = GetRootCall(callee->body.as<CallNode>(), 2, std::vector<std::string>{"nn.conv2d", "multiply", "cos"});
      return GenerateBody(conv_call, "pim_conv2d", GetArgumentNames(caller),
                          ConvArgs(conv_call, ACT_SWISH));
    } else if (pattern_name == "pim.conv2d_bias_swish") {
      const auto* conv_call = GetRootCall(callee->body.as<CallNode>(), 3, std::vector<std::string>{"nn.conv2d", "add", "multiply", "cos"});
      return GenerateBody(conv_call, "pim_conv2d", GetArgumentNames(caller),
                          ConvArgs(conv_call, ACT_SWISH));
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
    code_stream_ << "#include <cstdint>\n";
    code_stream_ << "#include <cstdlib>\n";
    code_stream_ << "#include <string>\n";
    code_stream_ << "#include <vector>\n";
    code_stream_ << "#include <cstring>\n";
    code_stream_ << "#include <iostream>\n";
    code_stream_ << "#include <tvm/runtime/c_runtime_api.h>\n";
    code_stream_ << "#include <tvm/runtime/packed_func.h>\n";
    code_stream_ << "#include <dlpack/dlpack.h>\n";

    // cudnn header
    code_stream_ << "#include <cudnn.h>\n";
    code_stream_ << "#include <cuda_fp16.h>\n";
    code_stream_ <<
"static void HandleError(cudaError_t err, const char *file, int line) {\n"
"  if (err != cudaSuccess) {\n"
"    printf(\"%s in %s at line %d\\n\", cudaGetErrorString(err), file, line);\n"
"    exit(EXIT_FAILURE);\n"
"  }\n"
"}\n"
"#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))\n"
"#define checkCUDNN(expression) \\ \n"
"{ \\ \n"
"cudnnStatus_t status(expression); \\ \n"
"if (status != CUDNN_STATUS_SUCCESS) { \\ \n"
"  std::cerr << \"Error on line \" << __LINE__ << \": \" \\ \n"
"            << cudnnGetErrorString(status) << std::endl; \\ \n"
"  std::exit(EXIT_FAILURE); \\ \n"
"} \\ \n"
"}\n";
    code_stream_ << "typedef float d_type;\n";
    code_stream_ << "#define DATA_TYPE CUDNN_DATA_FLOAT\n";

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
