# Exploring Triton with MLIR: A Deep Dive into How Triton Works

- [Exploring Triton with MLIR: A Deep Dive into How Triton Works](#exploring-triton-with-mlir-a-deep-dive-into-how-triton-works)
  - [What is MLIR and Why It Matters?](#what-is-mlir-and-why-it-matters)
  - [Triton's Compilation Flow](#tritons-compilation-flow)
  - [A Closer Look at IR Transformations](#a-closer-look-at-ir-transformations)
  - [MLIR Passes and Their Roles](#mlir-passes-and-their-roles)
  - [How to Explore Triton Passes Yourself](#how-to-explore-triton-passes-yourself)
    - [Example in MLIR:](#example-in-mlir)
    - [Example in Triton](#example-in-triton)
    - [Example in third-party NVIDIA](#example-in-third-party-nvidia)
  - [Conclusion](#conclusion)

Triton, an OpenAI project, has recently gained a lot of attention for its CUDA-free inference capabilities for large language models, as discussed in [this PyTorch article](https://pytorch.org/blog/cuda-free-inference-for-llms/). If you're curious about how Triton achieves this, you're in the right place. This blog explores Triton's inner workings from the perspective of MLIR (Multi-Level Intermediate Representation).

Triton leverages LLVM’s MLIR infrastructure for many of its transformations and optimizations. In this article, we'll dive into how Triton manipulates Intermediate Representation (IR) using various passes and how it maps to different hardware backends like NVIDIA, AMD, and Intel.

## What is MLIR and Why It Matters?

MLIR is an infrastructure for building reusable and extensible compilers. It is a part of the LLVM project, and it enables transformation of code through various "passes." Passes can either optimize code or transform it from one abstraction level to another. For example, Triton takes high-level code and compiles it down to GPU-level instructions, with MLIR facilitating much of the process. 

LLVM (Low-Level Virtual Machine) is the compiler infrastructure that comes into play at the lower level, generating hardware-specific code such as PTX (NVIDIA's parallel thread execution).

In short:

* MLIR helps translate between abstraction levels or optimize within the same level.
* LLVM performs low-level optimizations and converts the code to machine instructions.

Now, let's dive into how Triton uses this infrastructure.

## Triton's Compilation Flow

Triton utilizes MLIR for transforming and optimizing its intermediate representation (IR). The following diagram from [Triton’s High-Level System Architecture](https://openai.com/research/triton) gives a rough overview of how Triton’s compilation pipeline works:

```mermaid
graph LR
    PythonAST["Python AST"]
    PythonAST --> TritonIR
    TritonIR["Triton-IR<br>(High-level IR)"]
    TritonIR --> LLVMIR
    LLVMIR["LLVM-IR"]
    LLVMIR --> PTX
    PTX["PTX<br>(for NVIDIA GPUs)"]
```

Different backends may have slight variations in this flow. For example, the compilation pipeline varies across NVIDIA, AMD, and Intel backends. Here’s a quick comparison:

* [NVIDIA Backend](https://github.com/triton-lang/triton/blob/c99c2148f363e4806e02300d302ae0b52bb19388/third_party/nvidia/backend/compiler.py#L369C9-L369C19): `ttir → ttgir → llir → ptx → cubin`
* [AMD Backend](https://github.com/triton-lang/triton/blob/c99c2148f363e4806e02300d302ae0b52bb19388/third_party/amd/backend/compiler.py#L298): `ttir → ttgir → llir → amdgcn → hsaco`
* [Intel Backend](https://github.com/intel/intel-xpu-backend-for-triton/blob/c39e89695507958378a2e3c8f114a82c47228257/third_party/intel/backend/compiler.py#L300): `ttir → ttgir → llir → spv`

These steps represent transformations from Triton-specific IR (Triton-IR and TritonGPU-IR) to target-specific assembly languages like PTX (NVIDIA), AMDGCN (AMD), and SPIR-V (Intel).

## A Closer Look at IR Transformations

To understand what happens at each stage of the compilation process, let's take a hands-on approach. The following script runs a simple Triton program and dumps the IR transformations that occur at each stage.

```python
import os
import subprocess
import urllib.request

# Enable MLIR IR dumping, you can enable other options to dump more info
os.environ['TORCH_COMPILE_DEBUG'] = '0'
os.environ['MLIR_ENABLE_DUMP'] = '1'
os.environ['LLVM_IR_ENABLE_DUMP'] = '0'

# Download example Triton program
url = 'https://triton-lang.org/main/_downloads/62d97d49a32414049819dd8bb8378080/01-vector-add.py'
urllib.request.urlretrieve(url, 'test.py')

# Run the script and capture the IR dump
p = subprocess.run(["python3", "test.py"], capture_output=True, text=True)
for line in p.stderr.splitlines():
    if 'IR Dump' in line:
        print(line)
```

Running the above code on a machine with an NVIDIA GPU (like in Google Colab) will print a series of IR dumps that reveal the transformations happening at each stage. These transformations are known as passes, and they are applied to the code in stages to optimize it and prepare it for hardware execution.

Example output might look like this:

```less
IR Dump Before Inliner (inline) ('builtin.module' operation)
IR Dump Before Canonicalizer (canonicalize) ('tt.func' operation: @add_kernel_0d1d2d3de)
...
IR Dump Before ConvertTritonToTritonGPU (convert-triton-to-tritongpu)
IR Dump Before TritonGPUCoalesce (tritongpu-coalesce)
...
IR Dump After ConvertTritonGPUToLLVM (convert-triton-gpu-to-llvm)
IR Dump After ConvertNVGPUToLLVM (convert-nv-gpu-to-llvm)
```

## MLIR Passes and Their Roles

Triton uses a series of MLIR passes to perform various optimizations and transformations. Below, I've listed the passes in Triton and linked them to their source code for further exploration:


<table>
  <tr>
    <td rowspan="8" colspan="1"><a href="https://github.com/triton-lang/triton/blob/c99c2148f363e4806e02300d302ae0b52bb19388/include/triton/Dialect/Triton/IR/TritonDialect.td#L6">Triton-IR</a><br>(<a href="https://github.com/triton-lang/triton/blob/c99c2148f363e4806e02300d302ae0b52bb19388/third_party/nvidia/backend/compiler.py#L178">make_ttir</a>)</td>
    <td><a href="https://github.com/llvm/llvm-project/blob/1e4e1ceeebb82dd86d0dc95eb1074b7326b50db3/mlir/lib/Transforms/InlinerPass.cpp#L194C29-L194C46">Inliner</a></td>
  </tr>
  <tr>
    <td>Canonicalizer</td>
  </tr>
  <tr>
    <td><a href="https://github.com/triton-lang/triton/blob/c99c2148f363e4806e02300d302ae0b52bb19388/lib/Dialect/Triton/Transforms/Combine.cpp#L244C29-L244C49">TritonCombineOps</a></td>
  </tr>
  <tr>
    <td>Canonicalizer</td>
  </tr>
  <tr>
    <td><a href="https://github.com/triton-lang/triton/blob/c99c2148f363e4806e02300d302ae0b52bb19388/lib/Dialect/Triton/Transforms/ReorderBroadcast.cpp#L243C29-L243C55">TritonReorderBroadcast</a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/llvm/llvm-project/blob/a20a97382fa14b62e7b3c9884ffddcd500124cef/mlir/lib/Transforms/CSE.cpp#L416">CSE</a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/llvm/llvm-project/blob/1e4e1ceeebb82dd86d0dc95eb1074b7326b50db3/mlir/lib/Transforms/LoopInvariantCodeMotion.cpp#L60C29-L60C62">LoopInvariantCodeMotion</a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/llvm/llvm-project/blob/1e4e1ceeebb82dd86d0dc95eb1074b7326b50db3/mlir/lib/Transforms/SymbolDCE.cpp#L149C29-L149C48">SymbolDCE</a></td>
  </tr>
  <tr>
    <td rowspan="23" colspan="1"><a href="https://github.com/triton-lang/triton/blob/c99c2148f363e4806e02300d302ae0b52bb19388/include/triton/Dialect/TritonGPU/IR/TritonGPUDialect.td#L6C5-L6C22">TritonGPU-IR<br>(<a href="https://github.com/triton-lang/triton/blob/c99c2148f363e4806e02300d302ae0b52bb19388/third_party/nvidia/backend/compiler.py#L194">make_ttgir</a>)</a></td>
    <td><a href="https://github.com/triton-lang/triton/blob/c99c2148f363e4806e02300d302ae0b52bb19388/lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp#L821C15-L821C49">ConvertTritonToTritonGPU</a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/triton-lang/triton/blob/c99c2148f363e4806e02300d302ae0b52bb19388/include/triton/Dialect/TritonGPU/Transforms/Passes.td#L94C5-L94C22">TritonGPUCoalesce</a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/triton-lang/triton/blob/c99c2148f363e4806e02300d302ae0b52bb19388/lib/Dialect/TritonNvidiaGPU/Transforms/PlanCTA.cpp#L1026C7-L1026C39">TritonGPUPlanCTAPass</a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/triton-lang/triton/blob/c99c2148f363e4806e02300d302ae0b52bb19388/lib/Dialect/Triton/Transforms/RewriteTensorPointer.cpp#L560C31-L560C61">TritonGPURewriteTensorPointer</a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/triton-lang/triton/blob/c99c2148f363e4806e02300d302ae0b52bb19388/lib/Dialect/TritonNvidiaGPU/Transforms/PlanCTA.cpp#L1026C7-L1026C39">TritonGPUPlanCTAPass</a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/triton-lang/triton/blob/c99c2148f363e4806e02300d302ae0b52bb19388/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp#L1207C7-L1207C43">TritonGPURemoveLayoutConversions</a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/triton-lang/triton/blob/c99c2148f363e4806e02300d302ae0b52bb19388/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp#L386C7-L386C36">TritonGPUAccelerateMatmul</a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/triton-lang/triton/blob/c99c2148f363e4806e02300d302ae0b52bb19388/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp#L1207C7-L1207C43">TritonGPURemoveLayoutConversions</a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/triton-lang/triton/blob/c99c2148f363e4806e02300d302ae0b52bb19388/lib/Dialect/TritonGPU/Transforms/OptimizeDotOperands.cpp#L308C7-L308C39">TritonGPUOptimizeDotOperands</a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/llvm/llvm-project/blob/a20a97382fa14b62e7b3c9884ffddcd500124cef/mlir/lib/Transforms/CSE.cpp#L416">CSE</a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/triton-lang/triton/blob/c99c2148f363e4806e02300d302ae0b52bb19388/include/triton/Dialect/TritonGPU/Transforms/Passes.td#L6C5-L6C22">TritonGPUPipeline</a></td>
  </tr>
  <tr>
    <td><a href="triton-nvidia-gpu-materialize-load-store">MaterializeLoadStore</a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/triton-lang/triton/blob/c99c2148f363e4806e02300d302ae0b52bb19388/lib/Dialect/TritonGPU/Transforms/Prefetch.cpp#L414C8-L414C20">TritonGPUPrefetch</a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/triton-lang/triton/blob/c99c2148f363e4806e02300d302ae0b52bb19388/lib/Dialect/TritonGPU/Transforms/OptimizeDotOperands.cpp#L308C7-L308C39">TritonGPUOptimizeDotOperands</a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/triton-lang/triton/blob/c99c2148f363e4806e02300d302ae0b52bb19388/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp#L1207C7-L1207C43">TritonGPURemoveLayoutConversions</a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/triton-lang/triton/blob/c99c2148f363e4806e02300d302ae0b52bb19388/include/triton/Dialect/TritonGPU/Transforms/Passes.td#L151C5-L151C35">TritonGPUDecomposeConversions</a></td>
  </tr>
  <tr>
    <td><a href="triton-nvidia-gpu-ws-fixup-missing-attrs">TritonGPUWSFixupMissingAttrs</a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/triton-lang/triton/blob/c99c2148f363e4806e02300d302ae0b52bb19388/lib/Dialect/TritonGPU/Transforms/ReorderInstructions.cpp#L42C7-L42C39">TritonGPUReorderInstructions</a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/llvm/llvm-project/blob/a20a97382fa14b62e7b3c9884ffddcd500124cef/mlir/lib/Transforms/CSE.cpp#L416">CSE</a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/llvm/llvm-project/blob/1e4e1ceeebb82dd86d0dc95eb1074b7326b50db3/mlir/lib/Transforms/SymbolDCE.cpp#L149C29-L149C48">SymbolDCE</a></td>
  </tr>
  <tr>
    <td><a href="triton-nvidia-gpu-ws-fixup-missing-attrs">TritonGPUWSFixupMissingAttrs</a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/triton-lang/triton/blob/c99c2148f363e4806e02300d302ae0b52bb19388/lib/Dialect/TritonGPU/Transforms/OptimizeThreadLocality.cpp#L95C7-L95C42">TritonGPUOptimizeThreadLocality</a></td>
  </tr>
  <tr>
    <td>Canonicalizer</td>
  </tr>
  <tr>
    <td rowspan="4" colspan="1"><a href="https://github.com/llvm/llvm-project/blob/fa478bd275f473861f6d4df4896244a730d4853f/mlir/include/mlir/Dialect/LLVMIR/LLVMDialect.td#L14C5-L14C17">LLVM-IR</a><br>(<a href="https://github.com/triton-lang/triton/blob/c99c2148f363e4806e02300d302ae0b52bb19388/third_party/nvidia/backend/compiler.py#L241">make_ttgir</a>)</td>
    <td><a href="https://github.com/triton-lang/triton/blob/c99c2148f363e4806e02300d302ae0b52bb19388/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp#L224">ConvertTritonGPUToLLVM</a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/triton-lang/triton/blob/c99c2148f363e4806e02300d302ae0b52bb19388/third_party/nvidia/lib/NVGPUToLLVM/NVGPUToLLVMPass.cpp#L563">ConvertNVGPUToLLVM</a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/llvm/llvm-project/blob/a20a97382fa14b62e7b3c9884ffddcd500124cef/mlir/lib/Transforms/CSE.cpp#L416">CSE</a></td>
  </tr>
  <tr>
    <td><a href="enable-line-info">LLVMDIScope</a></td>
  </tr>
</table>

Each of these passes performs a specialized task. For example:

* **Inliner**: Reduces function call overhead by embedding function bodies inline.
* **Canonicalizer**: Simplifies the IR by merging duplicate expressions and eliminating unnecessary code.
* **ConvertTritonToTritonGPU**: Translates high-level Triton operations into operations that can be executed on a GPU.

## How to Explore Triton Passes Yourself

If you want to dive deeper into how each pass works, Triton’s codebase and MLIR provide great resources for exploration:

1. Search for pass constructor names.
2. Locate their `.td` definitions and read short descriptions of each pass.
3. Check out the LIT tests for examples of how passes are applied.

### Example in MLIR:

* Search for [`createCSEPass`](https://github.com/llvm/llvm-project/blob/a20a97382fa14b62e7b3c9884ffddcd500124cef/mlir/lib/Transforms/CSE.cpp#L416)
* Find [`let constructor = "mlir::createCSEPass()";`](https://github.com/llvm/llvm-project/blob/55ec015c4dc9bb7c39b313c4662daec3b3c6043b/mlir/include/mlir/Transforms/Passes.td#L89C3-L89C45)
* Search for [`cse`](https://github.com/llvm/llvm-project/blob/55ec015c4dc9bb7c39b313c4662daec3b3c6043b/mlir/include/mlir/Transforms/Passes.td#L80C17-L80C20)
* See the test usage in LIT test [`mlir-opt -allow-unregistered-dialect %s -pass-pipeline='builtin.module(func.func(cse))'`](https://github.com/llvm/llvm-project/blob/1e4e1ceeebb82dd86d0dc95eb1074b7326b50db3/mlir/test/Transforms/cse.mlir#L1C9-L1C96)

### Example in Triton

* Search for [`createCombineOpsPass`](https://github.com/triton-lang/triton/blob/c99c2148f363e4806e02300d302ae0b52bb19388/lib/Dialect/Triton/Transforms/Combine.cpp#L244C29-L244C49)
* Find [`let constructor = "mlir::triton::createCombineOpsPass()";`](https://github.com/triton-lang/triton/blob/c99c2148f363e4806e02300d302ae0b52bb19388/include/triton/Dialect/Triton/Transforms/Passes.td#L22C3-L22C60)
* Search for [`triton-combine`](https://github.com/triton-lang/triton/blob/c99c2148f363e4806e02300d302ae0b52bb19388/include/triton/Dialect/Triton/Transforms/Passes.td#L6C41-L6C55)
* See the test usage in LIT test [`triton-opt %s -split-input-file -canonicalize -triton-combine`](https://github.com/triton-lang/triton/blob/c99c2148f363e4806e02300d302ae0b52bb19388/test/Triton/combine.mlir#L1C9-L1C70)

### Example in third-party NVIDIA

* Search for [`createConvertTritonGPUToLLVMPass`](https://github.com/triton-lang/triton/blob/c99c2148f363e4806e02300d302ae0b52bb19388/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp#L224)
* Find [`let constructor = "mlir::triton::createConvertNVGPUToLLVMPass()";`](https://github.com/triton-lang/triton/blob/c99c2148f363e4806e02300d302ae0b52bb19388/third_party/nvidia/include/NVGPUToLLVM/Passes.td#L12C5-L12C70)
* Search for [`convert-nv-gpu-to-llvm`](https://github.com/triton-lang/triton/blob/c99c2148f363e4806e02300d302ae0b52bb19388/third_party/nvidia/include/NVGPUToLLVM/Passes.td#L7C32-L7C54)
* See the test usage in LIT test [`triton-opt %s --convert-nv-gpu-to-llvm`](https://github.com/triton-lang/triton/blob/c99c2148f363e4806e02300d302ae0b52bb19388/test/Conversion/nvgpu_to_llvm.mlir#L1)

## Conclusion

Understanding the various stages in Triton’s compilation pipeline offers valuable insights into how high-level Python code is optimized and transformed for execution on different hardware backends. By examining the IR transformations using MLIR passes, you gain a clearer picture of how Triton works internally.

This post barely scratches the surface, but if you're curious to learn more, I encourage you to dive into the Triton and MLIR source code and explore the vast optimization infrastructure they provide. Happy hacking!

## References and Further Reading

Throughout this blog, I have referenced several insightful articles and blogs to provide more context and detailed explanations. For those interested in diving deeper into Triton and MLIR, here are the resources I found helpful:

1. [A deep dive into Triton and MLIR on Zhihu](https://zhuanlan.zhihu.com/p/695255185)
2. [Superjomn's post about Triton with MLIR](https://superjomn.github.io/posts/triton-mlir-publish/)
3. [Exploring OpenAI Triton MLIR - Chapter 0](http://giantpandacv.com/project/%E9%83%A8%E7%BD%B2%E4%BC%98%E5%8C%96/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%BC%96%E8%AF%91%E5%99%A8/OpenAI%20Triton%20MLIR%20%E7%AC%AC%E9%9B%B6%E7%AB%A0%20%E6%BA%90%E7%A0%81%E7%BC%96%E8%AF%91/)
4. [An Introduction to OpenAI Triton FusedAttention](http://giantpandacv.com/project/CUDA/%E3%80%90BBuf%E7%9A%84CUDA%E7%AC%94%E8%AE%B0%E3%80%91%E5%8D%81%E4%BA%94%EF%BC%8COpenAI%20Triton%E5%85%A5%E9%97%A8%E7%AC%94%E8%AE%B0%E4%B8%89%20FusedAttention/)
5. [A Zhihu Discussion on Triton and Its Applications](https://www.zhihu.com/question/622685131)
