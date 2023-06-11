# Getting Started with OpenAI Triton

## Project Structure


Pre read
[introduction](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html)
[related-work](https://triton-lang.org/main/programming-guide/chapter-2/related-work.html)


`.github/CODEOWNERS`
Files in there are probably important.


dialect list: mlir::triton::TritonDialect, mlir::cf::ControlFlowDialect,
                  mlir::triton::gpu::TritonGPUDialect, mlir::math::MathDialect,
                  mlir::arith::ArithDialect, mlir::scf::SCFDialect,
                  mlir::gpu::GPUDialect

add_llvm_executable(triton-opt triton-opt.cpp PARTIAL_SOURCES_INTENDED)
add_llvm_executable(triton-reduce triton-reduce.cpp PARTIAL_SOURCES_INTENDED)
add_llvm_executable(triton-translate triton-translate.cpp PARTIAL_SOURCES_INTENDED)

some complexity in triton-translate.cpp

`include`
important .h and .td file, useful for understanding big picture

`include/triton/Conversion`

according to [glossary](https://mlir.llvm.org/getting_started/Glossary/#conversion), dialect to another dialect

Two conversion pass: TritonGPUToLLVM and TritonToTritonGPU

`include/triton/Dialect`

Two dialect: triton and tritonGPU

include/triton/Dialect/Triton/IR
include/triton/Dialect/TritonGPU/IR

definition of Dialect, Interfaces, Traits, Ops, Types

`include/triton/Dialect/Triton/Transforms`
`include/triton/Dialect/TritonGPU/Transforms`

Passes, eg optimization triton-combine

TODO: where is the generated document?

`include/triton/Target`

Generation target, including: TODO

`lib`

Actual implementation

`python`

python binding, skip for now TODO:

`test`

Look at the tests in the tests directory. They not only ensure the code works as expected but also serve as good examples of how the different parts of the code are **supposed to work**.

[FileCheck](https://llvm.org/docs/CommandGuide/FileCheck.html)

TODO: getting it built


[AMDGPU support for triton MLIR #1073](https://github.com/openai/triton/issues/1073)

[[Triton-MLIR] Remaining issues in migration #673](https://github.com/openai/triton/issues/673)