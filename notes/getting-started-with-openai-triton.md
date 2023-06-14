# Getting Started with OpenAI Triton

> WIP

<!-- vscode-markdown-toc -->
* 1. [Build the Triton Project](#BuildtheTritonProject)
	* 1.1. [Clone LLVM and Triton](#CloneLLVMandTriton)
	* 1.2. [Build LLVM and Triton](#BuildLLVMandTriton)
	* 1.3. [Build TableGen docs](#BuildTableGendocs)
	* 1.4. [Regression Test](#RegressionTest)
	* 1.5. [Setup CLion](#SetupCLion)
* 2. [Project Structure](#ProjectStructure)
	* 2.1. [`.github`](#.github)
	* 2.2. [`bin`](#bin)
		* 2.2.1. [`bin/CMakeLists.txt`](#binCMakeLists.txt)
		* 2.2.2. [`bin/RegisterTritonDialects.h`](#binRegisterTritonDialects.h)
	* 2.3. [`include`](#include)
		* 2.3.1. [`include/triton/Conversion`](#includetritonConversion)
		* 2.3.2. [`include/triton/Dialect`](#includetritonDialect)
		* 2.3.3. [`include/triton/Dialect/{dialect}/IR`](#includetritonDialectdialectIR)
		* 2.3.4. [`include/triton/Dialect/{dialect}/Transforms`](#includetritonDialectdialectTransforms)
		* 2.3.5. [`include/triton/Target`](#includetritonTarget)
	* 2.4. [`lib`](#lib)
	* 2.5. [`python`](#python)
	* 2.6. [`test`](#test)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

##  1. <a name='BuildtheTritonProject'></a>Build the Triton Project

##### References:

1. [Getting the Source Code and Building LLVM](https://llvm.org/docs/GettingStarted.html#getting-the-source-code-and-building-llvm)
2. [Stand-alone Builds](https://llvm.org/docs/GettingStarted.html#stand-alone-builds)
3. [Getting Started - MLIR](https://mlir.llvm.org/getting_started/)

###  1.1. <a name='CloneLLVMandTriton'></a>Clone LLVM and Triton

```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout c5dede880d17 # change this to the commit id you found in triton/python/setup.py
cd ..
git clone https://github.com/openai/triton.git
```
We cannot do a shallow clone for LLVM here because we need a specific commit of LLVM. I would have the following error when building LLVM and Triton with the latest commit:

```log
CMake Error at /path/to/llvm/install/lib/cmake/llvm/AddLLVM.cmake:556 (add_dependencies):
  The dependency target "MLIRGPUOps" of target "obj.TritonGPUToLLVM" does not
  exist.
Call Stack (most recent call first):
  /path/to/llvm/install/lib/cmake/mlir/AddMLIR.cmake:303 (llvm_add_library)
  /path/to/llvm/install/lib/cmake/mlir/AddMLIR.cmake:588 (add_mlir_library)
  lib/Conversion/TritonGPUToLLVM/CMakeLists.txt:1 (add_mlir_conversion_library)


CMake Error at /path/to/llvm/install/lib/cmake/llvm/AddLLVM.cmake:556 (add_dependencies):
  The dependency target "MLIRGPUOps" of target "obj.TritonGPUIR" does not
  exist.
Call Stack (most recent call first):
  /path/to/llvm/install/lib/cmake/mlir/AddMLIR.cmake:303 (llvm_add_library)
  /path/to/llvm/install/lib/cmake/mlir/AddMLIR.cmake:582 (add_mlir_library)
  lib/Dialect/TritonGPU/IR/CMakeLists.txt:1 (add_mlir_dialect_library)
```

If we check the `setup.py` in Triton project, it [specifies](https://github.com/openai/triton/blob/main/python/setup.py#L82) the commit id of LLVM. For the current Triton HEAD (`b27a91a1137e`), it requires LLVM `c5dede880d17`.

> TODO: Is there a way to shallow clone a project with a specific commit id? 

###  1.2. <a name='BuildLLVMandTriton'></a>Build LLVM and Triton

Most of the required software are pre-installed with the Ubuntu 22.04.2 LTS. Some software that I manually installed are:

* ninja
* clang
* lld
* ccache

Then you can use this script to build both LLVM and Triton. Please see the references above for the details of each option.

```bash
#!/bin/sh

# Copy and amend the template from
# https://llvm.org/docs/GettingStarted.html#stand-alone-builds

build_llvm=/path/to/your/build-llvm # set your own path
build_triton=/path/to/your/build-triton # set your own path
installprefix=/path/to/your/install # set your own path
llvm=/path/to/your/llvm-project # set your own path
triton=/path/to/your/triton # set your own path
mkdir -p $build_llvm
mkdir -p $build_triton
mkdir -p $installprefix

cmake -G Ninja -S $llvm/llvm -B $build_llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON \ # optional
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_CCACHE_BUILD=ON \ # optional
   -DLLVM_INSTALL_UTILS=ON \
   -DCMAKE_INSTALL_PREFIX=$installprefix

ninja -C $build_llvm install

cmake -G Ninja -S $triton -B $build_triton \
    -DMLIR_DIR=$installprefix/lib/cmake/mlir \
    -DLLVM_EXTERNAL_LIT=$build_llvm/bin/llvm-lit

ninja -C $build_triton

# ninja -C $build_triton mlir-doc

# ninja -C $build_triton check-triton-lit-tests
```

###  1.3. <a name='BuildTableGendocs'></a>Build TableGen docs

MLIR uses a powerful [declaratively specification mechanism](https://mlir.llvm.org/docs/DefiningDialects/) via [TableGen](https://llvm.org/docs/TableGen/index.html). Along with generating source code, it can also generate `.md` docs.

For some reason, Triton project doesn't use this feature, but we can enable this by adding commands in cmake files.

One example is, in `triton/include/triton/Dialect/Triton/IR/CMakeLists.txt`, you can add the following:

```cmake
add_mlir_doc(TritonAttrDefs TritonAttrDefs Triton/IR/ -gen-attrdef-doc)
add_mlir_doc(TritonDialect TritonDialect Triton/IR/ -gen-dialect-doc)
add_mlir_doc(TritonInterfaces TritonInterfaces Triton/IR/ -gen-op-interface-docs)
add_mlir_doc(TritonOps TritonOps Triton/IR/ -gen-op-doc)
add_mlir_doc(TritonTypes TritonTypes Triton/IR/ -gen-typedef-doc)
```

The definition of `add_mlir_doc` is in `llvm-project/mlir/cmake/modules/AddMLIR.cmake`.

You can also use the `--help` of TableGen to list all the available args:

```bash
$installprefix/bin/mlir-tblgen --help | grep doc
```

We also need to add `set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})` to `triton/CMakeLists.txt`

Then you can use `ninja -C $build_triton mlir-doc` as shown in the above script. The generated docs will be under `$build_triton/docs`

Feel free to look at the example commit for more information: [29d4f90e0460](https://github.com/Merle-Zhang/triton/commit/29d4f90e0460bb5ba578f1fea3ad0b0d71eca5fe)

###  1.4. <a name='RegressionTest'></a>Regression Test

Like the `check-standalone` for `standalone` example, triton also have its regression test, you can run it with `ninja -C $build_triton check-triton-lit-tests` as shown in the above script. However some of the tests failed on the current HEAD.


###  1.5. <a name='SetupCLion'></a>Setup CLion

> TODO: how to setup CLion


##  2. <a name='ProjectStructure'></a>Project Structure

Some helpful pre-read:

* [introduction](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html)
* [related-work](https://triton-lang.org/main/programming-guide/chapter-2/related-work.html)

###  2.1. <a name='.github'></a>`.github`

`.github` normally contains some files defining the workflows, which is not what we care about at this moment. But there's one interesting file [`CODEOWNERS`](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners), listing some files which should be important given that they are specifically listed in the `CODEOWNERS`

We can use this file as the Table of Content of the important files in Triton project.

* Analyses
  * Alias analysis
  * Allocation analysis
  * Membar analysis
  * AxisInfo analysis
  * Utilities
* Dialects
  * Pipeline pass
  * Prefetch pass
  * Coalesce pass
  * Layout simplification pass
* Conversions
  * TritonGPUToLLVM
  * TritonToTritonGPU
* Targets
  * LLVMIR
  * PTX

###  2.2. <a name='bin'></a>`bin`

####  2.2.1. <a name='binCMakeLists.txt'></a>`bin/CMakeLists.txt` 

lists three executables of Triton project:

* `triton-opt`
* `triton-reduce`
* `triton-translate`

####  2.2.2. <a name='binRegisterTritonDialects.h'></a>`bin/RegisterTritonDialects.h`

lists the dialects that will be used in Triton:

* `mlir::triton::TritonDialect`
* `mlir::cf::ControlFlowDialect`
* `mlir::triton::gpu::TritonGPUDialect`
* `mlir::math::MathDialect`
* `mlir::arith::ArithDialect`
* `mlir::scf::SCFDialect`
* `mlir::gpu::GPUDialect`

> TODO: There are some complexities in `triton-translate.cpp` which might worth to investigate.

###  2.3. <a name='include'></a>`include`

Like the standalone and toy example, this directory contains important `.h` and `.td` files which are important for understanding the big picture.

####  2.3.1. <a name='includetritonConversion'></a>`include/triton/Conversion`

As defined in [glossary](https://mlir.llvm.org/getting_started/Glossary/#conversion), conversion refers o the process of one dialect converting to another dialect.

Two conversion passes:

* TritonGPUToLLVM
* TritonToTritonGPU

####  2.3.2. <a name='includetritonDialect'></a>`include/triton/Dialect`

Two dialect: 

* triton
* tritonGPU

####  2.3.3. <a name='includetritonDialectdialectIR'></a>`include/triton/Dialect/{dialect}/IR`

Definitions of Dialect, Interfaces, Traits, Ops, Types.

####  2.3.4. <a name='includetritonDialectdialectTransforms'></a>`include/triton/Dialect/{dialect}/Transforms`

Passes, eg optimization like triton-combine


####  2.3.5. <a name='includetritonTarget'></a>`include/triton/Target`

Generation targets:

* AMDGCN
* HSACO (AMD)
* LLVMIR
* PTX (Nvidia)

###  2.4. <a name='lib'></a>`lib`

The implementation details for the definitions in `include`.

###  2.5. <a name='python'></a>`python`

python binding, skip for now 

> TODO: 

###  2.6. <a name='test'></a>`test`

Also a good idea to look at the tests in the tests directory. They not only ensure the code works as expected but also serve as good examples of how the different parts of the code are **supposed to work**.

Might need to read about [FileCheck](https://llvm.org/docs/CommandGuide/FileCheck.html) for better understanding of the test format.

---

[AMDGPU support for triton MLIR #1073](https://github.com/openai/triton/issues/1073)

[[Triton-MLIR] Remaining issues in migration #673](https://github.com/openai/triton/issues/673)