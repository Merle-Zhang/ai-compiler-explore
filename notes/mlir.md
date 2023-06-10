# Notes about MLIR for my self

```sh
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON \
   -DCMAKE_BUILD_TYPE=Release 
```


```sh
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_CCACHE_BUILD=ON
```
# build the standalone

follow [MLIR Getting Started](https://mlir.llvm.org/getting_started/)
```sh
installprefix=../install cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_CCACHE_BUILD=ON \
   -DLLVM_INSTALL_UTILS=ON \
   -DCMAKE_INSTALL_PREFIX=$installprefix

installprefix=../install ninja install
```

[standalone readme](../standalone/README.md)
