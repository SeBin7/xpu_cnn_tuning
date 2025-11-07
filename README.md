# CNN Tuning Lab

Intel GPU에 최적화된 CNN 튜닝과 하이브리드 Python/SYCL 커널 실험을 위한 레포지토리입니다. 아래 절차를 따라 oneAPI, IPEX, SYCL, C++ 확장 환경을 설정할 수 있습니다.

## 사전 준비

- Ubuntu 22.04 이상 또는 Intel GPU 드라이버가 활성화된 WSL2
- Python 3.10 이상 (conda 권장)
- CMake 3.18+, Ninja(또는 make)
- oneAPI 패키지 설치를 위한 sudo 권한

## 필수 시스템 패키지 설치

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build pkg-config
```

## Intel oneAPI Base Toolkit 설치

1. Intel APT 저장소 추가:
   ```bash
   wget https://apt.repos.intel.com/intel-gpg-keys/Intel_GPG_Keys.asc -O- | sudo gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg
   echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
   sudo apt update
   ```
2. BaseKit과 DPC++ 컴파일러 설치 (SYCL 런타임 포함):
   ```bash
    sudo apt install intel-basekit intel-dpcpp-cpp-compiler
   ```
3. Level Zero/oneCCL/oneDNN 런타임 패키지 설치(필요 라이브러리 확보):
   ```bash
   sudo apt install intel-level-zero-gpu level-zero-dev \
        intel-oneapi-ccl intel-oneapi-ccl-devel \
        intel-oneapi-dnnl intel-oneapi-dnnl-devel
   ```
4. SYCL/컴파일러 사용 전 셸에서 oneAPI 환경 로드:
   ```bash
   source /opt/intel/oneapi/setvars.sh
   ```
   자주 사용할 경우 `~/.bashrc`에 위 라인을 추가합니다.
5. 설치 확인:
   ```bash
   icpx --version
   sycl-ls
   ```

## oneAPI 런타임 환경 변수 설정

- `source /opt/intel/oneapi/setvars.sh`를 실행하면 `LD_LIBRARY_PATH`, `LIBRARY_PATH`, `CPATH` 등 대부분의 경로가 자동으로 잡힙니다.
- 서비스(예: systemd)나 Docker 컨테이너처럼 `setvars.sh`를 호출하기 어려운 상황에서는 아래처럼 수동으로 지정할 수 있습니다.

```bash
export ONEAPI_HOME=/opt/intel/oneapi
export LD_LIBRARY_PATH=$ONEAPI_HOME/compiler/latest/lib:$ONEAPI_HOME/compiler/latest/lib/x64:$ONEAPI_HOME/compiler/latest/lib/intel64:$ONEAPI_HOME/ccl/latest/lib:$ONEAPI_HOME/dnnl/latest/lib:$ONEAPI_HOME/level-zero/latest/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$LD_LIBRARY_PATH
export CPATH=$ONEAPI_HOME/compiler/latest/include:$ONEAPI_HOME/ccl/latest/include:$ONEAPI_HOME/dnnl/latest/include:$CPATH
```

- 확인용 체크:
  ```bash
  ldconfig -p | grep -E "libsycl|libur_loader|libccl|libdnnl"
  ```
  원하는 경로가 목록에 나타나면 환경 설정이 완료된 것입니다.

## Python 환경 및 Intel Extension for PyTorch(IPEX)

1. 가상환경 생성 및 활성화(venv):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   ```
   *(Conda를 선호한다면 `conda create -n cnn-tuning python=3.10` → `conda activate cnn-tuning` 절차를 사용해도 됩니다.)*
2. PyTorch 및 기본 의존성 설치(필요 시 Torch XPU 빌드 선택):
   ```bash
   pip install -r requirements.txt
   ```
3. IPEX(XPU) 설치: (nightly가 필요하면 URL 변경)
   ```bash
   pip install --upgrade intel-extension-for-pytorch[xpu] \
       -f https://intel-extensions-for-pytorch.github.io/get_started/whl/stable/
   ```
4. Intel GPU 실행 시 추천 런타임 변수:
   ```bash
   export SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=1
   export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
   ```
5. Python 스택 확인:
   ```python
   python -c "import torch, intel_extension_for_pytorch as ipex; print(torch.__version__, ipex.__version__); print(torch.xpu.is_available())"
   ```

## SYCL / DPC++ 참고 사항

- `icpx -fsycl` 또는 `dpcpp`로 단독 SYCL 코드를 컴파일할 수 있습니다.
- 다중 GPU 환경에서는 `SYCL_DEVICE_FILTER` 혹은 `ONEAPI_DEVICE_SELECTOR`로 타깃 디바이스를 선택하세요.
- `SYCL_PI_TRACE=1`을 설정하면 백엔드 라우팅을 디버깅하기 쉽습니다.

## C++/SYCL 커스텀 연산 빌드

1. `setvars.sh`가 로드된 셸과 PyTorch 설치된 Python 환경을 동시에 활성화합니다.
2. 한 번만 Torch CMake 경로를 환경 변수로 지정:
   ```bash
   export CMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
   ```
3. CMake 설정 및 빌드(Ninja 예시):
   ```bash
   cmake -S . -B build/xpu -G Ninja \
         -DCMAKE_CXX_COMPILER=icpx \
         -DCMAKE_CXX_FLAGS="--gcc-toolchain=$(dirname $(dirname $(which gcc)))" \
         -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}"
   cmake --build build/xpu
   ```
   결과 공유 라이브러리(예: `custom_conv3x3_bn_relu_xpu.so`)는 `build/xpu`에 생성되며, Python에서 `torch.ops.load_library`로 로드할 수 있습니다.
4. 커널 튜닝 파라미터 조정:
   ```bash
   cmake --build build/xpu --target clean
   cmake -S . -B build/xpu -G Ninja \
         -DTS=16 -DCI_STEP=4 -DKO_STEP=2 -DUSE_HALF_TILE=0 -DSUBGROUP_HINT=16
   cmake --build build/xpu
   ```

## Python 워크플로 실행

- 학습 스크립트:
  ```bash
  ./scripts/train_xpu_overlap.sh
  ```
- 추론 스크립트:
  ```bash
  ./scripts/infer_cifar10.sh
  ```

각 스크립트를 실행하기 전에 `setvars.sh`와 Python 가상환경이 모두 활성화되어 있어야 oneAPI 런타임과 IPEX가 제대로 동작합니다.

## “튜닝했는데 왜 더 느려졌지?” 체크 리스트

`train_xpu_overlap.py`에 `--diag-fused`(체크리스트 출력)와 `--warmup-fused <N>`(사전 JIT 워밍업) 스위치를 추가했습니다. 대표적으로 다음 네 축을 점검하세요.

1. **메모리 포맷** – 모델/배치가 `channels_last`(NHWC)로 유지되지 않으면 NCHW↔NHWC 왕복 비용 때문에 SYCL 커널 내부 인덱싱이 미스코얼레스팅 됩니다. `--channels-last on` 또는 config의 `misc.channels_last=true`를 권장합니다.
2. **글로벌 메모리 접근 패턴** – `XPU_FUSED_LAYOUT=direct`/`im2row` 경로는 KO_STEP과 weight layout에 민감합니다. `--diag-fused`는 현재 `XPU_FUSED_LAYOUT`, `XPU_FUSED_KERNEL`, `XPU_FUSED_KO_STEP` 값을 에코하여 vec4 접근/채널 블로킹이 맞는지 확인할 수 있게 합니다.
   - direct 경로는 입력이 `channels_last`일 때 자동으로 NHWC → CI16 패딩 + HWCN 블록 재배치를 수행하고, 16채널 벡터 FMA를 사용합니다. 따라서 모델 전체를 `channels_last`로 유지하고 KO_STEP이 16 이상의 배수로 잡혀야 벡터 경로가 발동합니다. stride=1과 stride=2에 대해 각각 `XPU_FUSED_KO_STEP` / `XPU_FUSED_KO_STEP_S2`를 별도로 지정할 수 있습니다.
   - stride=2 계층은 기본값(auto)일 때 자동으로 `im2row` 경로로 라우팅되어 다운샘플링 시 발생하는 비효율적인 NHWC gather를 피합니다. 필요하다면 `XPU_FUSED_TILE_S2`, `XPU_FUSED_KO_STEP_S2`, `XPU_FUSED_MICRO_N_S2`로 전용 파라미터를 설정하세요.
3. **SLM/Winograd 타일** – 타일이 H/W 대비 과도하게 크면 halo 채움과 barrier가 증가하고 occupancy가 떨어집니다. `XPU_FUSED_TILE_S1/S2`를 직접 지정하거나(예: `8x8`) 비워두면 자동 휴리스틱이 Ho/Wo에 맞춰 조정됩니다.
4. **스트림 동기화 & 첫 스텝 컴파일** – 첫 step은 커널 컴파일/메모리 풀 워밍업으로 느려질 수 있습니다. `--warmup-fused 2`처럼 probe를 몇 번 실행해 JIT/USM allocator를 예열하면 이후 측정이 안정적입니다.

로그에서 oneDNN(ATen) 경로가 수 ms인데 SYCL 경로가 수천 ms라면 위 순서대로 원인을 제거하면서 비교해야 공정한 벤치마킹이 가능합니다.

### 진단 / 계측 팁

- `ENABLE_VEC16_STATS=1` 환경변수를 주고 학습 스크립트를 실행하면, direct 경로 NHWC 커널이 16채널 벡터 경로를 몇 번 탔는지 로그(`aligned`, `fallback`)로 확인할 수 있습니다.
- `TORCH_PROFILER=/tmp/xpu_trace`와 같이 출력 디렉터리를 지정하면 PyTorch profiler가 CPU/XPU trace를 기록합니다. 실행 후 `tensorboard --logdir /tmp/xpu_trace`로 레이어별 시간을 확인할 수 있습니다.

## 주요 패키지 및 라이브러리 역할

- **Intel oneAPI DPC++/C++ Compiler (`libsycl.so.8`, `icpx`, `dpcpp`)**  
  SYCL 표준 기반의 DPC++ 컴파일러를 제공하며, GPU/CPU/XPU 대상 SYCL 프로그램을 빌드할 때 필요합니다. `libsycl.so.8`은 런타임에서 SYCL 디바이스와 상호작용하는 핵심 공유 라이브러리입니다.

- **Level Zero Loader (`libur_loader.so.0`)**  
  Intel GPU용 Level Zero 런타임의 입력 지점입니다. SYCL 또는 IPEX가 Level Zero 백엔드를 사용할 때 디바이스를 초기화하고 커맨드를 제출하도록 도와줍니다.

- **oneCCL (`libccl.so.1`)**  
  분산 학습 또는 멀티 GPU 환경에서 collective 통신을 가속합니다. PyTorch Distributed + IPEX 조합에서 AllReduce/AllGather 등의 연산을 최적화할 때 연결됩니다.

- **oneDNN (`libdnnl.so.3`)**  
  딥러닝 연산에 특화된 고성능 수치 라이브러리입니다. IPEX 및 PyTorch가 Convolution, BatchNorm, GEMM 같은 핵심 커널을 XPU에서 실행할 때 활용합니다.
