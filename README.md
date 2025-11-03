# CNN Tuning Lab

Intel GPU에 최적화된 CNN 튜닝과 하이브리드 Python/SYCL 커널 실험을 위한 레포지토리입니다. 아래 절차를 따라 oneAPI, IPEX, SYCL, C++ 확장 환경을 설정할 수 있습니다.

## 사전 준비

- Ubuntu 22.04 이상 또는 Intel GPU 드라이버가 활성화된 WSL2
- Python 3.10 이상 (conda 권장)
- CMake 3.18+, Ninja(또는 make)
- oneAPI 패키지 설치를 위한 sudo 권한

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
3. SYCL/컴파일러 사용 전 셸에서 oneAPI 환경 로드:
   ```bash
   source /opt/intel/oneapi/setvars.sh
   ```
   자주 사용할 경우 `~/.bashrc`에 위 라인을 추가합니다.
4. 설치 확인:
   ```bash
   icpx --version
   sycl-ls
   ```

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
