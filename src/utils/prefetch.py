import torch


class DevicePrefetcher:
    """
    호스트→디바이스 전송과 연산을 겹치기 위한 간단한 디바이스 프리페처입니다.
    CUDA와 XPU 백엔드 모두 CUDA와 호환되는 스트림 API를 제공하므로 동일하게 동작합니다.
    """

    def __init__(self, loader, device, use_channels_last=False):
        self.loader = iter(loader)
        self.device = device
        self.length = len(loader)
        self.use_channels_last = use_channels_last
        self.mem_format = torch.channels_last if use_channels_last else torch.contiguous_format

        self.backend = None
        if device.type == "cuda":
            self.backend = torch.cuda
        elif device.type == "xpu" and hasattr(torch, "xpu"):
            self.backend = torch.xpu

        self.stream = None
        if self.backend is not None and hasattr(self.backend, "Stream"):
            try:
                self.stream = self.backend.Stream(device=device)
            except TypeError:
                # 일부 빌드는 device 인자를 지원하지 않음
                self.stream = self.backend.Stream()

        self._next_input = None
        self._next_target = None
        self._prefetched = False

        if self.stream is not None:
            self._prefetch()

    def __len__(self):
        return self.length

    def _prefetch(self):
        try:
            next_input, next_target = next(self.loader)
        except StopIteration:
            self._next_input = None
            self._next_target = None
            return

        if self.stream is None:
            self._next_input = next_input.to(self.device, non_blocking=True, memory_format=self.mem_format)
            self._next_target = next_target.to(self.device, non_blocking=True)
            return

        with self.backend.stream(self.stream):
            self._next_input = next_input.to(self.device, non_blocking=True, memory_format=self.mem_format)
            self._next_target = next_target.to(self.device, non_blocking=True)
        self._prefetched = True

    def next(self):
        if self._next_input is None and not self._prefetched:
            # 아직 프리페치가 수행되지 않았다면(예: 스트림 백엔드 없음) 지금 수행합니다.
            self._prefetch()
            if self._next_input is None:
                return None, None

        if self.stream is not None:
            current_stream = self.backend.current_stream(self.device)
            current_stream.wait_stream(self.stream)
            if self._next_input is None:
                return None, None

        inpt, target = self._next_input, self._next_target
        if inpt is None:
            return None, None

        self._prefetch()
        return inpt, target
