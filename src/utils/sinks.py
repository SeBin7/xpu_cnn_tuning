# 파일: src/utils/sinks.py
# 모든 주석은 한글로 작성합니다.
import json
from pathlib import Path
from collections import deque
from typing import Dict, Any
class MemorySink:
    def __init__(self, maxlen: int = 1000):
        self._buf = deque(maxlen=maxlen)
    def write(self, item: Dict[str, Any]):
        self._buf.append(item)
    def take(self):
        return list(self._buf)
class FileSink:
    def __init__(self, path: str):
        self._p = Path(path)
        self._p.parent.mkdir(parents=True, exist_ok=True)
    def write(self, item: Dict[str, Any]):
        with self._p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
