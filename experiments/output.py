"""Saving runs: each experiment run gets results/<experiment>/<timestamp>/ with the exact config
it used (config.yaml), everything it printed (log.txt), its results (results.json) and figures."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import yaml


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)

    def flush(self):
        for s in self.streams:
            s.flush()


class RunRecorder:
    def __init__(self, root: str | Path, experiment: str, conf: dict, enabled: bool = True):
        self.root, self.experiment, self.conf, self.enabled = Path(root), experiment, conf, enabled
        self.dir = self.root

    def __enter__(self):
        if self.enabled:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            base = self.root / self.experiment / stamp
            self.dir, n = base, 1
            while self.dir.exists():
                n += 1
                self.dir = base.with_name(f"{stamp}-{n}")
            self.dir.mkdir(parents=True)
            (self.dir / "config.yaml").write_text(yaml.safe_dump(self.conf, sort_keys=False))
            self._log = open(self.dir / "log.txt", "w")
            self._stdout, sys.stdout = sys.stdout, _Tee(sys.stdout, self._log)
        return self

    def save(self, results: dict):
        if self.enabled:
            with open(self.dir / "results.json", "w") as f:
                json.dump(results, f, indent=2, default=list)
            print(f"\nsaved {self.dir}")

    def __exit__(self, *exc):
        if self.enabled:
            sys.stdout = self._stdout
            self._log.close()
        return False
