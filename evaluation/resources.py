"""Best-effort process-tree CPU/RSS and device-wide NVIDIA resource samples."""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time

import psutil


class ResourceMonitor:
    def __init__(self, pid=None, *, interval=0.25):
        self.pid = pid or os.getpid()
        self.interval = interval
        self.started = time.perf_counter()
        self.started_epoch = time.time()
        self.stop_event = threading.Event()
        self.samples = 0
        self.rss_peak = 0
        self.cpu_peak = 0.
        self.processes = {}
        self.last_cpu = 0.
        self.last_time = self.started
        self.gpu_samples = []
        self.gpu_error = None
        self.nvidia = shutil.which("nvidia-smi")
        self.next_gpu = 0.

    def sample(self):
        try:
            root = psutil.Process(self.pid)
            processes = [root, *root.children(recursive=True)]
        except psutil.Error:
            processes = []
        rss = 0
        for process in processes:
            try:
                created = process.create_time()
                cpu = process.cpu_times()
                key = (process.pid, created)
                current = cpu.user+cpu.system
                baseline, maximum = self.processes.get(key, (0. if created >= self.started_epoch else current, current))
                self.processes[key] = (baseline, max(maximum, current))
                rss += process.memory_info().rss
            except psutil.Error:
                continue
        now = time.perf_counter()
        cpu = sum(maximum-baseline for baseline, maximum in self.processes.values())
        self.cpu_peak = max(self.cpu_peak, (cpu-self.last_cpu)/max(now-self.last_time, 1e-6)*100)
        self.last_cpu, self.last_time = cpu, now
        self.rss_peak = max(self.rss_peak, rss)
        self.samples += 1
        if self.nvidia and now >= self.next_gpu:
            self.next_gpu = now+1
            try:
                result = subprocess.run([self.nvidia, "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                                        capture_output=True, text=True, timeout=2, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                if result.returncode:
                    raise ValueError("nvidia-smi query returned nonzero exit")
                for line in result.stdout.splitlines():
                    index, memory, utilization = line.split(",")
                    self.gpu_samples.append(dict(device=int(index), memory_mb=float(memory), utilization_percent=float(utilization)))
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                self.gpu_error = str(error)

    def start(self):
        self.sample()
        def loop():
            while not self.stop_event.wait(self.interval):
                self.sample()
        self.thread = threading.Thread(target=loop, name="stage-resources", daemon=True)
        self.thread.start()
        return self

    def finish(self):
        self.stop_event.set()
        self.thread.join(timeout=4)
        self.sample()
        return dict(wall_seconds=time.perf_counter()-self.started, sample_count=self.samples,
            cpu_seconds=self.last_cpu, cpu_peak_percent_one_core=self.cpu_peak, cpu_logical_cores=psutil.cpu_count(),
            process_tree_rss_peak_bytes=self.rss_peak, observed_process_count=len(self.processes),
            gpu=dict(status="measured" if self.gpu_samples else "unavailable", scope="device-wide, includes unrelated processes; NOT per-stage allocations",
                reason=self.gpu_error if self.nvidia else "nvidia-smi not available; non-NVIDIA GPUs are not measured",
                samples=self.gpu_samples, memory_peak_mb=max((s["memory_mb"] for s in self.gpu_samples), default=None),
                utilization_peak_percent=max((s["utilization_percent"] for s in self.gpu_samples), default=None)),
            limitations=["Sampled process-tree RSS is a sum (shared pages may be counted twice).",
                         "Processes shorter than the sampling interval and CPU consumed after the final live sample may be missed.",
                         "CPU percent uses one logical core as 100%; multicore totals can exceed 100%."])


def terminate_tree(process):
    """Terminate only this explicitly spawned process and its descendants."""
    try:
        root = psutil.Process(process.pid)
        targets = root.children(recursive=True)+[root]
        for target in targets:
            try:
                target.terminate()
            except psutil.Error:
                pass
        _, alive = psutil.wait_procs(targets, timeout=2)
        for target in alive:
            try:
                target.kill()
            except psutil.Error:
                pass
    except psutil.Error:
        pass
    process.wait(timeout=5)

