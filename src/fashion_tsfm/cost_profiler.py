"""
Perfilador de Costo Computacional, Throughput y Escalabilidad Big Data (`CostProfiler`).

Cumple con el tercer resultado (Outcome 3) del Marco PICO:
- Mide el tiempo de ejecución (*wall-clock seconds*).
- Calcula el rendimiento (`series_por_segundo`) y el tiempo normalizado por cada
  **1.000 series** pronosticadas (`seconds_per_1000_series`).
- Estima el costo económico en la nube por cada **1.000 series** (`cost_usd_per_1000_series`).
- Monitorea el uso de memoria RAM (CPU) y VRAM (GPU CUDA si está disponible).
- Registra automáticamente las métricas y metadatos de hardware en **MLflow**.
"""

from __future__ import annotations

import functools
import os
import platform
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore


def get_compute_environment_info() -> Dict[str, Any]:
    """
    Detecta de forma automática las capacidades del hardware donde se ejecuta el notebook
    (Databricks Free Edition / Serverless CPU vs. Clúster con GPU NVIDIA).
    """
    cpu_cnt = (psutil.cpu_count(logical=True) if psutil is not None else os.cpu_count()) or 1
    ram_gb = round(psutil.virtual_memory().total / (1024**3), 2) if psutil is not None else 16.0
    info: Dict[str, Any] = {
        "os_platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count_logical": cpu_cnt,
        "ram_total_gb": ram_gb,
        "device": "cpu",
        "gpu_available": False,
        "gpu_name": "None (Serverless CPU)",
        "gpu_vram_total_gb": 0.0,
    }

    try:
        import torch

        if torch.cuda.is_available():
            info["device"] = "cuda"
            info["gpu_available"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_vram_total_gb"] = round(props.total_memory / (1024**3), 2)
    except Exception:
        pass

    return info


@dataclass
class CostProfiler:
    """
    Context manager para medir el tiempo de cómputo, memoria y costo normalizado
    por cada 1.000 series pronosticadas, registrando opcionalmente en MLflow.

    Ejemplo de uso:
    ```python
    with CostProfiler(
        step_name="Chronos_Bolt_ZeroShot",
        n_series=1000,
        model_family="TSFM",
        log_to_mlflow=True,
    ) as profiler:
        forecasts = run_inference(...)
    print(profiler.metrics)
    ```
    """

    step_name: str
    n_series: int
    model_family: str = "TSFM"
    log_to_mlflow: bool = True
    cluster_cost_usd_per_hour: Optional[float] = None
    extra_tags: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict, init=False)

    def __enter__(self) -> "CostProfiler":
        self._env_info = get_compute_environment_info()
        if self.cluster_cost_usd_per_hour is None:
            # Costo referencial por hora: GPU ($1.60/h) vs Serverless CPU ($0.45/h)
            self.cluster_cost_usd_per_hour = 1.60 if self._env_info["gpu_available"] else 0.45

        self._ram_start_mb = (
            psutil.Process().memory_info().rss / (1024**2) if psutil is not None else 0.0
        )
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        elapsed_sec = max(time.perf_counter() - self._t0, 1e-6)
        ram_end_mb = (
            psutil.Process().memory_info().rss / (1024**2) if psutil is not None else 0.0
        )
        ram_delta_mb = max(0.0, ram_end_mb - self._ram_start_mb)

        vram_peak_mb = 0.0
        try:
            import torch

            if torch.cuda.is_available():
                vram_peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
        except Exception:
            pass

        effective_series = max(int(self.n_series), 1)
        throughput = effective_series / elapsed_sec
        sec_per_1000 = (elapsed_sec / effective_series) * 1000.0
        cost_per_1000 = (sec_per_1000 / 3600.0) * float(self.cluster_cost_usd_per_hour or 0.45)

        self.metrics = {
            "step_name": self.step_name,
            "model_family": self.model_family,
            "n_series": effective_series,
            "device": self._env_info["device"],
            "gpu_name": self._env_info["gpu_name"],
            "wall_clock_seconds": round(elapsed_sec, 4),
            "throughput_series_per_sec": round(throughput, 2),
            "seconds_per_1000_series": round(sec_per_1000, 4),
            "cost_usd_per_1000_series": round(cost_per_1000, 6),
            "ram_rss_mb": round(ram_end_mb, 2),
            "ram_delta_mb": round(ram_delta_mb, 2),
            "vram_peak_mb": round(vram_peak_mb, 2),
        }

        if self.log_to_mlflow and exc_type is None:
            self._log_metrics_to_mlflow()

    def _log_metrics_to_mlflow(self) -> None:
        """
        Registra las métricas de rendimiento y hardware en el experimento activo de MLflow.
        """
        try:
            import mlflow

            active_run = mlflow.active_run()
            if active_run is not None:
                prefix = self.step_name.lower().replace(" ", "_")
                mlflow.log_metrics(
                    {
                        f"{prefix}_wall_clock_sec": self.metrics["wall_clock_seconds"],
                        f"{prefix}_series_per_sec": self.metrics["throughput_series_per_sec"],
                        f"{prefix}_sec_per_1000_series": self.metrics["seconds_per_1000_series"],
                        f"{prefix}_cost_usd_per_1000": self.metrics["cost_usd_per_1000_series"],
                        f"{prefix}_ram_mb": self.metrics["ram_rss_mb"],
                        f"{prefix}_vram_peak_mb": self.metrics["vram_peak_mb"],
                    }
                )
                mlflow.set_tags(
                    {
                        "compute_device": self.metrics["device"],
                        "gpu_name": self.metrics["gpu_name"],
                        **self.extra_tags,
                    }
                )
        except Exception:
            # Si MLflow no está configurado en una ejecución local aislada, continuamos sin fallar
            pass


def profile_forecasting_step(
    step_name: str,
    model_family: str = "TSFM",
    n_series_arg: str = "n_series",
    log_to_mlflow: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorador funcional para envolver funciones de entrenamiento o inferencia
    y medir automáticamente su tiempo y costo por cada 1.000 series.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            n_series_val = int(kwargs.get(n_series_arg, 1000))
            with CostProfiler(
                step_name=step_name,
                n_series=n_series_val,
                model_family=model_family,
                log_to_mlflow=log_to_mlflow,
            ) as prof:
                result = func(*args, **kwargs)
            if isinstance(result, dict):
                result["_cost_profile"] = prof.metrics
            return result

        return wrapper

    return decorator
