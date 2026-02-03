"""
Instagram Reels Encoder - CINEON FILM EMULATION EDITION v2.0.0 (FASE 26)

NOVIDADE v2.0 - PIPELINE DWG/CINEON (2025-01-22):
Integração completa do pipeline cinematográfico de 5 nodes com film emulation Kodak 2383.

FILOSOFIA:
- Duas engines: FFmpeg Filters (rápido) e PyAV+Cineon (qualidade cinematográfica)
- Backward compatible: todos os modos v1.4.1 preservados
- Zero breaking changes: --cineon-pipeline on/off (default: off)
- Film-grade color science: DWG → Cineon → Kodak 2383 LUT

PIPELINES DISPONÍVEIS (v2.0):
1. SDR Float (v1.4.1): 32-bit precision FFmpeg (default, --float on)
2. SDR 8-bit (v1.3): Legacy FFmpeg pipeline (--float off)
3. HDR: Tone mapping FFmpeg (auto-detect)
4. 🎬 CINEON (NOVO): PyAV + DWG + Cineon + Kodak 2383 (--cineon-pipeline on)

PIPELINE CINEON (5 NODES):
    PyAV Decode → Node 1 (DWG) → Node 2 (Grade) → Node 3 (Rec.709) →
    Node 4 (Cineon) → Node 5 (Kodak LUT) → FFmpeg Pipe (libx264)

BENEFÍCIOS CINEON:
- ✅ Film emulation profissional (Kodak 2383)
- ✅ 32-bit float precision (zero banding)
- ✅ DaVinci Wide Gamut color science
- ✅ Cineon Film Log (Colour-Science certified)
- ✅ Adjustable grading (exposure, saturation)
- ⚠️ Performance: ~5-15 fps (CPU) ou ~30-60 fps (GPU com CuPy)

BACKWARD COMPATIBILITY:
- Todos os argumentos v1.4.1 mantidos
- Modo FFmpeg (default): sem mudanças
- Modo Cineon (opt-in): --cineon-pipeline on
- VBV, loudnorm, metadados: preservados

COMPARAÇÃO:
┌────────────────────────────────────────────────────────────┐
│ MODO FFMPEG (default, v1.4.1):                            │
│ • Performance: ~30-60 fps (GPU filters)                    │
│ • Qualidade: Excelente (float + LUT v6.6)                  │
│ • Uso: Produção rápida, batch processing                   │
├────────────────────────────────────────────────────────────┤
│ MODO CINEON (novo, v2.0):                                  │
│ • Performance: ~5-15 fps (CPU) ou ~30-60 fps (GPU)         │
│ • Qualidade: Film-grade (DWG + Cineon + Kodak 2383)        │
│ • Uso: Projetos premium, film look autêntico               │
└────────────────────────────────────────────────────────────┘

USAGE:
  # Modo FFmpeg (default, rápido):
  python Reels_Encoder_v2.py input.mp4

  # Modo Cineon (film emulation):
  python Reels_Encoder_v2.py input.mp4 --cineon-pipeline on

  # Ajustes de grading (Cineon):
  python Reels_Encoder_v2.py input.mp4 --cineon-pipeline on --exposure +0.5 --saturation 1.1

DEPENDENCIES:
  pip install av>=11.0.0  # PyAV (para modo Cineon)
  pip install colour-science>=0.4.7  # Colour (para modo Cineon)

VERSÕES:
- v2.0.0: Integração Pipeline Cineon (FASE 26)
- v1.4.1: CAS Conservador (0.30 para SDR float)
- v1.4: Float Pipeline (32-bit precision)
- v1.3: Color Preservation Fix (desat=0)
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, Optional

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from pymediainfo import MediaInfo
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich import box

# =============================================================================
# CINEON PIPELINE IMPORTS (FASE 26)
# =============================================================================
try:
    from cineon_pipeline import LUT3D, process_frame_full_pipeline, COLOUR_AVAILABLE

    CINEON_AVAILABLE = True
except ImportError:
    CINEON_AVAILABLE = False

try:
    import av
    import numpy as np

    PYAV_AVAILABLE = True
except ImportError:
    PYAV_AVAILABLE = False
    np = None

DEVNULL_FF = "NUL" if os.name == "nt" else "/dev/null"

console = Console()

# =============================================================================
# VBV PRESETS PARA INSTAGRAM REELS
# =============================================================================
VBV_PRESETS = {
    "ultra_short": {
        "duration_max": 15,
        "target": 12000,
        "maxrate": 13000,
        "bufsize": 14000,
        "vbv_init": 0.9,
        "description": "Ultra Short (≤15s) — Maximum Quality",
    },
    "short": {
        "duration_max": 30,
        "target": 9800,
        "maxrate": 11000,
        "bufsize": 12000,
        "vbv_init": 0.85,
        "description": "Short (15-30s) — High Quality",
    },
    "medium": {
        "duration_max": 45,
        "target": 8500,
        "maxrate": 9500,
        "bufsize": 10200,
        "vbv_init": 0.85,
        "description": "Medium (30-45s) — Balanced Quality",
    },
    "long": {
        "duration_max": 60,
        "target": 8000,
        "maxrate": 9000,
        "bufsize": 10000,
        "vbv_init": 0.85,
        "description": "Long (45-60s) — Safe Premium",
    },
    "extra_long": {
        "duration_max": 90,
        "target": 6500,
        "maxrate": 7500,
        "bufsize": 8000,
        "vbv_init": 0.85,
        "description": "Extra Long (60-90s) — Conservative",
    },
}

# =============================================================================
# LOUDNORM TARGETS (EBU R128)
# =============================================================================
LOUDNORM_TARGETS = {
    "instagram": {"I": -14, "TP": -1, "LRA": 11},
    "youtube": {"I": -14, "TP": -1, "LRA": 11},
    "broadcast": {"I": -23, "TP": -1, "LRA": 7},
}

# =============================================================================
# HDR DETECTION & TONE MAPPING
# =============================================================================
HDR_PRIMARIES = ("bt2020",)
HDR_TRANSFERS = ("smpte2084", "arib-std-b67", "smpte-st-2084", "bt2020-10", "bt2020-12")

TONEMAP_ALGORITHMS = {
    "mobius": "Highlights suaves, melhor para skin tones (recomendado)",
    "reinhard": "Suave, preserva sombras",
    "hable": "Contraste cinematográfico (Uncharted 2)",
    "bt2390": "ITU standard para broadcast",
}

# =============================================================================
# HARDWARE DETECTION & OPTIMIZATION
# =============================================================================
HARDWARE_TIERS = {
    "ultra": {
        "min_cores": 16,
        "min_ram": 32,
        "preset": "veryslow",
        "lookahead": 60,
        "threads_mult": 2.0,
    },
    "high": {
        "min_cores": 6,
        "min_ram": 16,
        "preset": "slow",
        "lookahead": 50,
        "threads_mult": 1.5,
    },
    "medium": {
        "min_cores": 4,
        "min_ram": 8,
        "preset": "medium",
        "lookahead": 40,
        "threads_mult": 1.0,
    },
    "low": {
        "min_cores": 2,
        "min_ram": 4,
        "preset": "fast",
        "lookahead": 30,
        "threads_mult": 1.0,
    },
}


@dataclass
class HardwareProfile:
    """Perfil de hardware detectado do sistema."""

    cpu_name: str = "Unknown"
    cpu_cores: int = 4
    cpu_threads: int = 4
    cpu_freq_mhz: int = 0
    cpu_arch: str = "x64"
    ram_total_gb: float = 8.0
    ram_available_gb: float = 4.0
    ram_percent_used: float = 50.0
    tier: str = "medium"
    perf_score: int = 50
    recommended_threads: int = 4
    recommended_preset: str = "medium"
    recommended_lookahead: int = 40
    recommended_filter_threads: int = 4
    recommended_decoder_threads: int = 4
    os_name: str = "Windows"
    os_version: str = "10"


def detect_hardware() -> HardwareProfile:
    """Detecta hardware do sistema e retorna perfil otimizado."""
    profile = HardwareProfile()
    profile.os_name = platform.system()
    profile.os_version = platform.release()
    profile.cpu_arch = platform.machine()

    if not PSUTIL_AVAILABLE:
        console.print(
            "[yellow]⚠ psutil não instalado. Usando valores conservadores.[/yellow]"
        )
        return _calculate_recommendations(profile)

    try:
        profile.cpu_cores = psutil.cpu_count(logical=False) or 4
        profile.cpu_threads = psutil.cpu_count(logical=True) or 4

        cpu_freq = psutil.cpu_freq()
        if cpu_freq:
            profile.cpu_freq_mhz = (
                int(cpu_freq.current) if cpu_freq.current else int(cpu_freq.max)
            )

        if profile.os_name == "Windows":
            try:
                result = subprocess.run(
                    ["wmic", "cpu", "get", "name"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                lines = [
                    l.strip()
                    for l in result.stdout.strip().split("\n")
                    if l.strip() and l.strip() != "Name"
                ]
                if lines:
                    profile.cpu_name = lines[0]
            except Exception:
                profile.cpu_name = platform.processor() or "Unknown CPU"
        else:
            profile.cpu_name = platform.processor() or "Unknown CPU"

        ram = psutil.virtual_memory()
        profile.ram_total_gb = round(ram.total / (1024**3), 1)
        profile.ram_available_gb = round(ram.available / (1024**3), 1)
        profile.ram_percent_used = ram.percent

    except Exception as e:
        console.print(f"[yellow]⚠ Erro na detecção de hardware: {e}[/yellow]")

    return _calculate_recommendations(profile)


def _calculate_recommendations(profile: HardwareProfile) -> HardwareProfile:
    """Calcula recomendações de encoding baseado no hardware detectado."""
    cores = profile.cpu_cores
    ram = profile.ram_total_gb

    if cores >= 16 and ram >= 32:
        profile.tier = "ultra"
    elif cores >= 6 and ram >= 16:
        profile.tier = "high"
    elif cores >= 4 and ram >= 8:
        profile.tier = "medium"
    else:
        profile.tier = "low"

    tier_config = HARDWARE_TIERS[profile.tier]

    core_score = min(40, cores * 2.5)
    thread_score = min(20, profile.cpu_threads * 0.625)
    ram_score = min(30, ram * 0.47)
    freq_score = min(10, profile.cpu_freq_mhz / 500) if profile.cpu_freq_mhz > 0 else 5
    raw_score = core_score + thread_score + ram_score + freq_score
    profile.perf_score = min(100, max(1, int(raw_score)))

    available_threads = max(1, profile.cpu_threads - 2)
    profile.recommended_threads = max(
        1, int(available_threads * tier_config["threads_mult"])
    )
    profile.recommended_threads = min(profile.recommended_threads, profile.cpu_threads)

    profile.recommended_preset = tier_config["preset"]
    if profile.ram_available_gb < 4:
        profile.recommended_preset = "fast"
    elif profile.ram_available_gb < 8 and profile.recommended_preset == "slow":
        profile.recommended_preset = "medium"

    profile.recommended_lookahead = tier_config["lookahead"]
    if profile.ram_available_gb < 8:
        profile.recommended_lookahead = min(30, profile.recommended_lookahead)
    elif profile.ram_available_gb < 16:
        profile.recommended_lookahead = min(40, profile.recommended_lookahead)

    profile.recommended_filter_threads = max(2, profile.cpu_cores // 2)
    if profile.tier == "ultra":
        profile.recommended_filter_threads = max(4, profile.cpu_cores)
    elif profile.tier == "high":
        profile.recommended_filter_threads = max(4, int(profile.cpu_cores * 0.75))

    profile.recommended_decoder_threads = profile.cpu_cores
    if profile.tier in ("low", "medium"):
        profile.recommended_decoder_threads = min(4, profile.cpu_cores)

    return profile


def print_hardware_profile(profile: HardwareProfile) -> None:
    """Exibe perfil de hardware formatado no terminal."""
    tier_colors = {
        "ultra": "bold magenta",
        "high": "bold green",
        "medium": "bold yellow",
        "low": "bold red",
    }
    tier_color = tier_colors.get(profile.tier, "white")

    hw_table = Table(
        title="🖥️ Hardware Detectado",
        show_header=True,
        header_style="bold cyan",
        box=box.SIMPLE,
    )
    hw_table.add_column("Componente", style="dim", width=15)
    hw_table.add_column("Valor", style="white", width=35)
    hw_table.add_column("Detalhes", style="dim", width=25)

    cpu_name = (
        profile.cpu_name[:45] + "..."
        if len(profile.cpu_name) > 45
        else profile.cpu_name
    )
    hw_table.add_row(
        "CPU",
        cpu_name,
        f"{profile.cpu_freq_mhz} MHz" if profile.cpu_freq_mhz > 0 else "",
    )
    hw_table.add_row(
        "Cores/Threads",
        f"{profile.cpu_cores}C / {profile.cpu_threads}T",
        f"Arch: {profile.cpu_arch}",
    )

    ram_status = (
        "🟢"
        if profile.ram_available_gb >= 8
        else "🟡" if profile.ram_available_gb >= 4 else "🔴"
    )
    hw_table.add_row(
        "RAM Total",
        f"{profile.ram_total_gb:.1f} GB",
        f"Disponível: {profile.ram_available_gb:.1f} GB {ram_status}",
    )

    hw_table.add_row("Sistema", f"{profile.os_name} {profile.os_version}", "")

    rec_table = Table(
        title="📋 Recomendações de Encoding",
        show_header=True,
        header_style="bold green",
        box=box.SIMPLE,
    )
    rec_table.add_column("Parâmetro", style="dim", width=18)
    rec_table.add_column("Valor", style="green", width=10)
    rec_table.add_column("Impacto", style="dim", width=35)

    rec_table.add_row(
        "Encoder Threads", str(profile.recommended_threads), "x264 threads"
    )
    rec_table.add_row(
        "Filter Threads",
        str(profile.recommended_filter_threads),
        "Filtros (scale, tonemap, sharpen)",
    )
    rec_table.add_row(
        "Decoder Threads",
        str(profile.recommended_decoder_threads),
        "Decodificação do input",
    )
    rec_table.add_row(
        "Preset x264", profile.recommended_preset, "Qualidade vs Velocidade"
    )
    rec_table.add_row(
        "Lookahead", str(profile.recommended_lookahead), "Análise de cena (frames)"
    )

    score = profile.perf_score
    bar_width = 20
    filled = int(score / 100 * bar_width)
    bar = f"[green]{'█' * filled}[/green][dim]{'░' * (bar_width - filled)}[/dim]"

    console.print()
    console.rule("[bold cyan]🔧 Hardware Profile[/bold cyan]")
    console.print()
    console.print(hw_table)
    console.print()
    console.print(f"[bold]⚡ Performance Score:[/bold] {bar} {score}/100")
    console.print(
        f"[bold]🏆 Tier:[/bold] [{tier_color}]{profile.tier.upper()}[/{tier_color}]"
    )
    console.print()
    console.print(rec_table)
    console.print()


# =============================================================================
# HDR DETECTION & CONVERSION
# =============================================================================
def detect_hdr_metadata(input_file: str) -> Optional[dict]:
    """Detecta se o vídeo é HDR e retorna metadados de cor."""
    console.print("[cyan]🔍 Detectando metadados HDR...[/cyan]")

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=color_primaries,color_transfer,color_space,color_range",
        "-show_entries",
        "stream_side_data=max_luminance,min_luminance",
        "-of",
        "json",
        input_file,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        if not data.get("streams"):
            console.print("[yellow]⚠ Nenhum stream de vídeo encontrado[/yellow]")
            return None

        stream = data["streams"][0]

        color_primaries = stream.get("color_primaries", "unknown")
        color_transfer = stream.get("color_transfer", "unknown")
        color_space = stream.get("color_space", "unknown")
        color_range = stream.get("color_range", "unknown")

        max_luminance = None
        side_data = stream.get("side_data_list", [])
        for sd in side_data:
            if "max_luminance" in sd:
                lum = sd.get("max_luminance", "")
                if "/" in str(lum):
                    num, den = lum.split("/")
                    max_luminance = float(num) / float(den)
                else:
                    max_luminance = float(lum) if lum else None
                break

        is_hdr = False
        hdr_type = "SDR"

        if color_transfer in ("smpte2084", "smpte-st-2084"):
            is_hdr = True
            hdr_type = "HDR10 (PQ)"
        elif color_transfer in ("arib-std-b67",):
            is_hdr = True
            hdr_type = "HLG"
        elif color_primaries in HDR_PRIMARIES and color_transfer in HDR_TRANSFERS:
            is_hdr = True
            hdr_type = "HDR (BT.2020)"

        hdr_info = {
            "is_hdr": is_hdr,
            "color_primaries": color_primaries,
            "color_transfer": color_transfer,
            "color_space": color_space,
            "color_range": color_range,
            "hdr_type": hdr_type,
            "max_luminance": max_luminance,
        }

        if is_hdr:
            console.print(f"[bold yellow]⚠ HDR DETECTADO: {hdr_type}[/bold yellow]")
            console.print(f"[dim]   Primaries: {color_primaries}[/dim]")
            console.print(f"[dim]   Transfer: {color_transfer}[/dim]")
            console.print(f"[dim]   Space: {color_space}[/dim]")
            if max_luminance:
                console.print(f"[dim]   Max Luminance: {max_luminance:.0f} nits[/dim]")
        else:
            console.print(
                f"[green]✓ SDR detectado[/green] (primaries={color_primaries}, transfer={color_transfer})"
            )

        return hdr_info

    except (subprocess.CalledProcessError, json.JSONDecodeError, Exception) as e:
        console.print(f"[yellow]⚠ Detecção HDR falhou: {e}[/yellow]")
        return None


def build_hdr_to_sdr_filter(hdr_info: dict, tonemap: str = "mobius") -> Optional[str]:
    """Gera filtro FFmpeg para conversão HDR → SDR."""
    if not hdr_info or not hdr_info.get("is_hdr"):
        return None

    if tonemap not in TONEMAP_ALGORITHMS:
        console.print(
            f"[yellow]⚠ Tonemap '{tonemap}' inválido, usando 'mobius'[/yellow]"
        )
        tonemap = "mobius"

    console.print(f"[cyan]🎨 Gerando filtro HDR→SDR (tonemap={tonemap})...[/cyan]")

    hdr_filter = (
        "zscale=t=linear:npl=100,"
        "format=gbrpf32le,"
        "zscale=p=bt709,"
        f"tonemap={tonemap}:desat=0,"
        "zscale=t=bt709:m=bt709:r=tv,"
        "format=yuv420p"
    )

    console.print(f"[green]✓ Filtro HDR→SDR:[/green]")
    console.print(f"[dim]   {hdr_filter}[/dim]")

    return hdr_filter


# =============================================================================
# PROGRESS HUD
# =============================================================================
class ResolveProgressHUD:
    def __init__(self, total_frames):
        self.total_frames = max(int(total_frames), 1)
        self.current_frame = 0
        self.start_time = time.time()
        self.last_time = self.start_time
        self.last_frame = 0
        self.fps = 0.0
        self.speed = 0.0
        self.eta = "--:--:--"

    def update_frame(self, frame):
        try:
            frame = int(frame)
        except (TypeError, ValueError):
            return
        self.current_frame = frame
        now = time.time()
        dt = now - self.last_time
        df = frame - self.last_frame
        if dt > 0:
            self.fps = max(df / dt, 0.1)
            self.speed = self.fps / 24.0
        else:
            self.fps = 0.0
            self.speed = 0.0
        self.last_frame = frame
        self.last_time = now
        remaining_frames = max(self.total_frames - frame, 1)
        eta_seconds = remaining_frames / max(self.fps, 0.01)
        self.eta = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

    def render(self):
        table = Table.grid(expand=True)
        table.add_row(
            f"[cyan]Frame:[/cyan] {self.current_frame}/{self.total_frames}",
            f"[green]FPS:[/green] {self.fps:.2f}",
            f"[yellow]Speed:[/yellow] {self.speed:.2f}x",
            f"[magenta]ETA:[/magenta] {self.eta}",
        )
        progress_bar_width = 40
        p = float(self.current_frame) / float(self.total_frames)
        p = max(0.0, min(1.0, p))
        filled = int(p * progress_bar_width)
        empty = progress_bar_width - filled
        bar = f"[green]{'█' * filled}[/green][white]{'░' * empty}[/white]"
        table.add_row(bar)
        elapsed = time.time() - self.start_time
        table.add_row(
            f"[blue]Elapsed:[/blue] {time.strftime('%H:%M:%S', time.gmtime(elapsed))}"
        )
        return table


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def ffmpeg_live_reader(pipe, hud: ResolveProgressHUD):
    for line in iter(pipe.readline, ""):
        if not line:
            break
        if "frame=" in line:
            parts = line.split("frame=")
            if len(parts) > 1:
                try:
                    frame_str = parts[1].split()[0]
                    frame = int("".join(ch for ch in frame_str if ch.isdigit()))
                    hud.update_frame(frame)
                except ValueError:
                    continue


def get_video_duration(input_file: str) -> float:
    """Obtém duração do vídeo em segundos."""
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
                input_file,
            ],
            stderr=subprocess.PIPE,
        )
        return float(out.decode().strip())
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        console.print(
            f"[yellow]Aviso: ffprobe falhou, usando duração padrão 30s[/yellow]"
        )
        return 30.0


def get_total_frames(input_file: str) -> int:
    """Obtém total de frames do vídeo."""
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
                input_file,
            ],
            stderr=subprocess.PIPE,
        )
        text = out.decode().strip()
        try:
            return int(text)
        except (ValueError, TypeError):
            duration = get_video_duration(input_file)
            estimated = int(max(1, round(duration * 24)))
            console.print(
                f"[yellow]Aviso:[/yellow] nb_read_frames não disponível, estimando frames = {estimated}"
            )
            return estimated
    except (subprocess.CalledProcessError, FileNotFoundError):
        duration = get_video_duration(input_file)
        return int(max(1, round(duration * 24)))


def get_vbv_preset(duration: float) -> dict:
    """Seleciona o preset VBV baseado na duração do vídeo."""
    for preset_name, preset in VBV_PRESETS.items():
        if duration <= preset["duration_max"]:
            return preset
    return VBV_PRESETS["extra_long"]


def get_input_fps(input_file: str) -> int:
    """Detecta frame rate do vídeo de entrada."""
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=r_frame_rate",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
                input_file,
            ],
            stderr=subprocess.PIPE,
        )

        fps_str = out.decode().strip()
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps_float = float(num) / float(den)
        else:
            fps_float = float(fps_str)

        standard_fps = [24, 25, 30, 50, 60]
        closest_fps = min(standard_fps, key=lambda x: abs(x - fps_float))

        if 23.5 <= fps_float <= 24.5:
            return 24
        elif 29.5 <= fps_float <= 30.5:
            return 30
        elif 59.5 <= fps_float <= 60.5:
            return 60

        return closest_fps

    except (
        subprocess.CalledProcessError,
        ValueError,
        FileNotFoundError,
        ZeroDivisionError,
    ):
        console.print(
            f"[yellow]Aviso: Não foi possível detectar fps, usando 30 fps[/yellow]"
        )
        return 30


def get_input_resolution(input_file: str) -> Tuple[int, int]:
    """Detecta resolução REAL (largura x altura) do vídeo de entrada."""
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height:stream_tags=rotate:side_data",
                "-of",
                "json",
                input_file,
            ],
            stderr=subprocess.PIPE,
        )

        data = json.loads(out.decode())

        width = 0
        height = 0
        rotation = 0

        if "streams" in data and len(data["streams"]) > 0:
            stream = data["streams"][0]
            width = int(stream.get("width", 0))
            height = int(stream.get("height", 0))

            tags = stream.get("tags", {})
            if "rotate" in tags:
                rotation = int(tags["rotate"])

            side_data = stream.get("side_data_list", [])
            for sd in side_data:
                if sd.get("side_data_type") == "Display Matrix":
                    rot = sd.get("rotation", 0)
                    if rot != 0:
                        rotation = int(rot)

        if rotation == 0:
            out2 = subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format_tags=rotate",
                    "-of",
                    "json",
                    input_file,
                ],
                stderr=subprocess.PIPE,
            )

            data2 = json.loads(out2.decode())
            format_tags = data2.get("format", {}).get("tags", {})
            if "rotate" in format_tags:
                rotation = int(format_tags["rotate"])

        if rotation in (90, -90, 270, -270):
            width, height = height, width
            console.print(
                f"[cyan]📱 Rotação detectada: {abs(rotation)}° → Orientação: {'Vertical' if height > width else 'Horizontal'}[/cyan]"
            )

        if width > 0 and height > 0:
            return width, height

        console.print(f"[yellow]Aviso: Não foi possível detectar resolução[/yellow]")
        return 0, 0

    except (
        subprocess.CalledProcessError,
        ValueError,
        FileNotFoundError,
        json.JSONDecodeError,
        KeyError,
        IndexError,
    ):
        console.print(f"[yellow]Aviso: Não foi possível detectar resolução[/yellow]")
        return 0, 0


def build_scale_filter(
    input_width: int,
    input_height: int,
    target_width: int = 1080,
    target_height: int = 1920,
) -> Optional[str]:
    """Gera filtro de downscale de alta qualidade para Instagram Reels."""
    needs_width_scale = input_width > target_width
    needs_height_scale = input_height > target_height

    if not needs_width_scale and not needs_height_scale:
        return None

    if input_height > input_width:
        if input_height > target_height:
            scale_factor = target_height / input_height
            final_width = int(input_width * scale_factor)
            final_height = target_height
            final_width = final_width - (final_width % 2)
        else:
            return None
    else:
        if input_width > target_width:
            scale_factor = target_width / input_width
            final_width = target_width
            final_height = int(input_height * scale_factor)
            final_height = final_height - (final_height % 2)
        else:
            return None

    scale_filter = (
        f"zscale=w={final_width}:h={final_height}:" f"filter=lanczos:format=gbrpf32le,"
    )

    console.print(
        f"[cyan]📐 Downscale: {input_width}x{input_height} → {final_width}x{final_height}[/cyan]"
    )
    console.print(f"[dim]   Filtro: zscale + Lanczos[/dim]")

    return scale_filter


# =============================================================================
# AUDIO LOUDNESS NORMALIZATION (EBU R128)
# =============================================================================
def analyze_audio_loudness(
    input_file: str, target: str = "instagram"
) -> Optional[dict]:
    """Pass 1: Analisa loudness do áudio usando filtro loudnorm."""
    t = LOUDNORM_TARGETS.get(target, LOUDNORM_TARGETS["instagram"])

    console.print(f"[cyan]🔊 Pass 1: Analisando loudness (target: {target})...[/cyan]")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        input_file,
        "-af",
        f"loudnorm=I={t['I']}:TP={t['TP']}:LRA={t['LRA']}:print_format=json",
        "-f",
        "null",
        "-",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore"
        )

        stderr = result.stderr
        json_start = stderr.rfind("{")
        json_end = stderr.rfind("}") + 1

        if json_start == -1 or json_end == 0:
            console.print(
                "[yellow]⚠ Não foi possível extrair estatísticas de loudness[/yellow]"
            )
            return None

        json_str = stderr[json_start:json_end]
        stats = json.loads(json_str)

        required_fields = ["input_i", "input_tp", "input_lra", "input_thresh"]
        for field in required_fields:
            if field not in stats:
                console.print(
                    f"[yellow]⚠ Campo '{field}' não encontrado na análise[/yellow]"
                )
                return None

        console.print(f"[dim]   Integrated: {stats['input_i']} LUFS[/dim]")
        console.print(f"[dim]   True Peak: {stats['input_tp']} dBTP[/dim]")
        console.print(f"[dim]   LRA: {stats['input_lra']} LU[/dim]")
        console.print(f"[dim]   Threshold: {stats['input_thresh']} LUFS[/dim]")

        return stats

    except (json.JSONDecodeError, subprocess.SubprocessError) as e:
        console.print(f"[red]Erro ao executar análise de loudness: {e}[/red]")
        return None


def build_loudnorm_filter(stats: dict, target: str = "instagram") -> str:
    """Pass 2: Gera filtro loudnorm com valores medidos para normalização linear."""
    t = LOUDNORM_TARGETS.get(target, LOUDNORM_TARGETS["instagram"])

    loudnorm_filter = (
        f"loudnorm=I={t['I']}:TP={t['TP']}:LRA={t['LRA']}:"
        f"measured_I={stats['input_i']}:"
        f"measured_TP={stats['input_tp']}:"
        f"measured_LRA={stats['input_lra']}:"
        f"measured_thresh={stats['input_thresh']}:"
        f"linear=true:print_format=summary"
    )

    console.print(
        f"[green]✓ Loudnorm filter: I={t['I']} LUFS, TP={t['TP']} dBTP, linear=true[/green]"
    )

    return loudnorm_filter


def _x264_params_string(
    duration_seconds: float = 30, threads: int = 0, lookahead: int = 60
) -> str:
    """Parâmetros x264 otimizados para Instagram Reels - ZERO RECOMPRESSÃO."""
    vbv = get_vbv_preset(duration_seconds)

    parts = [
        "ref=3",
        "bframes=2",
        "b-adapt=2",
        "b-pyramid=2",
        "weightb=1",
        "weightp=2",
        "direct=auto",
        "me=umh",
        "subme=9",
        "me-range=24",
        "trellis=2",
        "psy-rd=1.00:0.15",
        "aq-mode=1",
        "aq-strength=0.60",
        f"rc-lookahead={lookahead}",
        "mbtree=1",
        "qcomp=0.60",
        "deblock=-1:-1",
        "no-fast-pskip=1",
        "keyint=60",
        "min-keyint=30",
        "scenecut=40",
        f"vbv-maxrate={vbv['maxrate']}",
        f"vbv-bufsize={vbv['bufsize']}",
        f"vbv-init={vbv['vbv_init']}",
        "nal-hrd=vbr",
        "aud=1",
        "repeat-headers=1",
        "open_gop=0",
        f"threads={threads}",
    ]
    params_string = ":".join(parts)

    # Verificar se há espaços em volta do '=' (isso causaria erro FFmpeg)
    if " = " in params_string or " =" in params_string or "= " in params_string:
        console.print(
            "[red]⚠️ ERRO CRÍTICO: Espaços detectados em volta do '=' nos x264-params![/red]"
        )
        console.print(f"[red]   String corrompida: {params_string}[/red]")

        # Tentar corrigir automaticamente
        params_string_fixed = (
            params_string.replace(" = ", "=").replace(" =", "=").replace("= ", "=")
        )
        console.print(f"[yellow]   Tentando corrigir: {params_string_fixed}[/yellow]")

        return params_string_fixed

    # Verificar caracteres problemáticos (shell escaping)
    problematic_chars = ['"', "'", "\\", "|", "&", ";", "$", "`"]
    for char in problematic_chars:
        if char in params_string:
            console.print(
                f"[yellow]⚠️ ATENÇÃO: Caractere '{char}' detectado em x264-params[/yellow]"
            )
            console.print(
                f"[yellow]   Isso pode causar problemas no Windows subprocess[/yellow]"
            )

    return params_string


def _debug_x264_params(
    params_string: str, encoder_threads: int, lookahead: int, duration: float
):
    """
    Função auxiliar de debug para x264-params.
    Exibe informações detalhadas sobre a string gerada.

    USAR APENAS DURANTE DEBUGGING (remover em produção).
    """
    vbv = get_vbv_preset(duration)

    console.print()
    console.print(
        "[cyan]═══════════════════════════════════════════════════════[/cyan]"
    )
    console.print("[bold cyan]🔍 DEBUG: x264-params String Analysis[/bold cyan]")
    console.print(
        "[cyan]═══════════════════════════════════════════════════════[/cyan]"
    )
    console.print()

    # Informações básicas
    console.print(f"[dim]Duração:[/dim] {duration:.1f}s")
    console.print(f"[dim]VBV Preset:[/dim] {vbv['description']}")
    console.print(f"[dim]Threads:[/dim] {encoder_threads}")
    console.print(f"[dim]Lookahead:[/dim] {lookahead}")
    console.print()

    # Análise da string
    console.print(f"[bold]String gerada:[/bold]")
    console.print(f"[green]Length:[/green] {len(params_string)} chars")
    console.print(f"[green]Type:[/green] {type(params_string)}")
    console.print()

    # Exibir string completa (quebrada em linhas para legibilidade)
    console.print(f"[bold]Conteúdo:[/bold]")
    params_list = params_string.split(":")
    for i, param in enumerate(params_list, 1):
        console.print(f"[dim]{i:2d}.[/dim] {param}")
    console.print()

    # Verificações de integridade
    console.print(f"[bold]Verificações:[/bold]")

    checks = [
        (" = ", "Espaço-Igual-Espaço"),
        (" =", "Espaço-Igual"),
        ("= ", "Igual-Espaço"),
        ("  ", "Espaços duplos"),
        ("\t", "Tabs"),
        ("\n", "Newlines"),
        ('"', "Aspas duplas"),
        ("'", "Aspas simples"),
    ]

    all_ok = True
    for pattern, name in checks:
        if pattern in params_string:
            console.print(f"[red]✗ {name}: DETECTADO![/red]")
            all_ok = False
        else:
            console.print(f"[green]✓ {name}: OK[/green]")

    if all_ok:
        console.print()
        console.print(
            "[bold green]✅ x264-params string está ÍNTEGRA (sem problemas detectados)[/bold green]"
        )
    else:
        console.print()
        console.print(
            "[bold red]❌ x264-params string tem PROBLEMAS (ver acima)[/bold red]"
        )

    console.print(
        "[cyan]═══════════════════════════════════════════════════════[/cyan]"
    )
    console.print()


def _build_metadata_args(
    duration: float, video_bitrate: int, mode: str, cineon_mode: bool = False
) -> list:
    """Gera metadados profissionais para o container MP4."""
    creation_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    vbv = get_vbv_preset(duration)
    vbv_preset_name = vbv["description"]
    vbv_maxrate = vbv["maxrate"]
    vbv_bufsize = vbv["bufsize"]
    # Comment personalizado por modo

    if cineon_mode:
        pipeline_tag = "Cineon+Kodak2383"
    else:
        pipeline_tag = "HollywoodLUT_v6.6"

    if mode == "crf":
        comment = f"{pipeline_tag} VBV:{vbv_preset_name} crf:15 max:{vbv_maxrate}k buf:{vbv_bufsize}k"
    else:
        comment = f"{pipeline_tag} VBV:{vbv_preset_name} target:{video_bitrate}k max:{vbv_maxrate}k buf:{vbv_bufsize}k"

    metadata = [
        "-metadata",
        f"creation_time={creation_time}",
        "-metadata",
        "encoder=Reels Encoder Hollywood LUT Transport",
        "-metadata",
        f"comment={comment}",
        "-metadata:s:v:0",
        "handler_name=VideoHandler",
        "-metadata:s:a:0",
        "handler_name=SoundHandler",
        "-movflags",
        "+faststart+write_colr",
        "-brand",
        "mp42",
    ]

    return metadata


def _run_encoding(ffmpeg_cmd, total_frames: int, cwd: Optional[str] = None):
    """Executa encoding com progress bar."""
    hud = ResolveProgressHUD(total_frames)
    process = subprocess.Popen(
        ffmpeg_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore",
        cwd=cwd,
    )
    t = threading.Thread(
        target=ffmpeg_live_reader, args=(process.stderr, hud), daemon=True
    )
    t.start()
    with Live(hud.render(), refresh_per_second=7, console=console) as live:
        while process.poll() is None:
            time.sleep(0.1)
            live.update(hud.render())

    try:
        stdout, stderr = process.communicate(timeout=1)
    except Exception:
        try:
            stderr = process.stderr.read()
        except Exception:
            stderr = ""

    t.join()

    if process.returncode != 0:
        console.print(
            f"[red]FFmpeg terminou com erro (codigo={process.returncode}):[/red]"
        )
        if stderr:
            lines = stderr.strip().splitlines()
            tail = "\n".join(lines[-40:])
            console.print(f"[red]{tail}[/red]")
        raise subprocess.CalledProcessError(
            process.returncode, ffmpeg_cmd, output=stdout, stderr=stderr
        )


# =============================================================================
# BUILD VIDEO FILTER - SCENE-REFERRED HDR PIPELINE
# =============================================================================
def build_scene_referred_hdr_pipeline(
    hdr_filter: str,
    scale_filter: Optional[str],
    target_resolution: Optional[Tuple[int, int]],
    tonemap_algorithm: str = "mobius",
) -> str:
    """
    Pipeline CORRETO para HDR sources: Scene-Referred Processing SEM LUT.

    FILOSOFIA:
    - Hollywood Cinema LUT v6.6 foi construída para SDR inputs (Rec.709/sRGB)
    - LUT espera coordenadas display-referred (gamma space), não HDR
    - Para HDR: TONEMAP apenas (sem LUT)

    Pipeline HDR (CORRETO):
        [SCALE] → LINEAR → CAS 0.35 → [TONEMAP] → CROP
                                                  (SEM LUT!)

    IMPORTANTE:
    - 🚫 LUT NÃO aplicada em HDR sources (seria coordenadas erradas)
    - ✅ Tonemap produz SDR (Rec.709) já gradado naturalmente
    - 🎨 Múltiplos algoritmos disponíveis (mobius, hable, reinhard)

    Args:
        hdr_filter: Filtro HDR→SDR (obrigatório, já validado)
        scale_filter: Filtro de downscale (opcional)
        target_resolution: Resolução alvo para crop (opcional)
        tonemap_algorithm: Algoritmo de tone mapping (mobius, hable, reinhard)
    """
    console.print(
        f"[bold magenta]🌟 PIPELINE HDR DETECTADO (TONEMAP: {tonemap_algorithm.upper()})[/bold magenta]"
    )
    console.print(
        "[dim]   LUT v6.6 não é aplicada em HDR (coordenadas SDR apenas)[/dim]"
    )

    parts = []

    # STAGE 1: Scale (se necessário)
    if scale_filter:
        parts.append(scale_filter)
        console.print("[green]✓ Scale:[/green] Lanczos downscale aplicado")

    # STAGE 2: Convert to LINEAR (scene-referred space)
    linear_conversion = "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709"
    parts.append(linear_conversion)
    console.print(
        "[green]✓ Linear:[/green] Convertido para scene-referred space (linear light)"
    )

    # STAGE 3: Sharpen em LINEAR SPACE (conservador para HDR)
    # CAS 0.35 é mais conservador que 0.45 - HDR tem mais detalhes intrínsecos
    parts.append("cas=strength=0.35")
    console.print(
        "[green]✓ Sharpen:[/green] CAS 0.35 em linear space (preserva highlights)"
    )

    # STAGE 4: Tone Mapping (linear → display-referred SDR)
    # Parâmetros otimizados por algoritmo PRESERVANDO CORES
    tonemap_configs = {
        "mobius": {
            "params": "param=0.5:desat=0:peak=100",
            "description": "Mobius suave (param=0.5, preserva cores)",
        },
        "hable": {
            "params": "desat=0:peak=100",
            "description": "Hable filmico (preserva cores)",
        },
        "reinhard": {
            "params": "param=0.6:desat=0:peak=100",
            "description": "Reinhard suave (param=0.6, preserva cores)",
        },
    }

    # Validar algoritmo
    if tonemap_algorithm not in tonemap_configs:
        console.print(
            f"[yellow]⚠ Tonemap '{tonemap_algorithm}' inválido, usando 'mobius'[/yellow]"
        )
        tonemap_algorithm = "mobius"

    config = tonemap_configs[tonemap_algorithm]
    tonemap_stage = f"tonemap={tonemap_algorithm}:{config['params']},zscale=t=bt709:m=bt709:r=tv,format=yuv420p"
    parts.append(tonemap_stage)
    console.print(f"[green]✓ Tonemap:[/green] HDR → SDR ({config['description']})")
    console.print("[dim]   Ajustado para evitar highlights estourados[/dim]")
    console.print(
        "[yellow]⚠ LUT v6.6 NÃO aplicada:[/yellow] Construída para SDR inputs apenas"
    )

    # STAGE 5: Crop final (remove macroblock padding)
    if target_resolution:
        tw, th = target_resolution
        parts.append(f"crop={tw}:{th}")
        console.print(f"[green]✓ Crop:[/green] {tw}×{th} (remove padding)")

    video_filter = ",".join(parts)

    console.print(
        Panel(
            f"[bold]Pipeline HDR ({tonemap_algorithm.upper()}):[/bold]\n[magenta]{video_filter}[/magenta]",
            title=f"🌟 HDR Pipeline (SEM LUT) - {tonemap_algorithm.upper()}",
            border_style="magenta",
        )
    )

    return video_filter


# =============================================================================
# BUILD VIDEO FILTER - SDR 32-BIT FLOAT PIPELINE (FASE 24)
# =============================================================================
def build_sdr_float_pipeline(
    scale_filter: Optional[str],
    target_resolution: Optional[Tuple[int, int]],
    lut_enabled: bool = True,
) -> str:
    """
    Pipeline SDR com 32-bit float (DaVinci Intermediate Simulado).

    FILOSOFIA:
    - Mantém color science Rec.709 (como LUT v6.6 espera)
    - Aumenta precisão matemática (32-bit float)
    - Elimina banding via float processing + high-end dither

    Pipeline:
        [SCALE] → IDT (32-bit) → CAS 0.30 → [LUT v6.6] → ODT (dither) → CROP
                  ↑ gbrpf32le  ↑          ↑            ↑ zscale+yuv420p ↑

    AJUSTE v1.4.1:
    - CAS reduzido de 0.45 → 0.30 (conservador, anti-banding)
    - Float permite ser mais suave sem perder definição
    - Sharpen forte + gradientes = banding artifacts

    Benefícios:
    - Zero banding (float elimina quantização)
    - Cores corretas (mantém Rec.709)
    - Sharpen suave (0.30 evita artifacts)
    - Compatível com LUT atual
    - Performance: ~10-20% mais lento (aceitável)

    Args:
        scale_filter: Filtro de downscale (opcional)
        target_resolution: Resolução alvo para crop (opcional)
        lut_enabled: Aplicar LUT v6.6 (default: True)
    """
    console.print(
        "[cyan]🎨 SDR Float Pipeline: IDT (32-bit) → Processing → ODT (8-bit)[/cyan]"
    )
    console.print("[dim]   DaVinci Intermediate simulado (float precision)[/dim]")

    parts = []

    # STAGE 1: Scale (se necessário)
    if scale_filter:
        parts.append(scale_filter)
        console.print("[green]✓ Scale:[/green] Lanczos downscale aplicado")

    # STAGE 2: IDT - Input Device Transform (8/10-bit → 32-bit float)
    parts.append("format=gbrpf32le")
    console.print("[green]✓ IDT:[/green] 8-bit → 32-bit float planar (gbrpf32le)")
    console.print("[dim]   Precisão infinita (elimina quantização)[/dim]")

    # STAGE 4: LUT v6.6B em Float Space (CONDICIONAL)
    if lut_enabled:
        lut_filename = (
            "HollywoodCinema_Ultimate_v6.7B_1.5IRE_Instagram8bit_NeutralShadows.cube"
        )

        # Verificar se LUT existe
        script_dir = os.path.dirname(os.path.abspath(__file__))
        lut_path = os.path.join(script_dir, lut_filename)

        if not os.path.exists(lut_path):
            console.print(f"[red]✗ LUT não encontrada: {lut_filename}[/red]")
            console.print(f"[yellow]  Execute: python hollywood_lut.py[/yellow]")
            raise FileNotFoundError(f"LUT não encontrada: {lut_path}")

        parts.append(f"lut3d=file={lut_filename}:interp=trilinear")
        console.print(f"[green]✓ LUT v6.6:[/green] {lut_filename} (trilinear em float)")
    else:
        console.print(f"[dim]○ LUT desativada (--lut off)[/dim]")

    # STAGE 5: ODT - Output Device Transform (32-bit float → 8-bit com dither)
    parts.append("zscale=t=bt709:m=bt709:r=tv")
    parts.append("format=yuv420p")
    console.print("[green]✓ ODT:[/green] 32-bit float → 8-bit YUV420p")

    # STAGE 6: Crop final (remove macroblock padding)
    if target_resolution:
        tw, th = target_resolution
        parts.append(f"crop={tw}:{th}")
        console.print(f"[green]✓ Crop:[/green] {tw}x{th} (remove padding)")

    video_filter = ",".join(parts)

    # Panel com destaque para float pipeline
    console.print(
        Panel(
            f"[bold]Float Pipeline:[/bold]\n[magenta]{video_filter}[/magenta]",
            title="🎬 SDR 32-bit Float (DaVinci Intermediate)",
            border_style="magenta",
        )
    )

    return video_filter


# =============================================================================
# BUILD VIDEO FILTER - HOLLYWOOD LUT TRANSPORT (SDR) - LEGACY 8-BIT
# =============================================================================
def build_hollywood_lut_filter(
    hdr_filter: Optional[str],
    scale_filter: Optional[str],
    target_resolution: Optional[Tuple[int, int]],
    lut_enabled: bool = True,
) -> str:
    """
    Pipeline ultra-simplificado: Hollywood LUT Transport (LEGACY 8-BIT).

    ⚠️ NOTA: Este é o pipeline 8-bit original (v1.3).
    Para máxima qualidade, use build_sdr_float_pipeline() (32-bit float).

    FILOSOFIA:
    - Confia 100% na LUT v6.6 para anti-banding (TPDF dithering)
    - Sem denoise (input deve ser relativamente limpo)
    - Sem deband (conflita com TPDF da LUT)
    - Sem grain (conflita com dithering da LUT)
    - Apenas: Scale, HDR→SDR, Sharpen, [LUT], Crop

    Pipeline 8-bit:
        [SCALE] → CAS 0.45 → [LUT v6.6] → CROP
        (todo processing em 8-bit)

    Args:
        hdr_filter: Filtro HDR→SDR (opcional)
        scale_filter: Filtro de downscale (opcional)
        target_resolution: Resolução alvo para crop (opcional)
        lut_enabled: Aplicar LUT v6.6 (default: True)
    """
    if lut_enabled:
        console.print("[cyan]🎨 Hollywood LUT Transport: Pipeline com LUT v6.6[/cyan]")
        console.print("[dim]   Confiando 100% na LUT v6.6 (TPDF Dithering)[/dim]")
    else:
        console.print("[cyan]⚡ Hollywood LUT Transport: Pipeline sem LUT[/cyan]")
        console.print("[dim]   Apenas: Scale → HDR→SDR → Sharpen → Crop[/dim]")

    parts = []

    # Prefix: Scale + HDR (ordem otimizada)
    if scale_filter:
        parts.append(scale_filter)
        console.print("[green]✓ Scale:[/green] Lanczos downscale aplicado")

    if hdr_filter:
        parts.append(hdr_filter)
        console.print("[green]✓ HDR→SDR:[/green] Tone mapping aplicado")

    # Core: Apenas CAS sharpen (0.45 é conservador)
    parts.append("cas=strength=0.45")
    console.print("[green]✓ Sharpen:[/green] CAS strength=0.40")

    # LUT v6.6 com trilinear interpolation (CONDICIONAL)
    if lut_enabled:
        lut_filename = (
            "HollywoodCinema_Ultimate_v6.7B_1.5IRE_Instagram8bit_NeutralShadows.cube"
        )

        # Verificar se LUT existe
        script_dir = os.path.dirname(os.path.abspath(__file__))
        lut_path = os.path.join(script_dir, lut_filename)

        if not os.path.exists(lut_path):
            console.print(f"[red]✗ LUT não encontrada: {lut_filename}[/red]")
            console.print(f"[yellow]  Execute: python hollywood_lut.py[/yellow]")
            raise FileNotFoundError(f"LUT não encontrada: {lut_path}")

        parts.append(f"lut3d=file={lut_filename}:interp=trilinear")
        console.print(f"[green]✓ LUT v6.6:[/green] {lut_filename} (trilinear)")
    else:
        console.print(f"[dim]○ LUT desativada (--lut off)[/dim]")

    # Crop final (remove macroblock padding)
    if target_resolution:
        tw, th = target_resolution
        parts.append(f"crop={tw}:{th}")
        console.print(f"[green]✓ Crop:[/green] {tw}x{th} (remove padding)")

    video_filter = ",".join(parts)

    # Panel com título e cor baseado em lut_enabled
    if lut_enabled:
        panel_title = "🎬 Hollywood LUT Transport (com LUT v6.6)"
        panel_border = "green"
    else:
        panel_title = "🎬 Hollywood LUT Transport (sem LUT)"
        panel_border = "yellow"

    console.print(
        Panel(
            f"[bold]Pipeline:[/bold]\n[green]{video_filter}[/green]",
            title=panel_title,
            border_style=panel_border,
        )
    )

    return video_filter


# =============================================================================
# BUILD VIDEO FILTER - AUTO DETECTION (HDR vs SDR) + FLOAT SUPPORT
# =============================================================================
def build_video_filter_auto(
    hdr_filter: Optional[str],
    scale_filter: Optional[str],
    target_resolution: Optional[Tuple[int, int]],
    lut_enabled: bool = True,
    tonemap_algorithm: str = "mobius",
    float_processing: bool = True,
) -> str:
    """
    Função INTELIGENTE de construção de pipeline com suporte a 32-bit float.
    Detecta automaticamente se deve usar:
    - Pipeline HDR (tonemap apenas, SEM LUT)
    - Pipeline SDR Float (32-bit, novo em v1.4)
    - Pipeline SDR 8-bit (legacy, v1.3)

    IMPORTANTE:
    - Hollywood Cinema LUT v6.6 foi construída para SDR inputs (Rec.709/sRGB)
    - LUT espera coordenadas display-referred, não HDR
    - HDR sources: TONEMAP apenas (sem LUT)
    - SDR sources: LUT aplicada (pipeline tradicional ou float)

    Args:
        hdr_filter: Filtro HDR→SDR (None = SDR source)
        scale_filter: Filtro de downscale (opcional)
        target_resolution: Resolução alvo para crop (opcional)
        lut_enabled: Aplicar LUT v6.6 (APENAS para SDR sources)
        tonemap_algorithm: Algoritmo de tone mapping para HDR (mobius, hable, reinhard)
        float_processing: Usar 32-bit float pipeline para SDR (default: True)

    Returns:
        String completa do filtro FFmpeg
    """
    if hdr_filter:
        # HDR SOURCE: Usa pipeline tonemap apenas (SEM LUT)
        console.print(
            f"[bold cyan]🎯 Modo: HDR Source (Tonemap: {tonemap_algorithm.upper()})[/bold cyan]"
        )
        console.print(
            "[yellow]⚠ LUT v6.6 desativada:[/yellow] Construída para SDR inputs apenas"
        )
        return build_scene_referred_hdr_pipeline(
            hdr_filter=hdr_filter,
            scale_filter=scale_filter,
            target_resolution=target_resolution,
            tonemap_algorithm=tonemap_algorithm,
        )
    else:
        # SDR SOURCE: Escolhe entre float (novo) ou 8-bit (legacy)
        if float_processing:
            console.print(
                "[bold cyan]🎯 Modo: SDR Source (32-bit Float Pipeline)[/bold cyan]"
            )
            console.print(
                "[dim]   DaVinci Intermediate simulado (máxima qualidade)[/dim]"
            )
            return build_sdr_float_pipeline(
                scale_filter=scale_filter,
                target_resolution=target_resolution,
                lut_enabled=lut_enabled,
            )
        else:
            console.print(
                "[bold cyan]🎯 Modo: SDR Source (8-bit Legacy Pipeline)[/bold cyan]"
            )
            console.print("[dim]   Pipeline v1.3 original (compatibilidade)[/dim]")
            return build_hollywood_lut_filter(
                hdr_filter=None,
                scale_filter=scale_filter,
                target_resolution=target_resolution,
                lut_enabled=lut_enabled,
            )


# =============================================================================
# MAIN ENCODING FUNCTION
# =============================================================================
def run_ffmpeg(
    input_file: str,
    output_file: str,
    mode: str = "crf",
    lut_enabled: bool = True,
    loudnorm_enabled: bool = True,
    hdr_mode: str = "auto",
    tonemap: str = "mobius",
    target_fps: str = "30",
    scale_mode: str = "auto",
    show_hardware: bool = True,
    threads_override: int = 0,
    performance_mode: str = "balanced",
    float_processing: bool = True,
):
    """
    Função principal de encoding - Hollywood LUT Transport.

    Pipeline ultra-simplificado sem denoise/deband/grain.
    Confia 100% na Hollywood Cinema LUT v6.6 para qualidade.

    NOVO v1.4: Suporta 32-bit float processing (DaVinci Intermediate simulado)
    """
    if mode == "crf":
        console.rule("[bold yellow]🎬 Encode CRF 15 - Hollywood LUT Transport")
    else:
        console.rule("[bold yellow]🎬 Encode 2-Pass - Hollywood LUT Transport")

    # Hardware detection
    hw_profile = detect_hardware()
    if show_hardware:
        print_hardware_profile(hw_profile)

    # Thread optimization
    if threads_override > 0:
        encoder_threads = threads_override
        filter_threads = max(2, threads_override // 2)
        decoder_threads = min(threads_override, hw_profile.cpu_cores)
        console.print(
            f"[cyan]🔧 Threads (override): encoder={encoder_threads}, filter={filter_threads}, decoder={decoder_threads}[/cyan]"
        )
    else:
        encoder_threads = hw_profile.recommended_threads
        filter_threads = hw_profile.recommended_filter_threads
        decoder_threads = hw_profile.recommended_decoder_threads

    # Performance mode adjustments
    if performance_mode == "quality":
        encoder_threads = min(encoder_threads + 2, hw_profile.cpu_threads)
        hw_profile.recommended_preset = "slow"
        hw_profile.recommended_lookahead = min(
            60, hw_profile.recommended_lookahead + 10
        )
        console.print(
            "[cyan]🎯 Performance Mode: QUALITY (máxima qualidade, mais lento)[/cyan]"
        )
    elif performance_mode == "speed":
        filter_threads = min(filter_threads + 2, hw_profile.cpu_cores)
        hw_profile.recommended_preset = "fast" if hw_profile.tier == "low" else "medium"
        hw_profile.recommended_lookahead = max(
            20, hw_profile.recommended_lookahead - 10
        )
        console.print(
            "[cyan]🚀 Performance Mode: SPEED (mais rápido, boa qualidade)[/cyan]"
        )

    # Paths absolutos
    input_file = os.path.abspath(input_file)
    output_file = os.path.abspath(output_file)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Frame rate (CFR)
    input_fps = get_input_fps(input_file)
    if target_fps == "auto":
        output_fps = input_fps
        console.print(
            f"[cyan]🎞️ Frame Rate: {output_fps} fps (auto-detectado) + CFR[/cyan]"
        )
    else:
        output_fps = int(target_fps)
        if input_fps > output_fps:
            console.print(
                f"[cyan]🎞️ Frame Rate: {input_fps} fps → {output_fps} fps (CFR + decimation)[/cyan]"
            )
        else:
            console.print(f"[cyan]🎞️ Frame Rate: {output_fps} fps (CFR fixo)[/cyan]")

    # Downscale automático
    scale_filter = None
    target_resolution = None

    if scale_mode == "auto":
        input_width, input_height = get_input_resolution(input_file)
        if input_width > 0 and input_height > 0:
            if input_height > input_width:
                scale_filter = build_scale_filter(input_width, input_height, 1080, 1920)
                target_resolution = (1080, 1920)
            else:
                scale_filter = build_scale_filter(input_width, input_height, 1920, 1080)
                target_resolution = (1920, 1080)

            if not scale_filter:
                console.print(
                    f"[green]✓ Resolução: {input_width}×{input_height} (sem downscale necessário)[/green]"
                )
                target_resolution = (input_width, input_height)
    else:
        console.print("[dim]○ Downscale desativado (--scale off)[/dim]")
        input_width, input_height = get_input_resolution(input_file)
        if input_width > 0 and input_height > 0:
            target_resolution = (input_width, input_height)

    # HDR detection & conversion
    hdr_filter = None
    if hdr_mode == "auto":
        hdr_info = detect_hdr_metadata(input_file)
        if hdr_info and hdr_info.get("is_hdr"):
            hdr_filter = build_hdr_to_sdr_filter(hdr_info, tonemap=tonemap)
    else:
        console.print("[dim]○ Detecção HDR desativada (--hdr off)[/dim]")

    # Build Video Filter (Auto HDR/SDR Detection + Float Support)
    video_filter = build_video_filter_auto(
        hdr_filter=hdr_filter,
        scale_filter=scale_filter,
        target_resolution=target_resolution,
        lut_enabled=lut_enabled,
        tonemap_algorithm=tonemap,
        float_processing=float_processing,
    )

    # Duração e VBV
    duration = get_video_duration(input_file)
    x264_params = _x264_params_string(
        duration, threads=encoder_threads, lookahead=hw_profile.recommended_lookahead
    )
    total_frames = get_total_frames(input_file)

    vbv = get_vbv_preset(duration)
    vbv_description = vbv["description"]
    video_bitrate = vbv["target"]

    console.print(f"[dim]📋 Duração: {duration:.1f}s | VBV: {vbv_description}[/dim]")
    console.print(
        f"[dim]🔧 Threads: encoder={encoder_threads}, filter={filter_threads}, decoder={decoder_threads}[/dim]"
    )

    # Audio loudness normalization
    audio_filter = None
    if loudnorm_enabled:
        console.print("[cyan]🔊 Loudnorm EBU R128 ativado[/cyan]")
        loudness_stats = analyze_audio_loudness(input_file, target="instagram")
        if loudness_stats:
            audio_filter = build_loudnorm_filter(loudness_stats, target="instagram")
        else:
            console.print("[yellow]⚠ Loudnorm desativado (falha na análise)[/yellow]")
    else:
        console.print("[dim]○ Loudnorm desativado (--loudnorm off)[/dim]")

    # CRF MODE
    if mode == "crf":
        metadata_args = _build_metadata_args(duration, video_bitrate, "crf")

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-threads",
            str(decoder_threads),
            "-filter_threads",
            str(filter_threads),
            "-filter_complex_threads",
            str(filter_threads),
            "-i",
            input_file,
            "-vf",
            video_filter,
            "-r",
            str(output_fps),
            "-fps_mode",
            "cfr",
            "-c:v",
            "libx264",
            "-preset",
            hw_profile.recommended_preset,
            "-crf",
            "15",
            "-profile:v",
            "high",
            "-level:v",
            "4.1",
            "-pix_fmt",
            "yuv420p",
            "-color_range",
            "tv",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-tune",
            "film",
            "-x264-params",
            x264_params,
        ]

        if audio_filter:
            ffmpeg_cmd.extend(["-af", audio_filter])

        ffmpeg_cmd.extend(
            [
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-profile:a",
                "aac_low",
                *metadata_args,
                output_file,
            ]
        )

        _run_encoding(ffmpeg_cmd, total_frames, cwd=script_dir)
        console.print("[green]✓ Render finalizado![/green]")

        if audio_filter:
            console.print(
                f"[dim]📋 Metadados: BT.709 TV | CRF 15 | VBV {vbv_description} | Loudnorm: -14 LUFS[/dim]"
            )
        else:
            console.print(
                f"[dim]📋 Metadados: BT.709 TV | CRF 15 | VBV {vbv_description}[/dim]"
            )

        console.print("[cyan]🔍 Validando MediaInfo (Studio Delivery)...[/cyan]")
        validate_media_info(output_file)
        return

    # 2-PASS MODE
    console.print(f"[yellow]📊 Bitrate:[/yellow] {video_bitrate}k")
    console.print(
        f"[dim]📐 Profile: High@4.1 | Color: BT.709 TV | Container: MP4 (ISO Base Media)[/dim]"
    )

    logfile = f"{output_file}_2pass"

    # Pass 1
    console.print("[cyan]📊 Pass 1: Analisando complexidade...[/cyan]")

    pass1_cmd = [
        "ffmpeg",
        "-y",
        "-threads",
        str(decoder_threads),
        "-filter_threads",
        str(filter_threads),
        "-filter_complex_threads",
        str(filter_threads),
        "-i",
        input_file,
        "-vf",
        video_filter,
        "-r",
        str(output_fps),
        "-fps_mode",
        "cfr",
        "-c:v",
        "libx264",
        "-preset",
        hw_profile.recommended_preset,
        "-b:v",
        f"{video_bitrate}k",
        "-profile:v",
        "high",
        "-level:v",
        "4.1",
        "-pix_fmt",
        "yuv420p",
        "-color_range",
        "tv",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-tune",
        "film",
        "-x264-params",
        x264_params,
        "-pass",
        "1",
        "-passlogfile",
        logfile,
        "-an",
        "-f",
        "null",
        DEVNULL_FF,
    ]

    _run_encoding(pass1_cmd, total_frames, cwd=script_dir)
    console.print("[green]✓ Pass 1 completo![/green]")

    # Pass 2
    console.print(
        "[cyan]🎬 Pass 2: Encoding final com metadados profissionais...[/cyan]"
    )

    metadata_args = _build_metadata_args(duration, video_bitrate, "2pass")

    pass2_cmd = [
        "ffmpeg",
        "-y",
        "-threads",
        str(decoder_threads),
        "-filter_threads",
        str(filter_threads),
        "-filter_complex_threads",
        str(filter_threads),
        "-i",
        input_file,
        "-vf",
        video_filter,
        "-r",
        str(output_fps),
        "-fps_mode",
        "cfr",
        "-c:v",
        "libx264",
        "-preset",
        hw_profile.recommended_preset,
        "-b:v",
        f"{video_bitrate}k",
        "-profile:v",
        "high",
        "-level:v",
        "4.1",
        "-pix_fmt",
        "yuv420p",
        "-color_range",
        "tv",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-tune",
        "film",
        "-x264-params",
        x264_params,
        "-pass",
        "2",
        "-passlogfile",
        logfile,
    ]

    if audio_filter:
        pass2_cmd.extend(["-af", audio_filter])

    pass2_cmd.extend(
        [
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-profile:a",
            "aac_low",
            *metadata_args,
            output_file,
        ]
    )

    _run_encoding(pass2_cmd, total_frames, cwd=script_dir)

    # Limpar logs temporários
    for ext in ["-0.log", "-0.log.mbtree"]:
        log_path = f"{logfile}{ext}"
        if os.path.exists(log_path):
            try:
                os.remove(log_path)
            except OSError:
                pass

    console.print("[green]✓ Render 2-Pass finalizado![/green]")

    if audio_filter:
        console.print(
            f"[dim]📋 Metadados: BT.709 TV | Profile High@4.1 | VBV {vbv_description} | Loudnorm: -14 LUFS[/dim]"
        )
    else:
        console.print(
            f"[dim]📋 Metadados: BT.709 TV | Profile High@4.1 | VBV {vbv_description}[/dim]"
        )

    console.print("[cyan]🔍 Validando MediaInfo (Studio Delivery)...[/cyan]")
    validate_media_info(output_file)


# =============================================================================
# MEDIAINFO VALIDATION
# =============================================================================
def validate_media_info(file_path: str):
    """Valida metadados do arquivo gerado."""
    console.print(
        "\n[bold cyan]📋 MediaInfo Validation — Studio Delivery[/bold cyan]\n"
    )

    try:
        media_info = MediaInfo.parse(file_path)
    except Exception as e:
        console.print(f"[red]Erro ao parsear MediaInfo: {e}[/red]")
        return

    video = next((t for t in media_info.tracks if t.track_type == "Video"), None)
    audio = next((t for t in media_info.tracks if t.track_type == "Audio"), None)
    general = next((t for t in media_info.tracks if t.track_type == "General"), None)

    def ok(label, value):
        console.print(f"[green]✓ {label}: {value}[/green]")

    def fail(label, value, expected):
        console.print(f"[red]✗ {label}: {value} (esperado: {expected})[/red]")

    # CONTAINER
    console.print("[bold]📦 CONTAINER[/bold]")

    if not general:
        console.print("[red]✗ Track General não encontrada[/red]")
    else:
        if general.format == "MPEG-4":
            ok("Container", "MPEG-4")
        else:
            fail("Container", general.format, "MPEG-4")

        if general.writing_application:
            ok("Writing application", general.writing_application)

        if hasattr(general, "comment") and general.comment:
            ok("Comment (VBV)", general.comment)

    # VIDEO
    console.print("\n[bold]🎥 VIDEO[/bold]")

    if not video:
        console.print("[red]✗ Track de vídeo não encontrada[/red]")
    else:
        if video.format in ("AVC", "H.264"):
            ok("Codec", video.format)
        else:
            fail("Codec", video.format, "AVC / H.264")

        if video.codec_id and video.codec_id.startswith("avc"):
            ok("Codec ID", video.codec_id)
        else:
            fail("Codec ID", video.codec_id, "avc1")

        if (
            video.format_profile
            and "High" in video.format_profile
            and "4.1" in video.format_profile
        ):
            ok("Profile", video.format_profile)
        else:
            fail("Profile", video.format_profile, "High@4.1")

        if video.bit_depth == 8:
            ok("Bit depth", "8-bit")
        else:
            fail("Bit depth", video.bit_depth, "8-bit")

        if video.color_range in ("Limited", "TV"):
            ok("Color range", video.color_range)
        else:
            fail("Color range", video.color_range, "Limited / TV")

        if video.color_primaries == "BT.709":
            ok("Color primaries", "BT.709")
        else:
            fail("Color primaries", video.color_primaries, "BT.709")

        if video.transfer_characteristics == "BT.709":
            ok("Transfer characteristics", "BT.709")
        else:
            fail("Transfer characteristics", video.transfer_characteristics, "BT.709")

        if video.matrix_coefficients == "BT.709":
            ok("Matrix coefficients", "BT.709")
        else:
            fail("Matrix coefficients", video.matrix_coefficients, "BT.709")

        if video.width == 1080 and video.height == 1920:
            ok("Resolution", "1080x1920")
        else:
            console.print(
                f"[yellow]• Resolution: {video.width}x{video.height} (verificar uso)[/yellow]"
            )

        try:
            fps = float(video.frame_rate)
            if 23.976 <= fps <= 60:
                ok("Frame rate", f"{fps:.3f} fps")
            else:
                fail("Frame rate", fps, "24-60 fps")
        except Exception:
            console.print("[yellow]• Frame rate: não detectável[/yellow]")

    # AUDIO
    console.print("\n[bold]🔊 AUDIO[/bold]")

    if not audio:
        console.print("[yellow]⚠ Sem trilha de áudio (Instagram aceita)[/yellow]")
    else:
        if audio.format == "AAC":
            ok("Codec", "AAC")
        else:
            fail("Codec", audio.format, "AAC")

        if audio.format_profile in ("LC", "AAC LC"):
            ok("Profile", "AAC-LC")
        else:
            console.print(
                f"[yellow]• Profile: {audio.format_profile} (aceitável)[/yellow]"
            )

        if audio.channel_s == 2:
            ok("Channels", "2.0")
        else:
            fail("Channels", audio.channel_s, "2")

        if audio.sampling_rate and int(audio.sampling_rate) == 48000:
            ok("Sampling rate", "48 kHz")
        else:
            fail("Sampling rate", audio.sampling_rate, "48000")

    console.print("\n[bold green]✅ Validação Studio Delivery concluída[/bold green]\n")


def analyze_with_mediainfo(output_file: str):
    """Exibe MediaInfo completo em JSON."""
    try:
        out = subprocess.check_output(
            ["mediainfo", "--Output=JSON", output_file],
            stderr=subprocess.PIPE,
            encoding="utf-8",
        )
        console.rule("[bold cyan]📊 MediaInfo (JSON)")
        console.print(json.dumps(json.loads(out), indent=4))
    except subprocess.CalledProcessError as e:
        console.print(f"[red]MediaInfo retornou erro:[/red] {e}")
    except FileNotFoundError:
        console.print("[yellow]mediainfo CLI não encontrado no PATH.[/yellow]")


def run_ffmpeg_with_cineon(
    input_file: str,
    output_file: str,
    mode: str = "crf",
    lut_path: str = "Rec709_Kodak2383_D60.cube",
    exposure_offset: float = 0.0,
    saturation: float = 1.0,
    lut_strength: float = 1.0,
    loudnorm_enabled: bool = True,
    target_fps: str = "30",
    scale_mode: str = "auto",
    show_hardware: bool = True,
    threads_override: int = 0,
    performance_mode: str = "balanced",
    cineon_lut_path: Optional[str] = None,
):
    """
    Encoding com pipeline Cineon (PyAV + 5 nodes + Kodak 2383).

    CORREÇÕES v2.0.1:
    - FIX 1: Debug detalhado de x264-params (identificar problemas de parsing)
    - FIX 2: Metadados de cor com +write_colr (forçar escrita no container)

    Pipeline:
        PyAV decode → Cineon (5 nodes) → FFmpeg pipe (stdin) → libx264 → MP4

    Args:
        input_file: Arquivo de entrada
        output_file: Arquivo de saída
        mode: "crf" ou "2pass"
        exposure_offset: Ajuste de exposição (-2.0 a +2.0 stops)
        saturation: Multiplicador de saturação (0.0 a 2.0)
        loudnorm_enabled: Normalização de áudio EBU R128
        target_fps: Frame rate do output
        scale_mode: Downscale automático
        show_hardware: Exibir perfil de hardware
        threads_override: Override manual de threads
        performance_mode: "quality", "balanced", "speed"
        cineon_lut_path: Caminho customizado para LUT Kodak 2383
    """
    console.rule("[bold magenta]🎬 Encode Cineon Film Emulation Pipeline")

    # ═══════════════════════════════════════════════════════════════
    # HARDWARE DETECTION & OPTIMIZATION
    # ═══════════════════════════════════════════════════════════════

    hw_profile = detect_hardware()
    if show_hardware:
        print_hardware_profile(hw_profile)

    # Thread optimization
    if threads_override > 0:
        encoder_threads = threads_override
        filter_threads = max(2, threads_override // 2)
        decoder_threads = min(threads_override, hw_profile.cpu_cores)
    else:
        encoder_threads = hw_profile.recommended_threads
        filter_threads = hw_profile.recommended_filter_threads
        decoder_threads = hw_profile.recommended_decoder_threads

    # Performance mode adjustments
    if performance_mode == "quality":
        encoder_threads = min(encoder_threads + 2, hw_profile.cpu_threads)
        hw_profile.recommended_preset = "slow"
        hw_profile.recommended_lookahead = min(
            60, hw_profile.recommended_lookahead + 10
        )
        console.print("[cyan]🎯 Performance Mode: QUALITY (máxima qualidade)[/cyan]")
    elif performance_mode == "speed":
        filter_threads = min(filter_threads + 2, hw_profile.cpu_cores)
        hw_profile.recommended_preset = "fast" if hw_profile.tier == "low" else "medium"
        hw_profile.recommended_lookahead = max(
            20, hw_profile.recommended_lookahead - 10
        )
        console.print("[cyan]🚀 Performance Mode: SPEED (mais rápido)[/cyan]")

    # ═══════════════════════════════════════════════════════════════
    # INPUT ANALYSIS
    # ═══════════════════════════════════════════════════════════════

    input_file = os.path.abspath(input_file)
    output_file = os.path.abspath(output_file)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Frame rate
    input_fps = get_input_fps(input_file)
    if target_fps == "auto":
        output_fps = input_fps
        console.print(
            f"[cyan]🎞️ Frame Rate: {output_fps} fps (auto-detectado) + CFR[/cyan]"
        )
    else:
        output_fps = int(target_fps)
        if input_fps > output_fps:
            console.print(
                f"[cyan]🎞️ Frame Rate: {input_fps} fps → {output_fps} fps (CFR + decimation)[/cyan]"
            )
        else:
            console.print(f"[cyan]🎞️ Frame Rate: {output_fps} fps (CFR fixo)[/cyan]")

    # Resolução e downscale
    scale_filter = None
    target_resolution = None

    if scale_mode == "auto":
        input_width, input_height = get_input_resolution(input_file)
        if input_width > 0 and input_height > 0:
            if input_height > input_width:
                scale_filter = build_scale_filter(input_width, input_height, 1080, 1920)
                target_resolution = (1080, 1920)
            else:
                scale_filter = build_scale_filter(input_width, input_height, 1920, 1080)
                target_resolution = (1920, 1080)

            if not scale_filter:
                console.print(
                    f"[green]✓ Resolução: {input_width}×{input_height} (sem downscale)[/green]"
                )
                target_resolution = (input_width, input_height)
    else:
        console.print("[dim]○ Downscale desativado (--scale off)[/dim]")
        input_width, input_height = get_input_resolution(input_file)
        if input_width > 0 and input_height > 0:
            target_resolution = (input_width, input_height)

    # Duração e VBV
    duration = get_video_duration(input_file)
    total_frames = get_total_frames(input_file)

    # ═══════════════════════════════════════════════════════════════
    # FIX 1: x264-params com DEBUG DETALHADO
    # ═══════════════════════════════════════════════════════════════

    x264_params = _x264_params_string(
        duration, threads=encoder_threads, lookahead=hw_profile.recommended_lookahead
    )

    # DEBUG: Exibir análise completa da string x264-params
    _debug_x264_params(
        x264_params, encoder_threads, hw_profile.recommended_lookahead, duration
    )

    vbv = get_vbv_preset(duration)
    vbv_description = vbv["description"]
    video_bitrate = vbv["target"]

    console.print(f"[dim]📋 Duração: {duration:.1f}s | VBV: {vbv_description}[/dim]")
    console.print(
        f"[dim]🔧 Threads: encoder={encoder_threads}, filter={filter_threads}, decoder={decoder_threads}[/dim]"
    )

    # ═══════════════════════════════════════════════════════════════
    # AUDIO LOUDNESS NORMALIZATION
    # ═══════════════════════════════════════════════════════════════

    audio_filter = None
    if loudnorm_enabled:
        console.print("[cyan]🔊 Loudnorm EBU R128 ativado[/cyan]")
        loudness_stats = analyze_audio_loudness(input_file, target="instagram")
        if loudness_stats:
            audio_filter = build_loudnorm_filter(loudness_stats, target="instagram")
        else:
            console.print("[yellow]⚠ Loudnorm desativado (falha na análise)[/yellow]")
    else:
        console.print("[dim]○ Loudnorm desativado (--loudnorm off)[/dim]")

    # ═══════════════════════════════════════════════════════════════
    # CINEON PIPELINE INITIALIZATION
    # ═══════════════════════════════════════════════════════════════

    console.print()
    console.print(
        "[bold magenta]🎬 Inicializando Cineon Film Emulation Pipeline[/bold magenta]"
    )
    console.print(
        f"[dim]   5 Nodes: DWG Transform → Grading → Gamut Map → Log → Kodak 2383[/dim]"
    )
    console.print(
        f"[dim]   Exposure: {exposure_offset:+.1f} stops | Saturation: {saturation:.2f}x | LUT Strength: {lut_strength*100:.0f}%[/dim]"
    )
    console.print()

    # Carregar LUT Kodak 2383
    if cineon_lut_path is None:
        cineon_lut_path = os.path.join(script_dir, "Rec709_Kodak2383_D60.cube")

    if not os.path.exists(cineon_lut_path):
        console.print(f"[red]✗ LUT Kodak 2383 não encontrada: {cineon_lut_path}[/red]")
        raise FileNotFoundError(f"LUT não encontrada: {cineon_lut_path}")

    from cineon_pipeline import LUT3D, process_frame_full_pipeline

    kodak_lut = LUT3D(cineon_lut_path)
    console.print(
        f"[green]✓ LUT Kodak 2383 carregada: {os.path.basename(cineon_lut_path)}[/green]"
    )
    console.print(
        f"[dim]   Size: {kodak_lut.lut_size}³ ({kodak_lut.lut_size**3:,} pontos)[/dim]"
    )
    console.print(
        f"[dim]   Domain: [{kodak_lut.domain_min[0]:.2f}, {kodak_lut.domain_max[0]:.2f}][/dim]"
    )

    # ═══════════════════════════════════════════════════════════════
    # FIX 2: METADADOS DE COR COM +write_colr
    # ═══════════════════════════════════════════════════════════════

    metadata_args = _build_metadata_args(
        duration, video_bitrate, mode, cineon_mode=True
    )

    console.print("[green]✓ Metadados configurados (com +write_colr)[/green]")

    # ═══════════════════════════════════════════════════════════════
    # FFMPEG COMMAND CONSTRUCTION
    # ═══════════════════════════════════════════════════════════════

    if mode == "crf":
        console.print(f"[yellow]📊 Modo: CRF 15 | VBV: {vbv_description}[/yellow]")
    else:
        console.print(
            f"[yellow]📊 Modo: 2-Pass | Bitrate: {video_bitrate}k | VBV: {vbv_description}[/yellow]"
        )

    # Comando FFmpeg base (recebe frames via stdin pipe)
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        (
            f"{target_resolution[0]}x{target_resolution[1]}"
            if target_resolution
            else "1080x1920"
        ),
        "-r",
        str(output_fps),
        "-i",
        "-",  # ← stdin pipe (binary mode)
        # Audio input (separado)
        "-i",
        input_file,
        # Stream mapping
        "-map",
        "0:v:0",  # Vídeo do pipe (stdin)
        "-map",
        "1:a:0?",  # Áudio do input file (opcional)
        # Video encoding
        "-c:v",
        "libx264",
        "-preset",
        hw_profile.recommended_preset,
    ]

    # CRF ou 2-pass
    if mode == "crf":
        ffmpeg_cmd.extend(
            [
                "-crf",
                "15",
            ]
        )
    else:
        ffmpeg_cmd.extend(
            [
                "-b:v",
                f"{video_bitrate}k",
            ]
        )

    # x264 profile & level
    ffmpeg_cmd.extend(
        [
            "-profile:v",
            "high",
            "-level:v",
            "4.1",
            "-pix_fmt",
            "yuv420p",
        ]
    )

    ffmpeg_cmd.extend(
        [
            "-color_range",
            "tv",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-bsf:v",
            "h264_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1",
        ]
    )
    console.print("[dim]   Color metadata: BT.709 TV range (posição otimizada)[/dim]")

    ffmpeg_cmd.extend(
        [
            "-tune",
            "film",
            "-x264-params",
            x264_params,
        ]
    )

    # Audio filter (loudnorm)
    if audio_filter:
        ffmpeg_cmd.extend(["-af", audio_filter])

    ffmpeg_cmd.extend(
        [
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-profile:a",
            "aac_low",
        ]
    )

    # ═══════════════════════════════════════════════════════════════
    # METADATA & CONTAINER FLAGS
    # ═══════════════════════════════════════════════════════════════
    # CRÍTICO: -movflags +write_colr DEVE vir DEPOIS de todos os
    # argumentos de encoding, mas ANTES do output filename
    metadata_args = _build_metadata_args(
        duration, video_bitrate, mode, cineon_mode=True
    )
    ffmpeg_cmd.extend(metadata_args)

    console.print(
        "[dim]   Container flags: +faststart+write_colr (escrita forçada)[/dim]"
    )

    # ═══════════════════════════════════════════════════════════════
    # OUTPUT
    # ═══════════════════════════════════════════════════════════════
    ffmpeg_cmd.append(output_file)

    # ═══════════════════════════════════════════════════════════════
    # DEBUG: Exibir comando FFmpeg completo
    # ═══════════════════════════════════════════════════════════════

    console.print()
    console.print(
        "[cyan]═══════════════════════════════════════════════════════[/cyan]"
    )
    console.print("[bold cyan]🔍 DEBUG: Comando FFmpeg Completo[/bold cyan]")
    console.print(
        "[cyan]═══════════════════════════════════════════════════════[/cyan]"
    )

    # Salvar comando em arquivo para análise
    debug_cmd_file = os.path.join(script_dir, "ffmpeg_command_debug.txt")
    with open(debug_cmd_file, "w", encoding="utf-8") as f:
        f.write(" ".join(ffmpeg_cmd))
        f.write("\n\n")
        f.write("# Argumentos separados:\n")
        for i, arg in enumerate(ffmpeg_cmd, 1):
            f.write(f"{i:3d}. {arg}\n")

    console.print(f"[green]✓ Comando salvo em: {debug_cmd_file}[/green]")
    console.print()

    # Exibir apenas partes críticas no terminal
    console.print("[dim]Partes críticas:[/dim]")
    for i, arg in enumerate(ffmpeg_cmd):
        if arg in (
            "-x264-params",
            "-colorspace",
            "-color_primaries",
            "-color_trc",
            "-movflags",
        ):
            console.print(f"[dim]{i:3d}.[/dim] [yellow]{arg}[/yellow]")
            if i + 1 < len(ffmpeg_cmd):
                console.print(f"[dim]{i+1:3d}.[/dim] [green]{ffmpeg_cmd[i+1]}[/green]")

    console.print(
        "[cyan]═══════════════════════════════════════════════════════[/cyan]"
    )
    console.print()

    # ═══════════════════════════════════════════════════════════════
    # PYAV VIDEO DECODE & CINEON PROCESSING
    # ═══════════════════════════════════════════════════════════════

    console.print("[cyan]🎬 Iniciando encoding com pipeline Cineon...[/cyan]")
    console.print()

    import av

    # Abrir container com PyAV
    try:
        container = av.open(input_file)
    except Exception as e:
        console.print(f"[red]Erro ao abrir input com PyAV: {e}[/red]")
        raise

    video_stream = container.streams.video[0]
    console.print(
        f"[dim]   Input: {video_stream.width}×{video_stream.height} @ {video_stream.average_rate} fps[/dim]"
    )
    console.print(f"[dim]   Codec: {video_stream.codec_context.name}[/dim]")
    console.print()

    # Iniciar subprocess FFmpeg (stdin=PIPE para receber frames)
    try:
        ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=script_dir,
        )
    except Exception as e:
        console.print(f"[red]Erro ao iniciar FFmpeg subprocess: {e}[/red]")
        container.close()
        raise

    console.print(
        f"[green]✓ FFmpeg subprocess iniciado (PID: {ffmpeg_process.pid})[/green]"
    )

    # Progress HUD
    hud = ResolveProgressHUD(total_frames)

    # Thread para capturar stderr do FFmpeg em tempo real
    def ffmpeg_stderr_reader(pipe, hud):
        """
        Lê stderr do FFmpeg em tempo real para atualizar progress HUD.

        CORREÇÃO: Trata gracefully o fechamento do pipe.
        """
        try:
            for line in iter(pipe.readline, b""):
                if not line:
                    break
                try:
                    line_str = line.decode("utf-8", errors="ignore")
                    if "frame=" in line_str:
                        parts = line_str.split("frame=")
                        if len(parts) > 1:
                            frame_str = parts[1].split()[0]
                            frame_num = int(
                                "".join(ch for ch in frame_str if ch.isdigit())
                            )
                            hud.update_frame(frame_num)
                except Exception:
                    pass
        except (OSError, ValueError):
            pass

    stderr_thread = threading.Thread(
        target=ffmpeg_stderr_reader, args=(ffmpeg_process.stderr, hud), daemon=True
    )
    stderr_thread.start()

    # ═══════════════════════════════════════════════════════════════
    # MAIN PROCESSING LOOP
    # ═══════════════════════════════════════════════════════════════

    frame_count = 0
    error_occurred = False

    with Live(hud.render(), refresh_per_second=7, console=console) as live:
        try:
            for frame in container.decode(video=0):
                # PyAV frame → NumPy array (RGB)
                frame_rgb = frame.to_ndarray(format="rgb24")[:, :, ::-1]

                frame_rgb_normalized = frame_rgb.astype(np.float32) / 255.0

                # Cineon pipeline (5 nodes)
                frame_processed = process_frame_full_pipeline(
                    frame_rgb_normalized,
                    kodak_lut,
                    exposure_offset=exposure_offset,
                    saturation=saturation,
                    lut_strength=lut_strength,
                )

                # Validação: Garantir array C-contiguous uint8
                if not frame_processed.flags["C_CONTIGUOUS"]:
                    frame_processed = np.ascontiguousarray(frame_processed)

                if frame_processed.dtype != np.uint8:
                    frame_processed = np.clip(frame_processed * 255.0, 0, 255).astype(
                        np.uint8
                    )

                # Validar dimensões esperadas
                expected_shape = (
                    (target_resolution[1], target_resolution[0], 3)
                    if target_resolution
                    else (1920, 1080, 3)
                )
                if frame_processed.shape != expected_shape:
                    console.print(
                        f"[red]✗ Frame {frame_count}: shape incorreta {frame_processed.shape}, esperado {expected_shape}[/red]"
                    )
                    error_occurred = True
                    break

                # Converter para bytes
                frame_bytes = frame_processed.tobytes()

                # Escrever no pipe do FFmpeg (binary mode)
                try:
                    ffmpeg_process.stdin.write(frame_bytes)
                except (BrokenPipeError, OSError) as e:
                    console.print(f"[red]✗ Erro ao escrever no pipe FFmpeg: {e}[/red]")
                    console.print(
                        f"[yellow]   FFmpeg process poll: {ffmpeg_process.poll()}[/yellow]"
                    )
                    error_occurred = True
                    break

                frame_count += 1
                hud.update_frame(frame_count)
                live.update(hud.render())

                # Check se FFmpeg morreu prematuramente
                if ffmpeg_process.poll() is not None:
                    console.print(
                        f"[red]✗ FFmpeg terminou prematuramente (returncode={ffmpeg_process.poll()})[/red]"
                    )
                    error_occurred = True
                    break

        except KeyboardInterrupt:
            console.print("\n[yellow]⚠ Interrompido pelo usuário[/yellow]")
            error_occurred = True

        except Exception as e:
            console.print(f"\n[red]✗ Erro durante processamento: {e}[/red]")
            import traceback

            traceback.print_exc()
            error_occurred = True

        finally:
            # Fechar pipe de escrita (sinaliza EOF para FFmpeg)
            try:
                if ffmpeg_process.stdin:
                    ffmpeg_process.stdin.close()
            except Exception:
                pass

            # Fechar container PyAV
            container.close()

    # ═══════════════════════════════════════════════════════════════
    # WAIT FOR FFMPEG COMPLETION
    # ═══════════════════════════════════════════════════════════════

    if not error_occurred:
        console.print()
        console.print(
            "[cyan]⏳ Aguardando finalização do FFmpeg (muxing final)...[/cyan]"
        )

        try:
            stdout, stderr = ffmpeg_process.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            console.print("[red]✗ FFmpeg timeout (60s). Forçando término...[/red]")
            ffmpeg_process.kill()
            stdout, stderr = ffmpeg_process.communicate()

        returncode = ffmpeg_process.returncode

        if returncode != 0:
            console.print(f"[red]✗ FFmpeg retornou erro (code={returncode})[/red]")
            console.print()
            console.print("[bold red]FFmpeg stderr (últimas 50 linhas):[/bold red]")

            stderr_str = stderr.decode("utf-8", errors="ignore")
            stderr_lines = stderr_str.strip().splitlines()
            for line in stderr_lines[-50:]:
                console.print(f"[red]{line}[/red]")

            raise subprocess.CalledProcessError(
                returncode, ffmpeg_cmd, output=stdout, stderr=stderr
            )
        else:
            console.print(
                f"[green]✓ FFmpeg finalizado com sucesso ({frame_count} frames)[/green]"
            )

            # ═══════════════════════════════════════════════════════════
            # FIX 9.1: REMUX PARA INJETAR 'COLR' ATOM (v2.0.3)
            # ═══════════════════════════════════════════════════════════
            # Problema: rawvideo pipe stdin não permite MP4 muxer escrever 'colr' atom
            # Solução: Remux com stream copy + metadados de cor explícitos

            console.print()
            console.print(
                "[cyan]🔄 Pós-processamento: Injetando 'colr' atom no container MP4...[/cyan]"
            )

            # Arquivo temporário para output original
            output_temp = output_file.replace(".mp4", "_temp.mp4")

            # Renomear output original para temp
            try:
                shutil.move(output_file, output_temp)
            except Exception as e:
                console.print(
                    f"[yellow]⚠️ Erro ao renomear arquivo temporário: {e}[/yellow]"
                )
                console.print(f"[yellow]   Continuando sem remux...[/yellow]")
            else:
                # Comando de remux (stream copy, sem re-encode)
                remux_cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    output_temp,
                    "-c",
                    "copy",  # Stream copy (sem re-encode)
                    "-color_primaries",
                    "bt709",
                    "-color_trc",
                    "bt709",
                    "-colorspace",
                    "bt709",
                    "-color_range",
                    "tv",
                    "-movflags",
                    "+faststart+write_colr",
                    output_file,
                ]

                try:
                    console.print("[dim]   Executando remux (stream copy)...[/dim]")
                    subprocess.run(
                        remux_cmd,
                        check=True,
                        capture_output=True,
                        cwd=script_dir,
                    )
                    console.print("[green]✓ 'colr' atom injetado com sucesso[/green]")
                    console.print(
                        "[dim]   Metadados MP4 container: BT.709 TV range[/dim]"
                    )

                    # Remover arquivo temporário
                    try:
                        os.remove(output_temp)
                    except Exception:
                        pass

                except subprocess.CalledProcessError as e:
                    console.print(f"[red]✗ Erro no remux: {e}[/red]")
                    console.print("[yellow]   Restaurando arquivo original...[/yellow]")

                    # Restaurar arquivo original
                    try:
                        shutil.move(output_temp, output_file)
                    except Exception:
                        pass
    else:
        # Houve erro, terminar FFmpeg
        console.print("[yellow]⚠ Encerrando FFmpeg devido a erro...[/yellow]")

        try:
            ffmpeg_process.terminate()
            ffmpeg_process.wait(timeout=10)
        except Exception:
            ffmpeg_process.kill()

        raise RuntimeError("Encoding interrompido por erro no processamento")

    # ═══════════════════════════════════════════════════════════════
    # VALIDATION
    # ═══════════════════════════════════════════════════════════════

    console.print()
    console.print("[green]✅ Render Cineon finalizado![/green]")

    if audio_filter:
        console.print(
            f"[dim]📋 Metadados: BT.709 TV | Pipeline Cineon | VBV {vbv_description} | Loudnorm: -14 LUFS[/dim]"
        )
    else:
        console.print(
            f"[dim]📋 Metadados: BT.709 TV | Pipeline Cineon | VBV {vbv_description}[/dim]"
        )

    # ═══════════════════════════════════════════════════════════════
    # VALIDAÇÃO ADICIONAL: ffprobe direto
    # ═══════════════════════════════════════════════════════════════

    console.print()
    console.print("[cyan]🔬 Validação técnica com ffprobe...[/cyan]")

    try:
        probe_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=color_primaries,color_transfer,color_space,color_range",
            "-of",
            "default=noprint_wrappers=1",
            output_file,
        ]

        probe_result = subprocess.run(
            probe_cmd, capture_output=True, text=True, check=True
        )
        probe_output = probe_result.stdout.strip()

        console.print("[dim]   ffprobe output:[/dim]")
        for line in probe_output.splitlines():
            if "color_primaries" in line:
                if "bt709" in line:
                    console.print(f"[green]   ✓ {line}[/green]")
                else:
                    console.print(f"[red]   ✗ {line} (esperado: bt709)[/red]")
            elif "color_transfer" in line:
                if "bt709" in line:
                    console.print(f"[green]   ✓ {line}[/green]")
                else:
                    console.print(f"[red]   ✗ {line} (esperado: bt709)[/red]")
            elif "color_space" in line:
                if "bt709" in line:
                    console.print(f"[green]   ✓ {line}[/green]")
                else:
                    console.print(f"[yellow]   • {line}[/yellow]")
            elif "color_range" in line:
                if "tv" in line.lower():
                    console.print(f"[green]   ✓ {line}[/green]")
                else:
                    console.print(f"[yellow]   • {line}[/yellow]")

        console.print()

    except subprocess.CalledProcessError:
        console.print("[yellow]   ⚠ ffprobe falhou (não crítico)[/yellow]")

    # MediaInfo validation (existente)
    console.print("[cyan]🔍 Validando MediaInfo (Studio Delivery)...[/cyan]")
    validate_media_info(output_file)

    console.print()
    console.print(
        f"[bold green]✅ FASE 26 COMPLETA - Output: {os.path.basename(output_file)}[/bold green]"
    )


# =============================================================================
# MAIN
# =============================================================================
def main():
    console.rule(
        "[bold magenta]🎞️ Instagram Reels Encoder - Cineon Film Emulation Edition v2.0"
    )
    parser = argparse.ArgumentParser(
        description="Instagram Reels Encoder com suporte a Film Emulation Cineon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            """Exemplos de uso:

MODO FFMPEG (default, rápido):
  python Reels_Encoder_v2.py input.mp4                           # Float 32-bit + LUT v6.6
  python Reels_Encoder_v2.py 4k_video.mp4                        # 4K 60fps → 1080p 30fps (auto)
  python Reels_Encoder_v2.py input.mp4 --mode 2pass              # 2-Pass + Loudnorm
  python Reels_Encoder_v2.py input.mp4 --lut off                 # Sem LUT (apenas scale/sharpen)
  python Reels_Encoder_v2.py input.mp4 --fps 60                  # 60 fps CFR (ação/esportes)
  python Reels_Encoder_v2.py iphone_dolby.mov                    # HDR→SDR + 4K→1080p automático
  python Reels_Encoder_v2.py input.mp4 --float off               # 8-bit legacy (v1.3)

MODO CINEON (film emulation):
  python Reels_Encoder_v2.py input.mp4 --cineon-pipeline on                    # Film look Kodak 2383
  python Reels_Encoder_v2.py input.mp4 --cineon-pipeline on --exposure-offset +0.5    # Ajuste exposure
  python Reels_Encoder_v2.py input.mp4 --cineon-pipeline on --saturation 1.2   # Ajuste saturation

OUTROS:
  python Reels_Encoder_v2.py --hardware-info                     # Exibe info de hardware e sai
  python Reels_Encoder_v2.py input.mp4 --performance speed       # Modo rápido (preview)
  python Reels_Encoder_v2.py input.mp4 --performance quality     # Máxima qualidade

COMPARAÇÃO:
  FFmpeg:  ~30-60 fps (GPU), excelente qualidade
  Cineon:  ~5-15 fps (CPU) ou ~30-60 fps (GPU), film-grade qualidade
"""
        ),
    )
    parser.add_argument(
        "input", nargs="?", default=None, help="Arquivo de vídeo de entrada"
    )
    parser.add_argument(
        "--mode",
        choices=["crf", "2pass"],
        default="crf",
        help="Modo de encoding: crf (qualidade constante) ou 2pass (otimizado)",
    )
    parser.add_argument(
        "--lut",
        choices=["on", "off"],
        default="on",
        help="Aplicar Hollywood Cinema LUT v6.6 (default: on). off = apenas scale/HDR/sharpen",
    )
    parser.add_argument(
        "--loudnorm",
        choices=["on", "off"],
        default="on",
        help="Normalização de áudio EBU R128 (default: on). Target: -14 LUFS, -1 dBTP",
    )
    parser.add_argument(
        "--hdr",
        choices=["auto", "off"],
        default="auto",
        help="Conversão HDR→SDR: auto (detecta e converte) ou off (ignora). Default: auto",
    )
    parser.add_argument(
        "--tonemap",
        choices=["mobius", "reinhard", "hable", "bt2390"],
        default="mobius",
        help="Algoritmo de tone mapping HDR→SDR (default: mobius). mobius=skin tones, hable=cinema",
    )
    parser.add_argument(
        "--fps",
        choices=["auto", "24", "25", "30", "60"],
        default="30",
        help="Frame rate do output: auto (preserva original), 30 (recomendado), 60 (ação). Default: 30. Sempre CFR.",
    )
    parser.add_argument(
        "--scale",
        choices=["auto", "off"],
        default="auto",
        help="Downscale automático para 1080p: auto (detecta e converte 4K→1080p), off (mantém original). Default: auto.",
    )
    parser.add_argument(
        "--show-hardware",
        choices=["on", "off"],
        default="on",
        help="Exibir perfil de hardware no início do encode. Default: on.",
    )
    parser.add_argument(
        "--hardware-info",
        action="store_true",
        help="Exibe informações de hardware e sai (não faz encode).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Override manual de threads (0 = auto-detectar). Aplica a encoder, filtros e decoder.",
    )
    parser.add_argument(
        "--performance",
        choices=["quality", "balanced", "speed"],
        default="balanced",
        help="Modo de performance: quality (lento, máxima qualidade), balanced (auto), speed (rápido). Default: balanced.",
    )
    parser.add_argument(
        "--float",
        choices=["on", "off"],
        default="on",
        help="[NOVO v1.4] 32-bit float processing para SDR (DaVinci Intermediate simulado). on = máxima qualidade (default), off = 8-bit legacy (v1.3)",
    )
    # NOVO v2.0: Argumentos Cineon
    parser.add_argument(
        "--cineon-pipeline",
        choices=["on", "off"],
        default="off",
        help="[NOVO v2.0] Ativar pipeline DWG/Cineon (film emulation Kodak 2383). Default: off (usa FFmpeg)",
    )
    parser.add_argument(
        "--cineon-lut",
        type=str,
        default="Rec709_Kodak2383_D60.cube",
        help="[Cineon] Caminho para LUT Kodak 2383 (.cube). Default: Rec709_Kodak2383_D60.cube",
    )
    parser.add_argument(
        "--exposure-offset",
        type=float,
        default=0.0,
        help="[Cineon] Ajuste de exposição em stops (+/- EV). Range: -2.0 a +2.0. Default: 0.0",
    )
    parser.add_argument(
        "--saturation",
        type=float,
        default=1.0,
        help="[Cineon] Ajuste de saturação. Range: 0.0 a 2.0. Default: 1.0 (sem alteração)",
    )
    parser.add_argument(
        "--lut-strength",
        type=float,
        default=1.0,
        help="[Cineon] Intensidade do film look (Key Output Gain). Range: 0.0 a 1.0. "
        "0.0=sem film look, 0.5=50%% (recomendado), 1.0=full Kodak (default: 1.0)",
    )
    args = parser.parse_args()

    # Hardware info mode
    if args.hardware_info:
        console.rule("[bold cyan]🔧 Hardware Detection")
        hw_profile = detect_hardware()
        print_hardware_profile(hw_profile)
        console.print(
            "[dim]Estes parâmetros serão aplicados automaticamente no encode.[/dim]"
        )
        sys.exit(0)

    # Verificar input
    input_file = args.input
    if input_file is None:
        console.print("[red]Erro:[/red] Arquivo de entrada é obrigatório.")
        console.print("[dim]Use: python script.py video.mp4[/dim]")
        console.print(
            "[dim]Ou: python script.py --hardware-info (para ver info de hardware)[/dim]"
        )
        sys.exit(1)

    if not os.path.exists(input_file):
        console.print(f"[red]Arquivo não encontrado:[/red] {input_file}")
        sys.exit(1)

    base, _ = os.path.splitext(input_file)
    if args.cineon_pipeline == "on":
        output_file = f"{base}_Cineon_Film.mp4"
    elif args.mode == "crf":
        output_file = f"{base}_Hollywood_CRF15.mp4"
    else:
        output_file = f"{base}_Hollywood_2Pass.mp4"

    try:
        # Escolher engine baseado no argumento --cineon-pipeline
        if args.cineon_pipeline == "on":
            # NOVO v2.0: Modo Cineon (PyAV + Colour-Science)
            run_ffmpeg_with_cineon(
                input_file,
                output_file,
                mode=args.mode,
                cineon_lut_path=args.cineon_lut,
                exposure_offset=args.exposure_offset,
                saturation=args.saturation,
                lut_strength=args.lut_strength,
                loudnorm_enabled=(args.loudnorm == "on"),
                target_fps=args.fps,
                scale_mode=args.scale,
                show_hardware=(args.show_hardware == "on"),
                threads_override=args.threads,
                performance_mode=args.performance,
            )
        else:
            # Original v1.4.1: Modo FFmpeg (default)
            run_ffmpeg(
                input_file,
                output_file,
                mode=args.mode,
                lut_enabled=(args.lut == "on"),
                loudnorm_enabled=(args.loudnorm == "on"),
                hdr_mode=args.hdr,
                tonemap=args.tonemap,
                target_fps=args.fps,
                scale_mode=args.scale,
                show_hardware=(args.show_hardware == "on"),
                threads_override=args.threads,
                performance_mode=args.performance,
                float_processing=(args.float == "on"),
            )
        analyze_with_mediainfo(output_file)
    except (FileNotFoundError, subprocess.CalledProcessError, OSError, ValueError) as e:
        console.print(f"[red]Erro durante o processamento:[/red] {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
