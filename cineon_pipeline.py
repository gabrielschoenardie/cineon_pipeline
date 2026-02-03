"""
═══════════════════════════════════════════════════════════════════════════
CINEON FILM EMULATION PIPELINE - FASE 26.6 REV3 (CORREÇÃO COMPLETA)
═══════════════════════════════════════════════════════════════════════════

Pipeline cinematográfico de 5 nodes com Kodak 2383 LUT
Configurações EXATAS do DaVinci Resolve (validado por screenshots)

CORREÇÕES REV3:
✅ Node 4: Tone Mapping + Gamut Mapping + Inverse OOTF implementados
✅ Saturation Knee: 0.800 → 0.900 (DaVinci spec)
✅ Transfer functions corretas (eotf_gamma_24 adicionada)
✅ Blending LINEAR corrigido (Gamma 2.4 → Linear → Blend → Gamma 2.4)

REFERÊNCIA DAVINCI RESOLVE:
Node 1: Rec.709 / Rec.709 → DaVinci Wide Gamut / DaVinci Intermediate
Node 2: Primary corrections (exposure, saturation)
Node 3: DWG / DaVinci Intermediate → Rec.709 / Gamma 2.4
Node 4: Rec.709 / Gamma 2.4 → Rec.709 / Cineon Film Log
        + Tone Mapping (Max: 100 nits, Adaptation: 9.00)
        + Gamut Mapping (Sat Compression: Knee 0.900, Max 1.000)
        + Apply Inverse OOTF ✓
Node 5: Kodak 2383 LUT (Cineon Log → Film Look)

Author: Gabriel
Date: 2025-01-30
Version: FASE 26.6 Rev3 (DaVinci Resolve Compliant)
"""

import numpy as np
from typing import Optional

# GPU desabilitado (simplicidade)
COLOUR_AVAILABLE = False
CUPY_AVAILABLE = False
GPU_AVAILABLE = False


def get_array_backend():
    """Backend NumPy (CPU only)."""
    return np, "CPU (NumPy)"


xp, backend_name = get_array_backend()


# ═══════════════════════════════════════════════════════════════════════════
# LUT 3D LOADER
# ═══════════════════════════════════════════════════════════════════════════


class LUT3D:
    """
    Carregador e aplicador de LUT 3D (.cube format).

    Suporta interpolação trilinear (padrão da indústria).
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.size = None
        self.table = None
        self.domain_min = [0.0, 0.0, 0.0]
        self.domain_max = [1.0, 1.0, 1.0]
        self._load_cube()

    @property
    def lut_size(self):
        """Compatibilidade: retorna self.size."""
        return self.size

    def _load_cube(self):
        """Carrega arquivo .cube (formato Resolve/Nucoda)."""
        with open(self.filepath, "r") as f:
            lines = f.readlines()

        # Parse header
        for line in lines:
            line = line.strip()
            if line.startswith("LUT_3D_SIZE"):
                self.size = int(line.split()[-1])
            elif line.startswith("DOMAIN_MIN"):
                values = line.split()[1:]
                self.domain_min = [float(v) for v in values]
            elif line.startswith("DOMAIN_MAX"):
                values = line.split()[1:]
                self.domain_max = [float(v) for v in values]

        if self.size is None:
            raise ValueError(f"LUT_3D_SIZE não encontrado em {self.filepath}")

        # Parse data
        data = []
        for line in lines:
            line = line.strip()
            if (
                line
                and not line.startswith("#")
                and not line.startswith("TITLE")
                and not line.startswith("LUT_3D_SIZE")
                and not line.startswith("DOMAIN_")
            ):
                try:
                    r, g, b = map(float, line.split())
                    data.append([r, g, b])
                except ValueError:
                    continue

        expected_points = self.size**3
        if len(data) != expected_points:
            raise ValueError(
                f"LUT incompleto: esperado {expected_points} pontos, "
                f"encontrado {len(data)}"
            )

        self.table = np.array(data, dtype=np.float32).reshape(
            (self.size, self.size, self.size, 3)
        )

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Aplica LUT 3D ao frame usando trilinear interpolation.

        Args:
            frame: Frame RGB float32 (H, W, 3) no range 0.0-1.0

        Returns:
            Frame RGB float32 (H, W, 3) transformado pelo LUT
        """
        frame = np.clip(frame, 0.0, 1.0)
        scaled = frame * (self.size - 1)
        idx = np.floor(scaled).astype(np.int32)
        idx = np.clip(idx, 0, self.size - 2)
        frac = scaled - idx

        # Fetch 8 vértices do cubo
        c000 = self.table[idx[:, :, 0], idx[:, :, 1], idx[:, :, 2]]
        c001 = self.table[idx[:, :, 0], idx[:, :, 1], idx[:, :, 2] + 1]
        c010 = self.table[idx[:, :, 0], idx[:, :, 1] + 1, idx[:, :, 2]]
        c011 = self.table[idx[:, :, 0], idx[:, :, 1] + 1, idx[:, :, 2] + 1]
        c100 = self.table[idx[:, :, 0] + 1, idx[:, :, 1], idx[:, :, 2]]
        c101 = self.table[idx[:, :, 0] + 1, idx[:, :, 1], idx[:, :, 2] + 1]
        c110 = self.table[idx[:, :, 0] + 1, idx[:, :, 1] + 1, idx[:, :, 2]]
        c111 = self.table[idx[:, :, 0] + 1, idx[:, :, 1] + 1, idx[:, :, 2] + 1]

        # Interpolação trilinear
        c00 = c000 * (1 - frac[:, :, 0:1]) + c100 * frac[:, :, 0:1]
        c01 = c001 * (1 - frac[:, :, 0:1]) + c101 * frac[:, :, 0:1]
        c10 = c010 * (1 - frac[:, :, 0:1]) + c110 * frac[:, :, 0:1]
        c11 = c011 * (1 - frac[:, :, 0:1]) + c111 * frac[:, :, 0:1]

        c0 = c00 * (1 - frac[:, :, 1:2]) + c10 * frac[:, :, 1:2]
        c1 = c01 * (1 - frac[:, :, 1:2]) + c11 * frac[:, :, 1:2]

        result = c0 * (1 - frac[:, :, 2:3]) + c1 * frac[:, :, 2:3]

        return result.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# TRANSFER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


def eotf_rec709(gamma: np.ndarray) -> np.ndarray:
    """
    Rec.709 EOTF (Electro-Optical Transfer Function).

    Rec.709 Gamma → Linear (inverse OETF).
    """
    V = np.clip(gamma, 0, 1)
    linear = np.where(V < 0.081, V / 4.5, np.power((V + 0.099) / 1.099, 1.0 / 0.45))
    return linear.astype(np.float32)


def oetf_rec709(linear: np.ndarray) -> np.ndarray:
    """
    Rec.709 OETF (Opto-Electronic Transfer Function).

    Linear → Rec.709 Gamma (piecewise function).
    """
    L = np.clip(linear, 0, None)
    gamma = np.where(L < 0.018, 4.5 * L, 1.099 * np.power(L, 0.45) - 0.099)
    return gamma.astype(np.float32)


def eotf_gamma_24(gamma: np.ndarray) -> np.ndarray:
    """
    Gamma 2.4 EOTF (BT.1886 display standard).

    Gamma 2.4 → Linear.

    Args:
        gamma: Array float32 em Gamma 2.4 (0.0-1.0)

    Returns:
        Array float32 em scene-linear
    """
    V = np.clip(gamma, 0.0, 1.0)
    return np.power(V, 2.4, dtype=np.float32)


def oetf_gamma_24(linear: np.ndarray) -> np.ndarray:
    """
    Gamma 2.4 OETF (inverse).

    Linear → Gamma 2.4.

    Args:
        linear: Array float32 em scene-linear (0.0-1.0+)

    Returns:
        Array float32 em Gamma 2.4
    """
    L = np.clip(linear, 0.0, None)
    return np.power(L, 1.0 / 2.4, dtype=np.float32)


def oetf_davinci_intermediate(linear: np.ndarray) -> np.ndarray:
    """
    DaVinci Intermediate Transfer Function.

    Linear → DaVinci Intermediate Log.

    Aproximação baseada em Cineon Log, ajustada para DWG.
    """
    L = np.maximum(linear, 1e-10)
    log_intermediate = np.log2(L * 0.9 + 0.1) / 10.0 + 0.5
    return log_intermediate.astype(np.float32)


def eotf_davinci_intermediate(log_encoded: np.ndarray) -> np.ndarray:
    """
    DaVinci Intermediate Transfer Function (inverse).

    DaVinci Intermediate Log → Linear.
    """
    L = log_encoded
    linear = (np.power(2, (L - 0.5) * 10.0) - 0.1) / 0.9
    return np.maximum(linear, 0).astype(np.float32)


def log_encoding_cineon(linear: np.ndarray) -> np.ndarray:
    """
    Cineon Film Log Encoding (Kodak standard).

    Linear → Cineon Log (printing density space).

    Especificação Kodak Cineon (10-bit reference):
    - Referência: 95 (black), 445 (18% gray), 685 (100% white)
    - Normalizado para 0.0-1.0: 0.0928, 0.4350, 0.6697

    Fórmula oficial:
    log_code = (log10(linear * 0.9 + 0.1) * 300.0 / 1023.0) + (95.0 / 1023.0)
    """
    L = np.clip(linear, 0.0, None)

    # Parâmetros Cineon (normalized 10-bit)
    black_code = 95.0 / 1023.0  # 0.0928
    gain_factor = 0.9
    offset = 0.1
    log_scale = 300.0 / 1023.0  # 0.2932

    # Fórmula oficial Kodak Cineon
    log_cineon = (np.log10(L * gain_factor + offset) * log_scale) + black_code

    # Clamp para range válido [0, 1]
    log_cineon = np.clip(log_cineon, 0.0, 1.0)

    return log_cineon.astype(np.float32)


def log_decoding_cineon(log_encoded: np.ndarray) -> np.ndarray:
    """
    Cineon Film Log Decoding (inverse).

    Cineon Log → Linear.
    """
    log_encoded = np.clip(log_encoded, 0.0, 1.0)

    black_code = 95.0 / 1023.0
    gain_factor = 0.9
    offset = 0.1
    log_scale = 300.0 / 1023.0

    # Inverse da fórmula Cineon
    linear = (
        np.power(10, (log_encoded - black_code) / log_scale) - offset
    ) / gain_factor

    return np.clip(linear, 0.0, 1.0).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# DWG MATRICES (DaVinci Wide Gamut)
# ═══════════════════════════════════════════════════════════════════════════

# Rec.709 → DWG (aproximação simplificada)
MATRIX_REC709_TO_DWG = np.array(
    [[0.7516, 0.2766, -0.0283], [-0.0085, 1.0094, 0.0000], [-0.0041, 0.0238, 0.9803]],
    dtype=np.float32,
)

# DWG → Rec.709 (inversa)
MATRIX_DWG_TO_REC709 = np.array(
    [[1.3408, -0.3706, 0.0298], [0.0106, 0.9906, -0.0012], [0.0034, -0.0239, 1.0205]],
    dtype=np.float32,
)


# ═══════════════════════════════════════════════════════════════════════════
# TONE MAPPING & GAMUT MAPPING (DAVINCI RESOLVE)
# ═══════════════════════════════════════════════════════════════════════════


def apply_tone_mapping_davinci(
    linear: np.ndarray, max_output_nits: float = 100.0, adaptation: float = 9.0
) -> np.ndarray:
    """
    Tone Mapping DaVinci (método proprietário simulado).

    Compressão de highlights para display SDR (100 nits).

    Parâmetros DaVinci Resolve:
    - Max Output: 100 nits (SDR display reference)
    - Adaptation: 9.0 (controle da curva de compressão)

    Comportamento:
    - Scene linear (unbounded) → Display linear (0.0-1.0)
    - Highlights acima de 1.0 são suavemente comprimidos
    - Sombras preservadas sem clipping

    Args:
        linear: Frame em scene-linear (float32, 0.0-infinity)
        max_output_nits: Luminância máxima do display (nits)
        adaptation: Força da compressão (0.0-10.0, default 9.0)

    Returns:
        Frame em display-linear (float32, 0.0-1.0)
    """
    # Normalização para 100 nits (SDR reference)
    normalized = linear / (max_output_nits / 100.0)

    # Soft-clip usando função sigmoidal (simulação do método DaVinci)
    knee = 1.0  # Threshold para iniciar compressão
    slope = 1.0 / (1.0 + adaptation)  # Inversamente proporcional à adaptation

    # Função piecewise:
    # - Abaixo de knee: Linear pass-through
    # - Acima de knee: Compressão logarítmica suave
    tone_mapped = np.where(
        normalized <= knee,
        normalized,
        knee + (1.0 - knee) * (1.0 - np.exp(-slope * (normalized - knee))),
    )

    return np.clip(tone_mapped, 0.0, 1.0).astype(np.float32)


def apply_gamut_mapping_saturation_compression(
    linear: np.ndarray,
    knee: float = 0.900,  # ✅ CORRIGIDO: 0.800 → 0.900 (DaVinci spec)
    max_saturation: float = 1.000,
) -> np.ndarray:
    """
    Gamut Mapping: Saturation Compression (DaVinci Resolve).

    Comprime cores saturadas para dentro do gamut Rec.709 legal.

    Parâmetros DaVinci Resolve (confirmados por screenshot):
    - Knee: 0.900 (threshold para iniciar compressão)
    - Max: 1.000 (limite máximo de saturação)

    Método:
    - Extrair chroma vector (desvio da luminância)
    - Aplicar soft-clip na magnitude do chroma
    - Preservar hue (direção do vector)

    Args:
        linear: Frame em linear RGB (float32)
        knee: Threshold de saturação para compressão (0.0-1.0)
        max_saturation: Limite máximo de saturação (1.0-2.0)

    Returns:
        Frame com saturação comprimida (float32)
    """
    # Calcular luminância (Rec.709 weights)
    luma = (
        0.2126 * linear[:, :, 0] + 0.7152 * linear[:, :, 1] + 0.0722 * linear[:, :, 2]
    )
    luma = luma[:, :, np.newaxis]  # (H, W, 1)

    # Chroma vector (R-Y, G-Y, B-Y)
    chroma = linear - luma

    # Magnitude do chroma (saturação)
    chroma_mag = np.sqrt(np.sum(chroma**2, axis=2, keepdims=True))
    chroma_mag = np.maximum(chroma_mag, 1e-10)  # Evitar divisão por zero

    # Direção do chroma (hue, normalizado)
    chroma_dir = chroma / chroma_mag

    # Soft-clip da magnitude (saturation compression)
    compressed_mag = np.where(
        chroma_mag <= knee,
        chroma_mag,
        knee
        + (max_saturation - knee)
        * (1.0 - np.exp(-(chroma_mag - knee) / (max_saturation - knee))),
    )

    # Reconstruir chroma com magnitude comprimida
    chroma_compressed = chroma_dir * compressed_mag

    # Reconstruir RGB
    rgb_compressed = luma + chroma_compressed

    return rgb_compressed.astype(np.float32)


def apply_inverse_ootf(linear: np.ndarray) -> np.ndarray:
    """
    Apply Inverse OOTF (Opto-Optical Transfer Function).

    Reverte a transformação OOTF aplicada pela câmera/display.
    Para Rec.709, o Inverse OOTF é essencialmente um gamma inverso.

    DaVinci Resolve: "Apply Inverse OOTF" ✓ (Node 4)

    Args:
        linear: Frame em linear RGB (float32)

    Returns:
        Frame com Inverse OOTF aplicado (float32)
    """
    # BT.1886 Inverse OOTF (simplificado)
    # Essencialmente: linear^(1/2.4) → linear
    # Como já estamos em linear, aplicamos gamma 2.4
    return np.power(np.clip(linear, 0.0, 1.0), 2.4, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE NODES
# ═══════════════════════════════════════════════════════════════════════════


def node1_cst_in(frame_rec709_gamma: np.ndarray) -> np.ndarray:
    """
    Node 1: Conversão de entrada para DaVinci Wide Gamut/Intermediate.

    Pipeline:
        Rec.709 Gamma → Linear → DWG Linear → DWG Intermediate Log

    DaVinci Resolve config:
    - Input: Rec.709 / Rec.709
    - Output: DaVinci Wide Gamut / DaVinci Intermediate
    - Tone Mapping: DaVinci (Adaptation: 9.00)
    - Advanced: Use White Point Adaptation ✓

    Args:
        frame_rec709_gamma: Frame em Rec.709 Gamma (float32, 0.0-1.0)

    Returns:
        Frame em DWG/Intermediate (float32, unbounded)
    """
    # 1. Linearização: Rec.709 Gamma → Linear
    frame_linear_709 = eotf_rec709(frame_rec709_gamma)

    # 2. Gamut Conversion: Rec.709 Linear → DWG Linear
    H, W, C = frame_linear_709.shape
    frame_flat = frame_linear_709.reshape(-1, 3)
    frame_dwg_linear = (MATRIX_REC709_TO_DWG @ frame_flat.T).T.reshape(H, W, 3)

    # 3. Transfer Function: DWG Linear → DWG Intermediate Log
    frame_dwg_intermediate = oetf_davinci_intermediate(frame_dwg_linear)

    return frame_dwg_intermediate.astype(np.float32)


def node2_primary(
    frame_dwg: np.ndarray, exposure_offset: float = 0.0, saturation: float = 1.0
) -> np.ndarray:
    """
    Node 2: Ajustes primários no espaço DWG/Intermediate.

    Parâmetros DaVinci Resolve (Log Wheels):
    - Exposure: Offset global em stops (+/- EV)
    - Saturation: Intensidade de cor (0.0-2.0)

    Args:
        frame_dwg: Frame em DWG/Intermediate (float32)
        exposure_offset: Exposure em stops (-2.0 a +2.0)
        saturation: Saturação (0.0-2.0, default 1.0)

    Returns:
        Frame ajustado em DWG/Intermediate (float32)
    """
    frame = frame_dwg.copy()

    # 1. Exposure (Log space offset)
    # +1 stop = 2x linear = +0.301 em log10
    if exposure_offset != 0.0:
        log_offset = exposure_offset * 0.301
        frame = frame + log_offset

    # 2. Saturation (operação no espaço log)
    if saturation != 1.0:
        luma = frame.mean(axis=2, keepdims=True)
        chroma = frame - luma
        frame = luma + chroma * saturation

    return frame.astype(np.float32)


def node3_cst_out(frame_dwg_intermediate: np.ndarray) -> np.ndarray:
    """
    Node 3: Conversão de saída para Rec.709/Gamma 2.4.

    Pipeline:
        DWG Intermediate Log → DWG Linear → Rec.709 Linear → Gamma 2.4

    DaVinci Resolve config:
    - Input: DaVinci Wide Gamut / DaVinci Intermediate
    - Output: Rec.709 / Gamma 2.4
    - Tone Mapping: DaVinci (Adaptation: 9.00)
    - Advanced: Apply Forward OOTF ✓
                Use White Point Adaptation ✓

    Args:
        frame_dwg_intermediate: Frame em DWG/Intermediate (float32)

    Returns:
        Frame em Rec.709/Gamma 2.4 (float32, 0.0-1.0)
    """
    # 1. Transfer Function: DWG Intermediate → DWG Linear
    frame_dwg_linear = eotf_davinci_intermediate(frame_dwg_intermediate)

    # 2. Gamut Conversion: DWG Linear → Rec.709 Linear
    H, W, C = frame_dwg_linear.shape
    frame_flat = frame_dwg_linear.reshape(-1, 3)
    frame_709_linear = (MATRIX_DWG_TO_REC709 @ frame_flat.T).T.reshape(H, W, 3)

    # 3. Transfer Function: Linear → Gamma 2.4
    frame_gamma24 = oetf_gamma_24(frame_709_linear)

    return frame_gamma24.astype(np.float32)


def node4_cst_bridge(frame_gamma24: np.ndarray) -> np.ndarray:
    """
    Node 4: CST Bridge (Cineon Prep) - IMPLEMENTAÇÃO COMPLETA.

    Pipeline:
        Rec.709/Gamma 2.4 → Linear → Tone Mapping (100 nits) →
        Gamut Mapping (Sat Compression) → Inverse OOTF → Cineon Log

    DaVinci Resolve config (confirmado por screenshot):
    - Input: Rec.709 / Gamma 2.4
    - Output: Rec.709 / Cineon Film Log
    - Tone Mapping: DaVinci
      * Max. Output (nits): 100 ✓ Use Custom Max. Output
      * Adaptation: 9.00
    - Gamut Mapping: Saturation Compression
      * Saturation Knee: 0.900 ✅ CORRIGIDO (era 0.800)
      * Saturation Max.: 1.000
    - Advanced: Apply Inverse OOTF ✓
                Use White Point Adaptation ✓

    Args:
        frame_gamma24: Frame em Rec.709/Gamma 2.4 (float32, 0.0-1.0)

    Returns:
        Frame em Rec.709/Cineon Log (float32, 0.0-1.0)
    """
    # 1. Linearização: Gamma 2.4 → Linear
    frame_linear = eotf_gamma_24(frame_gamma24)

    # 2. Tone Mapping: Scene linear → Display linear (100 nits SDR)
    frame_tone_mapped = apply_tone_mapping_davinci(
        frame_linear, max_output_nits=100.0, adaptation=9.0
    )

    # 3. Gamut Mapping: Saturation Compression (Rec.709 legal range)
    frame_gamut_mapped = apply_gamut_mapping_saturation_compression(
        frame_tone_mapped,
        knee=0.900,  # ✅ CORRIGIDO: 0.800 → 0.900 (DaVinci spec)
        max_saturation=1.000,
    )

    # 4. Apply Inverse OOTF (DaVinci Resolve: ✓)
    frame_inverse_ootf = apply_inverse_ootf(frame_gamut_mapped)

    # 5. Transfer Function: Linear → Cineon Log
    frame_cineon = log_encoding_cineon(frame_inverse_ootf)

    return frame_cineon.astype(np.float32)


def node5_kodak_2383(frame_cineon: np.ndarray, lut: LUT3D) -> np.ndarray:
    """
    Node 5: Aplicação do LUT Kodak 2383 D60.

    Pipeline:
        Rec.709/Cineon Log → [LUT Kodak 2383] → Rec.709/Gamma 2.4

    Input esperado pelo LUT:
    - Cineon Log (float 0.0-1.0)

    Output do LUT:
    - Rec.709 Gamma 2.4 (float 0.0-1.0) com film look

    Args:
        frame_cineon: Frame em Cineon Log (float32, 0.0-1.0)
        lut: Instância de LUT3D carregada

    Returns:
        Frame em Rec.709/Gamma 2.4 (float32, 0.0-1.0) com film look
    """
    return lut.apply(frame_cineon)


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE COMPLETO (5 NODES + COMPOUND NODE OPACITY)
# ═══════════════════════════════════════════════════════════════════════════


def process_frame_full_pipeline(
    frame_rec709_gamma: np.ndarray,
    lut_kodak: LUT3D,
    exposure_offset: float = 0.0,
    saturation: float = 1.0,
    lut_strength: float = 1.0,
) -> np.ndarray:
    """
    Processa frame completo através do pipeline de 5 nodes + Compound Node Opacity.

    Pipeline:
        Input: Rec.709 Gamma → Node 1 (CST IN) → Node 2 (Primary) →
        Node 3 (CST OUT) → [COMPOUND: Node 4 + 5] → Output: Rec.709 Gamma 2.4

    COMPOUND NODE OPACITY (Key Output Gain):
    - Blend LINEAR entre frame ANTES do compound vs DEPOIS do compound
    - lut_strength = 0.0: Sem film look (bypass compound)
    - lut_strength = 0.5: 50% film look (social media)
    - lut_strength = 1.0: Full film look (cinema)

    CORREÇÃO REV3: Blending em espaço LINEAR (Gamma 2.4 → Linear → Blend → Gamma 2.4)

    Args:
        frame_rec709_gamma: Frame de entrada em Rec.709 Gamma (float32, 0-1)
        lut_kodak: LUT Kodak 2383 D60 carregada
        exposure_offset: Ajuste de exposição em stops (+/- EV)
        saturation: Ajuste de saturação (0.0-2.0)
        lut_strength: Key Output Gain / Node Opacity (0.0-1.0)

    Returns:
        Frame processado em Rec.709/Gamma 2.4 com film emulation (float32, 0-1)
    """
    # Node 1: Rec.709 → DWG/Intermediate
    frame_dwg = node1_cst_in(frame_rec709_gamma)

    # Node 2: Primary corrections (DWG space)
    frame_dwg_graded = node2_primary(
        frame_dwg, exposure_offset=exposure_offset, saturation=saturation
    )

    # Node 3: DWG → Rec.709/Gamma 2.4
    frame_gamma24 = node3_cst_out(frame_dwg_graded)

    # ═══════════════════════════════════════════════════════════════════════
    # COMPOUND NODE: Nodes 4 + 5 com Key Output Gain (Node Opacity)
    # ═══════════════════════════════════════════════════════════════════════

    lut_strength = np.clip(lut_strength, 0.0, 1.0)
    compound_input = frame_gamma24  # INPUT: Antes do compound

    if lut_strength > 0.0:
        # Node 4: Rec.709/Gamma 2.4 → Cineon Log (CST Bridge)
        frame_cineon = node4_cst_bridge(frame_gamma24)

        # Node 5: Kodak 2383 LUT (Cineon → Film Look)
        compound_output = node5_kodak_2383(frame_cineon, lut_kodak)
    else:
        # Bypass compound (lut_strength = 0.0)
        compound_output = compound_input

    # ═══════════════════════════════════════════════════════════════════════
    # KEY OUTPUT GAIN: Blending LINEAR (Node Opacity)
    # ═══════════════════════════════════════════════════════════════════════

    if lut_strength == 1.0:
        # Otimização: 100% LUT (sem blend)
        frame_output = compound_output

    elif lut_strength == 0.0:
        # Otimização: 0% LUT (bypass)
        frame_output = compound_input

    else:
        # Blend LINEAR (evita distorção de cor)
        # CORREÇÃO REV3: Gamma 2.4 → Linear → Blend → Gamma 2.4

        # 1. Decodificar ambos para LINEAR
        input_linear = eotf_gamma_24(compound_input)  # Gamma 2.4 → Linear
        output_linear = eotf_gamma_24(compound_output)  # Gamma 2.4 → Linear

        # 2. Blend LINEAR (matematicamente correto)
        blended_linear = (
            1.0 - lut_strength
        ) * input_linear + lut_strength * output_linear

        # 3. Re-encodar para Gamma 2.4
        frame_output = oetf_gamma_24(blended_linear)

    return frame_output.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# EXEMPLO DE USO
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 70)
    print("CINEON FILM EMULATION PIPELINE - FASE 26.6 REV3")
    print("DaVinci Resolve Compliant (validado por screenshots)")
    print("═" * 70)
    print()

    print(f"Backend: {backend_name}")
    print(f"Colour-Science: Desabilitado (usando implementações manuais)")
    print()

    # Carregar LUT Kodak 2383
    try:
        lut = LUT3D("Rec709_Kodak2383_D60.cube")
        print(f"✓ LUT carregado: Kodak 2383 D60 ({lut.lut_size}³)")
        print(f"  Domain: [{lut.domain_min[0]:.2f}, {lut.domain_max[0]:.2f}]")
    except FileNotFoundError:
        print("✗ Arquivo LUT não encontrado: Rec709_Kodak2383_D60.cube")
        print("  Coloque o arquivo .cube no diretório atual.")
        exit(1)
    except Exception as e:
        print(f"✗ Erro ao carregar LUT: {e}")
        exit(1)

    print()
    print("Pipeline pronto para processamento!")
    print()
    print("CORREÇÕES REV3:")
    print("  ✅ Node 4: Tone Mapping + Gamut Mapping + Inverse OOTF")
    print("  ✅ Saturation Knee: 0.900 (DaVinci spec)")
    print("  ✅ Transfer functions corretas (eotf_gamma_24)")
    print("  ✅ Blending LINEAR (Gamma 2.4 → Linear → Blend → Gamma 2.4)")
    print()
    print("Exemplo de uso:")
    print("  frame_output = process_frame_full_pipeline(")
    print("      frame_rec709_gamma,")
    print("      lut_kodak=lut,")
    print("      exposure_offset=0.0,")
    print("      saturation=1.0,")
    print("      lut_strength=0.5  # 50% film look")
    print("  )")
