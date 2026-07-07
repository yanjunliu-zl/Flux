"""Physical consistency modules for Seedance 2.0.

Implements 6 techniques to improve physical realism in video generation:

1. PhaseLock     — 2-step motion prior locking (inference-time, ~1.06× overhead)
2. PhysicsProbe  — linear decoder on DiT hidden states (train-time auxiliary signal)
3. VPT           — role-aware signals + modality-decoupled denoising (train-time)
4. PhysCorr      — PhysicsRM reward model + DPO fine-tuning (post-train)
5. CausalMotion  — VLM-guided keyframe trajectory injection (inference-time, no training)
6. SimLoop       — physical simulator in the generation loop (high quality, high cost)

References:
    PhaseLock:  Han et al., ICML 2026 — "Physics in 2-Steps"
    PhysCorr:   Wang et al., 2025 — "Enhancing Video Physical Consistency via DPO"
    VPT:        Zheng et al., July 2026 — "Role-aware Joint Training"
    CausalMotion: Zhuang et al., June 2026
    PSIVG:      CVPR 2026, MPI — "Physical Simulator In-the-Loop"
    Hidden Physics: Esmati et al., June 2026 — linear decoding of physics from DiT states
"""

from seedance.physics.phase_lock import PhaseLockSampler
from seedance.physics.physics_probe import PhysicsProbe, PhysicsProbeLoss
from seedance.physics.vpt import VPTNoiseScheduler, RoleAwareCaptioner
from seedance.physics.physcorr import PhysicsRM, PhyDPOTrainer
from seedance.physics.causal_motion import CausalMotionGuide

__all__ = [
    "PhaseLockSampler",
    "PhysicsProbe",
    "PhysicsProbeLoss",
    "VPTNoiseScheduler",
    "RoleAwareCaptioner",
    "PhysicsRM",
    "PhyDPOTrainer",
    "CausalMotionGuide",
]
