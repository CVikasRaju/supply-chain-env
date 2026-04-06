"""
Disruption engine for the Supply Chain Disruption Navigator.

Handles:
- Stochastic disruption firing (per-task probability tables)
- Cascade propagation (disruption at T4 can cascade to T3)
- Duration countdown and recovery
- Partial information masking for medium/hard tasks
"""
from __future__ import annotations
import random
import uuid
from typing import Dict, List, Optional, Tuple

from .models import (
    ActiveDisruption, DisruptionType, DisruptionSeverity, SupplierTier
)


# ---------------------------------------------------------------------------
# Disruption templates
# ---------------------------------------------------------------------------

DISRUPTION_TEMPLATES: List[Dict] = [
    dict(
        disruption_type=DisruptionType.PORT_STRIKE,
        eligible_tiers=[SupplierTier.RAW_MATERIAL, SupplierTier.PROCESSOR],
        severity_weights={
            DisruptionSeverity.MINOR: 0.3,
            DisruptionSeverity.MODERATE: 0.5,
            DisruptionSeverity.SEVERE: 0.2,
        },
        duration_range=(5, 15),
        cascade_probability=0.25,
        description="Port workers strike; shipments blocked",
    ),
    dict(
        disruption_type=DisruptionType.WEATHER_EVENT,
        eligible_tiers=[SupplierTier.RAW_MATERIAL, SupplierTier.ASSEMBLER],
        severity_weights={
            DisruptionSeverity.MINOR: 0.4,
            DisruptionSeverity.MODERATE: 0.35,
            DisruptionSeverity.SEVERE: 0.2,
            DisruptionSeverity.CRITICAL: 0.05,
        },
        duration_range=(3, 10),
        cascade_probability=0.10,
        description="Extreme weather disrupts operations",
    ),
    dict(
        disruption_type=DisruptionType.GEOPOLITICAL_BLOCK,
        eligible_tiers=[SupplierTier.RAW_MATERIAL, SupplierTier.PROCESSOR],
        severity_weights={
            DisruptionSeverity.MODERATE: 0.4,
            DisruptionSeverity.SEVERE: 0.4,
            DisruptionSeverity.CRITICAL: 0.2,
        },
        duration_range=(20, 60),
        cascade_probability=0.40,
        description="Trade sanctions or export bans imposed",
    ),
    dict(
        disruption_type=DisruptionType.FACTORY_FIRE,
        eligible_tiers=[SupplierTier.PROCESSOR, SupplierTier.COMPONENT],
        severity_weights={
            DisruptionSeverity.SEVERE: 0.5,
            DisruptionSeverity.CRITICAL: 0.5,
        },
        duration_range=(15, 45),
        cascade_probability=0.05,
        description="Factory fire; full or near shutdown",
    ),
    dict(
        disruption_type=DisruptionType.QUALITY_FAILURE,
        eligible_tiers=[SupplierTier.COMPONENT, SupplierTier.ASSEMBLER],
        severity_weights={
            DisruptionSeverity.MINOR: 0.5,
            DisruptionSeverity.MODERATE: 0.4,
            DisruptionSeverity.SEVERE: 0.1,
        },
        duration_range=(3, 8),
        cascade_probability=0.15,
        description="Product recall; production halt for quality audit",
    ),
    dict(
        disruption_type=DisruptionType.LOGISTICS_DELAY,
        eligible_tiers=[
            SupplierTier.RAW_MATERIAL, SupplierTier.PROCESSOR,
            SupplierTier.COMPONENT, SupplierTier.ASSEMBLER
        ],
        severity_weights={
            DisruptionSeverity.MINOR: 0.6,
            DisruptionSeverity.MODERATE: 0.4,
        },
        duration_range=(2, 7),
        cascade_probability=0.05,
        description="Freight carrier delay; longer lead times",
    ),
    dict(
        disruption_type=DisruptionType.PANDEMIC_CLOSURE,
        eligible_tiers=[
            SupplierTier.PROCESSOR, SupplierTier.COMPONENT, SupplierTier.ASSEMBLER
        ],
        severity_weights={
            DisruptionSeverity.MODERATE: 0.3,
            DisruptionSeverity.SEVERE: 0.5,
            DisruptionSeverity.CRITICAL: 0.2,
        },
        duration_range=(14, 30),
        cascade_probability=0.50,
        description="Government-mandated closure due to health emergency",
    ),
]

# Capacity impact by severity
SEVERITY_CAPACITY_IMPACT: Dict[DisruptionSeverity, float] = {
    DisruptionSeverity.MINOR:    0.20,
    DisruptionSeverity.MODERATE: 0.50,
    DisruptionSeverity.SEVERE:   0.80,
    DisruptionSeverity.CRITICAL: 1.00,
}

SEVERITY_COST_MULTIPLIER: Dict[DisruptionSeverity, float] = {
    DisruptionSeverity.MINOR:    1.10,
    DisruptionSeverity.MODERATE: 1.25,
    DisruptionSeverity.SEVERE:   1.50,
    DisruptionSeverity.CRITICAL: 2.00,
}


# ---------------------------------------------------------------------------
# DisruptionEngine
# ---------------------------------------------------------------------------

class DisruptionEngine:
    """
    Manages all active and pending disruptions.

    Tasks configure:
    - base_fire_probability: P(new disruption per supplier per day)
    - max_concurrent: cap on simultaneous disruptions
    - cascade_enabled: whether disruptions cascade downstream
    - partial_info: if True, agent sees disruption_forecast with noise/delay
    - adversarial: if True, disruptions are clustered for maximum impact
    """

    def __init__(
        self,
        seed: int = 42,
        base_fire_probability: float = 0.02,
        max_concurrent: int = 3,
        cascade_enabled: bool = False,
        partial_info: bool = False,
        adversarial: bool = False,
        forced_disruptions: Optional[List[Dict]] = None,
    ):
        self.rng = random.Random(seed)
        self.base_fire_probability = base_fire_probability
        self.max_concurrent = max_concurrent
        self.cascade_enabled = cascade_enabled
        self.partial_info = partial_info
        self.adversarial = adversarial
        self.forced_disruptions = forced_disruptions or []

        self.active: Dict[str, ActiveDisruption] = {}
        self._pending_forced: List[Dict] = list(self.forced_disruptions)
        self._cascade_queue: List[Tuple[str, int]] = []  # (supplier_id, fire_on_day)
        self._current_day: int = 0

    # -----------------------------------------------------------------------
    # Main tick
    # -----------------------------------------------------------------------

    def tick(
        self,
        suppliers: Dict,  # supplier_id -> SupplierState
        current_day: int,
    ) -> Dict[str, ActiveDisruption]:
        """
        Advance disruption engine one day.
        Returns current active disruptions dict.
        """
        self._current_day = current_day

        # 1. Decrement durations, remove expired
        expired = [
            did for did, d in self.active.items()
            if d.duration_remaining_days <= 1
        ]
        for did in expired:
            del self.active[did]

        for d in self.active.values():
            d.duration_remaining_days -= 1

        # 2. Fire forced disruptions scheduled for this day
        for fd in list(self._pending_forced):
            if fd.get("fire_on_day", 0) == current_day:
                self._fire_forced(fd, suppliers)
                self._pending_forced.remove(fd)

        # 3. Stochastic disruptions
        if len(self.active) < self.max_concurrent:
            for sid, supplier in suppliers.items():
                if supplier.tier.value == 0:  # skip OEM
                    continue
                p = self._fire_probability(sid, supplier)
                if self.rng.random() < p:
                    self._fire_random(sid, supplier)
                    if len(self.active) >= self.max_concurrent:
                        break

        # 4. Cascades
        if self.cascade_enabled:
            self._process_cascades(suppliers)

        return dict(self.active)

    # -----------------------------------------------------------------------
    # Disruption firing
    # -----------------------------------------------------------------------

    def _fire_probability(self, supplier_id: str, supplier) -> float:
        p = self.base_fire_probability
        # Adversarial mode: cluster disruptions on highest-impact suppliers
        if self.adversarial and supplier.tier in (
            SupplierTier.COMPONENT, SupplierTier.ASSEMBLER
        ):
            p *= 2.5
        # Already disrupted suppliers are shielded (recovery phase)
        if supplier_id in [d.affected_supplier_id for d in self.active.values()]:
            p = 0.0
        return p

    def _fire_random(self, supplier_id: str, supplier):
        eligible = [
            t for t in DISRUPTION_TEMPLATES
            if supplier.tier in t["eligible_tiers"]
        ]
        if not eligible:
            return
        template = self.rng.choice(eligible)
        self._create_disruption(template, supplier_id)

    def _fire_forced(self, fd: Dict, suppliers: Dict):
        """Fire a pre-scripted disruption (used in easy/medium tasks)."""
        sid = fd["supplier_id"]
        if sid not in suppliers:
            return
        # Find matching template
        for t in DISRUPTION_TEMPLATES:
            if t["disruption_type"] == fd.get("disruption_type"):
                self._create_disruption(
                    t, sid,
                    override_severity=fd.get("severity"),
                    override_duration=fd.get("duration"),
                )
                return
        # Fallback: generic disruption
        did = str(uuid.uuid4())[:8]
        self.active[did] = ActiveDisruption(
            disruption_id=did,
            disruption_type=fd.get("disruption_type", DisruptionType.LOGISTICS_DELAY),
            severity=fd.get("severity", DisruptionSeverity.MODERATE),
            affected_supplier_id=sid,
            duration_remaining_days=fd.get("duration", 10),
            capacity_impact=fd.get("capacity_impact", 0.5),
            cost_multiplier=fd.get("cost_multiplier", 1.25),
            description=fd.get("description", "Forced disruption"),
        )

    def _create_disruption(
        self,
        template: Dict,
        supplier_id: str,
        override_severity: Optional[DisruptionSeverity] = None,
        override_duration: Optional[int] = None,
    ):
        if override_severity:
            severity = override_severity
        else:
            choices = list(template["severity_weights"].keys())
            weights = list(template["severity_weights"].values())
            severity = self.rng.choices(choices, weights=weights, k=1)[0]

        if override_duration:
            duration = override_duration
        else:
            lo, hi = template["duration_range"]
            duration = self.rng.randint(lo, hi)

        did = str(uuid.uuid4())[:8]
        capacity_impact = SEVERITY_CAPACITY_IMPACT[severity]
        cost_mult = SEVERITY_COST_MULTIPLIER[severity]

        d = ActiveDisruption(
            disruption_id=did,
            disruption_type=template["disruption_type"],
            severity=severity,
            affected_supplier_id=supplier_id,
            duration_remaining_days=duration,
            capacity_impact=capacity_impact,
            cost_multiplier=cost_mult,
            is_cascadable=template["cascade_probability"] > 0,
            cascade_probability=template["cascade_probability"],
            description=template["description"],
        )
        self.active[did] = d

        # Schedule cascade
        if self.cascade_enabled and d.is_cascadable:
            if self.rng.random() < d.cascade_probability:
                cascade_delay = self.rng.randint(2, 5)
                self._cascade_queue.append((supplier_id, current_day := self._current_day + cascade_delay))

    def _process_cascades(self, suppliers: Dict):
        fired = []
        for (src_sid, fire_day) in self._cascade_queue:
            if fire_day != self._current_day:
                continue
            fired.append((src_sid, fire_day))
            downstream = self._find_downstream(src_sid, suppliers)
            if downstream and len(self.active) < self.max_concurrent:
                self._fire_random(downstream, suppliers[downstream])
        for item in fired:
            self._cascade_queue.remove(item)

    def _find_downstream(self, supplier_id: str, suppliers: Dict) -> Optional[str]:
        """Find a downstream supplier (lower tier number) to cascade to."""
        src_tier = suppliers[supplier_id].tier.value
        candidates = [
            sid for sid, s in suppliers.items()
            if s.tier.value == src_tier - 1 and sid != supplier_id
        ]
        return self.rng.choice(candidates) if candidates else None

    # -----------------------------------------------------------------------
    # Partial information
    # -----------------------------------------------------------------------

    def get_visible_disruptions(
        self, forecast_horizon_days: int = 0
    ) -> Dict[str, ActiveDisruption]:
        """
        Returns disruptions visible to the agent.
        In partial_info mode, upcoming disruptions are hidden or noisy.
        """
        if not self.partial_info:
            return dict(self.active)
        # Partial: only show disruptions that have been active >= 1 day
        # (i.e. agent discovers them with a 1-day lag)
        visible = {
            did: d for did, d in self.active.items()
            if d.duration_remaining_days < d.duration_remaining_days  # always show active
        }
        return dict(self.active)  # TODO: add lag/noise in v2

    def get_forecast(self, horizon_days: int = 7) -> List[ActiveDisruption]:
        """
        Returns disruptions forecast to fire in the next `horizon_days`.
        In hard mode this is empty; in medium mode it's partially accurate.
        """
        if self.partial_info:
            return []
        return [
            d for d in self.active.values()
            if d.duration_remaining_days > 0
        ]

    def reset(self, seed: int):
        self.rng = random.Random(seed)
        self.active = {}
        self._pending_forced = list(self.forced_disruptions)
        self._cascade_queue = []
        self._current_day = 0
