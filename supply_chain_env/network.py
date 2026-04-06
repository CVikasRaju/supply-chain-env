"""
Supply chain network graph.
Models 12 suppliers across 4 tiers feeding one OEM factory.
Handles material flow, inventory, and capacity updates each timestep.
"""
from __future__ import annotations
import copy
import random
from typing import Dict, List, Optional, Tuple

from .models import (
    SupplierState, SupplierTier, OrderRoute, ProductionLine
)


# ---------------------------------------------------------------------------
# Default network topology
# ---------------------------------------------------------------------------

DEFAULT_SUPPLIERS: List[Dict] = [
    # Tier 4 – Raw material
    dict(supplier_id="S401", name="Mining Co Alpha", tier=SupplierTier.RAW_MATERIAL,
         country="Chile", base_lead_time_days=21, unit_cost=8.0,
         reliability_score=0.90, esg_score=0.55, inventory_held=500.0),
    dict(supplier_id="S402", name="Agri Corp Beta",  tier=SupplierTier.RAW_MATERIAL,
         country="Brazil", base_lead_time_days=14, unit_cost=3.0,
         reliability_score=0.85, esg_score=0.70, inventory_held=800.0),
    dict(supplier_id="S403", name="PetroChem Gamma", tier=SupplierTier.RAW_MATERIAL,
         country="Saudi Arabia", base_lead_time_days=18, unit_cost=5.0,
         reliability_score=0.88, esg_score=0.40, inventory_held=600.0),

    # Tier 3 – Processors
    dict(supplier_id="S301", name="Smelter Delta",    tier=SupplierTier.PROCESSOR,
         country="China", base_lead_time_days=14, unit_cost=22.0,
         reliability_score=0.92, esg_score=0.50, inventory_held=200.0),
    dict(supplier_id="S302", name="Chem Plant Epsilon", tier=SupplierTier.PROCESSOR,
         country="Germany", base_lead_time_days=10, unit_cost=35.0,
         reliability_score=0.95, esg_score=0.80, inventory_held=150.0),
    dict(supplier_id="S303", name="Resin Extruder Zeta", tier=SupplierTier.PROCESSOR,
         country="South Korea", base_lead_time_days=12, unit_cost=18.0,
         reliability_score=0.91, esg_score=0.65, inventory_held=300.0),

    # Tier 2 – Components
    dict(supplier_id="S201", name="Battery Mfg Eta",  tier=SupplierTier.COMPONENT,
         country="China", base_lead_time_days=10, unit_cost=80.0,
         reliability_score=0.93, esg_score=0.60, inventory_held=100.0),
    dict(supplier_id="S202", name="PCB Fab Theta",    tier=SupplierTier.COMPONENT,
         country="Taiwan", base_lead_time_days=8, unit_cost=45.0,
         reliability_score=0.96, esg_score=0.72, inventory_held=120.0),
    dict(supplier_id="S203", name="Housing Mold Iota", tier=SupplierTier.COMPONENT,
         country="Mexico", base_lead_time_days=6, unit_cost=15.0,
         reliability_score=0.89, esg_score=0.68, inventory_held=250.0),

    # Tier 1 – Assemblers
    dict(supplier_id="S101", name="Assembler Alpha",  tier=SupplierTier.ASSEMBLER,
         country="Vietnam", base_lead_time_days=5, unit_cost=60.0,
         reliability_score=0.94, esg_score=0.65, inventory_held=80.0),
    dict(supplier_id="S102", name="Assembler Beta",   tier=SupplierTier.ASSEMBLER,
         country="India", base_lead_time_days=7, unit_cost=50.0,
         reliability_score=0.91, esg_score=0.70, inventory_held=90.0),

    # OEM
    dict(supplier_id="OEM0", name="OEM Factory",     tier=SupplierTier.OEM,
         country="USA", base_lead_time_days=1, unit_cost=0.0,
         reliability_score=1.0, esg_score=0.75, inventory_held=0.0),
]

# Material flow edges: (from_tier4/3/2/1 → to_tier3/2/1/OEM, material_type)
DEFAULT_EDGES: List[Tuple[str, str, str]] = [
    # T4 → T3
    ("S401", "S301", "lithium_ore"),
    ("S402", "S302", "bio_resin_raw"),
    ("S403", "S302", "solvents"),
    ("S403", "S303", "polymer_base"),
    # T3 → T2
    ("S301", "S201", "refined_lithium"),
    ("S302", "S201", "battery_electrolyte"),
    ("S302", "S202", "pcb_chemicals"),
    ("S303", "S203", "molded_resin"),
    # T2 → T1
    ("S201", "S101", "battery_cells"),
    ("S202", "S101", "circuit_boards"),
    ("S202", "S102", "circuit_boards"),
    ("S203", "S102", "housings"),
    # T1 → OEM
    ("S101", "OEM0", "sub_assembly_A"),
    ("S102", "OEM0", "sub_assembly_B"),
]

# Possible substitute edges (alternative routes)
SUBSTITUTE_EDGES: List[Tuple[str, str, str, float]] = [
    # (from, to, material, quality_penalty)
    ("S201", "S102", "battery_cells",   0.05),
    ("S303", "S202", "polymer_resin",   0.10),
    ("S101", "OEM0", "sub_assembly_B",  0.08),  # S101 can sub for S102
]


# ---------------------------------------------------------------------------
# Network class
# ---------------------------------------------------------------------------

class SupplyChainNetwork:
    """
    Manages the full supplier graph, routes, and daily material flows.
    Each tick() advances one simulation day.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.suppliers: Dict[str, SupplierState] = {}
        self.edges: List[Tuple[str, str, str]] = []
        self.substitute_edges: List[Tuple[str, str, str, float]] = []
        self.active_routes: List[OrderRoute] = []
        self.production_lines: List[ProductionLine] = []
        self.baseline_daily_cost: float = 0.0
        self._initialize()

    def _initialize(self):
        for s in DEFAULT_SUPPLIERS:
            self.suppliers[s["supplier_id"]] = SupplierState(
                capacity_pct=1.0,
                lead_time_days=s["base_lead_time_days"],
                **s
            )
        self.edges = list(DEFAULT_EDGES)
        self.substitute_edges = list(SUBSTITUTE_EDGES)
        self._setup_default_routes()
        self._setup_production_lines()
        self.baseline_daily_cost = self._compute_daily_cost()

    def _setup_default_routes(self):
        """Create default order routes along all edges."""
        self.active_routes = []
        for (src, dst, mat) in self.edges:
            src_s = self.suppliers[src]
            dst_s = self.suppliers[dst]
            qty = self._default_quantity(mat)
            self.active_routes.append(OrderRoute(
                from_supplier_id=src,
                to_supplier_id=dst,
                material_type=mat,
                quantity=qty,
                lead_time_days=src_s.lead_time_days,
                unit_cost=src_s.unit_cost,
            ))

    def _setup_production_lines(self):
        """OEM has two production lines consuming T1 sub-assemblies."""
        self.production_lines = [
            ProductionLine(
                line_id="LINE_A",
                required_materials={"sub_assembly_A": 50.0, "sub_assembly_B": 30.0},
                current_throughput_pct=1.0,
                daily_demand=80.0,
            ),
            ProductionLine(
                line_id="LINE_B",
                required_materials={"sub_assembly_B": 50.0},
                current_throughput_pct=1.0,
                daily_demand=50.0,
            ),
        ]

    def _default_quantity(self, material: str) -> float:
        defaults = {
            "lithium_ore": 100.0, "bio_resin_raw": 80.0, "solvents": 60.0,
            "polymer_base": 70.0, "refined_lithium": 90.0, "battery_electrolyte": 55.0,
            "pcb_chemicals": 40.0, "molded_resin": 65.0, "battery_cells": 80.0,
            "circuit_boards": 75.0, "housings": 70.0,
            "sub_assembly_A": 55.0, "sub_assembly_B": 55.0,
        }
        return defaults.get(material, 50.0)

    # -----------------------------------------------------------------------
    # Tick
    # -----------------------------------------------------------------------

    def tick(self, disruptions: Dict[str, object]) -> Dict[str, float]:
        """
        Advance one day. Apply disruptions, flow materials, produce.
        Routes are processed tier-by-tier (T4→T3→T2→T1→OEM) so upstream
        shortages naturally cascade within a single timestep.
        Returns per-line fill-rate dict.
        """
        # 1. Apply capacity from disruptions
        for sid, supplier in self.suppliers.items():
            if disruptions:
                active_ids = [
                    d for d in disruptions.values()
                    if d.affected_supplier_id == sid
                ]
                if active_ids:
                    worst = min(active_ids, key=lambda d: 1 - d.capacity_impact)
                    supplier.capacity_pct = max(0.0, 1.0 - worst.capacity_impact)
                    supplier.lead_time_days = int(
                        supplier.base_lead_time_days * worst.cost_multiplier
                    )
                    supplier.is_available = supplier.capacity_pct > 0.0
                else:
                    supplier.capacity_pct = min(1.0, supplier.capacity_pct + 0.05)
                    supplier.lead_time_days = supplier.base_lead_time_days
                    supplier.is_available = True

        # 2. Flow materials tier-by-tier: T4→T3→T2→T1→OEM
        #    Sort routes by source tier (highest tier value first)
        tier_sorted_routes = sorted(
            self.active_routes,
            key=lambda r: self.suppliers[r.from_supplier_id].tier.value,
            reverse=True,
        )

        material_availability: Dict[str, float] = {}
        for route in tier_sorted_routes:
            src = self.suppliers[route.from_supplier_id]
            
            # Node output is capped purely by accessible inventory
            available = src.inventory_held * src.capacity_pct
            delivered = min(route.quantity, available)
            src.inventory_held = max(0.0, src.inventory_held - delivered)

            dst_id = route.to_supplier_id
            if dst_id in self.suppliers:
                self.suppliers[dst_id].inventory_held += delivered

            mat = route.material_type
            material_availability[mat] = (
                material_availability.get(mat, 0.0) + delivered
            )

        # 5. Production at OEM
        fill_rates: Dict[str, float] = {}
        for line in self.production_lines:
            bottleneck = 1.0
            for mat, req_qty in line.required_materials.items():
                avail = material_availability.get(mat, 0.0)
                oem = self.suppliers.get("OEM0")
                if oem:
                    avail += oem.inventory_held * 0.02  # small buffer draw
                ratio = min(1.0, avail / req_qty) if req_qty > 0 else 1.0
                bottleneck = min(bottleneck, ratio)
            line.current_throughput_pct = bottleneck
            line.units_produced_today = line.daily_demand * bottleneck
            fill_rates[line.line_id] = bottleneck

        return fill_rates

    # -----------------------------------------------------------------------
    # Action handlers
    # -----------------------------------------------------------------------

    def reroute_order(self, from_id: str, to_id: str,
                      material: str, quantity: float) -> bool:
        """Remove existing route and add new one."""
        self.active_routes = [
            r for r in self.active_routes
            if not (r.from_supplier_id == from_id and r.material_type == material)
        ]
        if to_id not in self.suppliers:
            return False
        new_supplier = self.suppliers[to_id]
        self.active_routes.append(OrderRoute(
            from_supplier_id=to_id,
            to_supplier_id=from_id,  # keeps downstream the same
            material_type=material,
            quantity=quantity,
            lead_time_days=new_supplier.lead_time_days,
            unit_cost=new_supplier.unit_cost,
        ))
        return True

    def adjust_safety_stock(self, supplier_id: str,
                            target_days: float) -> float:
        """
        Set inventory target at a node.
        Returns additional cost incurred (holding cost proxy).
        """
        if supplier_id not in self.suppliers:
            return 0.0
        s = self.suppliers[supplier_id]
        current_days = s.inventory_held / max(1.0, self._daily_throughput(supplier_id))
        delta_days = target_days - current_days
        if delta_days <= 0:
            return 0.0
        units_to_buy = delta_days * self._daily_throughput(supplier_id)
        s.inventory_held += units_to_buy
        return units_to_buy * s.unit_cost * 0.15  # holding cost

    def expedite_shipment(self, order_id_or_material: str,
                          expedite_factor: float) -> float:
        """
        Speed up a shipment by expedite_factor (e.g. 2.0 = 2× faster).
        Returns extra cost incurred.
        """
        extra_cost = 0.0
        for route in self.active_routes:
            if route.material_type == order_id_or_material:
                route.lead_time_days = max(
                    1, int(route.lead_time_days / expedite_factor)
                )
                route.is_expedited = True
                extra_cost += route.quantity * route.unit_cost * (expedite_factor - 1) * 0.3
        return extra_cost

    def accept_substitute(self, supplier_id: str,
                          quality_penalty: float) -> bool:
        """Route from substitute supplier; apply quality penalty to throughput."""
        for (src, dst, mat, pen) in self.substitute_edges:
            if src == supplier_id:
                self.reroute_order(dst, src, mat, self._default_quantity(mat))
                # Quality penalty degrades production line throughput
                for line in self.production_lines:
                    line.current_throughput_pct *= (1.0 - quality_penalty)
                return True
        return False

    def negotiate_lead_time(self, supplier_id: str,
                             target_days: int, cost_premium_pct: float) -> float:
        """
        Pay a premium to reduce lead time.
        Returns extra cost. Lead time reduction capped at 30%.
        """
        if supplier_id not in self.suppliers:
            return 0.0
        s = self.suppliers[supplier_id]
        min_possible = max(1, int(s.base_lead_time_days * 0.7))
        s.lead_time_days = max(min_possible, target_days)
        # Apply cost to all routes through this supplier
        extra = 0.0
        for route in self.active_routes:
            if route.from_supplier_id == supplier_id:
                route.unit_cost *= (1 + cost_premium_pct / 100)
                extra += route.quantity * route.unit_cost * (cost_premium_pct / 100)
        return extra

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _daily_throughput(self, supplier_id: str) -> float:
        for route in self.active_routes:
            if route.from_supplier_id == supplier_id:
                return route.quantity
        return 50.0

    def _compute_daily_cost(self) -> float:
        return sum(r.quantity * r.unit_cost for r in self.active_routes)

    def get_daily_cost(self) -> float:
        return sum(r.quantity * r.unit_cost for r in self.active_routes)

    def get_esg_weighted_score(self) -> float:
        scores = [
            s.esg_score * s.capacity_pct
            for s in self.suppliers.values()
            if s.tier != SupplierTier.OEM
        ]
        return sum(scores) / max(1, len(scores))

    def snapshot(self) -> Dict[str, SupplierState]:
        return copy.deepcopy(self.suppliers)

    def reset(self, seed: int):
        self.rng = random.Random(seed)
        self._initialize()
