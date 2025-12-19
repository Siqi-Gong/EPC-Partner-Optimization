# Workflow SOP: Evolutionary Logic of the EPC Selection Engine

This document outlines the strategic rationale and technical iterations involved in developing the EPC Selection Engine. It documents the transition from simple linear scoring to a sophisticated, non-linear optimization framework.

---

## Phase 1: From Linear Scoring to Synergy Incentives

### Initial State: Weighted Linear Averaging
* **Approach**: Assigning weights to criteria (Tech, Agility, Cost) and summing scores.
* **Decision Logic (Why B over A)**: 
    * *Option A (Discarded)*: Pure linear summation.
    * *Option B (Adopted)*: Introduction of **Synergy Bonuses**.
    * *Rationale*: In commercial reality, the value of a portfolio is greater than the sum of its parts. A "Transatlantic" presence (covering both DE and US) offers strategic advantages that individual market scores cannot capture.

### Identified Flaw: The "Cheap Trap"
* **Pain Point**: The solver favored low-cost, low-competency firms just to "collect" synergy bonuses, as the saved management friction outweighed the utility gain of high-tier partners.

---

## Phase 2: Structural Hardening (Survival Thresholds)

### Implementation: Non-Compensatory Constraints
* **Approach**: Converting market presence and core tech requirements into **Hard Constraints**.
* **Decision Logic (Why B over A)**:
    * *Option A (Discarded)*: Increasing bonus values to "force" better selections.
    * *Option B (Adopted)*: **Binary Survival Filters** (e.g., `Fluid_Exp >= 8` for Tech Purist scenario).
    * *Rationale*: For a DeepTech startup, certain technical gaps are **non-compensable**. A high business score cannot mitigate the risk of technical failure during a pilot.

---

## Phase 3: Modeling Management Friction (The "Integration Tax")

### Implementation: Discrete Integration Penalties
* **Approach**: Introducing an `integration_tax` per partner and a `cost_sensitivity` (λ) parameter.
* **Decision Logic (Why B over A)**:
    * *Option A (Discarded)*: Treating management friction as a simple linear cost.
    * *Option B (Adopted)*: **Discrete Integration Tax**.
    * *Rationale*: Startup bandwidth is a finite resource. Managing 4 partners is exponentially more complex than managing 2. The model must penalize "over-selection" even if scores are high.

---

## Phase 4: Mathematical Rigor (Big-M Implementation)

### Implementation: Logical Binding
* **Approach**: Implementing **Big-M Constraints** for fact-equivalence.
* **Decision Logic (Why B over A)**:
    * *Option A (Discarded)*: Unidirectional constraints (`sum(x) >= y`).
    * *Option B (Adopted)*: **Bidirectional Big-M Binding** (`sum(x) <= M*y`).
    * *Rationale*: Prevents "logical drift" during sensitivity tests. Ensures that the state variable ($y$) strictly reflects the reality of the selection ($x$), even when bonuses are zero.

---

## Phase 5: Strategic Decision Engine (Strategy -> Wallet)

### Final Workflow: Two-Step Optimization
1. **Step 1: Strategic Archetype Definition**: Defining "Agile Pilot," "Global Scale," and "Tech Purist" personas with scenario-specific hard thresholds.
2. **Step 2: Pareto Frontier Analysis**: Scanning budget limits within each scenario to find the **"Sweet Spot"** (the point of maximum marginal utility).
    * *Critical Decision*: Set `cost_sensitivity` to 0 during Pareto runs to avoid "double-counting" friction as both a constraint and an objective penalty.

---

## Final Executive Insights

The evolutionary process resulted in three key strategic pillars:
1. **The Tactical Anchor**: **Anaergia** emerged as the most robust partner (>99% inclusion), acting as a natural hedge against both technical and agility risks.
2. **The Efficiency Frontier**: Identified a specific friction threshold (**Friction = 16**) where a marginal increase in management budget triggers a **~20% leap** in strategic value.
3. **The Scaling Roadmap**: Clearly delineated that while **Edina** is optimal for rapid pilots, **BayWa r.e.** becomes indispensable as the strategy shifts toward "Bankability" and global scaling.
