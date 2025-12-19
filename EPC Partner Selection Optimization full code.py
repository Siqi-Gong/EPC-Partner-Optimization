import pandas as pd
import numpy as np
import pulp
import matplotlib.pyplot as plt
from collections import Counter

# =========================================================
# 1) Data Layer
# =========================================================
data = {
    'Company': ['BayWa r.e.', 'Orsted', 'Wood Group', 'Greencells', 'Ethical Power', 'Edina', 'Anaergia'],
    'Fluid_Exp':       [8, 9, 10, 4, 3, 6, 10],   # Tech / Fluid Engineering
    'Startup_Agility': [5, 2, 3, 7, 8, 9, 10],    # Agility
    'Logistics_Fit':   [10, 3, 3, 7, 4, 7, 1],    # Amazon-related
    'Biz_Model':       [10, 6, 4, 10, 9, 3, 5],   # Business model / Scale
    'Has_DE':          [1, 1, 1, 1, 0, 0, 0],     # Covers Germany
    'Has_US':          [1, 1, 1, 0, 0, 0, 1],     # Covers USA
    'Mgmt_Friction':   [9, 10, 9, 7, 6, 4, 5]     # Management friction / Complexity (higher = harder)
}
df = pd.DataFrame(data)

def total_mgmt_friction(selected_names):
    if not selected_names:
        return 0
    return int(df.loc[df["Company"].isin(selected_names), "Mgmt_Friction"].sum())


# =========================================================
# 2) Optimization Engine (Hybrid)
# =========================================================
def solve_scenario_hybrid(
    scenario_name,
    benefit_weights,
    min_thresholds=None,
    cost_sensitivity=1.0,
    budget_limit=None,
    risk_penalty=0.2,
    synergy_bonus=20,
    agility_bonus=15,
    integration_tax=8,
    friction_scale=5
):
    """
    Returns:
      sel_names, objective_value,
      is_trans_real, is_agile_real, has_de_real, has_us_real,
      min_fluid_in_portfolio, min_agile_in_portfolio
    """
    if min_thresholds is None:
        min_thresholds = {}

    total_b = sum(benefit_weights.values())
    if total_b <= 0:
        raise ValueError("benefit_weights must sum to a positive number.")
    w = {k: v / total_b for k, v in benefit_weights.items()}

    prob = pulp.LpProblem(str(scenario_name).replace(" ", "_"), pulp.LpMaximize)
    x = pulp.LpVariable.dicts("Select", df.index, cat="Binary")

    # -------- Hard thresholds (non-compensatory) ----------
    for i in df.index:
        for col, th in min_thresholds.items():
            if col not in df.columns:
                raise ValueError(f"Threshold column '{col}' not found in df.")
            if df.loc[i, col] < th:
                prob += x[i] == 0


    # Bonus helper variables
    y_de = pulp.LpVariable("Cover_DE", 0, 1, cat="Binary")
    y_us = pulp.LpVariable("Cover_US", 0, 1, cat="Binary")
    y_trans = pulp.LpVariable("Bonus_Trans", 0, 1, cat="Binary")
    y_agile = pulp.LpVariable("Bonus_Agile", 0, 1, cat="Binary")

    # -------- Objective components ----------
    score_expr = []
    cost_expr = []
    for i in df.index:
        # 1) Base benefit score
        s = (
            w.get("w_fluid", 0) * df.loc[i, "Fluid_Exp"] +
            w.get("w_agile", 0) * df.loc[i, "Startup_Agility"] +
            w.get("w_log",   0) * df.loc[i, "Logistics_Fit"] +
            w.get("w_biz",   0) * df.loc[i, "Biz_Model"]
        ) * 10       

        if df.loc[i, "Fluid_Exp"] < 4:
            s *= (1 - risk_penalty)

        score_expr.append(s * x[i])

        c = df.loc[i, "Mgmt_Friction"] * friction_scale
        cost_expr.append(c * x[i])

    integration_penalty = integration_tax * pulp.lpSum([x[i] for i in df.index])

    prob += (
        pulp.lpSum(score_expr)
        - cost_sensitivity * pulp.lpSum(cost_expr)
        + synergy_bonus * y_trans
        + agility_bonus * y_agile
        - integration_penalty
    )

    # -------- Constraints (STRICT LOGIC via Big-M) ----------
    M = len(df)

    # y_de <-> any DE covered
    prob += pulp.lpSum([df.loc[i, "Has_DE"] * x[i] for i in df.index]) >= y_de
    prob += pulp.lpSum([df.loc[i, "Has_DE"] * x[i] for i in df.index]) <= M * y_de

    # y_us <-> any US covered
    prob += pulp.lpSum([df.loc[i, "Has_US"] * x[i] for i in df.index]) >= y_us
    prob += pulp.lpSum([df.loc[i, "Has_US"] * x[i] for i in df.index]) <= M * y_us

    # transatlantic bonus = AND
    prob += y_trans <= y_de
    prob += y_trans <= y_us
    prob += y_trans >= y_de + y_us - 1

    # y_agile <-> any partner with agility >= 8 selected
    is_agile_flag = {i: 1 if df.loc[i, "Startup_Agility"] >= 8 else 0 for i in df.index}
    prob += pulp.lpSum([is_agile_flag[i] * x[i] for i in df.index]) >= y_agile
    prob += pulp.lpSum([is_agile_flag[i] * x[i] for i in df.index]) <= M * y_agile

    # Portfolio size
    prob += pulp.lpSum([x[i] for i in df.index]) >= 2
    prob += pulp.lpSum([x[i] for i in df.index]) <= 4

    # Survival: at least one market
    prob += y_de + y_us >= 1

    # Wallet cap (hard budget on raw Mgmt_Friction)
    if budget_limit is not None:
        prob += pulp.lpSum([df.loc[i, "Mgmt_Friction"] * x[i] for i in df.index]) <= budget_limit

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if prob.status != pulp.LpStatusOptimal:
        return [], 0.0, False, False, False, False, 0, 0

    sel_indices = [i for i in df.index if x[i].varValue == 1]
    sel_names = df.loc[sel_indices, "Company"].tolist()
    val = float(pulp.value(prob.objective))

    # Truth checks from selection
    has_de_real = bool((df.loc[sel_indices, "Has_DE"] == 1).any()) if sel_indices else False
    has_us_real = bool((df.loc[sel_indices, "Has_US"] == 1).any()) if sel_indices else False
    is_trans_real = has_de_real and has_us_real
    is_agile_real = bool((df.loc[sel_indices, "Startup_Agility"] >= 8).any()) if sel_indices else False

    min_fluid = int(df.loc[sel_indices, "Fluid_Exp"].min()) if sel_indices else 0
    min_agile = int(df.loc[sel_indices, "Startup_Agility"].min()) if sel_indices else 0

    return sel_names, val, is_trans_real, is_agile_real, has_de_real, has_us_real, min_fluid, min_agile


# =========================================================
# 3) Scenarios
# =========================================================
SCENARIOS = [
    {
        "name": "Agile Pilot",
        "weights": {"w_fluid": 1, "w_agile": 10, "w_log": 5, "w_biz": 1},
        "thresholds": {"Startup_Agility": 7},   # Each partner must be agile
        "lam": 1.5,                              # More friction-sensitive (strategic phase)
        "tax": 12                                # Small teams fear too many partners
    },
    {
        "name": "Global Scale",
        "weights": {"w_fluid": 2, "w_agile": 1, "w_log": 2, "w_biz": 10},
        "thresholds": {"Biz_Model": 6},          # Each partner must have sufficient scale/model
        "lam": 0.5,                               # Less friction-sensitive (strategic phase)
        "tax": 5                                  # Large teams can absorb coordination overhead
    },
    {
        "name": "Tech Purist",
        "weights": {"w_fluid": 10, "w_agile": 1, "w_log": 1, "w_biz": 1},
        "thresholds": {"Fluid_Exp": 8},          # Each partner must meet technical threshold
        "lam": 1.0,
        "tax": 8
    }
]


# =========================================================
# 4) Strategy -> Wallet (Pareto) + Plot
# =========================================================
def plot_all_paretos(pareto_results):
    """
    pareto_results: dict[str -> DataFrame]
      each df has columns: BudgetCap, Friction, Score, Portfolio
    Draw all scenarios on ONE figure with different colors.
    """
    # Filter out empty entries
    non_empty = {k: v for k, v in pareto_results.items() if v is not None and not v.empty}
    if not non_empty:
        print("[Warn] No Pareto data to plot.")
        return

    plt.figure(figsize=(10, 6))

    color_map = {
        "Agile Pilot": "#b71c1c",
        "Global Scale": "#1565c0",
        "Tech Purist": "#2e7d32"
    }

    for scen, p_df in non_empty.items():
        p_df = p_df.drop_duplicates(subset=["Friction", "Score", "Portfolio"]).sort_values("Friction")

        # Plot curve
        plt.plot(
            p_df["Friction"], p_df["Score"],
            marker="o",
            linewidth=2,
            label=scen,
            color=color_map.get(scen, None)  # Fall back to default if not specified
        )

        # Add "Smart Elbow" annotation per curve (computed independently)
        if len(p_df) >= 2:
            denom = (p_df[["Friction", "Score"]].max() - p_df[["Friction", "Score"]].min() + 1e-9)
            p_norm = (p_df[["Friction", "Score"]] - p_df[["Friction", "Score"]].min()) / denom
            best_idx = (p_norm["Score"] - p_norm["Friction"]).idxmax()

            fx = p_df.loc[best_idx, "Friction"]
            sc = p_df.loc[best_idx, "Score"]
            port = p_df.loc[best_idx, "Portfolio"]

            plt.annotate(
                f"{scen}\n{port}",
                xy=(fx, sc),
                xytext=(fx + 0.8, sc - 12),
                arrowprops=dict(facecolor="black", shrink=0.05),
                fontsize=9
            )

    plt.title("Pareto Frontiers (All Scenarios): Value vs. Friction", fontsize=14)
    plt.xlabel("Management Friction (Lower is better)", fontsize=12)
    plt.ylabel("Strategic Score", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_strategy_then_wallet(scenarios, budget_min=6, budget_max=30, plot_scenario="Agile Pilot"):
    # ---------------- Step 1: Strategy-Optimal ----------------
    print("\n" + "="*110)
    print("STEP 1) Strategy-Optimal (no wallet cap)  —— Determine the strategy-optimal portfolio first")
    print("="*110)
    print(f"{'Scenario':<15} | {'Strategy-Optimal Portfolio':<55} | Friction | #P | MinTech | MinAgile")
    print("-"*110)

    strategy_best = {}
    for sc in scenarios:
        sel, val, trans, agile, de, us, min_f, min_a = solve_scenario_hybrid(
            scenario_name=sc["name"],
            benefit_weights=sc["weights"],
            min_thresholds=sc["thresholds"],
            cost_sensitivity=sc["lam"],
            integration_tax=sc["tax"],
            budget_limit=None
        )
        fr = total_mgmt_friction(sel)
        strategy_best[sc["name"]] = sel
        print(f"{sc['name']:<15} | {str(sel):<55} | {fr:^8} | {len(sel):^2} | {min_f:^6} | {min_a:^7}")

    # ---------------- Step 2: Wallet-Pareto under SAME rules ----------------
    print("\n" + "="*110)
    print("STEP 2) Wallet-Feasible (Pareto) —— Under the same strategy rules, see what the wallet can buy")
    print("="*110)

    pareto_results = {}
    budgets = list(range(budget_min, budget_max + 1))

    for sc in scenarios:
        name = sc["name"]
        pareto_rows = []

        for b in budgets:
            # Wallet phase: budget is a hard constraint; don't subtract friction in objective (avoid double-penalty)
            sel, val, trans, agile, de, us, min_f, min_a = solve_scenario_hybrid(
                scenario_name=f"{name}_Pareto",
                benefit_weights=sc["weights"],
                min_thresholds=sc["thresholds"],
                cost_sensitivity=0.0,        # IMPORTANT
                integration_tax=sc["tax"],   # keep integration tax (realistic)
                budget_limit=b
            )
            if sel:
                fr = total_mgmt_friction(sel)
                pareto_rows.append({"BudgetCap": b, "Friction": fr, "Score": val, "Portfolio": str(sel)})

        pareto_df = pd.DataFrame(pareto_rows).drop_duplicates(subset=["Friction", "Score", "Portfolio"])
        pareto_results[name] = pareto_df

        print(f"\n--- {name} Pareto (BudgetCap -> Best Portfolio) ---")
        if pareto_df.empty:
            print("No feasible solution under scanned budgets (check thresholds too strict vs budget).")
            continue
        print(pareto_df.sort_values(["Friction", "Score"]).to_string(index=False))

    return strategy_best, pareto_results


# =========================================================
# 5) Monte Carlo Robustness
# =========================================================
def run_monte_carlo(
    scenario,
    n_runs=1000,
    lam_range=(0.5, 1.5),
    tax_range=None,
    budget_limit=None,
    seed=42,
    jitter_thresholds=True  # Whether to jitter thresholds in MC to create “hesitation”
):
    """
    Monte Carlo robustness for ONE scenario.
    Randomize benefit weights + lambda (+ optional tax).
    Thresholds are HARD (non-compensatory). Optionally jitter thresholds in MC.
    Returns: pandas Series of inclusion probability (%)
    """
    np.random.seed(seed)
    inclusion = Counter()

    keys = list(scenario["weights"].keys())

    for _ in range(n_runs):
        # 1) Randomize weights (Dirichlet)
        w_vals = np.random.dirichlet(np.ones(len(keys)), size=1)[0]
        w_rand = dict(zip(keys, w_vals))

        # 2) Randomize lambda
        lam = float(np.random.uniform(lam_range[0], lam_range[1]))

        # 3) Randomize tax (optional)
        if tax_range is None:
            tax = scenario["tax"]
        else:
            tax = float(np.random.uniform(tax_range[0], tax_range[1]))

        # 4) Thresholds: HARD, but optionally jitter by ±1 ONLY in MC
        th = dict(scenario["thresholds"])  # copy
        if jitter_thresholds and th:
            for k in th:
                th[k] = int(np.clip(th[k] + np.random.choice([-1, 0, 1]), 1, 10))

        # 5) Solve
        sel, *_ = solve_scenario_hybrid(
            scenario_name="MC",
            benefit_weights=w_rand,
            min_thresholds=th,          
            cost_sensitivity=lam,
            integration_tax=tax,
            budget_limit=budget_limit
        )

        for nm in sel:
            inclusion[nm] += 1

    probs = pd.Series({c: inclusion[c] / n_runs * 100 for c in df["Company"]}).sort_values(ascending=False)
    return probs

# =========================================================
# 6) Main Program
# =========================================================
if __name__ == "__main__":
    # A) Strategy -> Wallet (Pareto) + Plot
    strategy_best, pareto_results = run_strategy_then_wallet(
        SCENARIOS,
        budget_min=6,
        budget_max=40,
    )
    plot_all_paretos(pareto_results)

    # B) Monte Carlo for each scenario
    print("\n" + "="*110)
    print("STEP 3) Monte Carlo Robustness —— Is the conclusion stable? (Perturbing weights/λ)")
    print("="*110)

    for sc in SCENARIOS:
        probs = run_monte_carlo(
            scenario=sc,
            n_runs=1000,
            lam_range=(0.5, 1.5),
            tax_range=None,     # If you want random tax: (3, 15)
            budget_limit=None,
            seed=42,
            jitter_thresholds=True # True = “more hesitant”; False = pure hard-threshold MC
        )
        print(f"\n--- {sc['name']} (MC Inclusion %) ---")
        print(probs.round(1))
