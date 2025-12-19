# EPC Partner Selection Optimization Engine
> A Decision Science Framework for Strategic Infrastructure Deployment

**!! All information and data are open-sourced !!**

## Project Overview
This project provides a **Strategic Decision Support System** designed for a CleanTech startup specializing in membrane-less flow batteries. The engine optimizes the selection of Engineering, Procurement, and Construction (EPC) partners for commercial pilots (e.g., Amazon Logistics sites) by balancing technical competency, management bandwidth, and market coverage.

## Technical Highlights
- **Hybrid Optimization Logic**: Combines non-compensatory **Hard Thresholds** (Min competency) with **Soft Incentives** (Synergy bonuses).
- **Decision Science Models**: 
    - **Big-M Constraints**: Ensures strict logical binding between selection states and market coverage.
    - **Integration Tax**: Models non-linear management friction that grows with portfolio complexity.
    - **Pareto Frontier**: Identifies the "sweet spot" between strategic value and management cost.
    - **Monte Carlo Simulation**: Validates solution robustness through 1,000+ randomized iterations.

## Strategic Scenarios
The engine evaluates partners across three distinct strategic "archetypes":
1. **Agile Pilot**: Prioritizes speed and flexibility for rapid Amazon deployment.
2. **Global Scale**: Focuses on bankability and geographic reach (DE/US) for expansion.
3. **Tech Purist**: Enforces extreme engineering standards for core fluid dynamics safety.


## Key Insights
- **The "Sweet Spot"**: Identified a specific budget threshold (Friction = 16) where a marginal increase in management effort yields a **~20% leap** in strategic portfolio value.
- **Anchor Partner Identification**: Monte Carlo results identified **Anaergia** as a "Strategically Indispensable" partner with a >99% inclusion rate across multiple scenarios.


## How to Run
1. Clone the repo: `git clone https://github.com/yourname/EPC-Partner-Optimization.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Execute the engine: `python main.py`

## Dependencies
- `pandas`: Data manipulation
- `numpy`: Numerical simulations
- `pulp`: Linear Programming (MILP solver)
- `matplotlib`: Data visualization
