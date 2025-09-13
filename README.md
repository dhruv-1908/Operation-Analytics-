# Operation-Analytics-


# Production Planning Optimization in Flexible Manufacturing Systems

## Overview

This project focuses on **optimizing production planning** in a flexible manufacturing system (FMS). The core objective was to create an advanced optimization model that **minimizes total operational costs** while supporting better decision-making for managers.

The work began with a baseline **Mixed-Integer Linear Programming (MILP)** model that reduced rejected batches. While effective, it lacked financial realism. To address this, the model was extended to include:

* **Machine operating costs**
* **Financial penalties for rejected batches**

The extended version, implemented using **Google OR-Tools CP-SAT solver in Python**, provides a more realistic framework by balancing production costs and rejection penalties. To validate the model, synthetic datasets were generated and analyzed across multiple operational scenarios.

---

## Key Findings from Sensitivity Analysis

1. **Machine Operating Costs** – Higher costs did not always raise total cost; sometimes rejecting batches or underusing expensive machines was more efficient.
2. **Rejection Penalties** – Low penalties encouraged full rejection, while higher penalties drove full scheduling.
3. **Planning Horizon** – Longer horizons generally lowered costs and rejections, though results were non-linear.
4. **Production Time Volatility** – Longer processing times increased costs and rejections.
5. **Batch Mix** – The ratio of batch types significantly affected feasibility and efficiency due to shared workshop constraints.

---

## Managerial Insights

* **Accurate Costing**: True machine operating costs must be known to avoid suboptimal decisions.
* **Penalty Structures**: Rejection penalties strongly influence scheduling choices and profitability.
* **Planning Flexibility**: Adjusting horizons has non-linear effects on efficiency.
* **Product Mix Management**: Portfolio balance is key under shared constraints.

---

## Project Structure

### Baseline Model

* Based on *Rezig et al. (2020)*, “Mathematical Model for Production Plan Optimization.”
* Objective: **Minimize rejected batches**.
* Implemented in Python (`production_plan_base.py`).
* Key constraints: batch assignment, machine capacity, time windows, workshop restrictions, and non-interruptibility.
* Results: Achieved feasible scheduling with minimal rejections.

### Extended Model

* Objective modified to **minimize total cost** = production expenses + rejection penalties.
* Added parameters:

  * `Cj` = machine operating cost per hour
  * `Pi` = penalty for rejecting batch i
* Implemented in `production_plan_extended.py`.
* Generates schedules that balance machine usage and penalties, sometimes choosing strategic rejections to reduce overall cost.

---

## Sensitivity Analysis Scenarios

Conducted using `production_plan_sensitivity_analysis.py` across five dimensions:

1. **Machine Costs** (e.g., raising S2 cost shifted scheduling strategy).
2. **Rejection Penalty Scale** (low penalties → more rejections; high penalties → forced scheduling).
3. **Planning Horizon** (longer horizons improved cost and fulfillment).
4. **Production Times** (shorter times reduced costs and rejections).
5. **Batch Mix** (imbalanced demand increased scheduling difficulties).

Results were stored in `sensitivity_analysis_results.xlsx` with visualizations generated via Matplotlib.

---

## Assumptions & Limitations

* Deterministic inputs (costs, times, demand known in advance).
* Machines assumed fully available (no breakdowns/maintenance).
* No preemption (batches run to completion).
* No setup/changeover times modeled.
* Inventory and supply chain constraints not included.
* Static schedule (not adaptive to real-time changes).

These assumptions simplify the problem but limit real-world applicability.

---

## Conclusion

This project extends a classic MILP scheduling model into a **financially aware optimization framework**. By combining production costs and rejection penalties, the model identifies **economically optimal schedules** rather than just feasible ones.

The sensitivity analysis highlighted:

* Cost–penalty trade-offs that guide resource allocation.
* The importance of accurate cost/penalty calibration.
* The role of planning horizon, product mix, and production efficiency in operational outcomes.

This work demonstrates how advanced optimization can support **strategic production planning** and lays the foundation for future enhancements in **real-time, uncertainty-aware scheduling models**.

---

## References

* Google. (n.d.). [Google OR-Tools: Introduction](https://developers.google.com/optimization/introduction)
* Rezig, S., Ezzeddine, W., Turki, S., & Rezg, N. (2020). *Mathematical Model for Production Plan Optimization—A Case Study of Discrete Event Systems*. Mathematics, 8(6), 955.


