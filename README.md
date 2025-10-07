# ECU Selection Optimizer — Streamlit App

## Overview
The **ECU Selection Optimizer** is an interactive Streamlit web application that identifies the **optimal mix of Environmental Control Units (ECUs)** to meet shelter cooling requirements. It balances multiple competing objectives such as cost, power consumption, weight, and size using **Mixed-Integer Linear Programming (MILP)**.

The app supports HVAC load profile and ECU catalog uploads, extracts target cooling requirements automatically, and provides an optimized configuration recommendation for each shelter.

[Link to tool](http://ecuoptimizationtool.streamlit.app)

---

## ✨ Features

- 📁 **Upload Inputs**
  - 24-hour HVAC profile CSVs (multi-table format)
    - This file is output from AutoDISE.
  - ECU specifications catalog (capacity, cost, power, weight, size)
    - Made by user, template available for download in app.

- 🧠 **Automatic Data Extraction**
  - Computes peak BTU load targets per shelter
  - Normalizes ECU characteristics for optimization

- ⚙️ **Optimization Engine**
  - Solves a **Mixed-Integer Linear Program (MILP)** using `scipy.optimize.milp`
  - Decision variables:
    - `x[s, m]`: number of ECUs of model *m* assigned to shelter *s*
    - `e[s]`: excess BTU capacity for shelter *s*
  - Constraints:
    1. Total ECU capacities ≥ required BTU per shelter  
    2. Excess variables track oversupply  
    3. ECU counts are non-negative integers  
  - Objective:
    - Minimize a weighted, normalized sum of:
      - Cost  
      - Power  
      - Weight  
      - Size  
      - Excess BTUs (penalty)

- 📊 **Visualization**
  - Displays solution tables and charts
  - Allows users to adjust objective weights dynamically
  - Highlights trade-offs between competing criteria

---

## 🚀 Getting Started

If you don't wish to run it using the link above:

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/ecu-selection-optimizer.git
cd ecu-selection-optimizer
```

### 2. Create and Activate a Virtual Environment

- Use a Python environment of your choosing.

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App
```bash
streamlit run ecu_app.py
```

Then open the link displayed in your terminal (usually `http://localhost:8501`).

---

## 📂 Input File Formats

### **1. HVAC Load Profiles**
CSV files containing hourly BTU demand data for one or more shelters. Output from AutoDISE. See [user guide](https://github.com/deryk96/ecu_optimization_tool/blob/main/Documentation/ECU_Tool_Guide.pdf) for instructions on how to generate this file.

### **2. ECU Catalog**
Catalog of available ECU models and their characteristics. Example available for download [here](https://github.com/deryk96/ecu_optimization_tool/blob/main/Inputs/ECUSpecs.csv).

| Model | Capacity (BTU) | Power (kW) | Cost ($) | Weight (lb) | Size (ft³) |
|--------|----------------|-------------|-----------|--------------|-------------|
| ECU-1 | 12000 | 1.5 | 2000 | 180 | 5.2 |
| ECU-2 | 18000 | 2.0 | 2500 | 210 | 6.0 |
| ... | ... | ... | ... | ... | ... |

---

## 🧩 Dependencies

- [Streamlit](https://streamlit.io/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [SciPy](https://scipy.org/)
- [Plotly](https://plotly.com/python/)
- [OpenPyXL](https://openpyxl.readthedocs.io/)
- See requirements.txt for full list.

---

## ⚖️ Optimization Logic

The MILP problem is formulated as:

\[
\min_x \; w_c C(x) + w_p P(x) + w_w W(x) + w_s S(x) + w_e E(x)
\]

Subject to:
\[
\sum_m \text{Capacity}_{m} \cdot x_{s,m} \geq \text{TargetBTU}_s \quad \forall s
\]
\[
x_{s,m} \in \mathbb{Z}_{\ge 0}
\]

Where each \( w_i \) is a user-specified weight.

Full formulation available [here](https://github.com/deryk96/ecu_optimization_tool/blob/main/Documentation/MILP%20Formulation.pdf).

---

## 🧮 Example Workflow

1. Upload your **HVAC profile** and **ECU catalog**.
2. Adjust objective weights for cost, power, weight, size, and excess BTU.
3. Click **Run Optimization**.
4. View results including:
   - ECU allocation per shelter
   - Total cost, power, and capacity summaries
   - Graphical trade-offs between optimization metrics

---

## 🛠 Developer Notes

- The app includes input validation to prevent unsafe characters using regex:
  ```python
  pattern = r"^[A-Za-z0-9_.\-() ]+$"
  ```
- Built and tested with **Python 3.13** and **Streamlit 1.50+**
- Optimizer uses SciPy’s [`milp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html) function







