# ECU Selection Optimizer — Streamlit App

## Overview
The **ECU Selection Optimizer** is an interactive Streamlit web application that identifies the **optimal mix of Environmental Control Units (ECUs)** to meet shelter cooling and heating requirements. It balances multiple competing objectives such as cost, power consumption, weight, and size using **Mixed-Integer Linear Programming (MILP)**.

The app supports HVAC load profile, ECU catalog, and generator catalog uploads, extracts target cooling requirements automatically, and provides an optimized configuration recommendation for each shelter.

[Link to tool hosted on Streamlit Community Cloud](http://ecuoptimizationtool.streamlit.app)

Note: Changes to this repository are mirrored from a [git.mil repository](https://web.git.mil/e2o/ecu_optimization_tool) to [a public GitHub repository](https://github.com/deryk96/ecu_optimization_tool.git). This was done because Streamlit Community Cloud can only interact with GitHub, it does not support GitLab. Most of the development was done on a NIPR computer, which cannot connect to GitHub due to network constraints, hence the workaround.

---

## ✨ Features

- 📁 **Upload Inputs**
  - 24-hour HVAC profile for multiple shelters (xls or csv format).
    - This file is output from AutoDISE.
  - ECU specifications catalog (capacity, cost, power, weight, size).
    - Made by user, template available for download in app.

- 🧠 **Automatic Data Extraction**
  - Computes peak BTU load targets per shelter.
  - Extracts compatible generators and calculates the fuel use.

- ⚙️ **Optimization Engine**
  - Solves a **Mixed-Integer Linear Program (MILP)** using `scipy.optimize.milp`
  - See below for formulation.

- 📊 **Visualization**
  - Displays solution tables and charts
  - Allows users to adjust objective weights dynamically
  - Highlights trade-offs between competing criteria
  - Displays the amount of fuel each compatible generator would burn.

---

## 🚀 Getting Started

If you don't wish to run it using the link to Streamlit Community Cloud above:

### 1. Clone the Repository
```bash
git clone https://github.com/deryk96/ecu-selection-optimizer.git
cd ecu-selection-optimizer
```

Alternatively, if you're on a NIPR computer:
```bash
git clone https://sync.git.mil/e2o/ecu_optimization_tool.git
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
CSV or XLS files containing hourly BTU demand data for one or more shelters. Output from AutoDISE. See [user guide](Documentation/ECU_Tool_Guide.pdf) for instructions on how to generate this file.

### **2. ECU Catalog**
Catalog of available ECU models and their characteristics. Example available for download [here](Inputs/ECUSpecs.xlsx).

---

## 🧩 Dependencies

- [Streamlit](https://streamlit.io/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [SciPy](https://scipy.org/)
- [Plotly](https://plotly.com/python/)
- [OpenPyXL](https://openpyxl.readthedocs.io/)
- See [requirements file](requirements.txt) for full list.

---

## ⚖️ Optimization Logic

Full formulation available [here](Documentation/MILP%20Formulation.pdf).

---

## 🧮 Example Workflow

1. Upload your **HVAC profile** and **ECU catalog**.
2. Select which shelters are compatible with windowed ECUs.
3. Adjust objective weights for cost, power, weight, size, and excess BTU.
4. Click **Optimize**.
5. View results including:
   - ECU allocation per shelter
   - Total cost, power, and capacity summaries
   - Graphical trade-offs between optimization metrics
   - Fuel burn rate by generator

Alternatively, you can test the app with example data by clicking **Run Example Scenario**.

---

## 🛠 Developer Notes

- The app includes input validation to prevent unsafe characters from being entered in by the user.
- Built and tested with **Python 3.13** and **Streamlit 1.50+**
- Optimizer uses SciPy’s [`milp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html) function
