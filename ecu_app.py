""" 
ECU Selection Optimizer — Streamlit App 
--------------------------------------- 

This application helps identify the optimal mix of Environmental Control Units (ECUs) 
to satisfy shelter cooling requirements while balancing multiple competing objectives. 

Features: 
- Upload HVAC 24-hour profile CSVs (multi-table format). 
- Upload ECU specifications catalog (capacity, cost, power, weight, size). 
- Extract peak BTU load targets for each shelter automatically. 
- Solve a Mixed-Integer Linear Program (MILP) using SciPy's `milp` solver: 
Decision Variables: 
x[s, m] : number of ECUs of model m assigned to shelter s 
e[s] : excess BTU capacity for shelter s 
Constraints: 
1. Sum of ECU capacities ≥ TargetBTU for each shelter 
2. Excess variable tracks oversupply of BTU 
3. ECU counts are integers ≥ 0 
Objective Function: 
Minimize weighted normalized sum of cost, power, weight, size, 
plus a penalty on excess BTUs produced. This function is heavily influenced 
by the user-provided weights for each parameter. 

Normalization: 
Each attribute (cost, power, weight, size) is divided by its maximum 
value in the catalog to normalize between 0 and 1. 
Excess BTU penalty is scaled by (excess / target). 
Constraints are NOT normalized (they use real BTU values). 

- Interactive weight sliders (0-10) for cost, power, weight, size, and BTU penalty. 
- Displays results table with selected ECU mix per shelter, objective value, and totals. 
- Visualizations: 
* Target vs Achieved BTU per shelter 
* Normalized objective function component breakdown 
* ECU mix distribution per shelter 
- Downloadable solution table as CSV or Excel. 

Usage: 
$ streamlit run ecu_app.py 
Also deployed online at ecuoptimizationtool.streamlit.app 

Author: Capt Deryk Clary 
Office: Expeditionary Energy Office 
Email: deryk.l.clary.mil@usmc.mil 
Date: 2025-09-16 
""" 

# TODO: Implement window unit constraints 

# Import statements 
import io 
import numpy as np 
import pandas as pd 
import streamlit as st 
import seaborn as sns 
import matplotlib.pyplot as plt 
from matplotlib.ticker import MaxNLocator 
import zipfile 
from datetime import datetime 
from pathlib import Path 
import matplotlib.patches as mpatches 
import random 

# MILP tools 
from scipy.optimize import milp, LinearConstraint, Bounds 

sns.set_theme(style="whitegrid", palette="colorblind") 

# HVAC data to Pandas 
# @st.cache_data  # Holds the result in the cache for recall  # TODO: Uncomment
def read_multiple_tables(f): 
    """ 
    Reads a CSV string with multiple tables separated by blank lines 
    into a list of Pandas DataFrames. 

    Args: 
        csv_string (str): The CSV data as a string. 

    Returns: 
        list: A list of Pandas DataFrames, one for each table. 
        Returns an empty list if no tables are found or if an error occurs. 
    """ 
    rows = [] 
    current_ecu_config = None 
    current_shelter = None 
    current_header = None 
    current_subtable = None 

    for line in f: 
        stripped = line.decode("utf-8").strip().strip(",") 

        if not stripped:  # blank line → table break 
            continue 

        # If this is a shelter name row 
        if stripped.startswith("24 Profile for Shelter") or stripped.startswith("\ufeff24 Profile for Shelter"): 
            split_str = stripped.split(" ") 
            current_ecu_config = split_str[-1] 
            current_shelter = ' '.join(split_str[-3:-1]) 
            current_header = None 
            continue 

        # Detect subtable title (row without commas or with commas only at end) 
        if not "," in stripped: 
            current_subtable = stripped 
            current_header = None 
            continue 

        # If this is a header row (contains commas and no table header yet) 
        if current_header is None: 
            current_header = stripped.split(",") 
            current_header.insert(0, "RowName") 
            continue 

        # Otherwise, it's a data row 
        row_data = stripped.split(",") 
        row_dict = { 
            "ShelterName": current_shelter, 
            "ECUConfig": current_ecu_config, 
            "SubTableName": current_subtable, 
            "RowName": row_data[0] 
        } 
        row_dict.update(dict(zip(current_header, row_data))) 
        rows.append(row_dict) 

    # Combine into DataFrame and Return 
    return pd.DataFrame(rows)


# --------------------- 
# Helper: parse multi-table HVAC CSVs into tidy long DF 
# TODO: Delete 
# --------------------- 
def parse_multi_table_csv_like(df_raw: pd.DataFrame) -> pd.DataFrame: 
    """ 
    Convert a DataFrame that was produced by reading a multi-table CSV (with shelter/subtable rows) 
    into a tidy long structure with columns: 
    ShelterName, SubTableName, RowName, <hour columns...> 
    The uploaded CSV might already be in tidy form; function tries to be robust. 
    """ 
    # If the dataframe already contains a RowName and ShelterName columns and numeric hour columns -> assume tidy wide 
    cols = list(df_raw.columns) 
    # find hour-like columns (0..23 or '0000 - 0100' style) 
    hour_cols = [c for c in cols if str(c).strip().isdigit() or (isinstance(c, str) and c.strip().split()[0].isdigit())] 
    # Try common column names 
    if set(["ShelterName", "SubTableName", "RowName"]).issubset(set(cols)) and hour_cols: 
        tidy = df_raw.copy() 
        return tidy 
    # Otherwise attempt to detect header rows: look for "ShelterName" like rows. 
    # We'll try the simple fallback: assume first 4 columns are meta (ShelterName, ECUConfig, SubTableName, RowName) 
    meta = [] 
    possible_meta = ["ShelterName", "ECUConfig", "SubTableName", "RowName"] 
    # If these names aren't present, try to map. 
    # We'll create them by taking first 4 columns as these meta labels 
    if len(cols) >= 4: 
        tidy = df_raw.copy() 
        tidy = tidy.rename(columns={cols[0]: "ShelterName", 
                                    cols[1]: "ECUConfig", 
                                    cols[2]: "SubTableName", 
                                    cols[3]: "RowName"}) 
        return tidy 
    else: 
        raise ValueError("Uploaded HVAC CSV does not have expected structure; need at least 4 columns of metadata.") 


# --------------------- 
# Extract target BTU per shelter (peak of 'Shelter HVAC Heat Load' rows) 
# --------------------- 
# @st.cache_data  # Holds result in cache  # TODO: Uncomment
def extract_targets_from_hvac_df(tidy_wide: pd.DataFrame, shelter_col="ShelterName", rowname="Shelter HVAC Heat Load"): 
    """ 
    Input: tidy_wide with Hour columns in wide format (string hour names or ints). 
    Returns DataFrame with columns [ShelterName, TargetBTU] 
    """ 
    # find hour columns 
    hour_cols = [c for c in tidy_wide.columns if str(c).strip().isdigit()] 
    if not hour_cols: 
        # also accept columns like '2400 - 0100' or '0000 - 0100' -> extract leftmost hour 
        for c in tidy_wide.columns: 
            if isinstance(c, str) and c.strip()[:2].isdigit(): 
                # treat as hour column 
                hour_cols.append(c) 
    if not hour_cols: 
        raise ValueError("Could not detect hour columns in HVAC dataframe.") 

    # coerce numeric values 
    tmp = tidy_wide.copy() 
    for c in hour_cols: 
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce") 

    mask = tmp["RowName"].astype(str).str.strip() == rowname 
    df_rows = tmp.loc[mask, [shelter_col] + hour_cols].copy() 
    if df_rows.empty: 
        # try contains 
        mask = tmp["RowName"].astype(str).str.contains("Shelter HVAC Heat Load", 
                                                        case=False, 
                                                        na=False) 
        df_rows = tmp.loc[mask, [shelter_col] + hour_cols].copy() 

    # melt and compute peak per shelter 
    long = df_rows.melt(id_vars=[shelter_col], 
                        value_vars=hour_cols, 
                        var_name="Hour", 
                        value_name="BTU") 
    long = long.dropna(subset=["BTU"]) 
    target = long.groupby(shelter_col, as_index=False)["BTU"].max().rename(columns={"BTU": "TargetBTU"}) 
    # ensure numeric target 
    target["TargetBTU"] = target["TargetBTU"].astype(float) 
    return target


# --------------------- 
# MILP solver (normalized objective, normalized excess in objective) 
# Assumes scipy.optimize.milp is available 
# --------------------- 
# @st.cache_data  # Holds result in cache for recall  # TODO: Uncomment
def optimize_ecu_mix_normalized( 
    targets_df: pd.DataFrame, 
    catalog_df: pd.DataFrame, 
    *, 
    weights=None, 
    btu_penalty: float = 1.0, 
    shelter_col: str = "ShelterName", 
    target_col: str = "TargetBTU", 
): 
    """ 
    Returns DataFrame with one row per shelter: 
    Shelter, Location (if present), TargetBTU, AchievedBTU, ExcessBTU, 
    TotalKW, TotalCost, TotalWeight, TotalSize, ObjectiveValue, ECU_Mix (dict) 
    """ 
    if weights is None: 
        weights = {"cost": 1.0, "power": 1.0, "weight": 1.0, "size": 1.0} 

    shelters = (targets_df[[shelter_col, target_col]] 
                .dropna() 
                .groupby(shelter_col, as_index=False)[target_col].max()) 
    shelter_names = shelters[shelter_col].astype(str).tolist() 
    S = len(shelter_names) 

    cat = catalog_df[["Model", "CapacityBTU", "PowerKW", "CostUSD", 
                      "Weight", "Size", "Window Mount"]].copy() 
    cat["Model"] = cat["Model"].astype(str) 
    models = cat["Model"].tolist() 
    M = len(models) 

    cap = cat["CapacityBTU"].to_numpy(float) 
    kw = cat["PowerKW"].to_numpy(float) 
    cost = cat["CostUSD"].to_numpy(float) 
    wt = cat["Weight"].to_numpy(float) 
    sz = cat["Size"].to_numpy(float) 

    # normalize by max (only used in objective) 
    cost_scale = cost.max() if cost.max() > 0 else 1.0 
    kw_scale = kw.max() if kw.max() > 0 else 1.0 
    wt_scale = wt.max() if wt.max() > 0 else 1.0 
    sz_scale = sz.max() if sz.max() > 0 else 1.0 

    norm_cost = cost / cost_scale 
    norm_kw = kw / kw_scale 
    norm_wt = wt / wt_scale 
    norm_sz = sz / sz_scale 

    w_cost = float(weights.get("cost", 1.0)) 
    w_power = float(weights.get("power", 1.0)) 
    w_weight = float(weights.get("weight", 1.0)) 
    w_size = float(weights.get("size", 1.0)) 

    def idx_x(s_idx, m_idx): return s_idx * M + m_idx 
    def idx_e(s_idx): return S * M + s_idx 

    n_vars = S * M + S 

    # Objective coefficients 
    c = np.zeros(n_vars, dtype=float) 
    for s_idx in range(S): 
        for m_idx in range(M): 
            j = idx_x(s_idx, m_idx) 
            c[j] = (w_cost * norm_cost[m_idx] + 
                    w_power * norm_kw[m_idx] + 
                    w_weight * norm_wt[m_idx] + 
                    w_size * norm_sz[m_idx]) 
        # normalize excess by target in objective (so solver minimizes fraction overshoot) 
        target = float(shelters.iloc[s_idx][target_col]) 
        norm_penalty = (btu_penalty / target) if target > 0 else btu_penalty 
        c[idx_e(s_idx)] = norm_penalty 

    # Build constraints 
    rows = [] 
    ub = [] 
    # (1) meet demand: -sum cap*x <= -target 
    for s_idx, s_name in enumerate(shelter_names): 
        target = float(shelters.loc[shelters[shelter_col] == s_name, target_col].iloc[0]) 
        row = np.zeros(n_vars) 
        for m_idx in range(M): 
            row[idx_x(s_idx, m_idx)] = -cap[m_idx] 
        rows.append(row) 
        ub.append(-target) 

    # (2) define excess: sum cap*x - e <= target 
    for s_idx, s_name in enumerate(shelter_names): 
        target = float(shelters.loc[shelters[shelter_col] == s_name, target_col].iloc[0]) 
        row = np.zeros(n_vars) 
        for m_idx in range(M): 
            row[idx_x(s_idx, m_idx)] = cap[m_idx] 
        row[idx_e(s_idx)] = -1.0 
        rows.append(row) 
        ub.append(target) 

    # (3) window unit compatibility 
    for s_idx, s_name in enumerate(shelter_names): 
        compatible = bool(targets_df.loc[targets_df[shelter_col] == s_name, 
                                         "Window Unit Compatibility"].iloc[0]) 
        if not compatible: 
            for m_idx, m in enumerate(models): 
                if bool(cat.loc[cat["Model"] == m, "Window Mount"].iloc[0]): 
                    row = np.zeros(n_vars) 
                    row[idx_x(s_idx, m_idx)] = 1.0 
                    rows.append(row) 
                    ub.append(0.0) 

    # Now build LinearConstraint 
    A = np.vstack(rows) 
    lc = LinearConstraint(A, 
                          lb=-np.inf * np.ones(A.shape[0]), 
                          ub=np.array(ub, dtype=float)) 
    bounds = Bounds(lb=np.zeros(n_vars), ub=np.full(n_vars, np.inf)) 

    integrality = np.zeros(n_vars, dtype=int) 
    integrality[:S * M] = 1 # x integer 

    res = milp(c=c, integrality=integrality, bounds=bounds, constraints=[lc]) 
    if res.status != 0: 
        raise RuntimeError(f"MILP failed: {res.message}") 

    x = res.x.copy() 
    # round integer parts and ensure nonnegative 
    x[:S * M] = np.maximum(0, np.rint(x[:S * M]).astype(int)) 
    x[S * M:] = np.maximum(0.0, x[S * M:]) 

    # Collect results 
    out_rows = [] 
    for s_idx, s_name in enumerate(shelter_names): 
        target = float(shelters.iloc[s_idx][target_col]) 
        mix = {} 
        total_btu = total_kw = total_cost = total_wt = total_sz = 0.0 
        for m_idx, m in enumerate(models): 
            qty = int(x[idx_x(s_idx, m_idx)]) 
            if qty > 0: 
                mix[m] = qty 
                total_btu += qty * cap[m_idx] 
                total_kw += qty * kw[m_idx] 
                total_cost += qty * cost[m_idx] 
                total_wt += qty * wt[m_idx] 
                total_sz += qty * sz[m_idx] 
        excess = max(0.0, total_btu - target) 
        # compute normalized objective (reported) 
        obj_val = (w_cost * (total_cost / cost_scale) + 
                   w_power * (total_kw / kw_scale) + 
                   w_weight * (total_wt / wt_scale) + 
                   w_size * (total_sz / sz_scale) + 
                   btu_penalty * (excess / target if target > 0 else 0.0)) 

        out_rows.append({ 
            "Shelter": s_name, 
            "TargetBTU": target, 
            "AchievedBTU": total_btu, 
            "ExcessBTU": excess, 
            "TotalKW": total_kw, 
            "TotalCost": total_cost, 
            "TotalWeight": total_wt, 
            "TotalSize": total_sz, 
            "Cost_Norm": w_cost * (total_cost / cost_scale), 
            "Power_Norm": w_power * (total_kw / kw_scale), 
            "Weight_Norm": w_weight * (total_wt / wt_scale), 
            "Size_Norm": w_size * (total_sz / sz_scale), 
            "Penalty_Norm": btu_penalty * (excess / target if target > 0 else 0.0), 
            "ObjectiveValue": obj_val, 
            "ECU_Mix": mix 
        }) 

    return pd.DataFrame(out_rows)


# --------------------- 
# Fuel consumption helper 
# --------------------- 
# --- Function to estimate fuel consumption at any load --- 
def fuel_at_load(gen_row, load_kw, data_complete=True): 
    max_kw = gen_row["Max Power (kW)"] 

    # Generator cannot handle the load 
    if load_kw > max_kw: 
        return np.nan 

    # Load percentage 
    load_pct = load_kw / max_kw * 100 

    # Fuel at discrete loads 
    levels = [25, 50, 75, 100] 
    if data_complete: 
        fuels = [gen_row[f"{lvl}% Load"] for lvl in levels] 
    else: 
        fuels = [gen_row[f"100% Load"] for lvl in levels] 

    # Linear interpolation 
    return np.interp(load_pct, levels, fuels) 


def calc_fuel(shelters_df, gens_df): 
    power_factor = 0.8 # Power factor for USMC generators 

    # --- Build result DataFrame --- 
    all_results = {} 

    for _, shelter in shelters_df.iterrows(): 
        shelter_name = shelter["Shelter"] 
        real_power = shelter["TotalKW"] 
        apparent_power = real_power / power_factor # kW generator must supply after power factor applied 

        fuel_col = [] 
        runtime_col = [] 

        for _, gen in gens_df.iterrows(): 
            fuel_per_hr = fuel_at_load(gen, apparent_power, gen["Full Data"]) 
            if np.isnan(fuel_per_hr): 
                fuel_col.append(np.nan) 
                runtime_col.append(np.nan) 
            else: 
                fuel_col.append(fuel_per_hr) 
                runtime_col.append(gen["Fuel Capacity (gal)"] / fuel_per_hr) 

        # Add two columns per shelter 
        all_results[(shelter_name, "Fuel Consumption (gal/hr)")] = fuel_col 
        all_results[(shelter_name, "Runtime (hr)")] = runtime_col 

    # Create MultiIndex DataFrame 
    result_df = pd.DataFrame(all_results, index=gens_df["Generator Name"]) 
    result_df.columns = pd.MultiIndex.from_tuples(result_df.columns) 

    return result_df 


# --------------------- 
# Plotting helpers (Seaborn + Matplotlib), return fig for Streamlit 
# --------------------- 
def plot_temperature_with_hvac(melted, shelter_name, title_suffix=None): 

    fig, ax1 = plt.subplots(figsize=(12, 5)) 

    hvac_load_df = (melted[(melted["RowName"] == "Shelter HVAC Heat Load") & 
                           (melted["ShelterName"] == shelter_name)]
                    .drop_duplicates(subset=("ShelterName", "Hour"))) 
    hvac_load_df["Hour"] = hvac_load_df["Hour"] - 1 

    source_df = (melted[melted["RowName"].isin(["Structure", "Ventilation", 
                                                "Personnel", "Electrical"])]
                 .drop_duplicates(subset=['Hour', 'RowName', 'ShelterName'])) 
    source_df = source_df[source_df['ShelterName'] == shelter_name] 

    sns.set_theme(style="whitegrid", palette="muted") 

    # Multiple bar plot 
    bottoms = pd.Series([0] * 24) 
    ax = sns.barplot( 
        data=source_df, 
        x="Hour", 
        y="Value", 
        hue="RowName", 
        palette="muted" 
    ) 

    # Overlay HVAC Load line 
    sns.lineplot(data=hvac_load_df, x="Hour", y="Value", 
                 color="black", marker="o", linewidth=2, 
                 label="Total HVAC Heat Load", ax=ax) 

    # Format axes 
    plt.ylabel("Head Load (BTU/hr)") 
    plt.xlabel("Time of Day") 
    plt.title(f"Shelter Heat Load Breakdown\nfor {shelter_name}") 
    plt.xticks(hvac_load_df["Hour"], [f"{h:02d}00" for h in hvac_load_df["Hour"]]) 

    for label in ax.get_xticklabels(): # Rotate ticks 90 degrees 
        label.set_rotation(45) 

    plt.legend() 
    plt.tight_layout() 

    return fig 


def plot_target_vs_achieved(solution_df, melted): 
    solution = solution_df.copy() 
    if "TotalBTU" not in solution.columns: 
        solution["TotalBTU"] = solution["AchievedBTU"] 

    fig, ax = plt.subplots(figsize=(12, 5)) 

    # Compute max HVAC load per shelter from palm_hvac_df 
    max_loads = ( 
        melted[melted["RowName"] == "Shelter HVAC Heat Load"] 
        .groupby("ShelterName")["Value"] 
        .max() 
        .reset_index(name="Max_Observed_Load") 
    ) 

    # Merge solution and max loads 
    solution = solution.rename(columns={'Shelter': 'ShelterName'}) 
    comparison_df = max_loads.merge(solution, on="ShelterName") 

    # Reshape for seaborn 
    comparison_melted = comparison_df.melt( 
        id_vars="ShelterName", 
        value_vars=["Max_Observed_Load", "AchievedBTU"], 
        var_name="Source", 
        value_name="Max_Load" 
    ) 
    name_mapping = { 
        "Max_Observed_Load": "Observed Head Load (BTU)", 
        "AchievedBTU": "Achieved Heat Load (BTU)" 
    } 
    comparison_melted['Source'] = comparison_melted['Source'].map(name_mapping) 

    # Plot with seaborn 
    ax = sns.barplot( 
        data=comparison_melted, 
        x="ShelterName", y="Max_Load", hue="Source", 
        palette="muted", dodge=True 
    ) 

    # Plot labeling/settings 
    plt.ylabel("Heat Load (BTU/hr)") 
    plt.xlabel("Shelter Name") 
    plt.title(f"Observed vs. Achieved Heat Load per Shelter") 
    plt.xticks(rotation=45, ha="right") 
    ax.legend(loc="lower left", title_fontsize=9, bbox_to_anchor=(1.01,0.0)) 
    ax.set_ylim(0, (comparison_melted["Max_Load"].max() + 5000)) 

    # Add bar labels, suppress output 
    _ = [ax.bar_label(container, fontsize=10) for container in ax.containers] 

    fig.tight_layout() 
    return fig


def plot_solution_metrics(solution_df): 
    df = solution_df.copy() 
    fig, ax = plt.subplots(figsize=(12, 5)) 
    metrics = ["Cost_Norm", "Power_Norm", "Weight_Norm", "Size_Norm", "Penalty_Norm"] 
    melted = df.melt(id_vars=["Shelter"], 
                     value_vars=metrics, 
                     var_name="Metric", 
                     value_name="Value") 
    sns.barplot(data=melted, 
                x="Shelter", 
                y="Value", 
                hue="Metric", 
                ax=ax, 
                palette="muted") 
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right") 
    plt.title(f"Value of Normalized Parameters in Objective Function Per Shelter") 

    handles, _ = ax.get_legend_handles_labels() 
    custom_labels = ["Cost", "Power", "Weight", "Size", "BTU Penalty"] 

    # Shrink and move legend 
    ax.legend( 
        title="Parameter", 
        title_fontsize=9, 
        bbox_to_anchor=(1.01,0.0), 
        loc="lower left", 
        handles=handles, 
        labels=custom_labels 
    ) 

    # Add bar labels rounded to 2 decimals 
    for container in ax.containers: 
        ax.bar_label( 
            container, 
            labels=[f"{v.get_height():.2f}" for v in container], 
            fontsize=10 
        ) 

    fig.tight_layout() 
    return fig 


def plot_ecu_mix(solution_df): 
    # Expand mix dict 
    rows = [] 
    for _, r in solution_df.iterrows(): 
        s = r["Shelter"] 
        mix = r["ECU_Mix"] if isinstance(r["ECU_Mix"], dict) else {} 
        for model, qty in mix.items(): 
            rows.append({"Shelter": s, "Model": model, "Qty": qty}) 

    if not rows: 
        fig, ax = plt.subplots() 
        ax.text(0.5, 0.5, "No ECUs selected", ha="center") 
        return fig 

    mix_df = pd.DataFrame(rows) 
    fig, ax = plt.subplots(figsize=(12, 5)) 
    sns.barplot(data=mix_df, 
                x="Shelter", 
                y="Qty", 
                hue="Model", 
                ax=ax, 
                palette="muted") 

    plt.title(f"Number of ECUs by Type Per Shelter") 
    plt.ylabel("Quantity") 
    ax.legend(title="ECU Name", 
              loc="lower left", 
              title_fontsize=9, 
              bbox_to_anchor=(1.01,0.0)) 
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right") 
    # Set the y-axis major locator to MaxNLocator with integer=True 
    ax.yaxis.set_major_locator(MaxNLocator(integer=True)) 

    # Add bar labels, suppress output 
    _ = [ax.bar_label(container, fontsize=10) for container in ax.containers] 

    fig.tight_layout() 
    return fig 

# @st.fragment  # Allows function to run without rerunning entire script  # TODO: Uncomment
def plot_fuel(result_df, generator_name): 
    # --- Select generator row --- 
    gen_row = result_df.loc[[generator_name]] 

    # --- Prepare data --- 
    # Extract unique shelters from the MultiIndex tuples 
    unique_shelters = sorted(set([shel for shel, _ in gen_row.columns]), 
                             key=lambda x: list(gen_row.columns.get_level_values(0)).index(x)) 

    fuel_values = [gen_row[(shel, "Fuel Consumption (gal/hr)")].values[0] 
                   for shel in unique_shelters] 
    runtime_values = [gen_row[(shel, "Runtime (hr)")].values[0] 
                      for shel in unique_shelters] 

    x = range(len(unique_shelters)) 
    width = 0.35 

    # --- Colors --- 
    fuel_color = "#597dbf" # blue 
    runtime_color = "#d98b5f" # orange 

    # --- Create figure --- 
    fig, ax1 = plt.subplots(figsize=(12,5)) 

    # Left axis: Fuel 
    bars1 = ax1.bar([i - width/2 for i in x], 
                    fuel_values, 
                    width=width, 
                    color=fuel_color, 
                    label="Fuel (gal/hr)") 
    ax1.set_ylabel("Fuel (gal/hr)") 
    ax1.set_xlabel("Shelter") 
    ax1.set_xticks(x) 
    ax1.set_xticklabels(unique_shelters, rotation=45) 

    # Label bars 
    for bar in bars1: 
        height = bar.get_height() 
        ax1.text(bar.get_x() + bar.get_width()/2, 
                 height + 0.01*max(fuel_values), 
                 f"{height:.2f}", 
                 ha='center', 
                 va='bottom', 
                 fontsize=9) 

    # Right axis: Runtime 
    ax2 = ax1.twinx() 
    bars2 = ax2.bar([i + width/2 for i in x], 
                    runtime_values, 
                    width=width, 
                    color=runtime_color, 
                    label="Runtime (hr)") 
    ax2.set_ylabel("Runtime (hours)") 

    # Label bars 
    for bar in bars2: 
        height = bar.get_height() 
        ax2.text(bar.get_x() + bar.get_width()/2, 
                 height + 0.01*max(runtime_values), 
                 f"{height:.2f}", 
                 ha='center', 
                 va='bottom', 
                 fontsize=9) 

    # Remove right axis grid 
    ax2.grid(False) 

    # Combine legends 
    legend_handles = [ 
        mpatches.Patch(color=fuel_color, label="Fuel Consumption (gal/hr)"), 
        mpatches.Patch(color=runtime_color, label="Runtime (Single Tank of Gas)") 
    ] 
    ax1.legend(handles=legend_handles, 
               loc="lower left", 
               bbox_to_anchor=(1.04,0.0), 
               ncol=1) 

    ax1.set_title(f"Fuel and Runtime per Shelter for {generator_name}") 
    fig.tight_layout() 
    return fig


# ---------------------
# Initialize session_state
# ---------------------
# Initialize files and vars in session_state
if "hvac_file" not in st.session_state:
    st.session_state["hvac_file"] = None
if "catalog_file" not in st.session_state:
    st.session_state["catalog_file"] = None
if "example_scenario" not in st.session_state:
    st.session_state.example_scenario = False
if "optimized" not in st.session_state:
    st.session_state.optimized = False
if "files_uploaded" not in st.session_state:
    st.session_state.files_uploaded = False
if "user_weights" not in st.session_state:
    st.session_state["user_weights"] = [1] * 5

# ---------------------
# Other UI Helper Functions
# ---------------------
def reset_results():
    st.session_state.optimized = False
    st.session_state.example_scenario = False

def load_random_weights():
    st.session_state["user_weights"] = [random.randint(0, 5) for _ in range(5)]
    st.session_state.optimized = False

def weight_change():
    st.session_state.optimized = False


# ---------------------
# Streamlit UI
# ---------------------
st.set_page_config(layout="wide", page_title="ECU Selection Optimizer")
st.title("ECU Selection Optimizer")

st.markdown("""
Upload your AutoDISE output file and ECU catalog. Then set your weights to the left and press **Optimize**.
""")

hvac_file = st.file_uploader(
    "**Upload :green[AutoDISE Output]:**",
    type=["csv"],
    key="hvac",
    on_change=reset_results
)
catalog_file = st.file_uploader(
    "**Upload :blue[ECU Specifications File]:**",
    type=["csv"],
    key="catalog",
    on_change=reset_results
)

# Download example buttons
ecu_example_file = Path("Inputs/ECUSpecs.csv")
hvac_example_file = Path("Inputs/HVAC24HourProfile_Palms_Custom.csv")

# Read files into memory
with open(ecu_example_file, "rb") as f:
    ecu_bytes = f.read()

with open(hvac_example_file, "rb") as f:
    hvac_bytes = f.read()

if hvac_file is None or catalog_file is None:
    col3, col4, col7 = st.columns(3)
    with col4:
        st.download_button(
            label="Download Example ECU Specifications",
            data=ecu_bytes,
            file_name="ECU_Specs_Example.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col3:
        st.download_button(
            label="Download Example AutoDISE Output",
            data=hvac_bytes,
            file_name="HVAC_Profile_Example.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col7:
        if st.button(
            "Run Example Scenario",
            type="primary",
            help="This will automatically load the example files found on the left into the tool.",
            use_container_width=True
        ):
            load_random_weights()
            st.session_state.example_scenario = True
            st.session_state.optimized = False
else:
    st.session_state.example_scenario = False

# Put files in session state
if st.session_state.example_scenario:
    st.session_state["hvac_file"] = hvac_bytes
    st.session_state["catalog_file"] = ecu_bytes
    st.session_state.files_uploaded = True
elif hvac_file is not None and catalog_file is not None:
    st.session_state["hvac_file"] = hvac_file.getvalue()
    st.session_state["catalog_file"] = catalog_file.getvalue()
    st.session_state.files_uploaded = True
else:
    st.session_state.files_uploaded = False

# Sliders
st.sidebar.header("Set Weights")
st.sidebar.markdown("Click 'Optimize' after changing weights.")
w_cost      = st.sidebar.slider("Cost", 0, 5, st.session_state["user_weights"][0], step=1, 
                                on_change=weight_change)
w_power     = st.sidebar.slider("Power", 0, 5, st.session_state["user_weights"][1], step=1, 
                                on_change=weight_change)
w_weight    = st.sidebar.slider("Weight", 0, 5, st.session_state["user_weights"][2], step=1, 
                                on_change=weight_change)
w_size      = st.sidebar.slider("Size", 0, 5, st.session_state["user_weights"][3], step=1, 
                                on_change=weight_change)
btu_penalty = st.sidebar.slider("BTU Penalty", 0, 5, st.session_state["user_weights"][4], step=1, 
                                on_change=weight_change)

# Button to randomize weights
st.sidebar.button("Randomize Weights", 
                  help="Randomizes all the weights above.", 
                  on_click=load_random_weights,
                  use_container_width=True)


# ---------------------
# Read in files
# ---------------------
if (st.session_state["hvac_file"] is not None and
    st.session_state["catalog_file"] is not None and
    st.session_state.files_uploaded):

    # Read uploaded csvs
    try:
        hvac_buf = io.BytesIO(st.session_state["hvac_file"])
        tidy_hvac = read_multiple_tables(hvac_buf)
    except Exception as e:
        st.error(f"Failed to parse AutoDISE File: {e}")
        st.stop()

    # Coerce numeric hour columns
    hvac_df_numeric = pd.read_csv(io.BytesIO(st.session_state["hvac_file"]))
    for c in tidy_hvac.columns:
        if str(c).strip().isdigit():
            tidy_hvac[c] = pd.to_numeric(tidy_hvac[c], errors="coerce")

    # Extract targets
    try:
        targets = extract_targets_from_hvac_df(tidy_hvac)
    except Exception as e:
        st.error(f"Could not extract targets from HVAC file: {e}")
        st.stop()

    # Add true/false column for window unit compatibility
    targets["Window Unit Compatibility"] = True

    # Read catalog
    catalog_buf = io.BytesIO(st.session_state["catalog_file"])
    catalog = pd.read_csv(catalog_buf)

    # Normalize catalog column names (best-effort)
    expected_cols = ["Model", "CapacityBTU", "PowerKW", "CostUSD", "Weight", "Size"]
    lower = {c.lower(): c for c in catalog.columns}

    def pick(*cands):
        for cc in cands:
            if cc in lower:
                return lower[cc]
        return None

    mappings = {
        "Model": pick("model", "name", "sku", "unit (ecu)"),
        "CapacityBTU": pick("capacity_btu", "capacity (btu/hr)", "capacity", "cooling capacity (btu/hr)"),
        "PowerKW": pick("power_kw", "kw", "power (kw)", "power", "cooling load (kva)"),
        "CostUSD": pick("cost_usd", "cost", "price_usd", "price", "cost"),
        "Weight": pick("weight", "mass", "weight (lbs)"),
        "Size": pick("size", "volume", "size (ft3)")
    }

    missing = [k for k, v in mappings.items() if v is None]
    if missing:
        st.error(f"Catalog is missing columns or unrecognized names. Detected columns: {list(catalog.columns)}")
        st.stop()

    catalog = catalog.rename(columns={v: k for k, v in mappings.items()})

    # Coerce numeric
    for c in ["CapacityBTU", "PowerKW", "CostUSD", "Weight", "Size"]:
        catalog[c] = pd.to_numeric(catalog[c], errors="coerce")

    # ---------------------
    # Display success or not
    # ---------------------
    if st.session_state.example_scenario:
        st.success("Example scenario loaded.")
    else:
        st.success("Files uploaded.")
    st.markdown("---")  # horizontal rule

    # Make columns for display
    col1, col2 = st.columns(2)

    # Display targets
    with col1:
        st.markdown(
            "### Target BTU and Window Compatibility "
            '<span style="color:gray;" title="These are the maximum BTU loads for each shelter. '
            'These values are extracted from the file you uploaded from AutoDISE.">ⓘ</span>',
            unsafe_allow_html=True
        )
        st.markdown(":red[Action:] Select whether each shelter is compatible with window ECU units.")
        targets = st.data_editor(targets, 
                                 hide_index=True, 
                                 disabled=["TargetBTU"],
                                 column_config={
                                     "ShelterName": st.column_config.TextColumn(
                                         "Shelter Name",
                                         help="Name of the shelter.",
                                         max_chars=50,
                                         required=True,
                                         #  pinned=True,  # TODO: uncomment
                                         validate="^[A-Za-z0-9_.\-() ]+$"  # Blocks any unsafe characters
                                     ),
                                     "TargetBTU": st.column_config.NumberColumn(
                                         "Target BTU",
                                         help="The maximum number of BTUs produced in the shelter. Calculated by AutoDISE.",
                                         required=True,
                                     ),
                                     "Window Unit Combatibility": st.column_config.CheckboxColumn(
                                         help="Select the checkmark if the shelter can accept window-mounted ECUs.",
                                         default=True
                                     )
                                 })

    # Display ECU catalog
    with col2:
        st.subheader("ECU Catalog")
        st.markdown("All ECUs loaded from ECU Specifications file uploaded above.")
        st.dataframe(catalog.set_index('Model'))

    # Load and show generator specs
    st.markdown(
        "### Generator Catalog "
        '<span style="color:gray;" title="These are all the generators loaded into the tool.">ⓘ</span>',
        unsafe_allow_html=True
    )
    st.markdown(
        """
        These generators are loaded from a background file. The 'XX% Load' columns indicate the number of gallons
        per hour each generator burns at the percentage of electrical load.
        
        :blue[Optional:] You can add or subtract generators from the list using the buttons below. \
        Scroll over each column name for help on what the column is.
        """
    )
    try:
        gen_spec_df = pd.read_csv(Path("Inputs/GeneratorSpecs.csv"))
    except Exception as e:
        st.error(
            f"Could not load generator specifications from file. \
            Please ensure 'GeneratorSpecs.csv' is placed in the Inputs folder on GitHub \
            or contact the developer. \n\n{e}"
        )
        st.stop()
    gen_spec_df = st.data_editor(gen_spec_df.set_index('Generator Name'),
                                 num_rows="dynamic",
                                 column_config={
                                     "Generator Name": st.column_config.TextColumn(
                                         help="Name of the generator",
                                         max_chars=50,
                                         required=True,
                                         #  pinned=True,  # TODO: uncomment
                                         validate="^[A-Za-z0-9_.\-() ]+$"  # Blocks any unsafe characters
                                     ),
                                     "Model": st.column_config.TextColumn(
                                         help="Model number",
                                         max_chars=50,
                                         validate="^[A-Za-z0-9_.\-() ]+$"  # Blocks any unsafe characters
                                     ),
                                     "TAMCN": st.column_config.TextColumn(
                                         help="TAMCN",
                                         max_chars=50,
                                         validate="^[A-Za-z0-9_.\-() ]+$"  # Blocks any unsafe characters
                                     ),
                                     "Max Power (kW)": st.column_config.NumberColumn(
                                         help="Maximum power the generator can output in kilowatts.",
                                         required=True,
                                         min_value=0,
                                         max_value=99999,
                                     ),
                                     "Fuel Capacity (gal)": st.column_config.NumberColumn(
                                         help="Capacity of a single fuel tank in gallons.",
                                         required=True,
                                         min_value=0,
                                         max_value=99999,
                                     ),
                                     "25% Load": st.column_config.NumberColumn(
                                         help="Fuel consumption at 25 percent load in gal/hour.",
                                         min_value=0,
                                         max_value=99999,
                                     ),
                                     "50% Load": st.column_config.NumberColumn(
                                         help="Fuel consumption at 50 percent load in gal/hour.",
                                         min_value=0,
                                         max_value=99999,
                                     ),
                                     "75% Load": st.column_config.NumberColumn(
                                         help="Fuel consumption at 75 percent load in gal/hour.",
                                         min_value=0,
                                         max_value=99999,
                                     ),
                                     "100% Load": st.column_config.NumberColumn(
                                         help="Fuel consumption at 100 percent load in gal/hour.",
                                         required=True,
                                         min_value=0,
                                         max_value=99999,
                                     ),
                                     "Full Data": None
                                 }).reset_index()
    # Assign full data column based on user inputs
    data_columns = ["Max Power (kW)", "Fuel Capacity (gal)", "25% Load", "50% Load", "75% Load", "100% Load"]
    gen_spec_df["Full Data"] = gen_spec_df[data_columns].notna().all(axis=1)

    # ---------------------
    # Init Solve Button
    # ---------------------
    solve_button = st.button("Optimize", type="primary")

    # When solve button pressed
    if solve_button:
        with st.spinner("Solving MILP..."):
            weights = {"cost": w_cost, "power": w_power, "weight": w_weight, "size": w_size}
            try:
                sol_df = optimize_ecu_mix_normalized(targets, catalog, weights=weights, btu_penalty=btu_penalty)
            except Exception as e:
                st.error(f"Optimization failed: {e}")
                st.stop()

        # Save results in session_state
        st.session_state["sol_df"] = sol_df
        st.session_state["fuel_result_df"] = calc_fuel(sol_df, gen_spec_df)

        # Set optimized bool
        st.session_state.optimized = True

    # ---------------------
    # Display results
    # ---------------------
    if st.session_state.optimized:
        sol_df = st.session_state["sol_df"]
        fuel_result_df = st.session_state["fuel_result_df"]

        st.success("Solution success.")
        st.markdown("---")
        st.markdown("### Solution Overview")
        st.markdown("The ECU_Mix column shows the type and number of ECUs that are optimal based on the user-input weights.")
        st.dataframe(
            sol_df.drop(["Cost_Norm", "Power_Norm", "Weight_Norm", "Size_Norm", "Penalty_Norm"], axis=1)
            .set_index('Shelter')
        )

        # Fuel consumption
        st.markdown(
            "### Fuel Consumption Metrics "
            '<span style="color:gray;" title="These values are all calculated using only the maximum electrical load ' \
            'from the ECUs. '
            'No additional electrical loads are plugged into the generators.">ⓘ</span>',
            unsafe_allow_html=True
        )
        st.markdown("Only generators that were capable of handling the electrical load of the ECUs are shown.")
        fuel_result_df = calc_fuel(sol_df, gen_spec_df)

        # Show result, drop generators with all na
        display_df = fuel_result_df.dropna(how='all').copy()
        display_df.columns = [f"{shelter} - {metric}" for shelter, metric in display_df.columns]
        st.dataframe(display_df.round(2), use_container_width=True)

        # ---------------------
        # Plotting area
        # ---------------------
        st.markdown("---")
        st.subheader("Plots")

        # Generator selection
        selected_gen = st.selectbox(
            "**Select Generator To Plot (this only affects the first plot below):**",
            options=display_df.index.tolist()
        )

        col5, col6 = st.columns(2)

        # Fuel plot
        if selected_gen is not None:
            fig_fuel = plot_fuel(fuel_result_df, selected_gen)
            with col5:
                st.pyplot(fig_fuel)
        fuel_result_df = fuel_result_df.dropna(how='all')  # Drop generators that can't handle the load

        # Melt HVAC data for plotting
        melted = tidy_hvac.melt(
            id_vars=["ShelterName", "ECUConfig", "SubTableName", "RowName"],
            var_name="Time", value_name="Value"
        ).drop_duplicates(subset=['ShelterName', 'ECUConfig', 'SubTableName', 'RowName', 'Time'])

        # Clean up Time axis
        melted["Hour"] = melted["Time"].str.extract(r"(\d{2})(?=00)")
        melted["Hour"] = pd.to_numeric(melted["Hour"], errors="coerce")

        # Clean up value axis
        melted["Value"] = pd.to_numeric(melted["Value"], errors="coerce")

        # Target vs Achieved
        fig_tva = plot_target_vs_achieved(sol_df, melted)
        with col5:
            st.pyplot(fig_tva)

        # Solution metrics
        fig_metrics = plot_solution_metrics(sol_df)
        with col6:
            st.pyplot(fig_metrics)

        # ECU mix
        fig_mix = plot_ecu_mix(sol_df)
        with col6:
            st.pyplot(fig_mix)

        # ---------------------
        # Download files
        # ---------------------
        date_str = datetime.now().strftime("%Y%m%d")
        zip_buffer = io.BytesIO()

        # User weights dataframe
        slider_df = pd.DataFrame({
            "Parameter": ["Cost", "Power", "Weight", "Size", "BTU Penalty"],
            "Value": [w_cost, w_power, w_weight, w_size, btu_penalty]
        })

        # ---------------------
        # Save all results into a zip folder
        # ---------------------
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
            # Save dataframes as XLSX with multiple sheets
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                sol_df.to_excel(writer, index=False, sheet_name="Solution")
                fuel_result_df.to_excel(writer, index=True, sheet_name="Fuel Consumption")
                slider_df.to_excel(writer, index=False, sheet_name="User Weights")
            zip_file.writestr(f"ECU_Opt_Output_{date_str}.xlsx", excel_buffer.getvalue())

            # Save plots as PNG
            # Fuel plots for each generator
            for gen in list(fuel_result_df.index.values):
                fig_fuel = plot_fuel(fuel_result_df, gen)
                buffer_fuel = io.BytesIO()
                fig_fuel.savefig(buffer_fuel, format="png", bbox_inches='tight')
                buffer_fuel.seek(0)
                zip_file.writestr(f"Fuel_Plot_{gen}_{date_str}.png", buffer_fuel.read())

            # Target vs Achieved plot
            buffer_tva = io.BytesIO()
            fig_tva.savefig(buffer_tva, format="png", bbox_inches='tight')
            buffer_tva.seek(0)
            zip_file.writestr(f"Target_Achieved_Plot_{date_str}.png", buffer_tva.read())

            # Metrics plot
            buffer_met = io.BytesIO()
            fig_metrics.savefig(buffer_met, format="png", bbox_inches='tight')
            buffer_met.seek(0)
            zip_file.writestr(f"Metrics_Plot_{date_str}.png", buffer_met.read())

            # ECU Mix plot
            buffer_mix = io.BytesIO()
            fig_mix.savefig(buffer_mix, format="png", bbox_inches='tight')
            buffer_mix.seek(0)
            zip_file.writestr(f"ECU_Mix_Plot_{date_str}.png", buffer_mix.read())

        # Download button
        zip_buffer.seek(0)
        st.download_button(
            label="📁 Download Tables and Plots (zip)",
            data=zip_buffer,
            file_name=f"ECU_Solution_{date_str}.zip",
            mime="application/zip",
            type="primary"
        )
else:
    st.info("Upload your AutoDISE file and ECU catalog to continue.")

st.markdown("---")

# ---------------------
# User guide download
# ---------------------
_, col8, _ = st.columns(3)
guide_path = Path("Documentation/ECU_Tool_Guide.pdf")
with open(guide_path, "rb") as f:
    guide_bytes = f.read()
with col8:
    st.download_button(
        label="Click here to download the user guide (PDF)",
        # icon="📘",  # TODO: Uncomment
        data=guide_bytes,
        file_name="ECU_App_User_Guide.pdf",
        mime="application/octet-stream",
        use_container_width=True
    )

# ---------------------
# Help section
# ---------------------
with st.expander(":question: Help"):
    st.markdown(
        """
### How To Use This Tool

1. :blue[**Upload Your Data**]
- Upload the *ECU Specifications file*.
- Upload the *HVAC Analysis file* (must be exported from AutoDISE).
- :green[Both files should be .csv files before uploading.]
- :red[If you are new to the tool, you can always click the "Run Example Scenario" button to automatically run the 
analysis with the example files.]

2. :blue[**Set Optimization Weights**]
- Use the sliders in the sidebar to adjust the weight (0–5) of:
  - **Cost**: The total procurement cost of the ECUs.
  - **Power**: The total power usage of the ECUs when running.
  - **Weight**: The total weight of the ECUs.
  - **Size**: The total size (ft³) of the ECUs.
  - **BTU Penalty**: Penalty for exceeding shelter heat load requirements.
- These weights are used to score the prospective mixes of ECUs. The solution becomes the mixture that scores the best 
for each shelter.
- :green[A higher number means it is more important to minimize.]

3. :blue[**Optimize**]
- Click the **Optimize** button to run the optimization.
- The solver will determine the best mix of ECUs to optimally satisfy the shelter heat loads.

4. :blue[**Review Results**]
- A per-shelter summary is displayed.
- The optimal ECU mix for each shelter is shown in the far right column.

5. :blue[**Visualize**]
- Explore the automatically generated plots:
  - Fuel consumption by generator
  - Shelter heat loads (observed vs. achieved)
  - Objective contributions (cost, power, etc.)
  - ECU allocations per shelter (number and type of ECU)

6. :blue[**Download**]
- Export all results as .xlsx files and .png images using the download button at the bottom of the page.
- All the generated tables will be in separate tabs within a single Excel file.

---
💡 **Tip:** If you don’t have your own data, you can use the example files available for download above or run the 
example scenario.
"""
    )

# ---------------------
# Open formulation file
# ---------------------
formulation_file = Path("Documentation/MILP Formulation.pdf")
with open(formulation_file, "rb") as f:
    formulation_bytes = f.read()

# Information expander
with st.expander(":information_source: Information"):
    st.markdown(
        """
1. :blue[**Goal**]
- The goal of this tool is to help Capability Integration Officers figure out what the requirements for the future 
Environmental Control Units (ECUs) for the Marine Corps are.
- This includes how many BTUs they need to produce and how many should be assigned to each echelon in the USMC.

2. :blue[**Problem Formulation**]
"""
    )
    st.download_button(
        "Click here to download the problem formulation sheet",
        data=formulation_bytes,
        file_name="ECU_Optimization_Formulation.pdf",
        mime="application/pdf",
        type="primary",
        help="PDF file"
    )

    st.link_button(
        label="Click here to learn more about Mixed Integer Programming",
        url="https://www.nvidia.com/en-us/glossary/mixed-integer-programming/",
        help="Learn more about MIP online"
    )

    st.markdown(
        """
3. :blue[**Assumptions**]
- BTU Algorithm:
    - The heat load due to electrical equipment in a shelter is simply the total power load of all consumers in that 
    shelter, in Watts, converted to BTU/hour.
    - Other assumptions and equations detailed in AutoDISE User Manual (available on the app).
- Equipment:
    - Equipment with a max operating temperature/humidity less than the environmental values require climate control.
    - Power factor = 0.8 for generators, ECUs = 1.0.
    - Equipment is always turned on day/night.
    - The fuel consumption rate is calculated via linear interpolation using the electrical load and known fuel burn 
    rates. Linear assumption is applied between each known load percentage.

4. :blue[**Limitations**]
- AutoDISE doesn't account for setting up your shelters in the shade.
- There is an option in AutoDISE to put cammie netting over your shelter; this can mimic shaded environments.
- Information on actual gear used by different COC echelons varies based on SOP.
- Acquisition, maintenance, and lifecycle costs of ECUs are not currently accounted for in the cost function due to data 
availability.
- You may add any associated costs into the total in the "cost" column of the uploaded ECU specifications file; it will 
then be accounted for.
- This tool only optimizes for the maximum BTU load the shelter experiences in the AutoDISE simulation and does not 
account for the full 24-hour profile due to problem complexity.
"""
    )

# ---------------------
# About developer section
# ---------------------
with st.expander("🪖 About The Developer"):
    st.markdown(
        """
This app was developed by **Captain Deryk Clary**, Operations Research Analyst at the 
**USMC Expeditionary Energy Office (E2O)**.

🌐 Website: [USMC E2O](https://www.cdi.marines.mil/Units/CDD/Marine-Corps-Expeditionary-Energy-Office/)  
📧 Contact: *deryk.l.clary.mil(at)usmc.mil*
"""
    )
