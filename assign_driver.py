"""
Assign drivers to student pickup locations (Excel input) using MIP.

Expect an Excel file with 3 sheets:
 - "students": columns ["student_name", "location"]
 - "drivers": columns ["driver_name", "capacity"]
 - "distance": n x n matrix: rows/cols are location names, values are travel times

Output: printed summary and "ride_assignment_output.xlsx" with:
 - driver_assignments: which locations & how many students each driver picks
 - uber_assignments: locations and number of students assigned to Uber
"""

import pandas as pd
import pulp
import math

# ---------- USER: change filename if needed ----------
INPUT_XLSX = "./Rides.xlsx"
OUTPUT_XLSX = "ride_assignment_output.xlsx"

SHEET_STUDENT='students'
SHEET_DRIVER='drivers'
SHEET_DISTANCE='campus_distance'

LOCATION_COL='location'
CAPACITY_COL='capacity'

VERBOSE_NUM = 0 # for silence,  1 for stdout
# ----------------------------------------------------
# Priority setting:
# Objective: lexicographic via big weight
# weights

BIG_UBER   = 10**8   # very large, avoid Uber first
BIG_DRIVER = 10**5   # next priority, avoid using too many drivers
SMALL_TRIP = 10**4   # finally minimize travel distance if the locations are nearby
BIG_DISTANCE = 10**6 # set big if a location is far and prefer to be picked by additional drivers


print(">>>> tEsT 2")
xls = pd.ExcelFile(INPUT_XLSX)
if SHEET_STUDENT not in xls.sheet_names or SHEET_DRIVER not in xls.sheet_names or SHEET_DISTANCE not in xls.sheet_names:
    raise ValueError(f"Excel must contain sheets named: {SHEET_STUDENT}, {SHEET_DRIVER} and {SHEET_DISTANCE}")

students_df = pd.read_excel(xls, sheet_name=SHEET_STUDENT)
drivers_df = pd.read_excel(xls, sheet_name=SHEET_DRIVER)
dist_df = pd.read_excel(xls, sheet_name=SHEET_DISTANCE, index_col=0)

# Basic checks
if LOCATION_COL not in students_df.columns:
    raise ValueError("students sheet must have a 'location' column")
if CAPACITY_COL not in drivers_df.columns and "num_students" not in drivers_df.columns:
    raise ValueError("drivers sheet must have 'capacity' (or 'num_students') column")

# Accept either 'capacity' or 'num_students'
if CAPACITY_COL in drivers_df.columns:
    drivers_df = drivers_df.rename(columns={CAPACITY_COL:"capacity"})
elif "num_students" in drivers_df.columns:
    drivers_df = drivers_df.rename(columns={"num_students":"capacity"})

# aggregate demand by location
demand_series = students_df.groupby(LOCATION_COL).size()
demand = demand_series.to_dict()  # location -> number of students
locations = list(demand.keys())

# Ensure distance matrix contains all locations (allow extra rows/cols but require these)
# --- CHANGED: allow missing locations ---
missing_rows = set(locations) - set(dist_df.index)
missing_cols = set(locations) - set(dist_df.columns)
isolated_locations = sorted(list(missing_rows | missing_cols))
if isolated_locations:
    print(f"Warning: these locations are not in distance sheet and will be treated as ISOLATED (no pairing allowed): {isolated_locations}")


# Create symmetric travel time dict for the relevant locations
# --- CHANGED: build travel dict only where both i,j exist in sheet ---
travel = {}
for i in locations:
    travel[i] = {}
    for j in locations:
        val = BIG_DISTANCE
        if i in dist_df.index and j in dist_df.columns:
            raw = dist_df.loc[i, j]
            if not pd.isna(raw):
                val = float(raw)

        # if missing, try the reverse direction
        if val is BIG_DISTANCE and j in dist_df.index and i in dist_df.columns:
            raw = dist_df.loc[j, i]
            if not pd.isna(raw):
                val = float(raw)

        # if still missing, leave as BIG_DISTANCE
        travel[i][j] = val


# drivers list and capacities
drivers = drivers_df.iloc[:,0].astype(str).tolist()  # driver names from first column
cap_series = drivers_df["capacity"].astype(int).tolist()
driver_caps = {drivers[i]: cap_series[i] for i in range(len(drivers))}

# Build optimization model (PuLP)
model = pulp.LpProblem("Ride_Assignment", pulp.LpMinimize)

# Decision variables
# x[j,i] = 1 if driver j stops at location i (binary)
x = pulp.LpVariable.dicts("x", (drivers, locations), cat="Binary")

# y[j,i] = integer number of students driver j picks up at location i
y = pulp.LpVariable.dicts("y", (drivers, locations), lowBound=0, cat="Integer")

# u[j] = 1 if driver j is used (binary)
u = pulp.LpVariable.dicts("u", drivers, cat="Binary")

# w[j,i,k] for i<k = 1 if driver j visits both i and k (to capture travel time)
# --- CHANGED: only allow pairs with valid distance ---
pairs = []
for a_idx in range(len(locations)):
    for b_idx in range(a_idx+1, len(locations)):
        i = locations[a_idx]
        k = locations[b_idx]
        if travel[i][k] is not None:
            pairs.append((i,k))

w = pulp.LpVariable.dicts("w", (drivers, range(len(pairs))), cat="Binary")
# map index back to pair
pair_index_to_pair = {idx: pairs[idx] for idx in range(len(pairs))}

# Constraints

# 1) If driver visits a location (x=1), y can be >0; but y limited by capacity
for j in drivers:
    cap_j = driver_caps[j]
    for i in locations:
        # y[j,i] <= cap_j * x[j,i]
        model += y[j][i] <= cap_j * x[j][i], f"cap_link_{j}_{i}"

# 2) driver total picks <= capacity * u_j
for j in drivers:
    cap_j = driver_caps[j]
    model += pulp.lpSum([y[j][i] for i in locations]) <= cap_j * u[j], f"driver_cap_{j}"

# 3) u_j must be 1 if any x[j,i] is 1
for j in drivers:
    model += pulp.lpSum([x[j][i] for i in locations]) <= 1000 * u[j], f"use_def_{j}"
    # also if u=1 it's allowed to have x=0; but we minimize u so it won't be 1 unnecessarily

# 4) each driver visits at most 2 locations
for j in drivers:
    model += pulp.lpSum([x[j][i] for i in locations]) <= 2, f"max2stops_{j}"

# 5) pair variable linking: w <= x[i], w <= x[k], and w >= x[i] + x[k] - 1
for j in drivers:
    for idx in range(len(pairs)):
        i,k = pair_index_to_pair[idx]
        model += w[j][idx] <= x[j][i], f"w_le_x1_{j}_{i}_{k}"
        model += w[j][idx] <= x[j][k], f"w_le_x2_{j}_{i}_{k}"
        model += w[j][idx] >= x[j][i] + x[j][k] - 1, f"w_ge_summinus1_{j}_{i}_{k}"

# 6) demand satisfied at most by total assigned to drivers (leftover -> Uber)
# --- CHANGED: introduce Uber assignment variables ---
uber_y = pulp.LpVariable.dicts("uber", locations, lowBound=0, cat="Integer")

for i in locations:
    # Demand must be covered by either drivers OR Uber
    model += pulp.lpSum([y[j][i] for j in drivers]) + uber_y[i] == demand[i], f"demand_cover_{i}"


# 7) y are integers (already declared int by PuLP)

# # Objective: lexicographic via big weight
# # weights
# BIG_UBER   = 10**8   # very large, avoid Uber first
# BIG_DRIVER = 10**6   # next priority, avoid using too many drivers
# SMALL_TRIP = 10**5       # finally minimize travel distance

# total_drivers_used
total_drivers_used = pulp.lpSum([u[j] for j in drivers])
# total_travel_time = sum_j sum_pairs travel[i,k]*w[j,idx]
total_travel_time = pulp.lpSum([ travel[pair_index_to_pair[idx][0]][pair_index_to_pair[idx][1]] * w[j][idx]
                                 for j in drivers for idx in range(len(pairs)) ])

# --- CHANGED: add a tiny penalty for Uber use ---
uber_penalty = pulp.lpSum([uber_y[i] for i in locations])
print('adjusted uber penalty')
model += BIG_UBER * uber_penalty + BIG_DRIVER * total_drivers_used + SMALL_TRIP * total_travel_time, "prefer_drivers_over_uber"


# Solve
solver = pulp.PULP_CBC_CMD(msg=VERBOSE_NUM, timeLimit=300)
res = model.solve(solver)

if pulp.LpStatus[model.status] not in ["Optimal","Not Solved","Integer Feasible","Optimal"]:
    print("Solver status:", pulp.LpStatus[model.status])

# Collect results
driver_assignments = []
for j in drivers:
    used = int(pulp.value(u[j]))
    stops = []
    total_picked = 0
    for i in locations:
        xi = int(round(pulp.value(x[j][i]) or 0))
        yi = int(round(pulp.value(y[j][i]) or 0))
        if xi:
            stops.append(i)
        if yi:
            total_picked += yi
        if yi:
            driver_assignments.append({
                "driver": j,
                "location": i,
                "students_picked": yi,
                "stop_flag": xi
            })
    # compute travel time for driver j
    travel_time_j = 0
    # find which pair w is 1
    wpairs = []
    for idx in range(len(pairs)):
        if int(round(pulp.value(w[j][idx]) or 0)) == 1:
            wpairs.append(pair_index_to_pair[idx])
            travel_time_j += travel[pair_index_to_pair[idx][0]][pair_index_to_pair[idx][1]]
    # if only one stop, travel_time_j stays 0
    # append summary row
    # (we may not want to duplicate per-location rows; so also produce summary below)
# Summary per driver
driver_summary = []
for j in drivers:
    used = int(round(pulp.value(u[j]) or 0))
    assigned_students = int(round(sum((pulp.value(y[j][i]) or 0) for i in locations)))
    stops_list = [i for i in locations if int(round(pulp.value(x[j][i]) or 0))==1]
    travel_time_j = 0
    for idx in range(len(pairs)):
        if int(round(pulp.value(w[j][idx]) or 0)) == 1:
            i,k = pair_index_to_pair[idx]
            travel_time_j += travel[i][k]
    driver_summary.append({
        "driver": j,
        "used": used,
        "stops": ", ".join(stops_list) if stops_list else "",
        "students_picked_total": assigned_students,
        "estimated_travel_time": travel_time_j
    })

# compute Uber leftover
# --- CHANGED: Uber assignments come from uber_y vars ---
uber = {}
for i in locations:
    left = int(round(pulp.value(uber_y[i]) or 0))
    if left > 0:
        uber[i] = left


# Print results
print("\n=== DRIVER SUMMARY ===")
for row in driver_summary:
    print(f"{row['driver']}: used={row['used']}, stops=[{row['stops']}], students_picked={row['students_picked_total']}, est_travel_time={row['estimated_travel_time']}")

print("\n=== DETAILED ASSIGNMENTS (driver, location, students_picked) ===")
for row in driver_assignments:
    if row["students_picked"]>0:
        print(f"{row['driver']}\t{row['location']}\t{row['students_picked']}")

print("\n=== UBER ASSIGNMENTS (leftover students by location) ===")
if uber:
    for loc, n in uber.items():
        print(f"{loc}: {n}")
else:
    print("No Uber needed — all students assigned to drivers.")

# Save to Excel
with pd.ExcelWriter(OUTPUT_XLSX) as writer:
    pd.DataFrame(driver_summary).to_excel(writer, sheet_name="driver_summary", index=False)
    # Only include rows with students picked or stops
    df_assign = pd.DataFrame(driver_assignments)
    if df_assign.empty:
        df_assign = pd.DataFrame(columns=["driver","location","students_picked","stop_flag"])
    df_assign.to_excel(writer, sheet_name="detailed_assignments", index=False)
    uber_df = pd.DataFrame([(k,v) for k,v in uber.items()], columns=["location","students_to_uber"])
    uber_df.to_excel(writer, sheet_name="uber_assignments", index=False)
    # also write original inputs so file is self-contained
    students_df.to_excel(writer, sheet_name="input_students", index=False)
    drivers_df.to_excel(writer, sheet_name="input_drivers", index=False)
    dist_df.to_excel(writer, sheet_name="input_distance")

print(f"\nOutput saved to {OUTPUT_XLSX}")
