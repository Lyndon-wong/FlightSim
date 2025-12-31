
import sys
import pandas as pd
import numpy as np
import csv
import time
from pathlib import Path
from tqdm import tqdm

# Setup paths
# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flightsim.sixdof import SixDOFModel
from flightsim.autopilot import FlightPhase, create_autopilot, LongHaulAutopilot, StandardAutopilot
from flightsim.navigation import NavUtils

WAYPOINTS_FILE = PROJECT_ROOT / "data" / "waypoints.csv"

# Aircraft Performance Dict (Approximate Mach Ranges)
# (Min, Optimization, Max)
AIRCRAFT_MACH = {
    "A320-200": [0.76, 0.78, 0.80],
    "B737-800": [0.76, 0.78, 0.80],
    "A321-200": [0.76, 0.78, 0.80],
    "B757-200": [0.78, 0.80, 0.82],
    "ERJ-145":  [0.74, 0.78, 0.78], # Slower regional
    "CRJ-900":  [0.74, 0.78, 0.80],
    "E190":     [0.74, 0.78, 0.80],
    "A220-300": [0.76, 0.78, 0.80],
    "A330-300": [0.80, 0.82, 0.84],
    "B767-300ER": [0.78, 0.80, 0.82],
    "A350-900": [0.82, 0.85, 0.89],
    "B777-300ER": [0.82, 0.84, 0.87],
    "B787-9":   [0.83, 0.85, 0.89],
    "A380-800": [0.83, 0.85, 0.89],
}

def load_all_routes():
    """Load all routes from CSV"""
    routes = []
    with open(WAYPOINTS_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse aircraft list
            ac_list_str = row['recommended_aircraft']
            ac_list = [ac.strip() for ac in ac_list_str.split(',') if ac.strip()]
            
            waypoints = []
            # Origin
            waypoints.append((float(row['origin_lat']), float(row['origin_lon'])))
            # Intermediate
            for i in range(1, 11):
                lat = row.get(f'waypoint{i}_lat')
                lon = row.get(f'waypoint{i}_lon')
                if lat and lon and lat.strip():
                    waypoints.append((float(lat), float(lon)))
            # Dest
            waypoints.append((float(row['dest_lat']), float(row['dest_lon'])))
            
            routes.append({
                "origin": row['origin_code'],
                "dest": row['dest_code'],
                "name": row['route_name'],
                "category": row['category'],
                "distance_km": float(row['distance_km']),
                "aircraft": ac_list,
                "waypoints": waypoints
            })
    return routes

def simulate_route(route_info, ac_type, mach):
    """Run a simulation, skipping steady cruise."""
    waypoints = route_info['waypoints']
    origin = waypoints[0]
    dest = waypoints[-1]
    
    # Runway headings mapping (rough approx for key airports to help stability)
    # Runway headings mapping (rough approx for key airports to help stability)
    airport_hdgs = {
        "AMS": 270.0, "IST": 354.0, "MNL": 240.0, "KIX": 239.0, "MAD": 323.0,
        "BKK": 195.0, "YVR": 261.0, "ATL": 270.0, "ORD": 270.0, "MIA": 267.0,
        "FCO": 160.0, "DEN": 350.0, "VIE": 290.0, "LHR": 270.0, "CDG": 266.0,
        "JFK": 224.0, "LAX": 250.0, "SFO": 280.0, "SIN": 202.0, "SYD": 346.0,
        "PEK": 360.0, "PVG": 350.0, "HKG": 070.0, "DXB": 300.0, "NRT": 340.0,
        "HND": 340.0, "FRA": 250.0, "ICN": 330.0, "GRU": 090.0, "BOM": 270.0,
        "DEL": 280.0, "CPT": 190.0, "JNB": 030.0
    }
    rwy_hdg = airport_hdgs.get(route_info['dest'], 0.0)

    # Init Model
    cruise_alt = 11000.0 if "ERJ" not in ac_type else 10000.0
    
    model = SixDOFModel(ac_type, origin[0], origin[1], cruise_alt, 270.0, dt=1.0)
    # Fix: create_autopilot signature changed (no ac_type as first arg for factory, 
    # but based on my earlier fix, I removed ac_type from the call in main loop)
    # Actually create_autopilot takes (model, range_category, cruise_speed_mach)
    autopilot = create_autopilot(model, cruise_speed_mach=mach)
    autopilot.load_route(waypoints, runway_heading=rwy_hdg)

    # Calculate Total Distance
    total_dist = 0
    for i in range(len(waypoints)-1):
        total_dist += NavUtils.haversine_distance(waypoints[i][0], waypoints[i][1], waypoints[i+1][0], waypoints[i+1][1])

    # Smart Start
    is_long_haul = isinstance(autopilot, LongHaulAutopilot)
    start_dist_from_dest = 500000 if is_long_haul else 200000
    
    if total_dist < start_dist_from_dest + 50000:
        start_wp_idx = 0
        model.lat, model.lon = origin
    else:
        cum_dist = 0
        start_wp_idx = 0
        for i in range(len(waypoints)-1):
            leg_dist = NavUtils.haversine_distance(waypoints[i][0], waypoints[i][1], waypoints[i+1][0], waypoints[i+1][1])
            if total_dist - (cum_dist + leg_dist) < start_dist_from_dest:
                 start_wp_idx = i
                 model.lat, model.lon = waypoints[i]
                 break
            cum_dist += leg_dist
            
    # Init State
    _, _, a = model._get_atmosphere(model.alt)
    model.tas = mach * a
    next_wp = waypoints[start_wp_idx+1]
    model.heading = NavUtils.calculate_bearing(model.lat, model.lon, next_wp[0], next_wp[1])
    autopilot.current_wp_idx = start_wp_idx + 1
    autopilot.phase = FlightPhase.CRUISE

    # Limit max steps (timeout)
    max_steps = 8000 
    landed = False
    
    for i in range(max_steps):
        throttle, pitch, roll = autopilot.update()
        model.update(throttle, pitch, roll)
        
        if autopilot.phase in [FlightPhase.TOUCHDOWN, FlightPhase.ROLLOUT]:
            landed = True
            return True, f"Landed at T={i}"
            
        if model.alt < 0:
            return False, "Crashed (Alt < 0)"
            
    return False, "Timeout (Max Steps Reached)"

def main():
    routes = load_all_routes()
    # Prioritize Long Haul (Sort by distance descending)
    routes.sort(key=lambda x: x['distance_km'], reverse=True)
    print(f"Loaded {len(routes)} routes. Sorted by distance (descending).")
    
    # Init CSV
    output_file = "all_test_results.csv"
    with open(output_file, "w") as f:
        f.write("route,aircraft,mach,success,reason\n")
        
    failed_cases = []
    total_tests_run = 0
    
    # Pre-calculate total work for progress bar
    work_items = []
    for route in routes:
        for ac in route['aircraft']:
            machs = AIRCRAFT_MACH.get(ac, [0.78, 0.80, 0.82])
            test_machs = [machs[0], machs[-1]] 
            if machs[0] == machs[-1]: test_machs = [machs[0]]
            for mach in test_machs:
                work_items.append((route, ac, mach))
                
    print(f"Total simulations to run: {len(work_items)}")
    start_time = time.time()
    
    # Use tqdm for progress bar
    for route, ac, mach in tqdm(work_items, desc="Simulating Routes", unit="test"):
        total_tests_run += 1
        route_key = f"{route['origin']}-{route['dest']}"
        
        try:
            success, msg = simulate_route(route, ac, mach)
            status_str = "PASS" if success else "FAIL"
            
            # Write immediately
            with open(output_file, "a") as f:
                f.write(f"{route_key},{ac},{mach},{success},{msg}\n")

            if not success:
                # print(f"!!! [FAIL] {route_key} | {ac} | M{mach} -> {msg}") # Optional: Remove spam if progress bar used, but user might want to see failures. 
                # Let's keep failure prints but formatted nicely to not break tqdm too much.
                tqdm.write(f"!!! [FAIL] {route_key} | {ac} | M{mach} -> {msg}")
                failed_cases.append({
                    "route": route_key,
                    "aircraft": ac,
                    "mach": mach,
                    "reason": msg
                })
            else:
                pass
                
        except Exception as e:
            tqdm.write(f"!!! [ERROR] {route_key} | {ac} | M{mach} -> Exception: {e}")
            with open(output_file, "a") as f:
                f.write(f"{route_key},{ac},{mach},False,Exception: {e}\n")

    elapsed = time.time() - start_time
    print(f"\n=== Test Complete in {elapsed:.1f}s ===")
    print(f"Total Tests: {total_tests_run}")
    print(f"Failed Cases: {len(failed_cases)}")
    
    if failed_cases:
        print("\n--- Failures ---")
        for case in failed_cases:
            print(case)
        print(f"All results saved to {output_file}")
    else:
        print("All tests passed!")

if __name__ == "__main__":
    main()
