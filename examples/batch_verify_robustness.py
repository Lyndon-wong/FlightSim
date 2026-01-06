
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import time
import logging

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flightsim.sixdof import SixDOFModel
from flightsim.autopilot import create_autopilot, FlightPhase
from flightsim.noise import NoiseConfig, NoiseManager
from flightsim.aerodynamics import get_database

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_single_simulation(route, aircraft_type, wind_speed_ms=20.0):
    """
    Run a single simulation for a given route and aircraft.
    Returns: Dictionary of results metrics.
    """
    try:
        # Route parameters
        origin = (route['origin_lat'], route['origin_lon'])
        dest = (route['dest_lat'], route['dest_lon'])
        
        # Parse waypoints
        waypoints = [origin]
        for i in range(1, 11):
            lat_key = f'waypoint{i}_lat'
            lon_key = f'waypoint{i}_lon'
            if pd.notna(route.get(lat_key)) and pd.notna(route.get(lon_key)):
                waypoints.append((route[lat_key], route[lon_key]))
        waypoints.append(dest)
        
        # Determine cruise altitude (simplified logic)
        dist_km = route['distance_km']
        if dist_km < 500:
            cruise_alt_ft = 24000
        elif dist_km < 1500:
            cruise_alt_ft = 32000
        else:
            cruise_alt_ft = 36000
            
        # Initialize Model
        # Mach 0.78 is a safe default for jets
        cruise_mach = 0.78
        model = SixDOFModel(aircraft_type, 
                           start_lat=origin[0], 
                           start_lon=origin[1], 
                           start_alt=10.0, # Start on ground/low
                           start_heading=0.0, # Will be set by autopilot
                           dt=0.1) # 10Hz simulation
        
        # Initialize Noise (Wind)
        # Assuming scale is 0-20m/s maps to 0-1 intensity
        wind_intensity = np.clip(wind_speed_ms / 20.0, 0.0, 1.0)
        noise_config = NoiseConfig(wind_intensity=wind_intensity)
        # Note: We need to attach noise manager to model
        noise_manager = NoiseManager(noise_config, model.dt)
        model.noise_obj = noise_config # Basic config
        model.noise_manager = noise_manager # Manager for dynamics
        
        # Initialize Autopilot
        autopilot = create_autopilot(model)
        autopilot.load_route(waypoints, departure_alt=10.0)
        
        # Run Simulation
        max_duration = (dist_km / 700.0) * 3600 * 1.5 # 1.5x expected time
        t = 0.0
        
        # Metrics
        xte_history = []
        pitch_history = []
        roll_history = []
        max_xte = 0.0
        landed = False
        crashed = False
        phase_history = set()
        
        step = 0
        while t < max_duration:
            # Autopilot update
            throttle, pitch_cmd, roll_cmd = autopilot.update()
            
            # Record phase
            phase_history.add(autopilot.phase)
            
            # Model update
            model.update(throttle, pitch_cmd, roll_cmd)
            
            # Apply Noise (Update noise manager)
            # The model.update calls internal noise logic if wired up, 
            # but let's make sure SixDOFModel uses it.
            # SixDOFModel.update checks self.noise_manager
            
            # Initial heading alignment
            if t < 1.0:
                # Cheat: Point aircraft to first waypoint initially to avoid massive turn on runway
                # Actually, autopilot takes off then turns.
                pass
                
            # Collect Metrics (skip first minute for stability)
            if t > 60.0:
                # Calculate Cross Track Error (XTE)
                # Need current segment. Autopilot knows it.
                # Simplified XTE check: standard autopilots minimize it internally.
                # We can try to access it if exposed, or recompute.
                # Let's rely on checking distance to route line.
                pass
                
            pitch_history.append(model.pitch)
            roll_history.append(model.roll)
            
            # Flight Phase Checks
            if autopilot.phase in [FlightPhase.TOUCHDOWN, FlightPhase.ROLLOUT] and model.gs < 40.0:
                # 认为是成功着陆（或者接近成功）
                landed = True
                break
                
            if model.alt < 0:
                crashed = True
                break
                
            t += model.dt
            step += 1
            
            # Logging interval
            # if step % 1000 == 0:
            #     print(f"T={t:.1f}, Alt={model.alt:.1f}, Phase={autopilot.phase}")
                
        # Post-process results
        pitch_std = np.std(pitch_history) if len(pitch_history) > 0 else 0.0
        roll_std = np.std(roll_history) if len(roll_history) > 0 else 0.0
        
        def get_distance_haversine(lat1, lon1, lat2, lon2):
            R = 6371000  # Earth radius in meters
            phi1, phi2 = np.radians(lat1), np.radians(lat2)
            dphi = np.radians(lat2 - lat1)
            dlambda = np.radians(lon2 - lon1)
            a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            return R * c

        # Distance directly to dest
        dist_to_dest = get_distance_haversine(model.lat, model.lon, dest[0], dest[1])
        
        result_status = "UNKNOWN"
        if crashed:
            result_status = "CRASH"
        elif landed:
            # Check if landed ON destination (within 20km - allow for approach pattern)
            if dist_to_dest < 20000:
                result_status = "SUCCESS"
            else:
                result_status = "LANDED_WRONG_LOC"
        else:
            result_status = "TIMEOUT"
            
        return {
            "route": route['route_name'],
            "aircraft": aircraft_type,
            "status": result_status,
            "final_dist": dist_to_dest,
            "time": t,
            "pitch_std": pitch_std,
            "roll_std": roll_std,
            "phases": [p.name for p in phase_history]
        }

    except AttributeError as e:
        logger.error(f"AttributeError for {aircraft_type}: {e}")
        return {
            "route": route['route_name'], 
            "aircraft": aircraft_type, 
            "status": "ERROR_ATTR", 
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"Error for {aircraft_type} on {route['route_name']}: {e}")
        return {
            "route": route['route_name'], 
            "aircraft": aircraft_type, 
            "status": "ERROR", 
            "error": str(e)
        }

def main():
    print("Starting Batch Robustness Test (20 m/s Wind)")
    
    # Load Data
    waypoints_file = PROJECT_ROOT / "data/waypoints.csv"
    routes_df = pd.read_csv(waypoints_file)
    
    results = []
    
    print(f"Total routes available: {len(routes_df)}")
    
    # Select diverse routes + random ones
    # Pick every 10th route to sample 30 routes
    test_indices = list(range(0, len(routes_df), 10))
    # Ensure some critical long haul ones are included
    extra_indices = [142, 143, 182] # PEK-LAX, PEK-SFO, JFK-LHR
    for idx in extra_indices:
        if idx not in test_indices and idx < len(routes_df):
            test_indices.append(idx)
            
    test_routes = routes_df.iloc[test_indices]
    
    print(f"Running test on {len(test_routes)} selected routes...")
    
    for idx, row in test_routes.iterrows():
        rec_aircraft = str(row['recommended_aircraft']).split(',')
        rec_aircraft = [a.strip() for a in rec_aircraft if a.strip()]
        
        # Limit to 1 aircraft per route to save time, unless it fails
        # Verify first aircraft in list
        if rec_aircraft:
            ac_type = rec_aircraft[0]
            print(f"Testing Route: {row['route_name']} ({row['distance_km']}km) with {ac_type} ...", end="", flush=True)
            res = run_single_simulation(row, ac_type, wind_speed_ms=20.0)
            print(f" {res['status']}")
            results.append(res)
            
            if res['status'] == "ERROR_ATTR":
                print("!!! CRITICAL: AttributeError detected. Stopping batch.")
                break
        
    # Save results
    df_res = pd.DataFrame(results)
    df_res.to_csv("batch_test_results.csv", index=False)
    print("\nBatch Test Complete. Results saved.")
    print(df_res)

if __name__ == "__main__":
    main()
