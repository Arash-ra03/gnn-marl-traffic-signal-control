import xml.etree.ElementTree as ET
import os

def log_baseline_simulation_metrics():
    """
    Reads network-level metrics from two fully generated XML files:
      - '../outputs/summary.xml': Computes the average 'meanTravelTime' from <step> elements,
         ignoring any value equal to -1.
      - '../outputs/tripinfos.xml': Computes the average 'waitingTime' across all <tripinfo> elements.
    
    Appends these calculated metrics to a log file named 'baseline_simulation_metrics.txt'
    (located in 'logs/plots'). Each episode’s metrics are appended on a new line.
    
    Example output line:
      Episode 3: Average meanTravelTime (ignoring -1): 7.83, Average waitingTime: 0.45
    """
    # Define file paths.
    summary_path = "actuated_sim/outputs/summary.xml"
    tripinfos_path = "actuated_sim/outputs/tripinfo.xml"
    output_file = os.path.join("learning_codes","logs", "actuated_baseline_simulation_metrics.txt")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print("Output file will be written to:", os.path.abspath(output_file))

    
    # --- Process summary.xml for meanTravelTime ---
    mean_travel_times = []
    try:
        tree = ET.parse(summary_path)
        root = tree.getroot()
        # For every <step> element, extract 'meanTravelTime'
        for step in root.findall("step"):
            mtt_str = step.get("meanTravelTime", "-1")
            try:
                mtt_val = float(mtt_str)
            except Exception:
                mtt_val = -1
            # Only use values that are not -1.
            if mtt_val != -1:
                mean_travel_times.append(mtt_val)
    except Exception as e:
        print(f"Error parsing {summary_path}: {e}")
    
    if mean_travel_times:
        avg_mtt = sum(mean_travel_times) / len(mean_travel_times)
    else:
        avg_mtt = None

    # --- Process tripinfos.xml for waitingTime ---
    waiting_times = []
    try:
        tree_trip = ET.parse(tripinfos_path)
        root_trip = tree_trip.getroot()
        # For every <tripinfo> element, extract 'waitingTime'
        for tripinfo in root_trip.findall("tripinfo"):
            wt_str = tripinfo.get("waitingTime", "0")
            try:
                wt_val = float(wt_str)
            except Exception:
                wt_val = 0
            waiting_times.append(wt_val)
    except Exception as e:
        print(f"Error parsing {tripinfos_path}: {e}")
    
    if waiting_times:
        avg_wait = sum(waiting_times) / len(waiting_times)
    else:
        avg_wait = None

    # --- Prepare output line and append to log file ---
    metrics_line = (
        f"Average meanTravelTime: {avg_mtt}, "
        f"Average waitingTime: {avg_wait}\n"
    )
    
    with open(output_file, "a") as f:
        f.write(metrics_line)


# At the end of each episode, after other logging calls:
log_baseline_simulation_metrics()