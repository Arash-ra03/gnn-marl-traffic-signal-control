import pandas as pd
import xml.etree.ElementTree as ET
import sumolib


def junction_flow_metrics(junction_ids, xml_file, networkRoot):
    # Parse the XML file to extract detector data
    tree = ET.parse(xml_file)
    root = tree.getroot()

    for junction_id in junction_ids:
        # Extract incoming edges to the given junction
        node = network.getNode(junction_id)
        incoming_edges = [edge.getID() for edge in node.getIncoming()]

        # Initialize list to store average flow data for each time step
        average_flows = []

        # Extract the total simulation time and time steps
        time_intervals = root.findall('.//interval')
        start_times = sorted(set(float(interval.get('begin')) for interval in time_intervals))

        # Iterate over each time step
        for start_time in start_times:
            total_flow = 0.0
            lane_count = 0

            # Iterate over each incoming edge
            for edge_id in incoming_edges:
                edge = networkRoot.find(f".//edge[@id='{edge_id}']")
                lanes = edge.findall('lane')
                lane_list = [lane.get('id') for lane in lanes]

                for lane in lane_list:
                    interval_xpath = f".//interval[@begin='{start_time}0'][@id='detector{lane}']"
                    interval = root.find(interval_xpath)
                    # for interval in intervals:
                    # if int(float(interval.get('begin'))) == int(start_time):
                    if interval != None:
                        flow = float(interval.get('flow', 1))
                        total_flow += flow
                        lane_count += 1
                    else: print('ERROR')

            # Calculate the average flow for this time step
            if lane_count > 0:
                avg_flow = total_flow / lane_count
                average_flows.append(avg_flow)
            else:
                average_flows.append(0)

        # Create a DataFrame for the junction with the calculated average flows for each time step
        df = pd.DataFrame({
            'Average Flow': average_flows
        })

        # Save the DataFrame to a CSV file
        df.to_csv(f'../junction_data/2.75_18000_20/{junction_id}_flow.csv', index=False)


# Usage example
tree = ET.parse('../5x5.net.xml')
root = tree.getroot()

# Extract junctions
junction_ids = []
for junction in root.findall('junction'):
    if junction.get('type') != 'internal':
        junction_id = junction.get('id')
        junction_ids.append(junction_id)

xml_file = '../outputs/2.75_18000_20/detectorsOut.xml'
network = sumolib.net.readNet("../5x5.net.xml")
junction_flow_metrics(junction_ids, xml_file, root)
