import pandas as pd
import xml.etree.ElementTree as ET
import sumolib

def junction_metrics(junction_ids, xml_file):
    for junction_id in junction_ids:
        # Extract incoming edges to the given junction
        node = network.getNode(junction_id)
        incoming_edges = [edge.getID() for edge in node.getIncoming()]


        # Parse the XML file to extract edge data
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Initialize lists to store data for each time interval
        densities = []
        waiting_times = []
        time_losses = []
        speeds = []
        travel_times = []

        # Iterate over intervals in the XML file
        for interval in root.findall('interval'):
            # Initialize metrics for this interval
            total_density = 0.0
            total_waiting_time = 0.0
            total_time_loss = 0.0
            total_speed = 0.0
            total_travel_time = 0.0
            edge_count = 0

            # Iterate over all edges in this interval
            for edge in interval.findall('edge'):
                edge_id = edge.attrib['id']
                if edge_id in incoming_edges:
                    density = float(edge.attrib.get('density', 0.0))
                    waiting_time = float(edge.attrib.get('waitingTime', 0.0))
                    time_loss = float(edge.attrib.get('timeLoss', 0.0))
                    speed = float(edge.attrib.get('speed', 0.0))
                    travel_time = float(edge.attrib.get('traveltime', 0.0))


                    total_density += density
                    total_waiting_time += waiting_time
                    total_time_loss += time_loss
                    total_speed += speed
                    total_travel_time += travel_time
                    edge_count += 1

            # Calculate averages
            if edge_count > 0:
                avg_density = total_density / edge_count
                avg_waiting_time = total_waiting_time / edge_count
                avg_time_loss = total_time_loss / edge_count
                avg_speed = total_speed / edge_count
                avg_travel_time = total_travel_time/ edge_count

                # Store data
                densities.append(avg_density)
                waiting_times.append(avg_waiting_time)
                time_losses.append(avg_time_loss)
                speeds.append(avg_speed)
                travel_times.append(avg_travel_time)
        # Create a DataFrame for the junction
        df = pd.DataFrame({
            'Density': densities,
            'Waiting Time': waiting_times,
            'Time Loss': time_losses,
            'Speed': speeds,
            'Travel Time': travel_times
        })
        df.to_csv(f'../junction_data/2.75_18000_20/{junction_id}.csv', index=False)



# Usage example
tree = ET.parse('../5x5.net.xml')
root = tree.getroot()

# Extract junctions
junction_ids = []
for junction in root.findall('junction'):
    if  junction.get('type') != 'internal':
        junction_id = junction.get('id')
        junction_ids.append(junction_id)

xml_file = '../outputs/2.75_18000_20/edgeData.xml'
network = sumolib.net.readNet("../5x5.net.xml")
junction_metrics(junction_ids, xml_file)
