import pandas as pd
import xml.etree.ElementTree as ET


def parse_xml(edge_ids, xml_file, edge_dict):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    time_steps = []

    for data in root.findall('data'):
        time_steps.append(data.get('timestep'))
        lanes_tag = data.findall('lanes')[0]
        lanes_dict = {}
        available_edge = []
        for lane in lanes_tag.findall('lane'):
            lane_id = lane.get('id')
            available_edge.append(lane_id[0:4])
            lane_queue = lane.get('queueing_length')
            lanes_dict[lane_id] = lane_queue
        for edge in edge_ids:
            if edge in available_edge:
                lane_0 = edge + '_0'
                lane_1 = edge + '_1'
                lane_2 = edge + '_2'
                sum = 0
                if lane_0 in lanes_dict.keys():
                    num = float(lanes_dict[lane_0])
                    sum += num
                if lane_1 in lanes_dict.keys():
                    num = float(lanes_dict[lane_1])
                    sum += num
                if lane_2 in lanes_dict.keys():
                    num = float(lanes_dict[lane_2])
                    sum += num
                edge_dict[edge].append(str(sum))
            else:
                edge_dict[edge].append('0')

    df = pd.DataFrame({
        'Timestep': time_steps
    })
    for edge in edge_dict.keys():
        df[edge] = edge_dict[edge]
    df.to_csv(f'../outputs/queue_length.csv', index=False)



tree = ET.parse('../5x5.net.xml')
root = tree.getroot()
edge_dict = {}
edge_ids = []
for edge in root.findall('edge'):
    if edge.get('priority') == '-1':
        edge_ids.append(edge.get('id'))

for edge in edge_ids:
    edge_dict[edge] = []

xml_file = '../outputs/2.75_18000_20/queue.xml'
parse_xml(edge_ids, xml_file, edge_dict)


