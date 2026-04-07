import xml.etree.ElementTree as ET
from xml.dom import minidom


def generate_indloop_file():
    # List of lane IDs
    lane_ids = [
        'A0A1_0', 'A0A1_1', 'A0A1_2', 'A0B0_0', 'A0B0_1', 'A0B0_2', 'A1A0_0', 'A1A0_1',
        'A1A0_2', 'A1A2_0', 'A1A2_1', 'A1A2_2', 'A1B1_0', 'A1B1_1', 'A1B1_2', 'A2A1_0',
        'A2A1_1', 'A2A1_2', 'A2A3_0', 'A2A3_1', 'A2A3_2', 'A2B2_0', 'A2B2_1', 'A2B2_2',
        'A3A2_0', 'A3A2_1', 'A3A2_2', 'A3A4_0', 'A3A4_1', 'A3A4_2', 'A3B3_0', 'A3B3_1',
        'A3B3_2', 'A4A3_0', 'A4A3_1', 'A4A3_2', 'A4B4_0', 'A4B4_1', 'A4B4_2', 'B0A0_0',
        'B0A0_1', 'B0A0_2', 'B0B1_0', 'B0B1_1', 'B0B1_2', 'B0C0_0', 'B0C0_1', 'B0C0_2',
        'B1A1_0', 'B1A1_1', 'B1A1_2', 'B1B0_0', 'B1B0_1', 'B1B0_2', 'B1B2_0', 'B1B2_1',
        'B1B2_2', 'B1C1_0', 'B1C1_1', 'B1C1_2', 'B2A2_0', 'B2A2_1', 'B2A2_2', 'B2B1_0',
        'B2B1_1', 'B2B1_2', 'B2B3_0', 'B2B3_1', 'B2B3_2', 'B2C2_0', 'B2C2_1', 'B2C2_2',
        'B3A3_0', 'B3A3_1', 'B3A3_2', 'B3B2_0', 'B3B2_1', 'B3B2_2', 'B3B4_0', 'B3B4_1',
        'B3B4_2', 'B3C3_0', 'B3C3_1', 'B3C3_2', 'B4A4_0', 'B4A4_1', 'B4A4_2', 'B4B3_0',
        'B4B3_1', 'B4B3_2', 'B4C4_0', 'B4C4_1', 'B4C4_2', 'C0B0_0', 'C0B0_1', 'C0B0_2',
        'C0C1_0', 'C0C1_1', 'C0C1_2', 'C0D0_0', 'C0D0_1', 'C0D0_2', 'C1B1_0', 'C1B1_1',
        'C1B1_2', 'C1C0_0', 'C1C0_1', 'C1C0_2', 'C1C2_0', 'C1C2_1', 'C1C2_2', 'C1D1_0',
        'C1D1_1', 'C1D1_2', 'C2B2_0', 'C2B2_1', 'C2B2_2', 'C2C1_0', 'C2C1_1', 'C2C1_2',
        'C2C3_0', 'C2C3_1', 'C2C3_2', 'C2D2_0', 'C2D2_1', 'C2D2_2', 'C3B3_0', 'C3B3_1',
        'C3B3_2', 'C3C2_0', 'C3C2_1', 'C3C2_2', 'C3C4_0', 'C3C4_1', 'C3C4_2', 'C3D3_0',
        'C3D3_1', 'C3D3_2', 'C4B4_0', 'C4B4_1', 'C4B4_2', 'C4C3_0', 'C4C3_1', 'C4C3_2',
        'C4D4_0', 'C4D4_1', 'C4D4_2', 'D0C0_0', 'D0C0_1', 'D0C0_2', 'D0D1_0', 'D0D1_1',
        'D0D1_2', 'D0E0_0', 'D0E0_1', 'D0E0_2', 'D1C1_0', 'D1C1_1', 'D1C1_2', 'D1D0_0',
        'D1D0_1', 'D1D0_2', 'D1D2_0', 'D1D2_1', 'D1D2_2', 'D1E1_0', 'D1E1_1', 'D1E1_2',
        'D2C2_0', 'D2C2_1', 'D2C2_2', 'D2D1_0', 'D2D1_1', 'D2D1_2', 'D2D3_0', 'D2D3_1',
        'D2D3_2', 'D2E2_0', 'D2E2_1', 'D2E2_2', 'D3C3_0', 'D3C3_1', 'D3C3_2', 'D3D2_0',
        'D3D2_1', 'D3D2_2', 'D3D4_0', 'D3D4_1', 'D3D4_2', 'D3E3_0', 'D3E3_1', 'D3E3_2',
        'D4C4_0', 'D4C4_1', 'D4C4_2', 'D4D3_0', 'D4D3_1', 'D4D3_2', 'D4E4_0', 'D4E4_1',
        'D4E4_2', 'E0D0_0', 'E0D0_1', 'E0D0_2', 'E0E1_0', 'E0E1_1', 'E0E1_2', 'E1D1_0',
        'E1D1_1', 'E1D1_2', 'E1E0_0', 'E1E0_1', 'E1E0_2', 'E1E2_0', 'E1E2_1', 'E1E2_2',
        'E2D2_0', 'E2D2_1', 'E2D2_2', 'E2E1_0', 'E2E1_1', 'E2E1_2', 'E2E3_0', 'E2E3_1',
        'E2E3_2', 'E3D3_0', 'E3D3_1', 'E3D3_2', 'E3E2_0', 'E3E2_1', 'E3E2_2', 'E3E4_0',
        'E3E4_1', 'E3E4_2', 'E4D4_0', 'E4D4_1', 'E4D4_2', 'E4E3_0', 'E4E3_1', 'E4E3_2'
    ]
    def prettify(elem):
        """Return a pretty-printed XML string for the Element."""
        rough_string = ET.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

    # Create the root element
    root = ET.Element("additional")

    # Add inductionLoop elements
    for i, lane_id in enumerate(lane_ids):
        induction_loop = ET.SubElement(
            root,
            "inductionLoop",
            id=f"detector{lane_id}",
            lane=lane_id,
            pos="400",
            file="outputs/detectorsOut.xml",
            period="20"
        )

    # Pretty-print the XML
    xml_str = prettify(root)

    # Write to file
    with open("../indloop.add.xml", "w") as file:
        file.write(xml_str)

generate_indloop_file()