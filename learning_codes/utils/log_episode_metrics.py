import xml.etree.ElementTree as ET
from lxml import etree


def log_episode_metrics(episode:int,runtime_log_msg:str):
    """
    Reads ../outputs/summary.xml and ../outputs/tripinfos.xml to compute:
        - The average meanTravelTime (ignoring entries where meanTravelTime == -1).
        - The average waitingTime across all <tripinfo> tags.
    Then appends these two metrics to a single simulation metrics file:
        logs/simulation_metrics.txt
    This log account for the WHOLE network as is not RegionController wise.
    This should be called right after plot_and_summarize_episode(...) each episode,
    before the next episode starts (because the files get overwritten).
    """
    def tolerant_parse_xml(filepath, expected_root_tag):
        """
        Reads an XML file at 'filepath' and attempts to tolerate an incomplete ending.
        The function:
        1. Reads the file line by line.
        2. Discards any trailing lines that don't have a closing '>' (which are likely incomplete).
        3. Joins the remaining lines to form the XML content.
        4. Appends the expected closing tag if it is missing.
        5. Uses lxml with recover=True to parse the (hopefully fixed) XML.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Remove trailing lines that don't contain a closing ">"
        while lines and ('>' not in lines[-1]):
            lines.pop()
        content = "".join(lines).strip()

        closing_tag = f"</{expected_root_tag}>"
        if not content.endswith(closing_tag):
            content += closing_tag

        parser = etree.XMLParser(recover=True)
        return etree.fromstring(content.encode('utf-8'), parser=parser)


    # 1) Parse ../outputs/summary.xml for meanTravelTime
    summary_path = "../outputs/summary.xml"
    mean_travel_time_vals = []
    try:
        root = tolerant_parse_xml(summary_path, "summary")
        # Each <step ... meanTravelTime="X" ... />
        for step in root.findall("step"):
            mtt_str = step.get("meanTravelTime", "-1")
            mtt_val = float(mtt_str)
            # Ignore -1 values
            if mtt_val >= 0:
                mean_travel_time_vals.append(mtt_val)
    except FileNotFoundError:
        print(f"Warning: Could not find {summary_path}, skipping meanTravelTime calculation.")
    except ET.ParseError:
        print(f"Warning: Could not parse {summary_path}, skipping meanTravelTime calculation.")

    if mean_travel_time_vals:
        avg_mtt = sum(mean_travel_time_vals) / len(mean_travel_time_vals)
    else:
        avg_mtt = None  # no valid data found

    # 2) Parse ../outputs/tripinfo.xml for waitingTime
    tripinfos_path = "../outputs/tripinfo.xml"
    waiting_times = []
    try:
        root = tolerant_parse_xml(tripinfos_path, "tripinfos")
        # Each <tripinfo ... waitingTime="X" ... />
        for tripinfo in root.findall("tripinfo"):
            wt_str = tripinfo.get("waitingTime", "0")
            wt_val = float(wt_str)
            waiting_times.append(wt_val)
    except FileNotFoundError:
        print(f"Warning: Could not find {tripinfos_path}, skipping waitingTime calculation.")
    except ET.ParseError:
        print(f"Warning: Could not parse {tripinfos_path}, skipping waitingTime calculation.")

    if waiting_times:
        avg_waiting_time = sum(waiting_times) / len(waiting_times)
    else:
        avg_waiting_time = None  # no data found

    # 3) Append these results to a single simulation metrics file.
    summary_file_path = f"logs/simulation_metrics.txt"
    mode = "w" if episode==1 else "a" 
    metrics_text = (
        f"Episode {episode}:\n"
        f"  Average meanTravelTime : {avg_mtt}\n"
        f"  Average waitingTime: {avg_waiting_time}\n"
        f"  Runtime: {runtime_log_msg}\n\n"
    )
    with open(summary_file_path, mode) as sf:
        sf.write(metrics_text)
