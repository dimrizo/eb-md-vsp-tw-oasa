import re

# --- Raw Input Data ---
RAW_DATA = """
Trip ID: 1   | Type: short | Starts at RP #4   -> Ends at RP #3   | Time:  940.24 to  969.95 | ETA: 22.64
Trip ID: 2   | Type: long  | Starts at RP #4   -> Ends at RP #2   | Time:  458.57 to  740.67 | ETA: 18.21
Trip ID: 3   | Type: long  | Starts at RP #4   -> Ends at RP #1   | Time:  386.17 to  627.66 | ETA: 25.88
Trip ID: 4   | Type: long  | Starts at RP #4   -> Ends at RP #3   | Time: 1180.38 to 1393.01 | ETA: 22.64
Trip ID: 5   | Type: short | Starts at RP #1   -> Ends at RP #4   | Time:  580.77 to  591.47 | ETA: 25.88
Trip ID: 6   | Type: short | Starts at RP #3   -> Ends at RP #1   | Time:  535.64 to  578.67 | ETA: 30.40
Trip ID: 7   | Type: long  | Starts at RP #1   -> Ends at RP #4   | Time: 1060.32 to 1295.21 | ETA: 25.88
Trip ID: 8   | Type: long  | Starts at RP #3   -> Ends at RP #2   | Time:  776.00 to 1040.14 | ETA: 35.12
Trip ID: 9   | Type: short | Starts at RP #1   -> Ends at RP #2   | Time:  455.48 to  477.28 | ETA: 16.61
Trip ID: 10  | Type: short | Starts at RP #1   -> Ends at RP #3   | Time:  640.38 to  685.16 | ETA: 30.40
Trip ID: 11  | Type: long  | Starts at RP #3   -> Ends at RP #4   | Time:  690.59 to  927.36 | ETA: 22.64
Trip ID: 12  | Type: long  | Starts at RP #1   -> Ends at RP #2   | Time:  414.44 to  604.79 | ETA: 16.61
"""

def parse_trip_data(raw_data):
    """
    Parses the raw trip data string into a list of dictionaries.
    """
    trips = []
    # Regex to capture all fields, handling variable spacing
    # (\d+): Trip ID
    # (short|long): Type
    # (\d+): Start RP
    # (\d+): End RP
    # (\d+\.\d+): Start Time
    # (\d+\.\d+): End Time
    # (\d+\.\d+): ETA (Duration)
    pattern = re.compile(
        r'Trip ID:\s*(\d+)\s*\|\s*Type:\s*(short|long)\s*\|\s*'
        r'Starts at RP #(\d+)\s*->\s*Ends at RP #(\d+)\s*\|\s*'
        r'Time:\s*(\d+\.\d+)\s*to\s*(\d+\.\d+)\s*\|\s*'
        r'ETA:\s*(\d+\.\d+)'
    )

    for line in raw_data.strip().split('\n'):
        match = pattern.search(line)
        if match:
            trips.append({
                'id': int(match.group(1)),
                'type': match.group(2).capitalize(),
                'start_rp': int(match.group(3)),
                'end_rp': int(match.group(4)),
                'start_time': float(match.group(5)),
                'end_time': float(match.group(6)),
                'duration': float(match.group(7)),
            })
    return trips

def generate_latex_table(trips):
    """
    Generates the LaTeX code for the table using the parsed trip data.
    """
    if not trips:
        return ""

    # \toprule, \midrule, \bottomrule require \usepackage{booktabs}
    latex_code = [
        r'\begin{table}[H]',
        r'    \centering',
        r'    \caption{Detailed Characteristics of Generated Test Trips}',
        r'    \label{tab:trip_details}',
        r'    \small',
        r'    \begin{tabular}{cc cc ccc}',
        r'        \toprule',
        r'        Trip ID & Type & Origin (RP) & Dest. (RP) & Start Time & End Time & $\eta_i$ \\',
        r'        \midrule'
    ]

    for trip in trips:
        # Format times and duration to two decimal places
        start_time_str = f'{trip["start_time"]:.2f}'
        end_time_str = f'{trip["end_time"]:.2f}'
        duration_str = f'{trip["duration"]:.2f}'

        row = (
            f'        {trip["id"]} & {trip["type"]} & {trip["start_rp"]} & {trip["end_rp"]} & '
            f'{start_time_str} & {end_time_str} & {duration_str} \\\\'
        )
        latex_code.append(row)

    latex_code.extend([
        r'        \bottomrule',
        r'    \end{tabular}',
        r'\end{table}'
    ])

    return '\n'.join(latex_code)

if __name__ == '__main__':
    parsed_trips = parse_trip_data(RAW_DATA)
    latex_output = generate_latex_table(parsed_trips)
    print("--- Start LaTeX Output ---")
    print(latex_output)
    print("--- End LaTeX Output ---")

    # The second file block below contains the generated LaTeX for convenience.