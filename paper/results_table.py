import re
import json

def parse_schedule_text(schedule_text):
    """
    Parses the raw text schedule, extracts vehicle paths and node details.
    Updates:
    1. Node Type is determined.
    2. Depot In (D1) Time/E_bar are set to '-' if Time is '10000.0'.
    3. Depot Out (O1) Time is set to '-' if Time is '0.00'.
    """
    schedules = {}
    # Regex to be non-greedy and stop before the next vehicle schedule line or EOF.
    # It now relies only on the '- Vehicle D1_V#' pattern.
    vehicle_pattern = re.compile(
        r'-\s*Vehicle\s*(D1_V\d+)\s*schedule:\s*(.*?)(?=\n- Vehicle|\Z)',
        re.DOTALL
    )

    # Regex: captures Node ID (Group 1) and all content inside parentheses (Group 2)
    node_pattern = re.compile(r'([OTCD]\d+)\s*\((.*?)\)')

    full_schedule_text = schedule_text

    for match in vehicle_pattern.finditer(full_schedule_text):
        vehicle_id = match.group(1)
        schedule_string = match.group(2).replace(' -> ', ' ')

        nodes = []

        for node_match in node_pattern.finditer(schedule_string):
            node_name = node_match.group(1)
            content = node_match.group(2) 

            # Extract Time
            time_match = re.search(r'Time:\s*([\d.]+)', content)
            time = time_match.group(1) if time_match else 'N/A'

            # Try to match the full energy tuple (E_pre, G, E_bar)
            energy_match = re.search(r'E_pre:\s*([\d.-]+),\s*G:\s*([-\d.]+),\s*E_bar:\s*([\d.]+)', content)

            if energy_match:
                e_pre = energy_match.group(1)
                g_val = energy_match.group(2)
                e_bar = energy_match.group(3)
            else:
                e_pre = '-'
                g_val = '-'
                e_bar_match = re.search(r'E_bar:\s*([\d.]+)', content)
                e_bar = e_bar_match.group(1) if e_bar_match else '-'

            # Manual override/finalization for Depot/Charge Nodes based on domain logic
            if node_name == 'O1':
                e_pre = '-'
                g_val = '-350.0'
                e_bar = '350.0'
                
                # NEW LOGIC: Check for 0.00 time value for Depot Out ('O1') node
                if time == '0.00':
                    time = '-'
                    
            elif node_name.startswith('C'): 
                if not energy_match: 
                    e_pre = '-' 
                    g_val = re.search(r'G:\s*([-\d.]+)', content).group(1) if re.search(r'G:\s*([-\d.]+)', content) else '-'
                    e_bar = '350.0'
            elif node_name == 'D1':
                if not energy_match: 
                    e_pre = '-'
                g_val = '-350.0' 
                e_bar = '0.0'
                
                # PREVIOUS LOGIC: Check for 10000.0 time value for Depot In ('D1') node
                if time.startswith('10000.0'): 
                    time = '-'
                    e_bar = '-'


            # Determine Node Type
            if node_name.startswith('T'):
                node_type = 'Trip'
            elif node_name.startswith('C'):
                node_type = 'Charge'
            elif node_name == 'O1':
                node_type = 'Depot Out'
            elif node_name == 'D1':
                node_type = 'Depot In'
            else:
                node_type = 'Unknown'

            nodes.append({
                'Node': node_name,
                'Time': time,
                'E_pre': e_pre,
                'G': g_val,
                'E_bar': e_bar,
                'Type': node_type
            })

        schedules[vehicle_id] = nodes

    # Post-processing for E_pre: use E_bar of the preceding node where E_pre is missing ('-')
    for vehicle_id, nodes in schedules.items():
        for i in range(1, len(nodes)):
            if nodes[i]['E_pre'] == '-':
                nodes[i]['E_pre'] = nodes[i-1]['E_bar']
                
    return schedules

def generate_latex_table(schedules):
    """
    Generates the LaTeX code for the schedule table, matching the requested format.
    The Node Type column is moved to the third position.
    """
    latex_code = []
    latex_code.append(r'\begin{table}[H]')
    latex_code.append(r'    \centering')
    latex_code.append(r'    \caption{Optimal Vehicle Schedules for the EB-MDVSPTW Instance.}')
    latex_code.append(r'    \label{tab:optimal_schedules}')
    latex_code.append(r'    \resizebox{0.9\textwidth}{!}{')
    # Updated column format to ccccccc
    latex_code.append(r'    \begin{tabular}{ccccccc}') 
    latex_code.append(r'        \toprule')
    # Updated header order: Vehicle ID, Node Sequence, Node Type, Arrival Time, E_pre, E_gain/loss (G), E_bar
    latex_code.append(r'        \textbf{Vehicle ID} & \textbf{Node Sequence} & \textbf{Node Type} & \textbf{Arrival Time} & \textbf{E\textsubscript{pre}} & \textbf{E\textsubscript{gain/loss} ($\mathbf{G}$)} & \textbf{E\textsubscript{bar}} \\')
    latex_code.append(r'        \midrule')

    first_vehicle = True

    for vehicle_id, nodes in schedules.items():
        if not first_vehicle:
            latex_code.append(r'        \midrule')
        first_vehicle = False

        num_nodes = len(nodes)

        for i, node in enumerate(nodes):
            # Format the node name for LaTeX (e.g., T2 -> T_2, T12 -> T_{12})
            if node['Node'] == 'O1':
                node_name_latex = 'O_1'
            elif node['Node'] == 'D1':
                node_name_latex = 'D_1'
            elif len(node['Node']) == 2 and node['Node'][0] == 'T':
                node_name_latex = f'T_{node["Node"][1]}'
            elif len(node['Node']) > 2 and node['Node'][0] == 'T':
                 node_name_latex = f'T_{{{node["Node"][1:]}}}'
            elif node['Node'] == 'C1':
                 node_name_latex = 'C_1'
            else:
                node_name_latex = node['Node']
            
            # REORDERED row_data to match new column order:
            # Node Sequence, Node Type, Arrival Time, E_pre, G, E_bar
            # All values that were 0.00 or 10000.0 for Depot nodes are now '-'
            row_data = [
                r'$' + node_name_latex + r'$', # 1. Node Sequence (e.g. $T_2$)
                node['Type'],                   # 2. Node Type
                r'$' + node['Time'] + r'$',     # 3. Arrival Time
                r'$' + node['E_pre'] + r'$',    # 4. E_pre
                r'$' + node['G'] + r'$',        # 5. G
                r'$' + node['E_bar'] + r'$'     # 6. E_bar
            ]

            # Add multirow command for Vehicle ID
            if i == 0:
                # First node: include multirow command
                latex_code.append(r'        \multirow{%d}{*}{$%s$} & %s \\' % (
                    num_nodes, vehicle_id.replace('_', '\_'), ' & '.join(row_data)
                ))
            else:
                # Subsequent nodes: leave the first column empty
                latex_code.append(r'        & %s \\' % ' & '.join(row_data))


    latex_code.append(r'        \bottomrule')
    latex_code.append(r'    \end{tabular}')
    latex_code.append(r'    }')
    latex_code.append(r'\end{table}')
    
    return '\n'.join(latex_code)

# --- Input Text ---
schedule_input = """
- Vehicle D1_V3 schedule: O1 (Time: 0.00, E_bar: 350.0) -> T9 (Time: 465.48 (E_pre: 298.5, G: 16.6, E_bar: 281.9)) -> C1 (Time: 487.94 (E_pre: 276.6, G: -73.4, E_bar: 350.0)) -> T6 (Time: 525.64 (E_pre: 316.8, G: 30.4, E_bar: 286.4)) -> T5 (Time: 570.77 (E_pre: 286.4, G: 25.9, E_bar: 260.5)) -> D1 (Time: 10000.00 (E_pre: 234.1, G: -350.0, E_bar: 0.0))
- Vehicle D1_V4 schedule: O1 (Time: 0.00, E_bar: 350.0) -> T2 (Time: 468.57 (E_pre: 323.5, G: 18.2, E_bar: 305.3)) -> T8 (Time: 766.00 (E_pre: 270.2, G: 35.1, E_bar: 235.1)) -> C1 (Time: 1030.80 (E_pre: 229.8, G: -120.2, E_bar: 350.0)) -> T7 (Time: 1050.32 (E_pre: 338.6, G: 25.9, E_bar: 312.7)) -> D1 (Time: 10000.00 (E_pre: 286.3, G: -350.0, E_bar: 0.0))
- Vehicle D1_V5 schedule: O1 (Time: 0.00, E_bar: 350.0) -> T3 (Time: 396.17 (E_pre: 323.5, G: 25.9, E_bar: 297.6)) -> T10 (Time: 650.38 (E_pre: 297.6, G: 30.4, E_bar: 267.2)) -> T11 (Time: 700.59 (E_pre: 267.2, G: 22.6, E_bar: 244.6)) -> C1 (Time: 947.76 (E_pre: 224.8, G: -125.2, E_bar: 350.0)) -> T1 (Time: 950.24 (E_pre: 330.2, G: 22.6, E_bar: 307.6)) -> T4 (Time: 1170.38 (E_pre: 284.9, G: 22.6, E_bar: 262.3)) -> D1 (Time: 10000.00 (E_pre: 219.7, G: -350.0, E_bar: 0.0))
- Vehicle D1_V6 schedule: O1 (Time: 0.00, E_bar: 350.0) -> T12 (Time: 424.44 (E_pre: 298.5, G: 16.6, E_bar: 281.9)) -> D1 (Time: 10000.00 (E_pre: 243.2, G: -350.0, E_bar: 0.0))
"""

# --- Execution ---
parsed_data = parse_schedule_text(schedule_input)
latex_output = generate_latex_table(parsed_data)

print("```python")
# In a real environment, you'd print the full code here.
print(
    '# ... parse_schedule_text and generate_latex_table functions (updated) ...\n'
    '# --- Input Text ---\n'
    'schedule_input = """\n'
    '- Vehicle D1_V2 schedule: O1 (Time: 0.00, E_bar: 350.0) -> ... -> D1 (Time: 10000.00, E_bar: 0.0)\n'
    '# ... remaining vehicles ...\n'
    '"""\n\n'
    '# --- Execution ---\n'
    'parsed_data = parse_schedule_text(schedule_input)\n'
    'latex_output = generate_latex_table(parsed_data)\n\n'
    '# The output below reflects the full updated code.\n'
)
print("```")
print("---")
print("## 📊 Generated LaTeX Table Output")
print("The **Arrival Time** for the starting **Depot Out** (`O_1`) nodes is now a hyphen (`-`).")
print("```latex")
print(latex_output)
print("```")