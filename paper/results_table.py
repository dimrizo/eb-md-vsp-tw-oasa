import re

def parse_schedule_text(schedule_text):
    """
    Parses the raw text schedule, extracts vehicle paths and node details.
    Updates:
    1. Node Type is determined.
    2. Depot In (D1) Time/E_bar are set to '-' if Time is '10000.0'.
    3. Depot Out (O1) Time is set to '-' if Time is '0.00'.
    """

    schedules = []   # list of blocks, each block = {"vehicle": ..., "nodes": ...}
    # Regex to be non-greedy and stop before the next vehicle schedule line or EOF.
    # It now relies only on the '- Vehicle D1_V#' pattern.
    vehicle_pattern = re.compile(
        r'-\s*Vehicle\s*(D\d+_V\d+)\s*schedule:\s*(.*?)(?=\n- Vehicle|\Z)',
        re.DOTALL
    )

    # Regex: captures Node ID (Group 1) and all content inside parentheses (Group 2)
    node_pattern = re.compile(r'([OTCD]\d+|T\d+)\s*\((.*?)\)')

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
            # Parse each field independently (so E_pre can be captured even if G/E_bar are '-')
            e_pre_match = re.search(r'E_pre:\s*([-\d.]+)', content)
            g_match     = re.search(r'G:\s*([-\d.]+|-)', content)
            e_bar_match = re.search(r'E_bar:\s*([-\d.]+|-)', content)

            e_pre = e_pre_match.group(1) if e_pre_match else '-'
            g_val = g_match.group(1) if g_match else '-'
            e_bar = e_bar_match.group(1) if e_bar_match else '-'

            # Manual override/finalization for Depot/Charge Nodes based on domain logic
            # Manual override/finalization for Depot/Charge Nodes based on domain logic
            if node_name.startswith('O'):
                # e_pre = '-'
                # g_val = '-350.0'
                # e_bar = '350.0'

                # Depot Out time -> '-' if it is 0
                try:
                    if float(time) == 0.0:
                        time = '-'
                except ValueError:
                    pass

            elif node_name.startswith('C'):
                # If E_bar is missing, apply your default charge-full rule
                if e_bar == '-':
                    e_bar = '350.0'
                # If E_pre is missing, keep it missing (or set '-' explicitly)
                if e_pre == '-':
                    e_pre = '-'

            elif node_name.startswith('D'):
                # Domain rule: depot-in doesn't have meaningful G/E_bar in your table
                g_val = '-'
                e_bar = '-'

                # Only force E_pre to '-' if it's genuinely missing
                if e_pre == '-':
                    e_pre = '-'

                # Depot In time -> '-' if it is ~10000
                try:
                    if float(time) >= 9999.0:
                        time = '-'
                        e_bar = '-'
                except ValueError:
                    pass

            # Determine Node Type
            if node_name.startswith('T'):
                node_type = 'Trip'
            elif node_name.startswith('C'):
                node_type = 'Charge'
            elif node_name.startswith('O'):
                node_type = 'Depot Out'
            elif node_name.startswith('D'):
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

        schedules.append({
            "vehicle": vehicle_id,
            "nodes": nodes
        })

    # Post-processing for E_pre: use E_bar of the preceding node where E_pre is missing ('-')
    for block in schedules:
        nodes = block["nodes"]
        for i in range(1, len(nodes)):
            if nodes[i]['E_pre'] == '-':
                nodes[i]['E_pre'] = nodes[i-1]['E_bar']

    # NEW: force E_bar of the last node in each vehicle schedule to '-'
    # for block in schedules:
    #     if block["nodes"]:
    #         block["nodes"][-1]["E_bar"] = '-'

    return schedules

def generate_latex_table(schedules, multipage_threshold=35):
    """
    Multipage LaTeX via longtable.
    Each vehicle block is kept together: if it doesn't fit, it starts on the next page.
    """

    schedules = sorted(
        schedules,
        key=lambda b: (
            b["vehicle"],
            float(b["nodes"][0]["Time"]) if b["nodes"] and b["nodes"][0]["Time"] not in ("-", "N/A")
            else float("inf")
        )
    )

    total_rows = sum(len(b["nodes"]) for b in schedules)
    use_longtable = total_rows > multipage_threshold

    header = (
        r'\textbf{Vehicle ID} & \textbf{Node Sequence} & \textbf{Node Type} & '
        r'\textbf{Arrival Time} & \textbf{E\textsubscript{pre}} & '
        r'\textbf{E\textsubscript{gain/loss} ($\mathbf{G}$)} & \textbf{E\textsubscript{bar}} \\'
    )

    def vehicle_tex(vehicle_id: str) -> str:
        return r'$%s$' % vehicle_id.replace('_', r'\_')

    def node_to_latex(node_name: str) -> str:
        if node_name.startswith("O") and node_name[1:].isdigit():
            return f'O_{node_name[1:]}'
        if node_name.startswith("D") and node_name[1:].isdigit():
            return f'D_{node_name[1:]}'
        if node_name.startswith("T") and node_name[1:].isdigit():
            idx = node_name[1:]
            return f'T_{{{idx}}}' if len(idx) > 1 else f'T_{idx}'
        if node_name.startswith("C") and node_name[1:].isdigit():
            idx = node_name[1:]
            return f'C_{{{idx}}}' if len(idx) > 1 else f'C_{idx}'
        return node_name

    latex = []

    if use_longtable:
        latex.append(r'\begin{longtable}{@{}ccccccc@{}}')
        latex.append(r'\caption{Optimal Vehicle Schedules for the EB-MDVSPTW Instance.}\label{tab:optimal_schedules}\\')
        latex.append(r'\toprule')
        latex.append(header)
        latex.append(r'\midrule')
        latex.append(r'\endfirsthead')

        latex.append(r'\toprule')
        latex.append(header)
        latex.append(r'\midrule')
        latex.append(r'\endhead')

        latex.append(r'\midrule')
        latex.append(r'\multicolumn{7}{r}{\small\itshape Continued on next page} \\')
        latex.append(r'\endfoot')

        latex.append(r'\bottomrule')
        latex.append(r'\endlastfoot')
    else:
        latex.append(r'\begin{table}[H]')
        latex.append(r'\centering')
        latex.append(r'\caption{Optimal Vehicle Schedules for the EB-MDVSPTW Instance.}')
        latex.append(r'\label{tab:optimal_schedules}')
        latex.append(r'\begin{tabular}{@{}ccccccc@{}}')
        latex.append(r'\toprule')
        latex.append(header)
        latex.append(r'\midrule')

    for bi, block in enumerate(schedules):
        vid = block["vehicle"]
        nodes = block["nodes"]
        if not nodes:
            continue

        vcell = vehicle_tex(vid)
        k = len(nodes)

        if bi > 0:
            latex.append(r'\midrule')

        if use_longtable:
            # Keep the WHOLE vehicle block together.
            # The +8 is padding for midrules/spacing/header artifacts that steal vertical space.
            latex.append(r'\noalign{\Needspace{%d\baselineskip}}' % (k + 8))

        for i, node in enumerate(nodes):
            node_name_latex = node_to_latex(node["Node"])
            row_data = [
                r'$' + node_name_latex + r'$',
                node['Type'],
                r'$' + str(node['Time']) + r'$',
                r'$' + str(node['E_pre']) + r'$',
                r'$' + str(node['G']) + r'$',
                r'$' + str(node['E_bar']) + r'$'
            ]

            if i == 0:
                latex.append(
                    r'\multirow[c]{%d}{*}{%s} & %s \\' % (k, vcell, ' & '.join(row_data))
                )
            else:
                latex.append(r' & %s \\' % (' & '.join(row_data)))

    if use_longtable:
        latex.append(r'\end{longtable}')
    else:
        latex.append(r'\bottomrule')
        latex.append(r'\end{tabular}')
        latex.append(r'\end{table}')

    return '\n'.join(latex)

# --- Input Text ---
schedule_input = """
- Vehicle D1_V3 schedule: O1 (Time: 273.88, E_bar: 350.0) -> T40 (Time: 290.00 (E_pre: 336.0, G: 27.7, E_bar: 308.3)) -> T34 (Time: 364.11 (E_pre: 308.3, G: 27.7, E_bar: 280.6)) -> T41 (Time: 437.88 (E_pre: 280.6, G: 27.7, E_bar: 252.9)) -> T35 (Time: 512.00 (E_pre: 252.9, G: 27.7, E_bar: 225.2)) -> D1 (Time: 601.89 (E_pre: 211.3, G: - , E_bar: -))
- Vehicle D1_V1 schedule: O1 (Time: 284.34, E_bar: 350.0) -> T9 (Time: 290.00 (E_pre: 345.1, G: 18.6, E_bar: 326.5)) -> T3 (Time: 350.29 (E_pre: 326.5, G: 18.6, E_bar: 308.0)) -> T12 (Time: 409.40 (E_pre: 308.0, G: 18.6, E_bar: 289.4)) -> T4 (Time: 469.69 (E_pre: 289.4, G: 18.6, E_bar: 270.9)) -> D1 (Time: 534.45 (E_pre: 266.0, G: - , E_bar: -))
- Vehicle D2_V7 schedule: O2 (Time: 288.98, E_bar: 350.0) -> T14 (Time: 300.49 (E_pre: 340.0, G: 15.7, E_bar: 324.3)) -> T25 (Time: 347.98 (E_pre: 324.3, G: 15.7, E_bar: 308.7)) -> T23 (Time: 400.00 (E_pre: 308.7, G: 15.7, E_bar: 293.0)) -> T26 (Time: 447.49 (E_pre: 293.0, G: 15.7, E_bar: 277.3)) -> D2 (Time: 511.02 (E_pre: 267.3, G: - , E_bar: -))
- Vehicle D2_V3 schedule: O2 (Time: 294.39, E_bar: 350.0) -> T1 (Time: 305.90 (E_pre: 340.0, G: 18.6, E_bar: 321.5)) -> T13 (Time: 365.00 (E_pre: 321.5, G: 18.6, E_bar: 302.9)) -> T5 (Time: 425.29 (E_pre: 302.9, G: 18.6, E_bar: 284.4)) -> T11 (Time: 484.40 (E_pre: 284.4, G: 18.6, E_bar: 265.8)) -> D2 (Time: 556.20 (E_pre: 255.8, G: - , E_bar: -))
- Vehicle D2_V12 schedule: O2 (Time: 295.18, E_bar: 350.0) -> T27 (Time: 302.98 (E_pre: 343.2, G: 15.7, E_bar: 327.6)) -> T20 (Time: 355.00 (E_pre: 327.6, G: 15.7, E_bar: 311.9)) -> T31 (Time: 402.49 (E_pre: 311.9, G: 15.7, E_bar: 296.2)) -> T24 (Time: 454.51 (E_pre: 296.2, G: 15.7, E_bar: 280.5)) -> D2 (Time: 509.81 (E_pre: 273.8, G: - , E_bar: -))
- Vehicle D1_V15 schedule: O1 (Time: 303.88, E_bar: 350.0) -> T42 (Time: 320.00 (E_pre: 336.0, G: 27.7, E_bar: 308.3)) -> T36 (Time: 394.11 (E_pre: 308.3, G: 27.7, E_bar: 280.6)) -> D1 (Time: 484.01 (E_pre: 266.7, G: - , E_bar: -))
- Vehicle D1_V9 schedule: O1 (Time: 304.78, E_bar: 350.0) -> T53 (Time: 310.44 (E_pre: 345.1, G: 18.6, E_bar: 326.5)) -> T49 (Time: 366.80 (E_pre: 326.5, G: 18.6, E_bar: 308.0)) -> T56 (Time: 423.63 (E_pre: 308.0, G: 18.6, E_bar: 289.4)) -> T50 (Time: 480.00 (E_pre: 289.4, G: 18.6, E_bar: 270.9)) -> T54 (Time: 536.83 (E_pre: 270.9, G: 18.6, E_bar: 252.3)) -> D1 (Time: 610.78 (E_pre: 237.1, G: - , E_bar: -))
- Vehicle D2_V2 schedule: O2 (Time: 308.99, E_bar: 350.0) -> T37 (Time: 320.50 (E_pre: 340.0, G: 27.7, E_bar: 312.3)) -> T39 (Time: 426.23 (E_pre: 284.6, G: 27.7, E_bar: 256.9)) -> T44 (Time: 500.00 (E_pre: 256.9, G: 27.7, E_bar: 229.2)) -> D2 (Time: 585.62 (E_pre: 219.2, G: - , E_bar: -))
- Vehicle D2_V4 schedule: O2 (Time: 312.68, E_bar: 350.0) -> T30 (Time: 320.49 (E_pre: 343.2, G: 15.7, E_bar: 327.6)) -> T22 (Time: 372.51 (E_pre: 327.6, G: 15.7, E_bar: 311.9)) -> T32 (Time: 420.00 (E_pre: 311.9, G: 15.7, E_bar: 296.2)) -> T21 (Time: 472.02 (E_pre: 296.2, G: 15.7, E_bar: 280.5)) -> D2 (Time: 527.32 (E_pre: 273.8, G: - , E_bar: -))
- Vehicle D2_V6 schedule: O2 (Time: 316.66, E_bar: 350.0) -> T47 (Time: 328.17 (E_pre: 340.0, G: 18.6, E_bar: 321.5)) -> T51 (Time: 385.00 (E_pre: 321.5, G: 18.6, E_bar: 302.9)) -> T46 (Time: 441.37 (E_pre: 302.9, G: 18.6, E_bar: 284.4)) -> T55 (Time: 498.20 (E_pre: 284.4, G: 18.6, E_bar: 265.8)) -> D2 (Time: 566.07 (E_pre: 255.8, G: - , E_bar: -))
- Vehicle D2_V10 schedule: O2 (Time: 318.98, E_bar: 350.0) -> T18 (Time: 330.49 (E_pre: 340.0, G: 15.7, E_bar: 324.3)) -> T29 (Time: 377.98 (E_pre: 324.3, G: 15.7, E_bar: 308.7)) -> T17 (Time: 430.00 (E_pre: 308.7, G: 15.7, E_bar: 293.0)) -> D2 (Time: 485.30 (E_pre: 286.2, G: - , E_bar: -))
- Vehicle D2_V1 schedule: O2 (Time: 323.49, E_bar: 350.0) -> T16 (Time: 335.00 (E_pre: 340.0, G: 15.7, E_bar: 324.3)) -> T28 (Time: 382.49 (E_pre: 324.3, G: 15.7, E_bar: 308.7)) -> T19 (Time: 434.51 (E_pre: 308.7, G: 15.7, E_bar: 293.0)) -> D2 (Time: 489.81 (E_pre: 286.2, G: - , E_bar: -))
- Vehicle D1_V6 schedule: O1 (Time: 324.34, E_bar: 350.0) -> T7 (Time: 330.00 (E_pre: 345.1, G: 18.6, E_bar: 326.5)) -> T2 (Time: 390.29 (E_pre: 326.5, G: 18.6, E_bar: 308.0)) -> T8 (Time: 449.40 (E_pre: 308.0, G: 18.6, E_bar: 289.4)) -> T6 (Time: 509.69 (E_pre: 289.4, G: 18.6, E_bar: 270.9)) -> D1 (Time: 574.45 (E_pre: 266.0, G: - , E_bar: -))
- Vehicle D1_V13 schedule: O1 (Time: 344.34, E_bar: 350.0) -> T52 (Time: 350.00 (E_pre: 345.1, G: 18.6, E_bar: 326.5)) -> T48 (Time: 406.37 (E_pre: 326.5, G: 18.6, E_bar: 308.0)) -> T57 (Time: 463.20 (E_pre: 308.0, G: 18.6, E_bar: 289.4)) -> T45 (Time: 519.56 (E_pre: 289.4, G: 18.6, E_bar: 270.9)) -> D1 (Time: 582.05 (E_pre: 266.0, G: - , E_bar: -))
- Vehicle D1_V14 schedule: O1 (Time: 353.88, E_bar: 350.0) -> T43 (Time: 370.00 (E_pre: 336.0, G: 27.7, E_bar: 308.3)) -> T38 (Time: 465.00 (E_pre: 308.3, G: 27.7, E_bar: 280.6)) -> D1 (Time: 554.89 (E_pre: 266.7, G: - , E_bar: -))
- Vehicle D2_V8 schedule: O2 (Time: 391.00, E_bar: 350.0) -> T15 (Time: 402.51 (E_pre: 340.0, G: 15.7, E_bar: 324.3)) -> T33 (Time: 450.00 (E_pre: 324.3, G: 15.7, E_bar: 308.7)) -> D2 (Time: 513.53 (E_pre: 298.7, G: - , E_bar: -))
- Vehicle D2_V15 schedule: O2 (Time: 457.19, E_bar: 350.0) -> T62 (Time: 465.00 (E_pre: 343.2, G: 15.7, E_bar: 327.6)) -> T60 (Time: 517.02 (E_pre: 327.6, G: 15.7, E_bar: 311.9)) -> D2 (Time: 572.32 (E_pre: 305.1, G: - , E_bar: -))
- Vehicle D2_V5 schedule: O2 (Time: 463.49, E_bar: 350.0) -> T59 (Time: 475.00 (E_pre: 340.0, G: 15.7, E_bar: 324.3)) -> T63 (Time: 522.49 (E_pre: 324.3, G: 15.7, E_bar: 308.7)) -> D2 (Time: 586.02 (E_pre: 298.7, G: - , E_bar: -))
- Vehicle D2_V10 schedule: O2 (Time: 485.30, E_bar: 286.2) -> T58 (Time: 496.81 (E_pre: 276.2, G: 15.7, E_bar: 260.6)) -> T64 (Time: 544.30 (E_pre: 260.6, G: 15.7, E_bar: 244.9)) -> D2 (Time: 607.83 (E_pre: 234.9, G: - , E_bar: -))
- Vehicle D2_V1 schedule: O2 (Time: 489.81, E_bar: 286.2) -> T61 (Time: 497.62 (E_pre: 279.5, G: 15.7, E_bar: 263.8)) -> D2 (Time: 561.14 (E_pre: 253.8, G: - , E_bar: -))
- Vehicle D1_V2 schedule: O1 (Time: 499.34, E_bar: 350.0) -> T10 (Time: 505.00 (E_pre: 345.1, G: 18.6, E_bar: 326.5)) -> D1 (Time: 582.88 (E_pre: 311.3, G: - , E_bar: -))
- Vehicle D2_V12 schedule: O2 (Time: 521.00, E_bar: 273.8) -> T87 (Time: 532.51 (E_pre: 263.8, G: 15.7, E_bar: 248.1)) -> T96 (Time: 580.00 (E_pre: 248.1, G: 15.7, E_bar: 232.4)) -> T89 (Time: 640.00 (E_pre: 232.4, G: 15.7, E_bar: 216.7)) -> T97 (Time: 690.00 (E_pre: 216.7, G: 15.7, E_bar: 201.1)) -> T90 (Time: 745.00 (E_pre: 201.1, G: 15.7, E_bar: 185.4)) -> D2 (Time: 800.30 (E_pre: 178.6, G: - , E_bar: -))
- Vehicle D1_V15 schedule: O1 (Time: 523.88, E_bar: 266.7) -> T113 (Time: 540.00 (E_pre: 252.7, G: 27.7, E_bar: 225.0)) -> T105 (Time: 614.11 (E_pre: 225.0, G: 27.7, E_bar: 197.3)) -> T116 (Time: 687.88 (E_pre: 197.3, G: 27.7, E_bar: 169.6)) -> C2 (Time: 780.01 (E_pre: 159.6, G: -190.4, E_bar: 350.0)) -> T101 (Time: 855.00 (E_pre: 340.0, G: 27.7, E_bar: 312.3)) -> D1 (Time: 944.89 (E_pre: 298.3, G: - , E_bar: -))
- Vehicle D2_V8 schedule: O2 (Time: 524.39, E_bar: 298.7) -> T72 (Time: 535.90 (E_pre: 288.7, G: 18.6, E_bar: 270.2)) -> T76 (Time: 595.00 (E_pre: 270.2, G: 18.6, E_bar: 251.6)) -> T74 (Time: 655.90 (E_pre: 251.6, G: 18.6, E_bar: 233.0)) -> T77 (Time: 715.00 (E_pre: 233.0, G: 18.6, E_bar: 214.5)) -> T75 (Time: 775.29 (E_pre: 214.5, G: 18.6, E_bar: 195.9)) -> T80 (Time: 834.40 (E_pre: 195.9, G: 18.6, E_bar: 177.4)) -> D2 (Time: 906.20 (E_pre: 167.4, G: - , E_bar: -))
- Vehicle D2_V9 schedule: O2 (Time: 524.72, E_bar: 350.0) -> T106 (Time: 536.23 (E_pre: 340.0, G: 27.7, E_bar: 312.3)) -> T118 (Time: 610.00 (E_pre: 312.3, G: 27.7, E_bar: 284.6)) -> T100 (Time: 684.11 (E_pre: 284.6, G: 27.7, E_bar: 256.9)) -> T110 (Time: 757.88 (E_pre: 256.9, G: 27.7, E_bar: 229.2)) -> T104 (Time: 832.00 (E_pre: 229.2, G: 27.7, E_bar: 201.5)) -> T115 (Time: 905.77 (E_pre: 201.5, G: 27.7, E_bar: 173.8)) -> D2 (Time: 991.39 (E_pre: 163.8, G: - , E_bar: -))
- Vehicle D2_V4 schedule: O2 (Time: 537.19, E_bar: 273.8) -> T93 (Time: 545.00 (E_pre: 267.0, G: 15.7, E_bar: 251.3)) -> C2 (Time: 616.46 (E_pre: 241.3, G: -108.7, E_bar: 350.0)) -> T94 (Time: 660.49 (E_pre: 343.2, G: 15.7, E_bar: 327.6)) -> T88 (Time: 712.51 (E_pre: 327.6, G: 15.7, E_bar: 311.9)) -> T95 (Time: 760.00 (E_pre: 311.9, G: 15.7, E_bar: 296.2)) -> D2 (Time: 823.53 (E_pre: 286.2, G: - , E_bar: -))
- Vehicle D2_V7 schedule: O2 (Time: 538.49, E_bar: 267.3) -> T86 (Time: 550.00 (E_pre: 257.4, G: 15.7, E_bar: 241.7)) -> D2 (Time: 605.30 (E_pre: 234.9, G: - , E_bar: -))
- Vehicle D2_V13 schedule: O2 (Time: 546.66, E_bar: 350.0) -> T126 (Time: 558.17 (E_pre: 340.0, G: 18.6, E_bar: 321.5)) -> T135 (Time: 615.00 (E_pre: 321.5, G: 18.6, E_bar: 302.9)) -> T120 (Time: 671.37 (E_pre: 302.9, G: 18.6, E_bar: 284.4)) -> T133 (Time: 735.00 (E_pre: 284.4, G: 18.6, E_bar: 265.8)) -> T121 (Time: 791.37 (E_pre: 265.8, G: 18.6, E_bar: 247.2)) -> T139 (Time: 848.20 (E_pre: 247.2, G: 18.6, E_bar: 228.7)) -> D2 (Time: 916.07 (E_pre: 218.7, G: - , E_bar: -))
- Vehicle D1_V1 schedule: O1 (Time: 554.34, E_bar: 266.0) -> T84 (Time: 560.00 (E_pre: 261.1, G: 18.6, E_bar: 242.5)) -> T68 (Time: 625.00 (E_pre: 242.5, G: 18.6, E_bar: 224.0)) -> T85 (Time: 685.00 (E_pre: 224.0, G: 18.6, E_bar: 205.4)) -> T71 (Time: 745.29 (E_pre: 205.4, G: 18.6, E_bar: 186.9)) -> T82 (Time: 804.40 (E_pre: 186.9, G: 18.6, E_bar: 168.3)) -> T69 (Time: 864.69 (E_pre: 168.3, G: 18.6, E_bar: 149.7)) -> D1 (Time: 929.45 (E_pre: 144.8, G: - , E_bar: -))
- Vehicle D2_V3 schedule: O2 (Time: 559.39, E_bar: 255.8) -> T66 (Time: 570.90 (E_pre: 245.9, G: 18.6, E_bar: 227.3)) -> T83 (Time: 630.00 (E_pre: 227.3, G: 18.6, E_bar: 208.7)) -> T70 (Time: 690.90 (E_pre: 208.7, G: 18.6, E_bar: 190.2)) -> T81 (Time: 750.00 (E_pre: 190.2, G: 18.6, E_bar: 171.6)) -> T67 (Time: 810.29 (E_pre: 171.6, G: 18.6, E_bar: 153.1)) -> T78 (Time: 869.40 (E_pre: 153.1, G: 18.6, E_bar: 134.5)) -> D2 (Time: 941.20 (E_pre: 124.5, G: - , E_bar: -))
- Vehicle D2_V1 schedule: O2 (Time: 561.14, E_bar: 253.8) -> T102 (Time: 572.65 (E_pre: 243.8, G: 27.7, E_bar: 216.1)) -> T111 (Time: 646.42 (E_pre: 216.1, G: 27.7, E_bar: 188.4)) -> T107 (Time: 720.54 (E_pre: 188.4, G: 27.7, E_bar: 160.7)) -> T119 (Time: 794.31 (E_pre: 160.7, G: 27.7, E_bar: 133.0)) -> D2 (Time: 879.93 (E_pre: 123.0, G: - , E_bar: -))
- Vehicle D1_V12 schedule: O1 (Time: 562.98, E_bar: 350.0) -> T137 (Time: 568.63 (E_pre: 345.1, G: 18.6, E_bar: 326.5)) -> T128 (Time: 625.00 (E_pre: 326.5, G: 18.6, E_bar: 308.0)) -> T132 (Time: 681.83 (E_pre: 308.0, G: 18.6, E_bar: 289.4)) -> T125 (Time: 738.20 (E_pre: 289.4, G: 18.6, E_bar: 270.9)) -> D1 (Time: 800.68 (E_pre: 266.0, G: - , E_bar: -))
- Vehicle D1_V14 schedule: O1 (Time: 563.88, E_bar: 266.7) -> T109 (Time: 580.00 (E_pre: 252.7, G: 27.7, E_bar: 225.0)) -> T103 (Time: 654.11 (E_pre: 225.0, G: 27.7, E_bar: 197.3)) -> T114 (Time: 727.88 (E_pre: 197.3, G: 27.7, E_bar: 169.6)) -> D1 (Time: 819.58 (E_pre: 154.3, G: - , E_bar: -))
- Vehicle D2_V15 schedule: O2 (Time: 581.49, E_bar: 305.1) -> T91 (Time: 593.00 (E_pre: 295.1, G: 15.7, E_bar: 279.5)) -> T98 (Time: 640.49 (E_pre: 279.5, G: 15.7, E_bar: 263.8)) -> T92 (Time: 692.51 (E_pre: 263.8, G: 15.7, E_bar: 248.1)) -> T99 (Time: 740.00 (E_pre: 248.1, G: 15.7, E_bar: 232.4)) -> D2 (Time: 803.53 (E_pre: 222.4, G: - , E_bar: -))
- Vehicle D2_V6 schedule: O2 (Time: 581.66, E_bar: 255.8) -> T122 (Time: 593.17 (E_pre: 245.9, G: 18.6, E_bar: 227.3)) -> T130 (Time: 650.00 (E_pre: 227.3, G: 18.6, E_bar: 208.7)) -> T124 (Time: 710.00 (E_pre: 208.7, G: 18.6, E_bar: 190.2)) -> T136 (Time: 768.63 (E_pre: 190.2, G: 18.6, E_bar: 171.6)) -> T127 (Time: 825.00 (E_pre: 171.6, G: 18.6, E_bar: 153.1)) -> T131 (Time: 881.83 (E_pre: 153.1, G: 18.6, E_bar: 134.5)) -> D2 (Time: 949.71 (E_pre: 124.5, G: - , E_bar: -))
- Vehicle D1_V9 schedule: O1 (Time: 639.34, E_bar: 237.1) -> T79 (Time: 645.00 (E_pre: 232.2, G: 18.6, E_bar: 213.6)) -> T73 (Time: 705.29 (E_pre: 213.6, G: 18.6, E_bar: 195.1)) -> D1 (Time: 770.05 (E_pre: 190.2, G: - , E_bar: -))
- Vehicle D1_V3 schedule: O1 (Time: 697.98, E_bar: 211.3) -> T138 (Time: 703.63 (E_pre: 206.4, G: 18.6, E_bar: 187.8)) -> T123 (Time: 760.00 (E_pre: 187.8, G: 18.6, E_bar: 169.2)) -> T134 (Time: 816.83 (E_pre: 169.2, G: 18.6, E_bar: 150.7)) -> T129 (Time: 873.20 (E_pre: 150.7, G: 18.6, E_bar: 132.1)) -> D1 (Time: 935.68 (E_pre: 127.2, G: - , E_bar: -))
- Vehicle D2_V12 schedule: O2 (Time: 800.30, E_bar: 178.6) -> T141 (Time: 811.81 (E_pre: 168.6, G: 15.7, E_bar: 153.0)) -> D2 (Time: 867.11 (E_pre: 146.2, G: - , E_bar: -))
- Vehicle D1_V12 schedule: O1 (Time: 800.68, E_bar: 266.0) -> T142 (Time: 818.27 (E_pre: 250.7, G: 15.7, E_bar: 235.1)) -> D1 (Time: 886.30 (E_pre: 217.3, G: - , E_bar: -))
- Vehicle D2_V15 schedule: O2 (Time: 803.53, E_bar: 222.4) -> T140 (Time: 811.33 (E_pre: 215.7, G: 15.7, E_bar: 200.0)) -> D2 (Time: 874.86 (E_pre: 190.0, G: - , E_bar: -))
- Vehicle D1_V9 schedule: O1 (Time: 804.76, E_bar: 190.2) -> T112 (Time: 820.89 (E_pre: 176.2, G: 27.7, E_bar: 148.5)) -> T108 (Time: 895.00 (E_pre: 148.5, G: 27.7, E_bar: 120.8)) -> D1 (Time: 984.89 (E_pre: 106.8, G: - , E_bar: -))
- Vehicle D1_V14 schedule: O1 (Time: 824.46, E_bar: 154.3) -> T143 (Time: 845.00 (E_pre: 136.5, G: 15.7, E_bar: 120.9)) -> D1 (Time: 914.60 (E_pre: 105.6, G: - , E_bar: -))
- Vehicle D2_V4 schedule: O2 (Time: 829.71, E_bar: 286.2) -> T117 (Time: 860.00 (E_pre: 260.0, G: 27.7, E_bar: 232.3)) -> D2 (Time: 945.62 (E_pre: 222.3, G: - , E_bar: -))
- Vehicle D2_V10 schedule: O2 (Time: 838.49, E_bar: 234.9) -> T144 (Time: 850.00 (E_pre: 224.9, G: 15.7, E_bar: 209.3)) -> D2 (Time: 905.30 (E_pre: 202.5, G: - , E_bar: -))
- Vehicle D2_V12 schedule: O2 (Time: 867.11, E_bar: 146.2) -> T145 (Time: 874.91 (E_pre: 139.4, G: 15.7, E_bar: 123.8)) -> D2 (Time: 938.44 (E_pre: 113.8, G: - , E_bar: -))
- Vehicle D2_V15 schedule: O2 (Time: 883.49, E_bar: 190.0) -> T149 (Time: 895.00 (E_pre: 180.0, G: 18.6, E_bar: 161.5)) -> T156 (Time: 965.00 (E_pre: 161.5, G: 18.6, E_bar: 142.9)) -> T150 (Time: 1025.29 (E_pre: 142.9, G: 18.6, E_bar: 124.4)) -> D2 (Time: 1099.23 (E_pre: 111.5, G: - , E_bar: -))
- Vehicle D1_V12 schedule: O1 (Time: 886.30, E_bar: 217.3) -> T154 (Time: 891.95 (E_pre: 212.4, G: 18.6, E_bar: 193.8)) -> T147 (Time: 952.24 (E_pre: 193.8, G: 18.6, E_bar: 175.2)) -> D1 (Time: 1017.00 (E_pre: 170.3, G: - , E_bar: -))
- Vehicle D2_V11 schedule: O2 (Time: 888.49, E_bar: 350.0) -> T168 (Time: 900.00 (E_pre: 340.0, G: 15.7, E_bar: 324.3)) -> D2 (Time: 955.30 (E_pre: 317.6, G: - , E_bar: -))
- Vehicle D2_V14 schedule: O2 (Time: 892.19, E_bar: 350.0) -> T167 (Time: 900.00 (E_pre: 343.2, G: 15.7, E_bar: 327.6)) -> D2 (Time: 963.53 (E_pre: 317.6, G: - , E_bar: -))
- Vehicle D2_V7 schedule: O2 (Time: 896.66, E_bar: 234.9) -> T158 (Time: 908.17 (E_pre: 224.9, G: 18.6, E_bar: 206.4)) -> T165 (Time: 965.00 (E_pre: 206.4, G: 18.6, E_bar: 187.8)) -> T161 (Time: 1030.00 (E_pre: 187.8, G: 18.6, E_bar: 169.3)) -> D2 (Time: 1101.67 (E_pre: 156.4, G: - , E_bar: -))
- Vehicle D2_V8 schedule: O2 (Time: 906.20, E_bar: 167.4) -> T151 (Time: 917.71 (E_pre: 157.4, G: 18.6, E_bar: 138.9)) -> D2 (Time: 991.65 (E_pre: 126.0, G: - , E_bar: -))
- Vehicle D2_V5 schedule: O2 (Time: 908.49, E_bar: 298.7) -> T169 (Time: 920.00 (E_pre: 288.7, G: 15.7, E_bar: 273.0)) -> D2 (Time: 975.30 (E_pre: 266.3, G: - , E_bar: -))
- Vehicle D2_V10 schedule: O2 (Time: 910.16, E_bar: 202.5) -> T155 (Time: 925.00 (E_pre: 189.6, G: 18.6, E_bar: 171.1)) -> T148 (Time: 985.29 (E_pre: 171.1, G: 18.6, E_bar: 152.5)) -> T153 (Time: 1044.40 (E_pre: 152.5, G: 18.6, E_bar: 134.0)) -> D2 (Time: 1116.20 (E_pre: 124.0, G: - , E_bar: -))
- Vehicle D2_V13 schedule: O2 (Time: 916.66, E_bar: 218.7) -> T157 (Time: 928.17 (E_pre: 208.7, G: 18.6, E_bar: 190.2)) -> T162 (Time: 985.00 (E_pre: 190.2, G: 18.6, E_bar: 171.6)) -> D2 (Time: 1052.88 (E_pre: 161.7, G: - , E_bar: -))
- Vehicle D2_V2 schedule: O2 (Time: 929.72, E_bar: 219.2) -> T172 (Time: 941.23 (E_pre: 209.3, G: 27.7, E_bar: 181.6)) -> T176 (Time: 1015.00 (E_pre: 181.6, G: 27.7, E_bar: 153.9)) -> D2 (Time: 1100.62 (E_pre: 143.9, G: - , E_bar: -))
- Vehicle D1_V4 schedule: O1 (Time: 929.76, E_bar: 350.0) -> T174 (Time: 945.89 (E_pre: 336.0, G: 27.7, E_bar: 308.3)) -> T170 (Time: 1020.00 (E_pre: 308.3, G: 27.7, E_bar: 280.6)) -> D1 (Time: 1109.89 (E_pre: 266.7, G: - , E_bar: -))
- Vehicle D2_V4 schedule: O2 (Time: 945.62, E_bar: 222.3) -> T178 (Time: 953.43 (E_pre: 215.5, G: 15.7, E_bar: 199.8)) -> D2 (Time: 1016.96 (E_pre: 189.9, G: - , E_bar: -))
- Vehicle D2_V14 schedule: O2 (Time: 963.53, E_bar: 317.6) -> T177 (Time: 993.82 (E_pre: 291.3, G: 27.7, E_bar: 263.6)) -> T173 (Time: 1067.93 (E_pre: 263.6, G: 27.7, E_bar: 235.9)) -> D2 (Time: 1171.99 (E_pre: 209.7, G: - , E_bar: -))
- Vehicle D2_V5 schedule: O2 (Time: 975.30, E_bar: 266.3) -> T171 (Time: 986.81 (E_pre: 256.3, G: 27.7, E_bar: 228.6)) -> T175 (Time: 1060.58 (E_pre: 228.6, G: 27.7, E_bar: 200.9)) -> D2 (Time: 1146.20 (E_pre: 190.9, G: - , E_bar: -))
- Vehicle D2_V9 schedule: O2 (Time: 991.39, E_bar: 163.8) -> T160 (Time: 1002.90 (E_pre: 153.9, G: 18.6, E_bar: 135.3)) -> T166 (Time: 1059.73 (E_pre: 135.3, G: 18.6, E_bar: 116.8)) -> D2 (Time: 1127.61 (E_pre: 106.8, G: - , E_bar: -))
- Vehicle D1_V15 schedule: O1 (Time: 994.34, E_bar: 298.3) -> T152 (Time: 1000.00 (E_pre: 293.4, G: 18.6, E_bar: 274.9)) -> T146 (Time: 1060.29 (E_pre: 274.9, G: 18.6, E_bar: 256.3)) -> D1 (Time: 1125.05 (E_pre: 251.4, G: - , E_bar: -))
- Vehicle D2_V4 schedule: O2 (Time: 1059.87, E_bar: 189.9) -> T184 (Time: 1074.71 (E_pre: 177.0, G: 18.6, E_bar: 158.5)) -> T180 (Time: 1135.00 (E_pre: 158.5, G: 18.6, E_bar: 139.9)) -> T185 (Time: 1194.10 (E_pre: 139.9, G: 18.6, E_bar: 121.3)) -> D2 (Time: 1265.91 (E_pre: 111.4, G: - , E_bar: -))
- Vehicle D2_V11 schedule: O2 (Time: 1077.52, E_bar: 317.6) -> T186 (Time: 1089.03 (E_pre: 307.6, G: 18.6, E_bar: 289.0)) -> T191 (Time: 1145.86 (E_pre: 289.0, G: 18.6, E_bar: 270.5)) -> T193 (Time: 1223.63 (E_pre: 251.9, G: 18.6, E_bar: 233.4)) -> T190 (Time: 1280.00 (E_pre: 233.4, G: 18.6, E_bar: 214.8)) -> D2 (Time: 1351.67 (E_pre: 202.0, G: - , E_bar: -))
- Vehicle D2_V13 schedule: O2 (Time: 1079.39, E_bar: 161.7) -> T179 (Time: 1090.90 (E_pre: 151.7, G: 18.6, E_bar: 133.1)) -> T183 (Time: 1150.00 (E_pre: 133.1, G: 18.6, E_bar: 114.6)) -> D2 (Time: 1221.80 (E_pre: 104.6, G: - , E_bar: -))
- Vehicle D1_V12 schedule: O1 (Time: 1092.98, E_bar: 170.3) -> T194 (Time: 1098.63 (E_pre: 165.4, G: 18.6, E_bar: 146.9)) -> T189 (Time: 1155.00 (E_pre: 146.9, G: 18.6, E_bar: 128.3)) -> D1 (Time: 1217.49 (E_pre: 123.4, G: - , E_bar: -))
- Vehicle D1_V11 schedule: O1 (Time: 1098.88, E_bar: 350.0) -> T202 (Time: 1115.00 (E_pre: 336.0, G: 27.7, E_bar: 308.3)) -> T195 (Time: 1191.23 (E_pre: 308.3, G: 27.7, E_bar: 280.6)) -> T203 (Time: 1265.00 (E_pre: 280.6, G: 27.7, E_bar: 252.9)) -> T196 (Time: 1339.11 (E_pre: 252.9, G: 27.7, E_bar: 225.2)) -> D1 (Time: 1429.01 (E_pre: 211.3, G: - , E_bar: -))
- Vehicle D2_V7 schedule: O2 (Time: 1103.49, E_bar: 156.4) -> T187 (Time: 1115.00 (E_pre: 146.4, G: 18.6, E_bar: 127.9)) -> D2 (Time: 1186.67 (E_pre: 115.0, G: - , E_bar: -))
- Vehicle D1_V4 schedule: O1 (Time: 1109.89, E_bar: 266.7) -> T182 (Time: 1115.55 (E_pre: 261.8, G: 18.6, E_bar: 243.2)) -> T181 (Time: 1175.84 (E_pre: 243.2, G: 18.6, E_bar: 224.6)) -> D1 (Time: 1240.60 (E_pre: 219.7, G: - , E_bar: -))
- Vehicle D1_V5 schedule: O1 (Time: 1112.42, E_bar: 350.0) -> T199 (Time: 1130.00 (E_pre: 334.8, G: 27.7, E_bar: 307.1)) -> T205 (Time: 1203.77 (E_pre: 307.1, G: 27.7, E_bar: 279.4)) -> T200 (Time: 1277.88 (E_pre: 279.4, G: 27.7, E_bar: 251.7)) -> D1 (Time: 1367.78 (E_pre: 237.7, G: - , E_bar: -))
- Vehicle D1_V7 schedule: O1 (Time: 1140.99, E_bar: 350.0) -> T206 (Time: 1157.12 (E_pre: 336.0, G: 27.7, E_bar: 308.3)) -> T201 (Time: 1231.23 (E_pre: 308.3, G: 27.7, E_bar: 280.6)) -> T207 (Time: 1305.00 (E_pre: 280.6, G: 27.7, E_bar: 252.9)) -> D1 (Time: 1396.70 (E_pre: 237.7, G: - , E_bar: -))
- Vehicle D2_V5 schedule: O2 (Time: 1146.20, E_bar: 190.9) -> T197 (Time: 1157.71 (E_pre: 180.9, G: 27.7, E_bar: 153.2)) -> T204 (Time: 1231.48 (E_pre: 153.2, G: 27.7, E_bar: 125.5)) -> D2 (Time: 1317.11 (E_pre: 115.6, G: - , E_bar: -))
- Vehicle D1_V15 schedule: O1 (Time: 1172.98, E_bar: 251.4) -> T192 (Time: 1178.63 (E_pre: 246.5, G: 18.6, E_bar: 228.0)) -> T188 (Time: 1235.00 (E_pre: 228.0, G: 18.6, E_bar: 209.4)) -> D1 (Time: 1297.49 (E_pre: 204.5, G: - , E_bar: -))
- Vehicle D2_V14 schedule: O2 (Time: 1203.49, E_bar: 209.7) -> T208 (Time: 1215.00 (E_pre: 199.7, G: 18.6, E_bar: 181.1)) -> D2 (Time: 1288.94 (E_pre: 168.3, G: - , E_bar: -))
- Vehicle D1_V15 schedule: O1 (Time: 1297.49, E_bar: 204.5) -> T198 (Time: 1315.07 (E_pre: 189.3, G: 27.7, E_bar: 161.6)) -> D1 (Time: 1404.96 (E_pre: 147.6, G: - , E_bar: -))
"""

# schedule_input = """
# - Vehicle D1_V1 schedule: O1 (Time: 336.02, E_bar: 350.0) -> T4 (Time: 341.67 (E_pre: 345.1, G: 30.4, E_bar: 314.7)) -> T2 (Time: 400.00 (E_pre: 314.7, G: 28.3, E_bar: 286.4)) -> T5 (Time: 451.67 (E_pre: 286.4, G: 30.4, E_bar: 256.0)) -> T14 (Time: 524.33 (E_pre: 256.0, G: 39.1, E_bar: 216.9)) -> T18 (Time: 596.67 (E_pre: 216.9, G: 41.0, E_bar: 175.9)) -> T15 (Time: 670.00 (E_pre: 175.9, G: 39.1, E_bar: 136.8)) -> D1 (Time: 758.46 (E_pre: 122.8, G: -350.0, E_bar: 0.0))
# - Vehicle D1_V2 schedule: O1 (Time: 283.21, E_bar: 350.0) -> T16 (Time: 299.33 (E_pre: 336.0, G: 41.0, E_bar: 295.0)) -> T13 (Time: 372.67 (E_pre: 295.0, G: 39.1, E_bar: 255.9)) -> T17 (Time: 445.00 (E_pre: 255.9, G: 41.0, E_bar: 214.9)) -> T7 (Time: 535.00 (E_pre: 214.9, G: 27.1, E_bar: 187.8)) -> T6 (Time: 595.00 (E_pre: 187.8, G: 30.4, E_bar: 157.4)) -> T8 (Time: 655.00 (E_pre: 157.4, G: 27.1, E_bar: 130.3)) -> C1 (Time: 720.65 (E_pre: 125.4, G: -224.6, E_bar: 350.0)) -> T9 (Time: 810.00 (E_pre: 334.8, G: 27.1, E_bar: 307.6)) -> D1 (Time: 875.65 (E_pre: 302.7, G: -350.0, E_bar: 0.0))
# - Vehicle D1_V3 schedule: O1 (Time: 1244.34, E_bar: 350.0) -> T12 (Time: 1250.00 (E_pre: 345.1, G: 28.3, E_bar: 316.8)) -> D1 (Time: 1321.58 (E_pre: 301.6, G: -350.0, E_bar: 0.0))
# - Vehicle D1_V4 schedule: O1 (Time: 990.34, E_bar: 350.0) -> T11 (Time: 996.00 (E_pre: 345.1, G: 28.3, E_bar: 316.8)) -> T3 (Time: 1050.00 (E_pre: 316.8, G: 28.3, E_bar: 288.5)) -> D1 (Time: 1107.32 (E_pre: 283.6, G: -350.0, E_bar: 0.0))
# - Vehicle D2_V1 schedule: O2 (Time: 298.49, E_bar: 350.0) -> T1 (Time: 310.00 (E_pre: 340.0, G: 28.3, E_bar: 311.7)) -> C1 (Time: 367.32 (E_pre: 306.8, G: -43.2, E_bar: 350.0)) -> T10 (Time: 387.38 (E_pre: 345.1, G: 28.3, E_bar: 316.8)) -> D2 (Time: 452.89 (E_pre: 306.8, G: -350.0, E_bar: 0.0))
# - Vehicle D2_V4 schedule: O2 (Time: 298.17, E_bar: 350.0) -> T19 (Time: 309.68 (E_pre: 340.0, G: 22.0, E_bar: 318.0)) -> T22 (Time: 353.34 (E_pre: 318.0, G: 20.6, E_bar: 297.3)) -> T20 (Time: 405.00 (E_pre: 297.3, G: 22.0, E_bar: 275.3)) -> T23 (Time: 449.68 (E_pre: 275.3, G: 20.6, E_bar: 254.7)) -> T21 (Time: 501.33 (E_pre: 254.7, G: 22.0, E_bar: 232.6)) -> T24 (Time: 545.00 (E_pre: 232.6, G: 20.6, E_bar: 212.0)) -> D2 (Time: 608.17 (E_pre: 202.0, G: -350.0, E_bar: 0.0))
# """

# schedule_input = """
# - Vehicle D1_V1 schedule: O1 (Time: 283.21, E_bar: 350.0) -> T4 (Time: 299.33 (E_pre: 336.0, G: 27.7, E_bar: 308.3)) -> T1 (Time: 372.67 (E_pre: 308.3, G: 27.7, E_bar: 280.6)) -> T5 (Time: 445.00 (E_pre: 280.6, G: 27.7, E_bar: 252.9)) -> T2 (Time: 518.33 (E_pre: 252.9, G: 27.7, E_bar: 225.2)) -> T6 (Time: 590.67 (E_pre: 225.2, G: 27.7, E_bar: 197.5)) -> T3 (Time: 670.00 (E_pre: 197.5, G: 27.7, E_bar: 169.8)) -> D1 (Time: 758.46 (E_pre: 155.9, G: - , E_bar: -))
# - Vehicle D2_V5 schedule: O2 (Time: 298.17, E_bar: 350.0) -> T7 (Time: 309.68 (E_pre: 340.0, G: 15.7, E_bar: 324.3)) -> T10 (Time: 353.34 (E_pre: 324.3, G: 15.7, E_bar: 308.7)) -> T8 (Time: 405.00 (E_pre: 308.7, G: 15.7, E_bar: 293.0)) -> T11 (Time: 448.67 (E_pre: 293.0, G: 15.7, E_bar: 277.3)) -> T9 (Time: 501.33 (E_pre: 277.3, G: 15.7, E_bar: 261.6)) -> T12 (Time: 545.00 (E_pre: 261.6, G: 15.7, E_bar: 246.0)) -> D2 (Time: 608.17 (E_pre: 236.0, G: - , E_bar: -))
# - Vehicle D2_V4 schedule: O2 (Time: 298.49, E_bar: 350.0) -> T13 (Time: 310.00 (E_pre: 340.0, G: 18.6, E_bar: 321.5)) -> C1 (Time: 368.20 (E_pre: 316.6, G: -33.4, E_bar: 350.0)) -> T20 (Time: 385.00 (E_pre: 345.1, G: 18.6, E_bar: 326.5)) -> D2 (Time: 450.51 (E_pre: 316.6, G: - , E_bar: -))
# - Vehicle D1_V3 schedule: O1 (Time: 336.02, E_bar: 350.0) -> T15 (Time: 341.67 (E_pre: 345.1, G: 18.6, E_bar: 326.5)) -> T14 (Time: 400.00 (E_pre: 326.5, G: 18.6, E_bar: 308.0)) -> T16 (Time: 455.00 (E_pre: 308.0, G: 18.6, E_bar: 289.4)) -> T18 (Time: 535.00 (E_pre: 289.4, G: 18.6, E_bar: 270.9)) -> T17 (Time: 595.00 (E_pre: 270.9, G: 18.6, E_bar: 252.3)) -> T19 (Time: 655.00 (E_pre: 252.3, G: 18.6, E_bar: 233.8)) -> D1 (Time: 720.66 (E_pre: 228.9, G: - , E_bar: -))
# - Vehicle D1_V1 schedule: O1 (Time: 792.42, E_bar: 155.9) -> T21 (Time: 810.00 (E_pre: 140.6, G: 18.6, E_bar: 122.1)) -> C1 (Time: 875.66 (E_pre: 117.2, G: -232.8, E_bar: 350.0)) -> T22 (Time: 985.00 (E_pre: 345.1, G: 18.6, E_bar: 326.5)) -> D1 (Time: 1056.58 (E_pre: 311.3, G: - , E_bar: -))
# - Vehicle D1_V5 schedule: O1 (Time: 1032.42, E_bar: 350.0) -> T23 (Time: 1050.00 (E_pre: 334.8, G: 18.6, E_bar: 316.2)) -> D1 (Time: 1107.32 (E_pre: 311.3, G: - , E_bar: -))
# - Vehicle D1_V5 schedule: O1 (Time: 1244.34, E_bar: 311.3) -> T24 (Time: 1250.00 (E_pre: 306.4, G: 18.6, E_bar: 287.8)) -> D1 (Time: 1321.58 (E_pre: 272.6, G: - , E_bar: -))
# """

# --- Execution ---
parsed_data = parse_schedule_text(schedule_input)
latex_output = generate_latex_table(parsed_data)

print("---")
print("## 📊 Generated LaTeX Table Output")
print("```latex")
print(latex_output)
print("```")

def sanitize_latex(s: str) -> str:
    # Replace non-breaking spaces with normal spaces
    s = s.replace("\u00A0", " ")
    # Replace tabs with 4 spaces (optional)
    s = s.replace("\t", "    ")
    return s

latex_output = sanitize_latex(latex_output)

with open("schedule_table.tex", "w", encoding="utf-8") as f:
    f.write(latex_output)
