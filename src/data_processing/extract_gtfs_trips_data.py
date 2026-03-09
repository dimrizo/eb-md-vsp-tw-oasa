# National Technical University of Athens
# Railways & transport Lab

import pandas as pd
import os

def process_gtfs_data(day, gtfs_folder_path, number_of_removed_stops=0):
    trips_file_path = os.path.join(gtfs_folder_path, 'trips.txt')
    stop_times_file_path = os.path.join(gtfs_folder_path, 'stop_times.txt')
    stops_file_path = os.path.join(gtfs_folder_path, 'stops.txt')
    calendar_file_path = os.path.join(gtfs_folder_path, 'calendar.txt')

    trips_df = pd.read_csv(trips_file_path)
    stop_times_df = pd.read_csv(stop_times_file_path)
    stops_df = pd.read_csv(stops_file_path)

    total_trips_count = trips_df['trip_id'].nunique()

    if os.path.exists(calendar_file_path):
        calendar_df = pd.read_csv(calendar_file_path)
        weekday_service_ids = calendar_df[(calendar_df[day] == 1)]['service_id'].tolist()
        weekday_trips_df = trips_df[trips_df['service_id'].isin(weekday_service_ids)]
        weekday_trips_count = weekday_trips_df['trip_id'].nunique()
    else:
        weekday_trips_df = trips_df
        weekday_trips_count = total_trips_count

    merged_df = pd.merge(stop_times_df, weekday_trips_df, on='trip_id')

    def time_to_minutes(t):
        h, m, s = map(int, t.split(':'))
        return h * 60 + m + s / 60

    grouped = merged_df.groupby(['route_id', 'direction_id', 'trip_id'])

    results = []

    # stop_id -> [lat, lon]
    stop_id_to_coords = stops_df.set_index('stop_id')[['stop_lat', 'stop_lon']] \
                                .apply(lambda x: [x['stop_lat'], x['stop_lon']], axis=1) \
                                .to_dict()

    # per-trip stop coords: {route_id: {direction_id: {trip_id: [[lat,lon], ...]}}}
    trip_stop_coords = {}

    for (route_id, direction_id, trip_id), group in grouped:
        group = group.sort_values(by='stop_sequence')

        # apply stop removal first
        if direction_id == 0:
            if number_of_removed_stops > 0:
                group = group.iloc[:-number_of_removed_stops] if len(group) > number_of_removed_stops else group.iloc[0:0]
        elif direction_id == 1:
            if number_of_removed_stops > 0:
                group = group.iloc[number_of_removed_stops:] if len(group) > number_of_removed_stops else group.iloc[0:0]

        if group.empty:
            continue

        # build stop coords sequence for this trip
        coords_seq = []
        for sid in group['stop_id']:
            if sid in stop_id_to_coords:
                coords_seq.append(stop_id_to_coords[sid])

        if not coords_seq:
            continue

        if route_id not in trip_stop_coords:
            trip_stop_coords[route_id] = {}
        if direction_id not in trip_stop_coords[route_id]:
            trip_stop_coords[route_id][direction_id] = {}
        trip_stop_coords[route_id][direction_id][trip_id] = coords_seq

        start_time = group.iloc[0]['arrival_time']
        end_time = group.iloc[-1]['arrival_time']
        first_stop_id = group.iloc[0]['stop_id']
        last_stop_id = group.iloc[-1]['stop_id']
        start_minutes = time_to_minutes(start_time)
        end_minutes = time_to_minutes(end_time)
        travel_time = end_minutes - start_minutes
        if travel_time < 0:
            travel_time += 24 * 60

        results.append([route_id, direction_id, trip_id, start_minutes, travel_time, first_stop_id, last_stop_id])

    results_df = pd.DataFrame(
        results,
        columns=['route_id', 'direction_id', 'trip_id', 'start_minutes',
                 'travel_time_minutes', 'first_stop_id', 'last_stop_id']
    )
    results_df['travel_time_minutes'] = results_df['travel_time_minutes'].astype(float)

    route_info = {}

    for route_id, group in results_df.groupby('route_id'):
        # direction 0
        go_group = group[group['direction_id'] == 0].sort_values(by='start_minutes')
        go_times = go_group['start_minutes'].tolist()
        go_stops = go_group['last_stop_id'].tolist()
        first_go_stops = go_group['first_stop_id'].tolist()
        go_trip_ids = go_group['trip_id'].tolist()

        # direction 1
        come_group = group[group['direction_id'] == 1].sort_values(by='start_minutes')
        come_times = come_group['start_minutes'].tolist()
        come_stops = come_group['last_stop_id'].tolist()
        first_come_stops = come_group['first_stop_id'].tolist()
        come_trip_ids = come_group['trip_id'].tolist()

        # coords for first/last stops (for your existing logic)
        go_stops_coords = [stop_id_to_coords[sid] for sid in go_stops if sid in stop_id_to_coords]
        come_stops_coords = [stop_id_to_coords[sid] for sid in come_stops if sid in stop_id_to_coords]
        first_go_stops_coords = [stop_id_to_coords[sid] for sid in first_go_stops if sid in stop_id_to_coords]
        first_come_stops_coords = [stop_id_to_coords[sid] for sid in first_come_stops if sid in stop_id_to_coords]

        # NEW: full stop sequences in the *same order* as go_times / come_times
        go_trip_stop_coords = []
        for tid in go_trip_ids:
            seq = trip_stop_coords.get(route_id, {}).get(0, {}).get(tid, [])
            if seq:
                go_trip_stop_coords.append(seq)

        come_trip_stop_coords = []
        for tid in come_trip_ids:
            seq = trip_stop_coords.get(route_id, {}).get(1, {}).get(tid, [])
            if seq:
                come_trip_stop_coords.append(seq)

        avg_go_travel_time = go_group['travel_time_minutes'].mean() if not go_group.empty else 0.0
        avg_come_travel_time = come_group['travel_time_minutes'].mean() if not come_group.empty else 0.0

        route_info[route_id] = [
            go_times,
            come_times,
            avg_go_travel_time,
            avg_come_travel_time,
            go_stops_coords,
            come_stops_coords,
            first_go_stops_coords,
            first_come_stops_coords,
            go_trip_stop_coords,
            come_trip_stop_coords
        ]

    return route_info

if __name__ == "__main__":
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # GTFS data
    route_to_analyse = 959 # Cyprus - 10090011 (i.e. 9), Athens - 959 (i.e. 550), 937 (i.e. 140)
    day = 'monday'
    gtfs_folder_path = os.path.join(current_dir, '..', '..', 'input', 'gtfs', 'oasa_september_2024')
    number_of_removed_stops = 0
    first_trip_to_consider_one_way = 0
    number_of_trips_one_way = 19
    first_trip_to_consider_other_way = 0
    number_of_trips_other_way = 19

    # extract respective gtfs data
    all_route_data = process_gtfs_data(day, gtfs_folder_path, number_of_removed_stops)
    route_data = all_route_data[route_to_analyse]

    number_of_trips = len(route_data[0]) + len(route_data[1])
    one_way_departures = route_data[0][first_trip_to_consider_one_way:number_of_trips_one_way]
    other_way_departures = route_data[1][first_trip_to_consider_other_way:number_of_trips_other_way]
    both_way_departures = one_way_departures + other_way_departures
    one_way_trip_time = route_data[2]
    other_way_trip_time = route_data[3]

    go_last_stops = route_data[4][first_trip_to_consider_one_way:number_of_trips_one_way]
    come_last_stops = route_data[5][first_trip_to_consider_other_way:number_of_trips_other_way]
    go_first_stops = route_data[6][first_trip_to_consider_one_way:number_of_trips_one_way]
    come_first_stops = route_data[7][first_trip_to_consider_other_way:number_of_trips_other_way]

    first_stops_one_way_lat = go_first_stops[0][0]
    first_stops_one_way_lon = go_first_stops[0][1]

    last_stop_one_way_lat = go_last_stops[0][0]
    last_stop_one_way_lon = go_last_stops[0][1]

    first_stops_other_way_lat = come_first_stops[0][0]
    first_stops_other_way_lon = come_first_stops[0][1]

    last_stop_other_way_lat = come_last_stops[0][0]
    last_stop_other_way_lon = come_last_stops[0][1]

    print("\r")
    print("Number of stops reduced in each direction: ", number_of_removed_stops)

    print("\r")
    print("First stop one-way: ")
    print(first_stops_one_way_lat, " ", first_stops_one_way_lon)
    print("\r")

    print("\r")
    print("Last stop one-way: ")
    print(last_stop_one_way_lat, " ", last_stop_one_way_lon)
    print("\r")

    print("\r")
    print("First stop other-way: ")
    print(first_stops_other_way_lat, " ", first_stops_other_way_lon)
    print("\r")

    print("\r")
    print("Last stop other-way: ")
    print(last_stop_other_way_lat, " ", last_stop_other_way_lon)
    print("\r")

    print("\r")
    print("One way trip time: ")
    print(one_way_trip_time)

    print("\r")
    print("Other way trip time: ")
    print(other_way_trip_time)
    print("\r")
