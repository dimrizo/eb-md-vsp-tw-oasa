import random
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from utilities import haversine

# --- Data Structures to Hold Instance Information ---

@dataclass(frozen=True)
class Point:
    """Represents a physical location with x, y coordinates."""
    x: float
    y: float

# --- NEW DATA STRUCTURE: Charging Station with Time Window ---
@dataclass
class ChargingStation:
    """Represents a charging location with its properties."""
    id: int
    location: Point
    # NEW FIELD: Time window for service [lower, upper] (e.g., [0, 1440])
    time_window: Tuple[float, float] 

@dataclass
class Trip:
    """Represents a single timetabled trip."""
    id: int
    start_point: Point
    end_point: Point
    start_time: float
    end_time: float
    trip_type: str
    # Optional fields for extensions
    start_time_window: Optional[Tuple[float, float]] = None
    # NEW FIELD: Energy replenishment/gain at node i (eta_i)
    eta: float = 0.0  # Default value is 0 (no gain)

@dataclass
class Depot:
    """Represents a depot with a location and number of vehicles."""
    id: int
    location: Point
    vehicle_count: int

@dataclass
class ProblemInstance:
    """A container for a fully generated problem instance."""
    grid_size: Tuple[int, int]
    trips: List[Trip]
    depots: List[Depot]
    relief_points: List[Point]
    # MODIFIED TYPE: Now a list of ChargingStation objects
    charging_stations: List[ChargingStation] = field(default_factory=list)

# --- Base Class for Instance Generation ---

class CarpanetoInstanceGenerator:
    """
    Generates synthetic test instances based on the procedure from Carpaneto et al. (1989).
    This is the foundational method upon which later procedures are built.
    """
    def __init__(self, n_trips: int, n_depots: int, n_relief_points: int, problem_class: str = 'A'):
        if problem_class not in ['A', 'B']:
            raise ValueError("Problem class must be 'A' or 'B'.")

        self.n_trips = n_trips
        self.n_depots = n_depots
        self.n_relief_points = n_relief_points
        self.problem_class = problem_class
        self.grid_size = (60, 60)
        self.relief_points: List[Point] = []
        self.trips: List[Trip] = []
        self.depots: List[Depot] = []

    def _calculate_distance(self, p1: Point, p2: Point) -> float:
        """
        Calculates Haversine distance.
        """
        lat_1, lon_1 = p1.y, p1.x
        lat_2, lon_2 = p2.y, p2.x

        # Calculate Haversine distance in meters
        distance_meters = haversine.main(lat_1, lon_1, lat_2, lon_2)
                
        return distance_meters

    def _calculate_travel_time(self, p1: Point, p2: Point) -> float:
        """
        Calculates Haversine distance and returns the required travel time (in minutes).
        Assumes average speed is 800,000 meters/minute (placeholder constant).
        """
        AVG_U_METERS_PER_MINUTE = 800000  # Placeholder speed constant
        
        lat_1, lon_1 = p1.y, p1.x
        lat_2, lon_2 = p2.y, p2.x

        # STEP 1: Calculate Haversine distance in meters
        distance_meters = haversine.main(lat_1, lon_1, lat_2, lon_2)
        
        # STEP 2: Calculate travel time in minutes (Time = Distance / Speed)
        travel_time_minutes = distance_meters / AVG_U_METERS_PER_MINUTE
        
        return travel_time_minutes

    def _generate_relief_points(self):
        """Generates v relief points uniformly in the grid."""
        width, height = self.grid_size
        for _ in range(self.n_relief_points):
            self.relief_points.append(Point(random.uniform(0, width), random.uniform(0, height)))

    def _generate_short_trip(self, trip_id: int) -> Trip:
        """Generates a short, urban-style trip."""
        start_point, end_point = random.sample(self.relief_points, 2)
        
        # Sample start time based on peak hour probabilities
        population = [(420, 480), (480, 1020), (1020, 1080)]
        weights = [0.15, 0.70, 0.15]
        start_interval = random.choices(population, weights, k=1)[0]
        s_j = random.uniform(start_interval[0], start_interval[1])
        
        travel_time = self._calculate_travel_time(start_point, end_point)
        e_j = s_j + travel_time + 5 + random.uniform(0, 40)

        return Trip(trip_id, start_point, end_point, s_j, e_j, "short")

    def _generate_long_trip(self, trip_id: int) -> Trip:
        """Generates a long, extra-urban style trip."""
        # Start and end points coincide for long trips
        start_point, end_point = random.sample(self.relief_points, 2)

        s_j = random.uniform(300, 1200)
        e_j = random.uniform(s_j + 180, s_j + 300)
        
        return Trip(trip_id, start_point, end_point, s_j, e_j, "long")

    def _generate_trips(self):
        """Generates n trips, mixing short and long types."""
        for i in range(1, self.n_trips + 1):
            if random.random() < 0.40: # 40% chance for a short trip
                self.trips.append(self._generate_short_trip(i))
            else: # 60% chance for a long trip
                self.trips.append(self._generate_long_trip(i))

    def _generate_depots(self):
        """Generates m depots according to the specified problem class."""
        width, height = self.grid_size
        
        # Calculate vehicles per depot based on the formula
        min_vehicles = math.floor(3 + self.n_trips / (3 * self.n_depots))
        max_vehicles = math.floor(3 + self.n_trips / (2 * self.n_depots))

        if self.problem_class == 'A':
            for i in range(1, self.n_depots + 1):
                location = Point(random.uniform(0, width), random.uniform(0, height))
                vehicle_count = random.randint(min_vehicles, max_vehicles)
                self.depots.append(Depot(i, location, vehicle_count))
        elif self.problem_class == 'B':
            if self.n_depots == 2:
                locations = [Point(0, 0), Point(width, height)]
            elif self.n_depots == 3:
                locations = [Point(0, 0), Point(width, height),
                             Point(random.uniform(0, width), random.uniform(0, height))]
            else: # Default for other cases
                locations = [Point(random.uniform(0, width), random.uniform(0, height)) for _ in range(self.n_depots)]
            
            for i, loc in enumerate(locations[:self.n_depots], 1):
                vehicle_count = random.randint(min_vehicles, max_vehicles)
                self.depots.append(Depot(i, loc, vehicle_count))

    def generate(self) -> ProblemInstance:
        """
        Executes the full generation procedure and returns a ProblemInstance object.
        """
        self._generate_relief_points()
        self._generate_trips()
        self._generate_depots()

        return ProblemInstance(
            grid_size=self.grid_size,
            trips=self.trips,
            depots=self.depots,
            relief_points=self.relief_points
        )

# --- Subclasses for Extended Procedures ---

class DesaulniersInstanceGenerator(CarpanetoInstanceGenerator):
    """
    Extends the Carpaneto generator to add time windows, as per Desaulniers et al. (1998).
    """
    def _add_time_windows_to_trips(self):
        """Adds a +/- 10 minute time window to each trip's start time."""
        for trip in self.trips:
            trip.start_time_window = (trip.start_time - 10, trip.start_time + 10)

    def generate(self) -> ProblemInstance:
        """
        Generates the base instance and then adds the time windows.
        """
        # Call the parent class's generate method
        base_instance = super().generate()
        # Add the modification
        self._add_time_windows_to_trips()
        
        return base_instance

class GkiotsalitisInstanceGenerator(DesaulniersInstanceGenerator): # <-- 1. Change Inheritance
    """
    Extends the Desaulniers generator for Electric Bus problems, as per Gkiotsalitis & Iliopoulou (2023).
    Includes charging stations and wide time windows.
    """
    def __init__(self, n_trips: int, n_depots: int, n_relief_points: int,
                 n_charging_stations: int, problem_class: str = 'A'):
        # Call the Desaulniers parent's init
        super().__init__(n_trips, n_depots, n_relief_points, problem_class)
        self.n_charging_stations = n_charging_stations
        self.charging_stations: List[ChargingStation] = []
        # Override grid_size to be explicitly in km
        self.grid_size = (60, 60) # 60 km x 60 km

    def _generate_charging_stations(self):
        """
        Generates random locations for charging stations and assigns a wide time window [0, 1440].
        """
        width, height = self.grid_size
        
        # 24 hours in minutes
        TIME_WINDOW = (0.0, 1440.0)
        
        for i in range(1, self.n_charging_stations + 1):
            location = Point(random.uniform(0, width), random.uniform(0, height))
            
            self.charging_stations.append(
                ChargingStation(
                    id=i,
                    location=location,
                    time_window=TIME_WINDOW
                )
            )
            
    def _add_wide_time_windows_to_trips(self):
        """Adds a wider time window based on problem size."""
        if self.n_trips <= 10:
            width = 200
        elif self.n_trips <= 20:
            width = 300
        else:
            width = 400
            
        for trip in self.trips:
            # Overwrites the narrow time window set by the Desaulniers parent
            trip.start_time_window = (trip.start_time, trip.start_time + width)

    def _add_eta_to_trips(self):
        """
        Calculates energy consumed per trip (eta_i).
        The value is a small random amount proportional to the Haversine distance of the trip.
        """
        # Base factor for energy consumption (e.g., related to regenerative braking)
        THETA_FACTOR = 0.00001 # Consumption factor
        
        for trip in self.trips:
            # Distance is a better proxy for energy gain than time
            distance = self._calculate_distance(trip.start_point, trip.end_point)
            
            # Gain is proportional to distance and a random multiplier
            base_gain = distance * THETA_FACTOR
            
            # Assign the calculated gain (eta_i)
            # Assuming eta > 0 means energy is replenished/gained, and consumption is handled elsewhere
            trip.eta = base_gain if base_gain > 0 else 0.0

    def generate(self) -> ProblemInstance:
        """
        Generates the base instance, then adds charging stations, wide time windows, and ETA.
        """
        # 1. Generate base instance (sets up self.trips, self.depots, etc.)
        # This calls Desaulniers.generate(), which runs Carpaneto's generation AND
        # Desaulniers._add_time_windows_to_trips() (narrow +/- 10 min window).
        super().generate()
        
        # 2. Add the specific modifications for this problem type
        self._generate_charging_stations()
        
        # Call the wide time window logic, which overwrites the narrow windows
        # self._add_wide_time_windows_to_trips() # <-- 2. Call the custom wide time window logic
        self._add_time_windows_to_trips()
        
        self._add_eta_to_trips() # Add the ETA values to each trip
        
        # 3. Return a complete ProblemInstance with the extra data
        return ProblemInstance(
            grid_size=self.grid_size,
            trips=self.trips,
            depots=self.depots,
            relief_points=self.relief_points,
            charging_stations=self.charging_stations
        )