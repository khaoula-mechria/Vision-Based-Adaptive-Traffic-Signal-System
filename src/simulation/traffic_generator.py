"""
Traffic generator for SUMO simulation
"""

import os
import random
from typing import Dict, List, Optional, Tuple


class TrafficGenerator:
    """
    Generates traffic demand files for SUMO simulation.
    
    Creates realistic traffic patterns including:
    - Vehicle flows with varying densities
    - Pedestrian crossing demand
    - Emergency vehicle events
    """
    
    def __init__(
        self,
        seed: int = 42,
    ):
        """
        Initialize traffic generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
        
        # Default routes
        self.routes = {
            "NS": {"from": "north_in", "to": "south_out"},
            "SN": {"from": "south_in", "to": "north_out"},
            "EW": {"from": "east_in", "to": "west_out"},
            "WE": {"from": "west_in", "to": "east_out"},
            "NE": {"from": "north_in", "to": "east_out"},
            "NW": {"from": "north_in", "to": "west_out"},
            "SE": {"from": "south_in", "to": "east_out"},
            "SW": {"from": "south_in", "to": "west_out"},
            "EN": {"from": "east_in", "to": "north_out"},
            "ES": {"from": "east_in", "to": "south_out"},
            "WN": {"from": "west_in", "to": "north_out"},
            "WS": {"from": "west_in", "to": "south_out"},
        }
        
        # Vehicle types
        self.vehicle_types = {
            "car": {"length": 4.5, "maxSpeed": 13.89, "probability": 0.7},
            "bus": {"length": 12.0, "maxSpeed": 11.11, "probability": 0.1},
            "truck": {"length": 8.0, "maxSpeed": 11.11, "probability": 0.1},
            "motorcycle": {"length": 2.0, "maxSpeed": 16.67, "probability": 0.1},
        }
        
        # Emergency vehicle types
        self.emergency_types = {
            "ambulance": {"length": 6.0, "maxSpeed": 19.44},
            "fire_truck": {"length": 10.0, "maxSpeed": 16.67},
            "police": {"length": 5.0, "maxSpeed": 22.22},
        }
    
    def generate_route_file(
        self,
        output_path: str,
        duration: int = 3600,
        vehicle_rate: float = 1.0,
        pedestrian_rate: float = 0.3,
        emergency_rate: float = 0.01,
        density: str = "medium",
    ) -> str:
        """
        Generate SUMO route file.
        
        Args:
            output_path: Path to save route file
            duration: Simulation duration in seconds
            vehicle_rate: Base vehicle generation rate
            pedestrian_rate: Pedestrian generation probability
            emergency_rate: Emergency vehicle probability
            density: Traffic density ('low', 'medium', 'high')
            
        Returns:
            Path to generated file
        """
        # Adjust rates based on density
        density_multipliers = {"low": 0.5, "medium": 1.0, "high": 1.5}
        multiplier = density_multipliers.get(density, 1.0)
        vehicle_rate *= multiplier
        pedestrian_rate *= multiplier
        
        # Generate XML content
        content = self._generate_routes_xml(
            duration=duration,
            vehicle_rate=vehicle_rate,
            pedestrian_rate=pedestrian_rate,
            emergency_rate=emergency_rate,
        )
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(content)
        
        return output_path
    
    def _generate_routes_xml(
        self,
        duration: int,
        vehicle_rate: float,
        pedestrian_rate: float,
        emergency_rate: float,
    ) -> str:
        """Generate routes XML content"""
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
            'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">',
            '',
            '    <!-- Vehicle Types -->',
        ]
        
        # Add vehicle types
        for vtype, props in self.vehicle_types.items():
            lines.append(
                f'    <vType id="{vtype}" length="{props["length"]}" '
                f'maxSpeed="{props["maxSpeed"]}" accel="2.6" decel="4.5"/>'
            )
        
        # Add emergency vehicle types
        for vtype, props in self.emergency_types.items():
            lines.append(
                f'    <vType id="{vtype}" length="{props["length"]}" '
                f'maxSpeed="{props["maxSpeed"]}" accel="3.0" decel="5.0" '
                f'vClass="emergency"/>'
            )
        
        lines.append('')
        lines.append('    <!-- Route Definitions -->')
        
        # Add routes
        for route_id, route_def in self.routes.items():
            lines.append(
                f'    <route id="route_{route_id}" edges="{route_def["from"]} '
                f'intersection {route_def["to"]}"/>'
            )
        
        lines.append('')
        lines.append('    <!-- Vehicle Flows -->')
        
        # Generate vehicle flows
        for route_id in self.routes.keys():
            for vtype, props in self.vehicle_types.items():
                flow_rate = vehicle_rate * props["probability"] / len(self.routes)
                if flow_rate > 0:
                    lines.append(
                        f'    <flow id="flow_{vtype}_{route_id}" '
                        f'type="{vtype}" route="route_{route_id}" '
                        f'begin="0" end="{duration}" '
                        f'probability="{flow_rate:.4f}"/>'
                    )
        
        lines.append('')
        lines.append('    <!-- Emergency Vehicles (random insertion) -->')
        
        # Generate emergency vehicle insertions
        num_emergencies = int(duration * emergency_rate)
        for i in range(num_emergencies):
            time = random.randint(0, duration)
            route_id = random.choice(list(self.routes.keys()))
            etype = random.choice(list(self.emergency_types.keys()))
            lines.append(
                f'    <vehicle id="emergency_{i}" type="{etype}" '
                f'route="route_{route_id}" depart="{time}"/>'
            )
        
        lines.append('')
        lines.append('</routes>')
        
        return '\n'.join(lines)
    
    def generate_network_file(
        self,
        output_path: str,
        intersection_size: float = 25.0,
        lane_width: float = 3.5,
        num_lanes: int = 2,
    ) -> str:
        """
        Generate SUMO network file for a standard intersection.
        
        Args:
            output_path: Path to save network file
            intersection_size: Size of intersection box
            lane_width: Width of each lane
            num_lanes: Number of lanes per direction
            
        Returns:
            Path to generated file
        """
        content = self._generate_network_xml(
            intersection_size=intersection_size,
            lane_width=lane_width,
            num_lanes=num_lanes,
        )
        
        with open(output_path, 'w') as f:
            f.write(content)
        
        return output_path
    
    def _generate_network_xml(
        self,
        intersection_size: float,
        lane_width: float,
        num_lanes: int,
    ) -> str:
        """Generate network XML content"""
        approach_length = 200.0
        offset = intersection_size / 2
        
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<net version="1.16" junctionCornerDetail="5">',
            '',
            '    <!-- Edges (Roads) -->',
            f'    <edge id="north_in" from="north" to="center" priority="1" numLanes="{num_lanes}"/>',
            f'    <edge id="south_in" from="south" to="center" priority="1" numLanes="{num_lanes}"/>',
            f'    <edge id="east_in" from="east" to="center" priority="1" numLanes="{num_lanes}"/>',
            f'    <edge id="west_in" from="west" to="center" priority="1" numLanes="{num_lanes}"/>',
            f'    <edge id="north_out" from="center" to="north" priority="1" numLanes="{num_lanes}"/>',
            f'    <edge id="south_out" from="center" to="south" priority="1" numLanes="{num_lanes}"/>',
            f'    <edge id="east_out" from="center" to="east" priority="1" numLanes="{num_lanes}"/>',
            f'    <edge id="west_out" from="center" to="west" priority="1" numLanes="{num_lanes}"/>',
            '',
            '    <!-- Junctions -->',
            f'    <junction id="center" type="traffic_light" x="0" y="0"/>',
            f'    <junction id="north" type="dead_end" x="0" y="{approach_length + offset}"/>',
            f'    <junction id="south" type="dead_end" x="0" y="{-approach_length - offset}"/>',
            f'    <junction id="east" type="dead_end" x="{approach_length + offset}" y="0"/>',
            f'    <junction id="west" type="dead_end" x="{-approach_length - offset}" y="0"/>',
            '',
            '    <!-- Traffic Light Program -->',
            '    <tlLogic id="center" type="static" programID="adaptive">',
            '        <phase duration="30" state="GGrrGGrr"/>',  # NS Green
            '        <phase duration="5" state="yyrryyrr"/>',   # NS Yellow
            '        <phase duration="30" state="rrGGrrGG"/>',  # EW Green
            '        <phase duration="5" state="rryyrryy"/>',   # EW Yellow
            '    </tlLogic>',
            '',
            '</net>',
        ]
        
        return '\n'.join(lines)
    
    def generate_pedestrian_flows(
        self,
        output_path: str,
        duration: int = 3600,
        rate: float = 0.1,
    ) -> str:
        """
        Generate pedestrian flow file.
        
        Args:
            output_path: Path to save file
            duration: Simulation duration
            rate: Pedestrian generation rate
            
        Returns:
            Path to generated file
        """
        crossings = [
            ("north_cross", "north_sidewalk_w", "north_sidewalk_e"),
            ("south_cross", "south_sidewalk_w", "south_sidewalk_e"),
            ("east_cross", "east_sidewalk_n", "east_sidewalk_s"),
            ("west_cross", "west_sidewalk_n", "west_sidewalk_s"),
        ]
        
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<routes>',
            '    <personFlow id="ped_flow_NS" begin="0" end="{duration}" '
            f'probability="{rate}">',
            '        <walk from="north_sidewalk" to="south_sidewalk"/>',
            '    </personFlow>',
            '</routes>',
        ]
        
        content = '\n'.join(lines)
        
        with open(output_path, 'w') as f:
            f.write(content)
        
        return output_path
