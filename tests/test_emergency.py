"""
Unit tests for emergency vehicle handling
"""

import pytest
import time

from src.emergency.emergency_state import (
    EmergencyVehicle,
    EmergencyState,
    EmergencyType,
    PreemptionState,
)
from src.emergency.emergency_handler import EmergencyVehicleHandler


class TestEmergencyVehicle:
    """Tests for EmergencyVehicle class"""
    
    def test_vehicle_creation(self):
        """Test emergency vehicle creation"""
        vehicle = EmergencyVehicle(
            track_id=1,
            vehicle_type=EmergencyType.AMBULANCE,
        )
        
        assert vehicle.track_id == 1
        assert vehicle.vehicle_type == EmergencyType.AMBULANCE
        assert vehicle.detection_count == 0
    
    def test_update_detection(self):
        """Test detection update"""
        vehicle = EmergencyVehicle(track_id=1)
        
        vehicle.update_detection(
            position=(100, 200),
            velocity=(10, -5),
            confidence=0.9,
            timestamp=time.time(),
        )
        
        assert vehicle.position == (100, 200)
        assert vehicle.velocity == (10, -5)
        assert vehicle.detection_count == 1
    
    def test_direction_inference(self):
        """Test approach direction inference from velocity"""
        vehicle = EmergencyVehicle(track_id=1)
        
        # Moving east (positive x)
        vehicle.update_detection((100, 200), (10, 2), 0.9, time.time())
        assert vehicle.approach_direction == "west"
        
        # Moving south (positive y)
        vehicle.update_detection((100, 200), (2, 10), 0.9, time.time())
        assert vehicle.approach_direction == "north"
    
    def test_confirmation(self):
        """Test confirmation after multiple detections"""
        vehicle = EmergencyVehicle(track_id=1)
        
        assert vehicle.is_confirmed(3) is False
        
        for i in range(3):
            vehicle.update_detection((100, 200), (10, 0), 0.9, time.time())
        
        assert vehicle.is_confirmed(3) is True


class TestEmergencyState:
    """Tests for EmergencyState class"""
    
    def test_initial_state(self):
        """Test initial state"""
        state = EmergencyState()
        
        assert state.preemption_state == PreemptionState.INACTIVE
        assert state.is_active() is False
    
    def test_start_preemption(self):
        """Test starting preemption"""
        state = EmergencyState()
        
        state.start_preemption(1, "north", time.time())
        
        assert state.preemption_state == PreemptionState.DETECTING
        assert state.active_vehicle_id == 1
        assert state.active_direction == "north"
        assert state.is_active() is True
    
    def test_activate_green_corridor(self):
        """Test green corridor activation"""
        state = EmergencyState()
        
        state.start_preemption(1, "north", time.time())
        state.activate_green_corridor(time.time())
        
        assert state.preemption_state == PreemptionState.ACTIVE
    
    def test_cooldown(self):
        """Test cooldown transition"""
        state = EmergencyState()
        
        state.start_preemption(1, "north", time.time())
        state.activate_green_corridor(time.time())
        state.start_cooldown(time.time())
        
        assert state.preemption_state == PreemptionState.COOLDOWN
    
    def test_deactivate(self):
        """Test deactivation"""
        state = EmergencyState()
        
        state.start_preemption(1, "north", time.time())
        state.deactivate()
        
        assert state.preemption_state == PreemptionState.INACTIVE
        assert state.is_active() is False


class TestEmergencyVehicleHandler:
    """Tests for EmergencyVehicleHandler class"""
    
    def test_handler_creation(self):
        """Test handler creation"""
        handler = EmergencyVehicleHandler(
            green_corridor_duration=15.0,
            cooldown_period=10.0,
        )
        
        assert handler.green_corridor_duration == 15.0
        assert handler.cooldown_period == 10.0
    
    def test_initial_state(self):
        """Test initial state"""
        handler = EmergencyVehicleHandler()
        
        assert handler.is_preemption_active() is False
        assert handler.get_required_phase() is None
    
    def test_get_blocked_directions(self):
        """Test blocked direction calculation"""
        handler = EmergencyVehicleHandler()
        
        # No emergency active
        assert handler.get_blocked_directions() == []
        
        # Simulate north emergency
        handler.state.start_preemption(1, "north", time.time())
        handler.state.activate_green_corridor(time.time())
        
        blocked = handler.get_blocked_directions()
        assert "east" in blocked
        assert "west" in blocked
        assert "north" not in blocked
    
    def test_clearance_time(self):
        """Test clearance time calculation"""
        handler = EmergencyVehicleHandler(green_corridor_duration=15.0)
        
        # Not active
        assert handler.get_clearance_time() == 0.0
        
        # Activate
        handler.state.start_preemption(1, "north", time.time())
        handler.state.activate_green_corridor(time.time())
        
        clearance = handler.get_clearance_time()
        assert clearance > 0
        assert clearance <= 15.0
    
    def test_reset(self):
        """Test handler reset"""
        handler = EmergencyVehicleHandler()
        
        handler.state.start_preemption(1, "north", time.time())
        handler.vehicles[1] = EmergencyVehicle(track_id=1)
        
        handler.reset()
        
        assert handler.is_preemption_active() is False
        assert len(handler.vehicles) == 0
