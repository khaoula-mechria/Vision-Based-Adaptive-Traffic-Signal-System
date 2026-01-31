"""
Unit tests for pedestrian safety module
"""

import pytest
import time

from src.safety.crosswalk import Crosswalk, CrosswalkState, CrosswalkPhase
from src.safety.safety_manager import PedestrianSafetyManager


class TestCrosswalkState:
    """Tests for CrosswalkState class"""
    
    def test_default_state(self):
        """Test default state values"""
        state = CrosswalkState()
        
        assert state.phase == CrosswalkPhase.DONT_WALK
        assert state.pedestrian_count == 0
        assert state.is_safe_to_change is True


class TestCrosswalk:
    """Tests for Crosswalk class"""
    
    def test_crosswalk_creation(self):
        """Test crosswalk creation"""
        cw = Crosswalk(
            crosswalk_id=0,
            direction="north",
            x1=100, y1=200, x2=300, y2=230,
            width_meters=12.0,
        )
        
        assert cw.crosswalk_id == 0
        assert cw.width_meters == 12.0
    
    def test_contains_point(self):
        """Test point containment"""
        cw = Crosswalk(
            crosswalk_id=0, direction="north",
            x1=100, y1=200, x2=300, y2=230,
        )
        
        assert cw.contains_point(200, 215) is True
        assert cw.contains_point(50, 215) is False
    
    def test_minimum_crossing_time(self):
        """Test minimum crossing time calculation"""
        cw = Crosswalk(
            crosswalk_id=0, direction="north",
            width_meters=10.0,
            min_time_per_meter=1.5,
            clearance_buffer=5.0,
        )
        
        # Base time = 10 * 1.5 + 5 = 20 seconds
        min_time = cw.calculate_minimum_crossing_time(0)
        assert min_time == 20.0
        
        # With crowd, time increases
        min_time_crowd = cw.calculate_minimum_crossing_time(10)
        assert min_time_crowd > min_time
    
    def test_phase_transitions(self):
        """Test phase transitions"""
        cw = Crosswalk(crosswalk_id=0, direction="north")
        
        # Initial state
        assert cw.state.phase == CrosswalkPhase.DONT_WALK
        
        # Set to WALK
        cw.set_phase(CrosswalkPhase.WALK, time.time())
        assert cw.state.phase == CrosswalkPhase.WALK
        
        # Set to FLASHING
        cw.set_phase(CrosswalkPhase.FLASHING_DONT_WALK, time.time())
        assert cw.state.phase == CrosswalkPhase.FLASHING_DONT_WALK


class TestPedestrianSafetyManager:
    """Tests for PedestrianSafetyManager class"""
    
    def test_add_crosswalk(self):
        """Test adding crosswalks"""
        manager = PedestrianSafetyManager()
        
        manager.add_crosswalk(
            crosswalk_id=0,
            direction="north",
            x1=100, y1=200, x2=300, y2=230,
        )
        
        assert len(manager.crosswalks) == 1
    
    def test_setup_default_crosswalks(self):
        """Test default crosswalk setup"""
        manager = PedestrianSafetyManager()
        manager.setup_default_crosswalks(640, 480)
        
        # Should have 4 crosswalks
        assert len(manager.crosswalks) == 4
    
    def test_can_change_phase_empty(self):
        """Test phase change with empty crosswalk"""
        manager = PedestrianSafetyManager()
        manager.add_crosswalk(0, "north", 100, 200, 300, 230)
        
        is_safe, violation = manager.can_change_phase(0)
        assert is_safe is True
        assert violation is None
    
    def test_safety_override(self):
        """Test that safety overrides optimization"""
        manager = PedestrianSafetyManager()
        manager.add_crosswalk(0, "north", 100, 200, 300, 230)
        
        # Simulate pedestrians in crosswalk
        cw = manager.crosswalks[0]
        cw.state.pedestrians_in_crosswalk = 3
        cw.state.is_safe_to_change = False
        
        is_safe, violation = manager.can_change_phase(0)
        assert is_safe is False
        assert violation is not None
        assert violation.violation_type == "pedestrian_in_crosswalk"
    
    def test_minimum_crossing_time_retrieval(self):
        """Test getting minimum crossing time"""
        manager = PedestrianSafetyManager()
        manager.add_crosswalk(0, "north", 100, 200, 300, 230, width_meters=12.0)
        
        min_time = manager.get_minimum_crossing_time(0)
        assert min_time > 0
    
    def test_phase_extension_tracking(self):
        """Test phase extension tracking"""
        manager = PedestrianSafetyManager()
        manager.add_crosswalk(0, "north", 100, 200, 300, 230)
        
        # Apply extension
        manager.apply_phase_extension(0, 5.0)
        assert manager._phase_extensions[0] == 5.0
        
        # Apply more
        manager.apply_phase_extension(0, 3.0)
        assert manager._phase_extensions[0] == 8.0
        
        # Reset
        manager.reset_phase_extension(0)
        assert manager._phase_extensions[0] == 0.0
