"""
Fine calculation based on violation type and severity
"""
from typing import Dict, Optional
from datetime import datetime, time
import logging

logger = logging.getLogger(__name__)

class FineCalculator:
    """Calculate fines based on violation rules"""
    
    def __init__(self, config):
        """
        Initialize with configuration
        """
        self.config = config
        
        # Fine multipliers for different factors
        self.multipliers = {
            'peak_hours': 1.5,      # 50% more during peak hours
            'school_zone': 2.0,      # Double in school zones
            'repeat_offender': 1.3,  # 30% more for repeat offenders
            'holiday': 0.8           # 20% less during holidays? (just for fun)
        }
    
    def calculate_speeding_fine(self, speed: float, speed_limit: float, 
                                location_type: str = "normal") -> float:
        """
        Calculate speeding fine
        """
        if speed <= speed_limit:
            return 0.0
        
        overspeed = speed - speed_limit
        base_fine = overspeed * self.config.SPEEDING_FINE_PER_KM
        
        # Apply location multiplier
        if location_type == "school_zone":
            base_fine *= self.multipliers['school_zone']
        elif location_type == "construction_zone":
            base_fine *= 1.8
        
        # Check if during peak hours
        if self._is_peak_hour():
            base_fine *= self.multipliers['peak_hours']
        
        return round(base_fine, 2)
    
    def calculate_red_light_fine(self, vehicle_type: str = "car", 
                                 crossing_distance: float = 1.0,
                                 repeat_offense: bool = False) -> float:
        """
        Calculate red light violation fine
        """
        base_fine = self.config.RED_LIGHT_FINE
        
        # Adjust based on vehicle type
        vehicle_multipliers = {
            "car": 1.0,
            "truck": 1.5,
            "bus": 1.5,
            "motorcycle": 0.8,
            "emergency": 0.0  # Emergency vehicles exempt
        }
        
        fine = base_fine * vehicle_multipliers.get(vehicle_type, 1.0)
        
        # Additional penalty for crossing far into intersection
        if crossing_distance > 5.0:  # meters
            fine += 500
        
        # Repeat offender penalty
        if repeat_offense:
            fine *= self.multipliers['repeat_offender']
        
        return round(fine, 2)
    
    def calculate_parking_fine(self, violation_type: str, duration_minutes: int,
                              zone_type: str = "normal") -> float:
        """
        Calculate parking violation fine
        """
        base_fines = {
            "no_parking": 1000,
            "handicap_zone": 5000,
            "fire_zone": 3000,
            "temporary_stop": 500,
            "blocking_driveway": 2000,
            "double_parked": 1500
        }
        
        base_fine = base_fines.get(violation_type, self.config.PARKING_FINE)
        
        # Additional per-hour charge for extended parking
        if duration_minutes > 60:
            extra_hours = (duration_minutes - 60) / 60
            base_fine += extra_hours * 200
        
        # Zone multiplier
        if zone_type == "airport":
            base_fine *= 2.0
        elif zone_type == "hospital":
            base_fine *= 1.5
        
        return round(base_fine, 2)
    
    def calculate_toll_evasion_fine(self, toll_amount: float, 
                                    vehicle_axles: int = 2) -> float:
        """
        Calculate toll evasion fine
        """
        base_fine = 2 * toll_amount  # Double the toll
        
        # Additional per-axle charge for commercial vehicles
        if vehicle_axles > 2:
            base_fine += (vehicle_axles - 2) * 500
        
        return round(base_fine, 2)
    
    def calculate_total_fine(self, violations: list, 
                            apply_discount: bool = False) -> Dict:
        """
        Calculate total fine for multiple violations
        """
        total = 0.0
        breakdown = []
        
        for violation in violations:
            v_type = violation.get('type')
            
            if v_type == 'speeding':
                fine = self.calculate_speeding_fine(
                    violation.get('speed', 0),
                    violation.get('speed_limit', 60),
                    violation.get('location_type', 'normal')
                )
            elif v_type == 'red_light':
                fine = self.calculate_red_light_fine(
                    violation.get('vehicle_type', 'car'),
                    violation.get('crossing_distance', 1.0),
                    violation.get('repeat_offense', False)
                )
            elif v_type == 'parking':
                fine = self.calculate_parking_fine(
                    violation.get('parking_type', 'no_parking'),
                    violation.get('duration', 30),
                    violation.get('zone_type', 'normal')
                )
            elif v_type == 'toll_evasion':
                fine = self.calculate_toll_evasion_fine(
                    violation.get('toll_amount', 100),
                    violation.get('axles', 2)
                )
            else:
                fine = 500  # Default fine
            
            breakdown.append({
                'type': v_type,
                'fine': fine,
                'details': violation
            })
            total += fine
        
        # Apply early payment discount if applicable
        if apply_discount:
            total *= 0.9  # 10% discount
            logger.info("Applied 10% early payment discount")
        
        return {
            'total': round(total, 2),
            'breakdown': breakdown,
            'violation_count': len(violations)
        }
    
    def _is_peak_hour(self) -> bool:
        """
        Check if current time is peak traffic hour
        """
        now = datetime.now().time()
        morning_peak = time(8, 0) <= now <= time(10, 0)
        evening_peak = time(17, 0) <= now <= time(19, 0)
        
        return morning_peak or evening_peak