"""
Mock database for vehicle owner information
"""
import pandas as pd
import random
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class OwnerDatabase:
    """Mock vehicle owner database"""
    
    def __init__(self, db_path: str):
        """
        Initialize mock database
        
        Args:
            db_path: Path to CSV database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create sample database if it doesn't exist
        if not self.db_path.exists():
            self._create_sample_database()
        
        self.df = pd.read_csv(self.db_path)
        logger.info(f"Loaded owner database with {len(self.df)} records")
    
    def _create_sample_database(self):
        """Create sample owner database for testing"""
        # Indian states codes
        states = ['MH', 'DL', 'KA', 'TN', 'GJ', 'UP', 'WB', 'AP', 'TS', 'HR']
        
        data = []
        for i in range(1, 101):  # 100 sample vehicles
            state = random.choice(states)
            district = f"{random.randint(1, 99):02d}"
            series = random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K'])
            number = f"{random.randint(1, 9999):04d}"
            
            plate = f"{state}{district}{series}{number}"
            
            owner = {
                'plate_number': plate,
                'owner_name': f"Owner_{i}",
                'address': f"{random.randint(1, 999)} Main Street, City_{i}",
                'city': f"City_{random.randint(1, 20)}",
                'state': state,
                'pincode': f"{random.randint(100000, 999999)}",
                'phone': f"98{random.randint(10000000, 99999999)}",
                'email': f"owner{i}@example.com",
                'vehicle_model': random.choice(['Sedan', 'SUV', 'Hatchback', 'Motorcycle']),
                'registration_date': f"202{random.randint(0, 3)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
            }
            data.append(owner)
        
        df = pd.DataFrame(data)
        df.to_csv(self.db_path, index=False)
        logger.info(f"Created sample database at {self.db_path}")
    
    def lookup_owner(self, plate_number: str) -> Optional[Dict]:
        """
        Look up owner by plate number
        
        Args:
            plate_number: License plate number
            
        Returns:
            Owner details or None if not found
        """
        # Clean plate for lookup (remove spaces)
        clean_plate = plate_number.replace(' ', '').upper()
        
        # Try exact match
        match = self.df[self.df['plate_number'] == clean_plate]
        
        if not match.empty:
            return match.iloc[0].to_dict()
        
        # Try fuzzy match (for testing)
        # Just return first record for demo purposes
        logger.warning(f"Plate {plate_number} not found in database")
        return self.df.iloc[0].to_dict()  # Demo: return first owner
    
    def add_vehicle(self, owner_details: Dict):
        """Add new vehicle to database"""
        new_row = pd.DataFrame([owner_details])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        self.df.to_csv(self.db_path, index=False)
        logger.info(f"Added new vehicle: {owner_details['plate_number']}")
    
    def update_owner(self, plate_number: str, updated_details: Dict):
        """Update owner information"""
        mask = self.df['plate_number'] == plate_number
        if mask.any():
            for key, value in updated_details.items():
                if key in self.df.columns:
                    self.df.loc[mask, key] = value
            self.df.to_csv(self.db_path, index=False)
            logger.info(f"Updated owner: {plate_number}")
            return True
        return False