"""
Log and track issued fines
"""
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class FineLogger:
    """Record and manage traffic fines"""
    
    def __init__(self, log_path: str):
        """
        Initialize fine logger
        
        Args:
            log_path: Path to violation logs CSV
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create log file if it doesn't exist
        if not self.log_path.exists():
            self._create_log_file()
        
        self.df = pd.read_csv(self.log_path)
        logger.info(f"Loaded fine log with {len(self.df)} records")
    
    def _create_log_file(self):
        """Create empty log file with headers"""
        columns = [
            'fine_id', 'timestamp', 'plate_number', 'violation_type',
            'location', 'fine_amount', 'status', 'owner_name',
            'owner_address', 'evidence_image'
        ]
        df = pd.DataFrame(columns=columns)
        df.to_csv(self.log_path, index=False)
    
    def log_fine(self, fine_data: Dict) -> str:
        """
        Record a new fine
        
        Args:
            fine_data: Dictionary with fine details
            
        Returns:
            Fine ID
        """
        # Generate fine ID
        fine_id = f"FINE_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.df)+1}"
        
        # Prepare record
        record = {
            'fine_id': fine_id,
            'timestamp': datetime.now().isoformat(),
            'plate_number': fine_data.get('plate_number', ''),
            'violation_type': fine_data.get('violation_type', ''),
            'location': fine_data.get('location', ''),
            'fine_amount': fine_data.get('fine_amount', 0),
            'status': 'ISSUED',
            'owner_name': fine_data.get('owner_name', ''),
            'owner_address': fine_data.get('owner_address', ''),
            'evidence_image': fine_data.get('evidence_image', '')
        }
        
        # Add to dataframe
        new_row = pd.DataFrame([record])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        self.df.to_csv(self.log_path, index=False)
        
        logger.info(f"Logged fine {fine_id} for {fine_data.get('plate_number')}")
        return fine_id
    
    def get_fines_by_plate(self, plate_number: str) -> List[Dict]:
        """Get all fines for a specific plate"""
        mask = self.df['plate_number'] == plate_number
        return self.df[mask].to_dict('records')
    
    def get_fines_by_date(self, date: str) -> List[Dict]:
        """Get fines for a specific date"""
        mask = self.df['timestamp'].str.contains(date)
        return self.df[mask].to_dict('records')
    
    def update_fine_status(self, fine_id: str, status: str):
        """Update fine status (PAID, DISPUTED, etc.)"""
        mask = self.df['fine_id'] == fine_id
        if mask.any():
            self.df.loc[mask, 'status'] = status
            self.df.to_csv(self.log_path, index=False)
            logger.info(f"Updated fine {fine_id} status to {status}")
            return True
        return False
    
    def generate_daily_report(self, date: Optional[str] = None) -> Dict:
        """
        Generate summary report for a day
        
        Args:
            date: Date string (YYYY-MM-DD), defaults to today
            
        Returns:
            Dictionary with report statistics
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        day_fines = self.get_fines_by_date(date)
        
        report = {
            'date': date,
            'total_fines': len(day_fines),
            'total_amount': sum(f['fine_amount'] for f in day_fines),
            'violation_breakdown': {},
            'status_breakdown': {}
        }
        
        # Count by violation type
        for fine in day_fines:
            vtype = fine['violation_type']
            report['violation_breakdown'][vtype] = report['violation_breakdown'].get(vtype, 0) + 1
            
            status = fine['status']
            report['status_breakdown'][status] = report['status_breakdown'].get(status, 0) + 1
        
        return report