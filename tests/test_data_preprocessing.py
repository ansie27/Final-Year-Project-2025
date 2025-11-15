"""
Unit tests for data preprocessing module.
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_preprocessing import (
    preprocess_supplier_data,
    preprocess_commodity_data,
    preprocess_co2_data,
    preprocess_esg_data,
    calculate_risk_metrics
)


class TestDataPreprocessing(unittest.TestCase):
    """Test cases for data preprocessing functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample supplier data
        self.sample_supplier_data = pd.DataFrame({
            'Supplier_ID': ['SUP001', 'SUP002', 'SUP003'],
            'Supplier_Name': ['Supplier A', 'Supplier B', 'Supplier C'],
            'Country': ['USA', 'China', 'Germany'],
            'Region': ['North America', 'Asia-Pacific', 'Europe'],
            'Industry_Sector': ['Technology', 'Manufacturing', 'Technology'],
            'Environmental_Score': [75.0, 60.0, 80.0],
            'ESG_Score': [70.0, 65.0, 75.0],
            'Carbon_Emission_Intensity': [10.5, 15.2, 8.3],
            'Renewable_Energy_Usage': [0.6, 0.4, 0.7],
            'Compliance_Level': [2, 1, 3],
            'On_Time_Delivery_Rate': [0.95, 0.85, 0.92],
            'Defect_Rate': [0.02, 0.05, 0.03],
            'Financial_Stability_Score': [80.0, 70.0, 85.0]
        })
    
    def test_preprocess_supplier_data(self):
        """Test supplier data preprocessing."""
        result = preprocess_supplier_data(self.sample_supplier_data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.sample_supplier_data))
        self.assertIn('Supplier_ID', result.columns)
    
    def test_calculate_risk_metrics(self):
        """Test risk metrics calculation."""
        # Add required columns for risk calculation
        test_data = self.sample_supplier_data.copy()
        test_data['Sustainability_Report_Availability'] = [1, 0, 1]
        
        result = calculate_risk_metrics(test_data)
        
        self.assertIn('Overall_Risk_Score', result.columns)
        self.assertIn('Risk_Classification', result.columns)
        self.assertIn('Environmental_Risk_Score', result.columns)
        self.assertIn('Compliance_Risk_Score', result.columns)
        self.assertIn('Operational_Risk_Score', result.columns)
        self.assertIn('Financial_Risk_Score', result.columns)


if __name__ == '__main__':
    unittest.main()

