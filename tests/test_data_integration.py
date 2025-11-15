"""
Unit tests for data integration module.
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_integration import integrate_datasets


class TestDataIntegration(unittest.TestCase):
    """Test cases for data integration functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample datasets
        self.supplier_data = pd.DataFrame({
            'Supplier_ID': ['SUP001', 'SUP002'],
            'Industry_Sector': ['Technology', 'Manufacturing'],
            'Country': ['USA', 'China']
        })
        
        self.commodity_data = pd.DataFrame({
            'Industry_Sector': ['Technology', 'Manufacturing'],
            'GHG': ['Carbon dioxide', 'Carbon dioxide'],
            'Supply Chain Emission Factors with Margins': [0.5, 0.8]
        })
        
        self.co2_data = pd.DataFrame({
            'Country': ['USA', 'China'],
            'year': [2023, 2023],
            'co2': [5000.0, 10000.0],
            'gdp': [20000.0, 15000.0],
            'population': [330000000, 1400000000]
        })
    
    def test_integrate_datasets(self):
        """Test dataset integration."""
        result = integrate_datasets(
            self.supplier_data,
            self.commodity_data,
            self.co2_data
        )
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)
        self.assertIn('Supplier_ID', result.columns)


if __name__ == '__main__':
    unittest.main()

