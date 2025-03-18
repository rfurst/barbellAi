# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 11:53:48 2025

@author: rkfurst
"""

# src/data_processing.py
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import glob
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataProcessor")

class DataProcessor:
    def __init__(self, data_dir: str = "data/raw", processed_dir: str = "data/processed"):
        """
        Initialize the DataProcessor.
        
        Args:
            data_dir: Directory containing raw data files
            processed_dir: Directory to save processed data
        """
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)
        
    def find_data_files(self) -> List[Tuple[str, str]]:
        """
        Find pairs of CSV and JSON files for processing.
        
        Returns:
            List of tuples (csv_path, json_path)
        """
        logger.info(f"Searching for data files in: {self.data_dir}")
        
        # List all files in the directory
        all_files = os.listdir(self.data_dir)
        logger.info(f"Found {len(all_files)} files in directory")
        
        # Find all CSV files (more flexible pattern)
        csv_files = []
        for file in all_files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(self.data_dir, file))
        
        logger.info(f"Found {len(csv_files)} CSV files")
        
        file_pairs = []
        
        for csv_file in csv_files:
            base_name = os.path.basename(csv_file).replace('.csv', '')
            base_name = base_name.replace('_processed_data', '')
            
            # Try different potential JSON file patterns
            potential_json_names = [
                f"{base_name}_rep_summary.json",
                f"{base_name}.json",
                f"{base_name}_results.json"
            ]
            
            found_json = False
            for json_name in potential_json_names:
                json_path = os.path.join(self.data_dir, json_name)
                if os.path.exists(json_path):
                    file_pairs.append((csv_file, json_path))
                    logger.info(f"Found data pair: {base_name} ({os.path.basename(csv_file)} + {json_name})")
                    found_json = True
                    break
            
            if not found_json:
                logger.warning(f"No matching JSON found for {os.path.basename(csv_file)}")
        
        logger.info(f"Found {len(file_pairs)} file pairs for processing")
        return file_pairs
    
    def load_data_pair(self, csv_path: str, json_path: str) -> Tuple[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]:
        """
        Load a CSV-JSON data pair.
        
        Args:
            csv_path: Path to the processed data CSV file
            json_path: Path to the rep summary JSON file
            
        Returns:
            Tuple of (DataFrame, metadata dict, repetitions list)
        """
        # Load the CSV file
        data = pd.read_csv(csv_path)
        
        # Load the JSON file
        with open(json_path, 'r') as f:
            rep_data = json.load(f)
        
        metadata = rep_data.get('metadata', {})
        repetitions = rep_data.get('repetitions', [])
        
        return data, metadata, repetitions
    
    def extract_repetition_segments(self, data: pd.DataFrame, repetitions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract individual repetition segments from the full dataset.
        
        Args:
            data: DataFrame containing the full dataset
            repetitions: List of repetition information from JSON
            
        Returns:
            List of dictionaries containing repetition data
        """
        rep_segments = []
        
        for rep in repetitions:
            rep_number = rep['rep_number']
            start_idx = rep['concentric_phase']['start_index']
            end_idx = rep['concentric_phase']['end_index']
            
            # Extract the segment for this repetition
            rep_df = data.iloc[start_idx:end_idx+1].copy()
            
            rep_segments.append({
                'rep_number': rep_number,
                'data': rep_df,
                'concentric_phase': rep['concentric_phase']
            })
            
        return rep_segments
    
    def preprocess_all_data(self) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        """
        Process all available data pairs.
        
        Returns:
            Tuple of (all repetition segments, file-specific repetitions)
        """
        file_pairs = self.find_data_files()
        all_repetitions = []
        file_repetitions = {}
        
        for csv_path, json_path in file_pairs:
            file_name = os.path.basename(csv_path).replace("_processed_data.csv", "")
            logger.info(f"Processing {file_name}")
            
            # Load data
            data, metadata, repetitions = self.load_data_pair(csv_path, json_path)
            
            # Extract repetition segments
            rep_segments = self.extract_repetition_segments(data, repetitions)
            
            # Store by file for reference
            file_repetitions[file_name] = rep_segments
            
            # Add to the overall list with file identifier
            for rep in rep_segments:
                rep['file_name'] = file_name
                rep['metadata'] = metadata
                all_repetitions.append(rep)
        
        # Save the processed data
        self.save_processed_data(all_repetitions)
        
        logger.info(f"Processed {len(all_repetitions)} repetitions from {len(file_pairs)} files")
        return all_repetitions, file_repetitions
    
    def save_processed_data(self, all_repetitions: List[Dict[str, Any]]) -> None:
        """
        Save processed repetition data for later use.
        
        Args:
            all_repetitions: List of processed repetition dictionaries
        """
        # Save repetition metadata (without DataFrame to enable JSON serialization)
        rep_meta = []
        for rep in all_repetitions:
            rep_meta.append({
                'file_name': rep['file_name'],
                'rep_number': rep['rep_number'],
                'concentric_phase': rep['concentric_phase'],
                'metadata': rep['metadata']
            })
        
        # Save as JSON
        with open(os.path.join(self.processed_dir, 'repetition_metadata.json'), 'w') as f:
            json.dump(rep_meta, f, indent=2)
        
        # Save each repetition's data separately
        for i, rep in enumerate(all_repetitions):
            file_name = rep['file_name']
            rep_number = rep['rep_number']
            rep['data'].to_csv(
                os.path.join(self.processed_dir, f"{file_name}_rep{rep_number}.csv"),
                index=False
            )
        
        logger.info(f"Saved processed data to {self.processed_dir}")

# Utility function to load processed data
def load_processed_data(processed_dir: str = "data/processed") -> List[Dict[str, Any]]:
    """
    Load previously processed repetition data.
    
    Args:
        processed_dir: Directory containing processed data
    
    Returns:
        List of repetition dictionaries with data reloaded
    """
    # Load repetition metadata
    with open(os.path.join(processed_dir, 'repetition_metadata.json'), 'r') as f:
        rep_meta = json.load(f)
    
    # Reload the data for each repetition
    all_repetitions = []
    for rep in rep_meta:
        file_name = rep['file_name']
        rep_number = rep['rep_number']
        
        # Load the data
        csv_path = os.path.join(processed_dir, f"{file_name}_rep{rep_number}.csv")
        rep_data = pd.read_csv(csv_path)
        
        # Recreate the full repetition dictionary
        all_repetitions.append({
            'file_name': file_name,
            'rep_number': rep_number,
            'concentric_phase': rep['concentric_phase'],
            'metadata': rep['metadata'],
            'data': rep_data
        })
    
    return all_repetitions