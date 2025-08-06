import json
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import os

class ModelConverter:
    def __init__(self, models_dir='saved_models'):
        self.models_dir = models_dir
        
    def export_random_forest_to_dict(self, rf_model):
        """Convert RandomForest model to dictionary format"""
        trees = []
        
        for tree in rf_model.estimators_:
            tree_dict = {
                'n_nodes': tree.tree_.node_count,
                'children_left': tree.tree_.children_left.tolist(),
                'children_right': tree.tree_.children_right.tolist(),
                'feature': tree.tree_.feature.tolist(),
                'threshold': tree.tree_.threshold.tolist(),
                'value': tree.tree_.value.reshape(-1).tolist(),
                'n_node_samples': tree.tree_.n_node_samples.tolist()
            }
            trees.append(tree_dict)
            
        return {
            'n_estimators': rf_model.n_estimators,
            'trees': trees,
            'n_features': rf_model.n_features_in_,
            'n_outputs': rf_model.n_outputs_
        }
    
    def export_scaler_to_dict(self, scaler):
        """Convert StandardScaler to dictionary format"""
        return {
            'mean': scaler.mean_.tolist() if scaler.mean_ is not None else None,
            'scale': scaler.scale_.tolist() if scaler.scale_ is not None else None,
            'var': scaler.var_.tolist() if scaler.var_ is not None else None,
            'n_features_in': int(scaler.n_features_in_) if hasattr(scaler, 'n_features_in_') else None
        }
    
    def convert_models_to_json(self, output_file='nutrition_models.json'):
        """Convert all joblib models and scalers to JSON format"""
        # Load the joblib files
        models = joblib.load(os.path.join(self.models_dir, 'models.joblib'))
        scalers = joblib.load(os.path.join(self.models_dir, 'scalers.joblib'))
        
        # Convert models
        converted_models = {}
        for target_name, model in models.items():
            if isinstance(model, RandomForestRegressor):
                converted_models[target_name] = self.export_random_forest_to_dict(model)
            else:
                print(f"Warning: Model type {type(model)} for {target_name} not supported")
        
        # Convert scalers
        converted_scalers = {
            'features': self.export_scaler_to_dict(scalers['features']),
            'targets': {}
        }
        
        for target_name, scaler in scalers['targets'].items():
            converted_scalers['targets'][target_name] = self.export_scaler_to_dict(scaler)
        
        # Create the final JSON structure
        json_data = {
            'feature_columns': ['type', 'target_weight', 'Body weight (lbs)', 'ADG (lbs)'],
            'target_columns': [
                'DM Intake (lbs/day)', 'TDN (% DM)', 'NEm (Mcal/lb)',
                'NEg (Mcal/lb)', 'CP (% DM)', 'Ca (%DM)', 'P (% DM)',
                'TDN (lbs)', 'NEm (Mcal)', 'NEg (Mcal)', 'CP (lbs)',
                'Ca (grams)', 'P (grams)'
            ],
            'models': converted_models,
            'scalers': converted_scalers,
            'model_metadata': {
                'model_type': 'random_forest',
                'n_features': 4,
                'n_targets': len(converted_models)
            }
        }
        
        # Save to JSON file
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Models exported to {output_file}")
        print(f"File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
        
        return json_data

# Usage example
if __name__ == "__main__":
    converter = ModelConverter()
    converter.convert_models_to_json('nutrition_models.json')