import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# Import additional models (install if needed: pip install xgboost lightgbm catboost)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost not available. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("LightGBM not available. Install with: pip install lightgbm")
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    print("CatBoost not available. Install with: pip install catboost")
    CATBOOST_AVAILABLE = False

class ModelValidator:
    def __init__(self, data_path):
        self.data_path = data_path
        self.feature_columns = ['type', 'target_weight', 'Body weight (lbs)', 'ADG (lbs)']
        self.target_columns = [
            'DM Intake (lbs/day)', 'TDN (% DM)', 'NEm (Mcal/lb)',
            'NEg (Mcal/lb)', 'CP (% DM)', 'Ca (%DM)', 'P (% DM)',
            'TDN (lbs)', 'NEm (Mcal)', 'NEg (Mcal)', 'CP (lbs)',
            'Ca (grams)', 'P (grams)'
        ]
        self.results = []
        
    def get_models(self):
        """Define all models to test"""
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0)
        
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=-1)
        
        if CATBOOST_AVAILABLE:
            models['CatBoost'] = CatBoostRegressor(iterations=100, random_state=42, verbose=False)
        
        return models
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate all evaluation metrics"""
        # Handle potential division by zero in MAPE
        y_true_safe = np.where(np.abs(y_true) < 1e-8, 1e-8, y_true)
        
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_true_safe, y_pred) * 100  # Convert to percentage
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape
        }
    
    def validate_models(self):
        """Validate all models on all targets"""
        print("Loading data...")
        df = pd.read_csv(self.data_path)
        
        # Prepare features
        X = df[self.feature_columns]
        
        # Scale features
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        
        # Split data (80-20)
        X_train, X_test, indices_train, indices_test = train_test_split(
            X_scaled, df.index, test_size=0.2, random_state=42
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        models = self.get_models()
        print(f"Available models: {list(models.keys())}")
        
        # Validate each model on each target
        for target in self.target_columns:
            print(f"\nValidating models for target: {target}")
            
            # Get target values
            y = df[target].values
            y_train = y[indices_train]
            y_test = y[indices_test]
            
            # Scale targets
            scaler_y = StandardScaler()
            y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
            
            for model_name, model in models.items():
                try:
                    print(f"  Training {model_name}...")
                    
                    # Train model
                    model.fit(X_train, y_train_scaled)
                    
                    # Make predictions
                    y_pred_scaled = model.predict(X_test)
                    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                    
                    # Calculate metrics
                    metrics = self.calculate_metrics(y_test, y_pred)
                    
                    # Store results
                    result = {
                        'Model': model_name,
                        'Target': target,
                        'MAE': metrics['MAE'],
                        'MSE': metrics['MSE'],
                        'RMSE': metrics['RMSE'],
                        'MAPE': metrics['MAPE']
                    }
                    self.results.append(result)
                    
                    print(f"    MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}, MAPE: {metrics['MAPE']:.2f}%")
                    
                except Exception as e:
                    print(f"    Error with {model_name}: {str(e)}")
                    
    def save_results(self, output_file='model_validation_results.csv'):
        """Save results to CSV"""
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        return results_df
    
    def get_summary_statistics(self):
        """Get summary statistics for each model across all targets"""
        if not self.results:
            print("No results available. Run validate_models() first.")
            return None
        
        results_df = pd.DataFrame(self.results)
        
        # Calculate mean metrics for each model
        summary = results_df.groupby('Model')[['MAE', 'MSE', 'RMSE', 'MAPE']].mean().round(4)
        summary = summary.sort_values('RMSE')  # Sort by RMSE (lower is better)
        
        print("\nSummary Statistics (Average across all targets):")
        print("=" * 60)
        print(summary)
        
        return summary
    
    def get_best_models_per_target(self):
        """Find the best model for each target based on RMSE"""
        if not self.results:
            print("No results available. Run validate_models() first.")
            return None
        
        results_df = pd.DataFrame(self.results)
        
        # Find best model (lowest RMSE) for each target
        best_models = results_df.loc[results_df.groupby('Target')['RMSE'].idxmin()]
        best_models = best_models[['Target', 'Model', 'MAE', 'MSE', 'RMSE', 'MAPE']].round(4)
        
        print("\nBest Model for Each Target (based on RMSE):")
        print("=" * 80)
        for _, row in best_models.iterrows():
            print(f"{row['Target']:20} -> {row['Model']:15} (RMSE: {row['RMSE']:.4f})")
        
        return best_models

def main():
    # Initialize validator
    validator = ModelValidator('combined_growing.csv')  # Update with your data path
    
    # Run validation
    print("Starting model validation...")
    validator.validate_models()
    
    # Save results
    results_df = validator.save_results()
    
    # Print summary statistics
    validator.get_summary_statistics()
    
    # Print best models per target
    validator.get_best_models_per_target()
    
    print(f"\nTotal results: {len(validator.results)} model-target combinations")
    print(f"Results shape: {results_df.shape}")

if __name__ == "__main__":
    main()