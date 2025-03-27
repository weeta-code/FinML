#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
import csv  # Add import for csv module

# Set the style
plt.style.use('ggplot')
sns.set_theme(style="darkgrid")

def analyze_calendar_spread_arbitrage(arbitrage_df):
    """
    Analyze calendar spread arbitrage opportunities in detail
    
    Args:
        arbitrage_df: DataFrame with arbitrage data
        
    Returns:
        DataFrame with metrics on calendar spread arbitrage
    """
    # Filter for calendar spread arbitrage
    calendar_arb = arbitrage_df[arbitrage_df['Type'] == 'CALENDAR_SPREAD']
    
    if calendar_arb.empty:
        print("No calendar spread arbitrage opportunities found.")
        return pd.DataFrame()
    
    # Parse descriptions to extract details
    results = []
    
    for _, row in calendar_arb.iterrows():
        desc = row['Description']
        try:
            if 'Calendar arbitrage on CALL options:' in desc or 'Calendar arbitrage on PUT options:' in desc:
                # Parse details from description
                option_type = 'CALL' if 'CALL options' in desc else 'PUT'
                parts = desc.split(', ')
                
                # Handle various description formats
                # Format 1: "Calendar arbitrage on CALL options: Strike=100.000000, T1=0.333333, T2=0.666667, IV1=0.25, IV2=0.20"
                # Format 2: "Calendar arbitrage on CALL options: Strike=100.000000, T1=0.333333, T2=0.666667"
                
                # Extract values
                strike = t1 = t2 = iv1 = iv2 = None
                
                for part in parts:
                    part = part.strip()
                    if part.startswith('Strike='):
                        strike = float(part.split('=')[1])
                    elif part.startswith('T1='):
                        t1 = float(part.split('=')[1])
                    elif part.startswith('T2='):
                        t2 = float(part.split('=')[1])
                    elif part.startswith('IV1='):
                        iv1 = float(part.split('=')[1])
                    elif part.startswith('IV2='):
                        iv2 = float(part.split('=')[1])
                
                # Handle missing fields
                if strike is None:
                    # Try to extract from beginning of description
                    if 'Strike=' in desc:
                        try:
                            strike_part = desc.split('Strike=')[1].split(',')[0].strip()
                            strike = float(strike_part)
                        except:
                            pass
                
                # If IVs are not provided, use estimated values
                if iv1 is None or iv2 is None:
                    # Estimate typical values based on the magnitude
                    base_iv = 0.25  # Base IV of 25%
                    iv1 = base_iv + row['Magnitude']/200.0  # Split the difference
                    iv2 = base_iv - row['Magnitude']/200.0
                
                # Calculate time difference and volatility difference
                time_diff = t2 - t1 if t1 is not None and t2 is not None else None
                vol_diff = iv2 - iv1 if iv1 is not None and iv2 is not None else None
                ratio = iv2 / iv1 if iv1 is not None and iv2 is not None and iv1 > 0 else None
                
                results.append({
                    'Strike': strike,
                    'Maturity1': t1,
                    'Maturity2': t2,
                    'IV1': iv1,
                    'IV2': iv2,
                    'Time_Difference': time_diff,
                    'Volatility_Difference': vol_diff,
                    'IV_Ratio': ratio,
                    'Magnitude': row['Magnitude'],
                    'Option_Type': option_type
                })
        except Exception as e:
            print(f"Error parsing calendar spread arbitrage: {e}")
            print(f"Description: {desc}")
            continue
    
    if not results:
        print("Could not parse any calendar spread arbitrage details.")
        return pd.DataFrame()
    
    # Create DataFrame from results
    result_df = pd.DataFrame(results)
    
    # Summary statistics
    print("\n=== Calendar Spread Arbitrage Analysis ===")
    print(f"Number of opportunities: {len(result_df)}")
    print(f"Average magnitude: {result_df['Magnitude'].mean():.6f}")
    print(f"Max magnitude: {result_df['Magnitude'].max():.6f}")
    
    # Report stats only if we have the data
    if 'Volatility_Difference' in result_df.columns and not result_df['Volatility_Difference'].isna().all():
        print(f"Average volatility difference: {result_df['Volatility_Difference'].mean():.6f}")
    
    if 'Time_Difference' in result_df.columns and not result_df['Time_Difference'].isna().all():
        print(f"Average time difference: {result_df['Time_Difference'].mean():.6f}")
    
    # Correlation analysis with available columns
    corr_columns = []
    for col in ['Strike', 'Time_Difference', 'Volatility_Difference', 'Magnitude']:
        if col in result_df.columns and not result_df[col].isna().all():
            corr_columns.append(col)
    
    if len(corr_columns) > 1:
        print("\nCorrelation between metrics:")
        corr = result_df[corr_columns].corr()
        print(corr.round(4))
    
    return result_df

def analyze_butterfly_arbitrage(arbitrage_df):
    """
    Analyze butterfly arbitrage opportunities in detail
    
    Args:
        arbitrage_df: DataFrame with arbitrage data
        
    Returns:
        DataFrame with metrics on butterfly arbitrage
    """
    # Filter for butterfly arbitrage
    butterfly_arb = arbitrage_df[arbitrage_df['Type'] == 'BUTTERFLY']
    
    if butterfly_arb.empty:
        print("No butterfly arbitrage opportunities found.")
        return pd.DataFrame()
    
    # Parse descriptions to extract details
    results = []
    
    for _, row in butterfly_arb.iterrows():
        desc = row['Description']
        try:
            if 'Butterfly arbitrage on CALL options:' in desc or 'Butterfly arbitrage on PUT options:' in desc:
                # Parse details from description
                option_type = 'CALL' if 'CALL options' in desc else 'PUT'
                parts = desc.split(', ')
                
                # Handle various description formats
                # Format 1: "Butterfly arbitrage on CALL options: Maturity=1.000000, K1=96.000000, K2=100.000000, K3=104.000000, IV1=0.27, IV2=0.30, IV3=0.28"
                # Format 2: "Butterfly arbitrage on CALL options: Maturity=0.500000, K1=92.000000, K2=96.000000, K3=100.000000"
                
                # Extract maturity
                maturity_part = parts[0].split(':')[-1].strip() if ':' in parts[0] else parts[0]
                maturity = float(maturity_part.split('=')[1])
                
                # Extract strikes
                k1 = k2 = k3 = None
                iv1 = iv2 = iv3 = None
                
                for part in parts:
                    if 'K1=' in part:
                        k1 = float(part.split('=')[1])
                    elif 'K2=' in part:
                        k2 = float(part.split('=')[1])
                    elif 'K3=' in part:
                        k3 = float(part.split('=')[1])
                    elif 'IV1=' in part:
                        iv1 = float(part.split('=')[1])
                    elif 'IV2=' in part:
                        iv2 = float(part.split('=')[1])
                    elif 'IV3=' in part:
                        iv3 = float(part.split('=')[1])
                
                # If IVs are not provided, use estimated values
                if iv1 is None or iv2 is None or iv3 is None:
                    # Estimate typical values based on the magnitude
                    base_iv = 0.25  # Base IV of 25%
                    iv1 = base_iv
                    iv3 = base_iv
                    iv2 = base_iv + row['Magnitude']/100.0  # Convert magnitude to decimal
                
                # Calculate strike differences
                k12_diff = k2 - k1 if k1 is not None and k2 is not None else None
                k23_diff = k3 - k2 if k2 is not None and k3 is not None else None
                k_ratio = k12_diff / k23_diff if k12_diff is not None and k23_diff is not None and k23_diff > 0 else None
                
                # Calculate convexity violation if all IVs are available
                convexity_violation = None
                weighted_iv = None
                
                if k1 is not None and k2 is not None and k3 is not None and iv1 is not None and iv2 is not None and iv3 is not None:
                    # For a convex curve, we expect iv2 <= w1*iv1 + w2*iv3 where w1 and w2 are weights
                    w1 = (k3 - k2) / (k3 - k1)
                    w2 = (k2 - k1) / (k3 - k1)
                    weighted_iv = w1 * iv1 + w2 * iv3
                    convexity_violation = iv2 - weighted_iv
                
                results.append({
                    'Maturity': maturity,
                    'Strike1': k1,
                    'Strike2': k2,
                    'Strike3': k3,
                    'IV1': iv1,
                    'IV2': iv2,
                    'IV3': iv3,
                    'Strike12_Diff': k12_diff,
                    'Strike23_Diff': k23_diff,
                    'Strike_Ratio': k_ratio,
                    'Weighted_IV': weighted_iv,
                    'Convexity_Violation': convexity_violation,
                    'Magnitude': row['Magnitude'],
                    'Option_Type': option_type
                })
        except Exception as e:
            print(f"Error parsing butterfly arbitrage: {e}")
            print(f"Description: {desc}")
            continue
    
    if not results:
        print("Could not parse any butterfly arbitrage details.")
        return pd.DataFrame()
    
    # Create DataFrame from results
    result_df = pd.DataFrame(results)
    
    # Summary statistics
    print("\n=== Butterfly Arbitrage Analysis ===")
    print(f"Number of opportunities: {len(result_df)}")
    print(f"Average magnitude: {result_df['Magnitude'].mean():.6f}")
    print(f"Max magnitude: {result_df['Magnitude'].max():.6f}")
    
    if 'Convexity_Violation' in result_df.columns and not result_df['Convexity_Violation'].isna().all():
        print(f"Average convexity violation: {result_df['Convexity_Violation'].mean():.6f}")
    
    # Correlation analysis with available columns
    corr_columns = ['Maturity', 'Magnitude']
    if 'Strike_Ratio' in result_df.columns and not result_df['Strike_Ratio'].isna().all():
        corr_columns.append('Strike_Ratio')
    if 'Convexity_Violation' in result_df.columns and not result_df['Convexity_Violation'].isna().all():
        corr_columns.append('Convexity_Violation')
    
    print("\nCorrelation between metrics:")
    corr = result_df[corr_columns].corr()
    print(corr.round(4))
    
    return result_df

def analyze_predicted_surface(pred_surface_df, arb_free_df):
    """
    Analyze the predicted volatility surface
    
    Args:
        pred_surface_df: DataFrame with predicted surface data
        arb_free_df: DataFrame with arbitrage-free surface data
        
    Returns:
        DataFrame with prediction error metrics
    """
    if pred_surface_df.empty or arb_free_df.empty:
        print("Missing data for prediction analysis.")
        return pd.DataFrame()
    
    # Merge the two dataframes on Strike and Maturity
    merged = pd.merge(
        pred_surface_df, 
        arb_free_df,
        on=['Strike', 'Maturity'],
        suffixes=('_pred', '_true')
    )
    
    if merged.empty:
        print("No matching data points found between predicted and true surfaces.")
        return pd.DataFrame()
    
    # Calculate error metrics
    merged['Error'] = merged['Call_IV_pred'] - merged['Call_IV_true']
    merged['Abs_Error'] = abs(merged['Error'])
    merged['Squared_Error'] = merged['Error'] ** 2
    merged['Pct_Error'] = 100 * merged['Error'] / merged['Call_IV_true']
    
    # Summarize by strike
    strike_summary = merged.groupby('Strike').agg({
        'Abs_Error': ['mean', 'median', 'std'],
        'Pct_Error': ['mean', 'median', 'std']
    }).reset_index()
    
    # Summarize by maturity
    maturity_summary = merged.groupby('Maturity').agg({
        'Abs_Error': ['mean', 'median', 'std'],
        'Pct_Error': ['mean', 'median', 'std']
    }).reset_index()
    
    # Overall metrics
    mse = merged['Squared_Error'].mean()
    rmse = np.sqrt(mse)
    mae = merged['Abs_Error'].mean()
    mean_pct_error = merged['Pct_Error'].mean()
    
    print("\n=== Prediction Error Analysis ===")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Mean Percentage Error: {mean_pct_error:.2f}%")
    
    return {
        'merged': merged,
        'strike_summary': strike_summary,
        'maturity_summary': maturity_summary,
        'overall_metrics': {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mean_pct_error': mean_pct_error
        }
    }

def analyze_arbitrage_bounds(lower_bound_df, upper_bound_df, pred_surface_df):
    """
    Analyze no-arbitrage bounds and whether predictions violate them
    
    Args:
        lower_bound_df: DataFrame with lower bound surface
        upper_bound_df: DataFrame with upper bound surface
        pred_surface_df: DataFrame with predicted surface
        
    Returns:
        DataFrame with bounds analysis
    """
    if lower_bound_df.empty or upper_bound_df.empty or pred_surface_df.empty:
        print("Missing data for bounds analysis.")
        return pd.DataFrame()
    
    # Merge all three dataframes
    merged = pd.merge(
        pd.merge(
            lower_bound_df,
            upper_bound_df,
            on=['Strike', 'Maturity'],
            suffixes=('_lower', '_upper')
        ),
        pred_surface_df,
        on=['Strike', 'Maturity']
    )
    
    if merged.empty:
        print("No matching data points found for bounds analysis.")
        return pd.DataFrame()
    
    # Check for violations
    merged['Lower_Violation'] = merged['Call_IV'] < merged['Call_IV_lower']
    merged['Upper_Violation'] = merged['Call_IV'] > merged['Call_IV_upper']
    merged['Any_Violation'] = merged['Lower_Violation'] | merged['Upper_Violation']
    
    # Calculate metrics
    total_points = len(merged)
    lower_violations = merged['Lower_Violation'].sum()
    upper_violations = merged['Upper_Violation'].sum()
    any_violations = merged['Any_Violation'].sum()
    
    print("\n=== No-Arbitrage Bounds Analysis ===")
    print(f"Total points: {total_points}")
    print(f"Lower bound violations: {lower_violations} ({lower_violations/total_points*100:.2f}%)")
    print(f"Upper bound violations: {upper_violations} ({upper_violations/total_points*100:.2f}%)")
    print(f"Total violations: {any_violations} ({any_violations/total_points*100:.2f}%)")
    
    # Calculate violation magnitudes
    merged['Lower_Violation_Magnitude'] = np.where(
        merged['Lower_Violation'],
        merged['Call_IV_lower'] - merged['Call_IV'],
        0
    )
    
    merged['Upper_Violation_Magnitude'] = np.where(
        merged['Upper_Violation'],
        merged['Call_IV'] - merged['Call_IV_upper'],
        0
    )
    
    merged['Violation_Magnitude'] = merged['Lower_Violation_Magnitude'] + merged['Upper_Violation_Magnitude']
    
    if any_violations > 0:
        print(f"Average violation magnitude: {merged['Violation_Magnitude'].mean():.6f}")
        print(f"Maximum violation magnitude: {merged['Violation_Magnitude'].max():.6f}")
    
    return merged

def plot_prediction_errors(pred_analysis_results, save_path=None):
    """
    Create visualizations of prediction errors
    
    Args:
        pred_analysis_results: Results from analyze_predicted_surface
        save_path: If provided, save the figure to this path
    """
    merged = pred_analysis_results['merged']
    strike_summary = pred_analysis_results['strike_summary']
    maturity_summary = pred_analysis_results['maturity_summary']
    
    # Create a figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Error distribution
    sns.histplot(merged['Error'], kde=True, ax=ax1)
    ax1.set_title('Distribution of Prediction Errors')
    ax1.set_xlabel('Error (Predicted - True)')
    
    # Add normal distribution for comparison
    x = np.linspace(merged['Error'].min(), merged['Error'].max(), 100)
    mu, sigma = stats.norm.fit(merged['Error'])
    y = stats.norm.pdf(x, mu, sigma)
    ax1.plot(x, y * len(merged['Error']) * (x[1] - x[0]), 'r-', linewidth=2)
    
    # 2. Error by strike
    sns.lineplot(
        data=strike_summary,
        x='Strike',
        y=('Abs_Error', 'mean'),
        marker='o',
        ax=ax2
    )
    ax2.fill_between(
        strike_summary['Strike'],
        strike_summary[('Abs_Error', 'mean')] - strike_summary[('Abs_Error', 'std')],
        strike_summary[('Abs_Error', 'mean')] + strike_summary[('Abs_Error', 'std')],
        alpha=0.3
    )
    ax2.set_title('Mean Absolute Error by Strike')
    ax2.set_xlabel('Strike Price')
    ax2.set_ylabel('Mean Absolute Error')
    
    # 3. Error by maturity
    sns.lineplot(
        data=maturity_summary,
        x='Maturity',
        y=('Abs_Error', 'mean'),
        marker='o',
        ax=ax3
    )
    ax3.fill_between(
        maturity_summary['Maturity'],
        maturity_summary[('Abs_Error', 'mean')] - maturity_summary[('Abs_Error', 'std')],
        maturity_summary[('Abs_Error', 'mean')] + maturity_summary[('Abs_Error', 'std')],
        alpha=0.3
    )
    ax3.set_title('Mean Absolute Error by Maturity')
    ax3.set_xlabel('Time to Maturity')
    ax3.set_ylabel('Mean Absolute Error')
    
    # 4. True vs. Predicted plot
    ax4.scatter(merged['Call_IV_true'], merged['Call_IV_pred'], alpha=0.5)
    ax4.plot([merged['Call_IV_true'].min(), merged['Call_IV_true'].max()],
            [merged['Call_IV_true'].min(), merged['Call_IV_true'].max()],
            'r--', linewidth=2)
    ax4.set_title('True vs. Predicted Implied Volatility')
    ax4.set_xlabel('True Implied Volatility')
    ax4.set_ylabel('Predicted Implied Volatility')
    
    # Add overall metrics as text
    overall_metrics = pred_analysis_results['overall_metrics']
    metrics_text = (
        f"RMSE: {overall_metrics['rmse']:.4f}\n"
        f"MAE: {overall_metrics['mae']:.4f}\n"
        f"Mean % Error: {overall_metrics['mean_pct_error']:.2f}%"
    )
    fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Prediction Error Analysis', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def plot_bounds_analysis(bounds_analysis_df, save_path=None):
    """
    Visualize bounds analysis results
    
    Args:
        bounds_analysis_df: DataFrame from analyze_arbitrage_bounds
        save_path: If provided, save the figure to this path
    """
    if bounds_analysis_df.empty:
        return
    
    # Filter to only include violations for detailed analysis
    violations_df = bounds_analysis_df[bounds_analysis_df['Any_Violation']]
    
    # Create a figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Heatmap of violations by strike and maturity
    if not violations_df.empty:
        pivot = bounds_analysis_df.pivot_table(
            values='Any_Violation',
            index='Strike',
            columns='Maturity',
            aggfunc="sum"  # Use string instead of function reference
        )
        sns.heatmap(pivot, cmap='YlOrRd', ax=ax1, annot=True, fmt='d', cbar_kws={'label': 'Violations'})
        ax1.set_title('Violations by Strike and Maturity')
    else:
        ax1.text(0.5, 0.5, 'No violations found', ha='center', va='center', fontsize=12)
        ax1.set_title('No Violations Detected')
    
    # 2. Violation magnitude distribution
    if not violations_df.empty:
        sns.histplot(violations_df['Violation_Magnitude'], kde=True, ax=ax2)
        ax2.set_title('Distribution of Violation Magnitudes')
        ax2.set_xlabel('Violation Magnitude')
    else:
        ax2.text(0.5, 0.5, 'No violations found', ha='center', va='center', fontsize=12)
        ax2.set_title('No Violations Detected')
    
    # 3. Bounds and predictions for a specific maturity
    sample_maturity = bounds_analysis_df['Maturity'].median()
    maturity_df = bounds_analysis_df[bounds_analysis_df['Maturity'] == sample_maturity].sort_values('Strike')
    
    ax3.plot(maturity_df['Strike'], maturity_df['Call_IV_lower'], 'b-', label='Lower Bound')
    ax3.plot(maturity_df['Strike'], maturity_df['Call_IV_upper'], 'r-', label='Upper Bound')
    ax3.plot(maturity_df['Strike'], maturity_df['Call_IV'], 'g--', label='Predicted IV')
    ax3.set_title(f'Bounds and Predictions (T = {sample_maturity:.2f})')
    ax3.set_xlabel('Strike Price')
    ax3.set_ylabel('Implied Volatility')
    ax3.legend()
    
    # 4. Plot percentage of violations by maturity
    violations_by_maturity = bounds_analysis_df.groupby('Maturity')['Any_Violation'].mean() * 100
    ax4.bar(violations_by_maturity.index, violations_by_maturity.values)
    ax4.set_title('Percentage of Violations by Maturity')
    ax4.set_xlabel('Time to Maturity')
    ax4.set_ylabel('Violation Percentage')
    
    plt.suptitle('No-Arbitrage Bounds Analysis', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def main():
    # Path to the build directory where CSV files are saved
    build_path = "/Users/victordesouza/Documents/Machine Learning Library/build"
    
    # Create output directory for analysis results
    analysis_path = "/Users/victordesouza/Documents/Machine Learning Library/analysis"
    os.makedirs(analysis_path, exist_ok=True)
    
    # Read CSV files
    try:
        arbitrage_free_surface = pd.read_csv(os.path.join(build_path, "arbitrage_free_surface.csv"))
        arbitrage_surface = pd.read_csv(os.path.join(build_path, "arbitrage_surface.csv"))
        predicted_surface = pd.read_csv(os.path.join(build_path, "predicted_surface.csv"))
        
        # Read arbitrage opportunities with proper quoting to handle embedded commas
        arbitrage_opportunities = pd.read_csv(
            os.path.join(build_path, "arbitrage_opportunities.csv"),
            quoting=csv.QUOTE_MINIMAL  # Handle quoted fields properly
        )
        
        lower_bound_surface = pd.read_csv(os.path.join(build_path, "lower_bound_surface.csv"))
        upper_bound_surface = pd.read_csv(os.path.join(build_path, "upper_bound_surface.csv"))
        
        print(f"Loaded data files successfully:")
        print(f"Arbitrage-free surface: {len(arbitrage_free_surface)} points")
        print(f"Surface with arbitrage: {len(arbitrage_surface)} points")
        print(f"Predicted surface: {len(predicted_surface)} points")
        print(f"Arbitrage opportunities: {len(arbitrage_opportunities)} instances")
        print(f"Lower bound surface: {len(lower_bound_surface)} points")
        print(f"Upper bound surface: {len(upper_bound_surface)} points")
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        return
    
    # Fix the Magnitude column in arbitrage_opportunities DataFrame
    if 'Magnitude' in arbitrage_opportunities.columns:
        try:
            # Convert to numeric, forcing non-numeric values to NaN
            arbitrage_opportunities['Magnitude'] = pd.to_numeric(arbitrage_opportunities['Magnitude'], errors='coerce')
            # Drop rows with NaN in Magnitude
            orig_len = len(arbitrage_opportunities)
            nan_rows = arbitrage_opportunities['Magnitude'].isna().sum()
            
            if nan_rows > 0:
                print(f"Warning: Found {nan_rows} rows with non-numeric Magnitude values")
                
                # Try to extract Magnitude from Description field if it exists there
                if 'Description' in arbitrage_opportunities.columns:
                    print("Attempting to extract Magnitude from Description field...")
                    
                    for idx in arbitrage_opportunities.index[arbitrage_opportunities['Magnitude'].isna()]:
                        desc = arbitrage_opportunities.loc[idx, 'Description']
                        
                        # Look for the magnitude pattern in the description
                        import re
                        magnitudes = re.findall(r'Magnitude[=:]?\s*([\d\.]+)', desc)
                        if magnitudes:
                            try:
                                arbitrage_opportunities.loc[idx, 'Magnitude'] = float(magnitudes[0])
                                print(f"  - Extracted magnitude {magnitudes[0]} from description")
                            except ValueError:
                                pass
                
                # Drop any remaining rows with NaN magnitude
                arbitrage_opportunities = arbitrage_opportunities.dropna(subset=['Magnitude'])
                if len(arbitrage_opportunities) < orig_len:
                    print(f"Warning: Dropped {orig_len - len(arbitrage_opportunities)} rows with non-recoverable Magnitude values")
        except Exception as e:
            print(f"Warning: Error handling Magnitude column: {e}")
    
    # Add log moneyness column if not present
    # Assuming S0 = 100 for this example
    S0 = 100.0
    
    for df in [arbitrage_free_surface, arbitrage_surface, predicted_surface]:
        if 'Log_Moneyness' not in df.columns:
            df['Log_Moneyness'] = np.log(df['Strike'] / S0)
    
    # Analyze calendar spread arbitrage
    print("\nAnalyzing calendar spread arbitrage...")
    calendar_spread_results = analyze_calendar_spread_arbitrage(arbitrage_opportunities)
    
    # Analyze butterfly arbitrage
    print("\nAnalyzing butterfly arbitrage...")
    butterfly_results = analyze_butterfly_arbitrage(arbitrage_opportunities)
    
    # Analyze predicted surface
    print("\nAnalyzing prediction accuracy...")
    prediction_results = analyze_predicted_surface(predicted_surface, arbitrage_free_surface)
    
    # Analyze no-arbitrage bounds
    print("\nAnalyzing no-arbitrage bounds...")
    bounds_results = analyze_arbitrage_bounds(
        lower_bound_surface, 
        upper_bound_surface,
        predicted_surface
    )
    
    # Generate visualizations
    print("\nGenerating additional visualizations...")
    
    # Prediction error plots
    if isinstance(prediction_results, dict) and 'merged' in prediction_results:
        plot_prediction_errors(
            prediction_results,
            os.path.join(analysis_path, "prediction_errors.png")
        )
    
    # Bounds analysis plots
    if isinstance(bounds_results, pd.DataFrame) and not bounds_results.empty:
        plot_bounds_analysis(
            bounds_results,
            os.path.join(analysis_path, "bounds_analysis.png")
        )
    
    # Save detailed analysis results to CSV
    if isinstance(calendar_spread_results, pd.DataFrame) and not calendar_spread_results.empty:
        calendar_spread_results.to_csv(
            os.path.join(analysis_path, "calendar_spread_analysis.csv"),
            index=False
        )
    
    if isinstance(butterfly_results, pd.DataFrame) and not butterfly_results.empty:
        butterfly_results.to_csv(
            os.path.join(analysis_path, "butterfly_analysis.csv"),
            index=False
        )
    
    if isinstance(prediction_results, dict) and 'merged' in prediction_results:
        prediction_results['merged'].to_csv(
            os.path.join(analysis_path, "prediction_errors.csv"),
            index=False
        )
    
    if isinstance(bounds_results, pd.DataFrame) and not bounds_results.empty:
        bounds_results.to_csv(
            os.path.join(analysis_path, "bounds_analysis.csv"),
            index=False
        )
    
    print(f"\nAll analysis results saved to {analysis_path}")
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main() 