#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import os
import matplotlib.patches as mpatches

# Set the style
plt.style.use('ggplot')
sns.set_theme(style="darkgrid")

# Function to create a 3D volatility surface plot
def plot_volatility_surface(df, title, save_path=None, highlight_arbitrage=None):
    """
    Create a 3D plot of a volatility surface
    
    Args:
        df: DataFrame with columns Strike, Maturity, Call_IV, Put_IV, Log_Moneyness
        title: Plot title
        save_path: If provided, save the figure to this path
        highlight_arbitrage: DataFrame with arbitrage opportunities to highlight
    """
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create mesh grid
    strikes = sorted(df['Strike'].unique())
    maturities = sorted(df['Maturity'].unique())
    strike_grid, maturity_grid = np.meshgrid(strikes, maturities)
    
    # Create volatility grid for call options
    call_vol_grid = np.zeros_like(strike_grid)
    for i, mat in enumerate(maturities):
        for j, k in enumerate(strikes):
            vol = df[(df['Maturity'] == mat) & (df['Strike'] == k)]['Call_IV'].values
            if len(vol) > 0:
                call_vol_grid[i, j] = vol[0]
    
    # Plot volatility surface
    surf = ax.plot_surface(
        strike_grid, maturity_grid, call_vol_grid,
        cmap=cm.viridis, alpha=0.8,
        linewidth=0, antialiased=True
    )
    
    # Highlight arbitrage opportunities if provided
    if highlight_arbitrage is not None:
        # For calendar spread arbitrage
        calendar_arb = highlight_arbitrage[highlight_arbitrage['Type'] == 'CALENDAR_SPREAD']
        # For butterfly arbitrage
        butterfly_arb = highlight_arbitrage[highlight_arbitrage['Type'] == 'BUTTERFLY']
        
        # Extract data points from descriptions
        all_points = []
        
        # Process calendar spread descriptions
        for _, row in calendar_arb.iterrows():
            desc = row['Description']
            # Parse Strike, T1, T2, IV1, IV2 from description
            parts = desc.replace('Calendar arbitrage on CALL options: ', '').split(', ')
            strike = float(parts[0].split('=')[1])
            t1 = float(parts[1].split('=')[1])
            t2 = float(parts[2].split('=')[1])
            iv1 = float(parts[3].split('=')[1])
            iv2 = float(parts[4].split('=')[1])
            
            all_points.append((strike, t1, iv1, 'calendar'))
            all_points.append((strike, t2, iv2, 'calendar'))
        
        # Process butterfly descriptions
        for _, row in butterfly_arb.iterrows():
            desc = row['Description']
            if 'Butterfly arbitrage on CALL options' in desc:
                # Parse Maturity, K1, K2, K3 from description
                parts = desc.replace('Butterfly arbitrage on CALL options: ', '').split(', ')
                maturity = float(parts[0].split('=')[1])
                k1 = float(parts[1].split('=')[1])
                k2 = float(parts[2].split('=')[1])
                k3 = float(parts[3].split('=')[1])
                
                # Find IV values for these points
                iv1 = df[(df['Maturity'] == maturity) & (df['Strike'] == k1)]['Call_IV'].values[0]
                iv2 = df[(df['Maturity'] == maturity) & (df['Strike'] == k2)]['Call_IV'].values[0]
                iv3 = df[(df['Maturity'] == maturity) & (df['Strike'] == k3)]['Call_IV'].values[0]
                
                all_points.append((k1, maturity, iv1, 'butterfly'))
                all_points.append((k2, maturity, iv2, 'butterfly'))
                all_points.append((k3, maturity, iv3, 'butterfly'))
        
        # Plot arbitrage points
        calendar_points = [(x, y, z) for x, y, z, t in all_points if t == 'calendar']
        butterfly_points = [(x, y, z) for x, y, z, t in all_points if t == 'butterfly']
        
        if calendar_points:
            x, y, z = zip(*calendar_points)
            ax.scatter(x, y, z, color='red', s=50, label='Calendar Arbitrage')
        
        if butterfly_points:
            x, y, z = zip(*butterfly_points)
            ax.scatter(x, y, z, color='yellow', s=50, label='Butterfly Arbitrage')
    
    # Add a color bar to show volatility values
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Implied Volatility')
    
    # Set labels and title
    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Time to Maturity')
    ax.set_zlabel('Implied Volatility')
    ax.set_title(title, fontsize=16)
    
    # Add legend
    if highlight_arbitrage is not None:
        ax.legend()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

# Function to compare volatility surfaces
def compare_surfaces(df1, df2, title1, title2, comparison_title, save_path=None):
    """
    Create a 2x1 grid of 3D plots to compare two volatility surfaces
    
    Args:
        df1, df2: DataFrames with columns Strike, Maturity, Call_IV, Put_IV, Log_Moneyness
        title1, title2: Titles for the two plots
        comparison_title: Overall title for the comparison
        save_path: If provided, save the figure to this path
    """
    
    fig = plt.figure(figsize=(18, 10))
    
    # First surface
    ax1 = fig.add_subplot(121, projection='3d')
    strikes1 = sorted(df1['Strike'].unique())
    maturities1 = sorted(df1['Maturity'].unique())
    strike_grid1, maturity_grid1 = np.meshgrid(strikes1, maturities1)
    
    call_vol_grid1 = np.zeros_like(strike_grid1)
    for i, mat in enumerate(maturities1):
        for j, k in enumerate(strikes1):
            vol = df1[(df1['Maturity'] == mat) & (df1['Strike'] == k)]['Call_IV'].values
            if len(vol) > 0:
                call_vol_grid1[i, j] = vol[0]
    
    surf1 = ax1.plot_surface(
        strike_grid1, maturity_grid1, call_vol_grid1,
        cmap=cm.viridis, alpha=0.8,
        linewidth=0, antialiased=True
    )
    
    ax1.set_xlabel('Strike Price')
    ax1.set_ylabel('Time to Maturity')
    ax1.set_zlabel('Implied Volatility')
    ax1.set_title(title1, fontsize=14)
    
    # Second surface
    ax2 = fig.add_subplot(122, projection='3d')
    strikes2 = sorted(df2['Strike'].unique())
    maturities2 = sorted(df2['Maturity'].unique())
    strike_grid2, maturity_grid2 = np.meshgrid(strikes2, maturities2)
    
    call_vol_grid2 = np.zeros_like(strike_grid2)
    for i, mat in enumerate(maturities2):
        for j, k in enumerate(strikes2):
            vol = df2[(df2['Maturity'] == mat) & (df2['Strike'] == k)]['Call_IV'].values
            if len(vol) > 0:
                call_vol_grid2[i, j] = vol[0]
    
    surf2 = ax2.plot_surface(
        strike_grid2, maturity_grid2, call_vol_grid2,
        cmap=cm.plasma, alpha=0.8,
        linewidth=0, antialiased=True
    )
    
    ax2.set_xlabel('Strike Price')
    ax2.set_ylabel('Time to Maturity')
    ax2.set_zlabel('Implied Volatility')
    ax2.set_title(title2, fontsize=14)
    
    # Add color bars
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5, label='Implied Volatility')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5, label='Implied Volatility')
    
    # Overall title
    plt.suptitle(comparison_title, fontsize=18)
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

# Function to create volatility smile plots
def plot_volatility_smile(df, maturities_to_show=None, title="Volatility Smile", save_path=None):
    """
    Create a plot of volatility smiles for selected maturities
    
    Args:
        df: DataFrame with columns Strike, Maturity, Call_IV, Put_IV, Log_Moneyness
        maturities_to_show: List of maturities to include in the plot (if None, show all)
        title: Plot title
        save_path: If provided, save the figure to this path
    """
    
    plt.figure(figsize=(12, 8))
    
    # Get all maturities if not specified
    if maturities_to_show is None:
        maturities_to_show = sorted(df['Maturity'].unique())
    
    # Use a color map for different maturities
    colors = plt.cm.viridis(np.linspace(0, 1, len(maturities_to_show)))
    
    # Plot each maturity
    for i, maturity in enumerate(maturities_to_show):
        maturity_df = df[df['Maturity'] == maturity]
        plt.plot(
            maturity_df['Log_Moneyness'],
            maturity_df['Call_IV'],
            marker='o',
            linewidth=2,
            color=colors[i],
            label=f"T = {maturity:.2f}"
        )
    
    plt.xlabel('Log Moneyness (log(K/S))')
    plt.ylabel('Implied Volatility')
    plt.title(title, fontsize=16)
    plt.legend(title='Maturity')
    plt.grid(True, alpha=0.3)
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

# Function to plot arbitrage opportunities by type and magnitude
def plot_arbitrage_summary(arbitrage_df, title="Arbitrage Opportunities Summary", save_path=None):
    """
    Create summary plots for arbitrage opportunities
    
    Args:
        arbitrage_df: DataFrame with arbitrage data
        title: Plot title
        save_path: If provided, save the figure to this path
    """
    
    # Create a copy of the dataframe to avoid modifying the original
    df = arbitrage_df.copy()
    
    # Ensure Magnitude column is numeric
    if 'Magnitude' in df.columns:
        try:
            # Try to convert Magnitude to numeric values, coerce errors to NaN
            df['Magnitude'] = pd.to_numeric(df['Magnitude'], errors='coerce')
            # Drop rows with NaN values in Magnitude
            df = df.dropna(subset=['Magnitude'])
        except Exception as e:
            print(f"Warning: Error converting Magnitude column to numeric: {e}")
            # If conversion fails, create a dummy Magnitude column with values of 1.0
            df['Magnitude'] = 1.0
    else:
        print("Warning: Magnitude column not found in arbitrage dataframe")
        # Create a dummy Magnitude column
        df['Magnitude'] = 1.0
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Count by type
    type_counts = df['Type'].value_counts()
    ax1.bar(type_counts.index, type_counts.values, color=sns.color_palette("viridis", len(type_counts)))
    ax1.set_title('Arbitrage Opportunities by Type', fontsize=14)
    ax1.set_xlabel('Type')
    ax1.set_ylabel('Count')
    
    # Rotate x labels if necessary
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Magnitude distribution
    if len(df) > 0 and df['Magnitude'].notna().any():
        ax2.hist(
            df['Magnitude'],
            bins=20,
            color='skyblue',
            edgecolor='black'
        )
        ax2.set_title('Distribution of Arbitrage Magnitudes', fontsize=14)
        ax2.set_xlabel('Magnitude')
        ax2.set_ylabel('Count')
        
        # Add a logarithmic scale if there are outliers
        if len(df['Magnitude']) > 1 and df['Magnitude'].max() > 0 and df['Magnitude'].median() > 0:
            try:
                ratio = df['Magnitude'].max() / df['Magnitude'].median()
                if ratio > 10:
                    ax2.set_xscale('log')
                    ax2.set_title('Distribution of Arbitrage Magnitudes (Log Scale)', fontsize=14)
            except Exception as e:
                print(f"Warning: Error setting log scale: {e}")
    else:
        ax2.text(0.5, 0.5, 'No valid magnitude data', ha='center', va='center', fontsize=12)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def main():
    # Path to the build directory where CSV files are saved
    build_path = "/Users/victordesouza/Documents/Machine Learning Library/build"
    
    # Create output directory for visualizations
    vis_path = "/Users/victordesouza/Documents/Machine Learning Library/visualizations"
    os.makedirs(vis_path, exist_ok=True)
    
    # Read CSV files
    arbitrage_free_surface = pd.read_csv(os.path.join(build_path, "arbitrage_free_surface.csv"))
    arbitrage_surface = pd.read_csv(os.path.join(build_path, "arbitrage_surface.csv"))
    predicted_surface = pd.read_csv(os.path.join(build_path, "predicted_surface.csv"))
    arbitrage_opportunities = pd.read_csv(os.path.join(build_path, "arbitrage_opportunities.csv"))
    lower_bound_surface = pd.read_csv(os.path.join(build_path, "lower_bound_surface.csv"))
    upper_bound_surface = pd.read_csv(os.path.join(build_path, "upper_bound_surface.csv"))
    
    print(f"Loaded data files successfully:")
    print(f"Arbitrage-free surface: {len(arbitrage_free_surface)} points")
    print(f"Surface with arbitrage: {len(arbitrage_surface)} points")
    print(f"Predicted surface: {len(predicted_surface)} points")
    print(f"Arbitrage opportunities: {len(arbitrage_opportunities)} instances")
    
    # Create visualizations
    
    # 1. Arbitrage-free surface
    print("Generating arbitrage-free surface visualization...")
    plot_volatility_surface(
        arbitrage_free_surface,
        "Arbitrage-Free Implied Volatility Surface",
        os.path.join(vis_path, "arbitrage_free_surface.png")
    )
    
    # 2. Surface with arbitrage
    print("Generating surface with arbitrage visualization...")
    plot_volatility_surface(
        arbitrage_surface,
        "Implied Volatility Surface with Arbitrage Opportunities",
        os.path.join(vis_path, "arbitrage_surface.png"),
        highlight_arbitrage=arbitrage_opportunities
    )
    
    # 3. Compare arbitrage-free and arbitrage surfaces
    print("Generating surface comparison...")
    compare_surfaces(
        arbitrage_free_surface,
        arbitrage_surface,
        "Arbitrage-Free Surface",
        "Surface with Arbitrage",
        "Comparison of Volatility Surfaces",
        os.path.join(vis_path, "surface_comparison.png")
    )
    
    # 4. Predicted surface
    print("Generating predicted surface visualization...")
    plot_volatility_surface(
        predicted_surface,
        "Predicted Implied Volatility Surface",
        os.path.join(vis_path, "predicted_surface.png")
    )
    
    # 5. Lower and upper bounds
    print("Generating bounds comparison...")
    compare_surfaces(
        lower_bound_surface,
        upper_bound_surface,
        "Lower Bound Surface",
        "Upper Bound Surface",
        "No-Arbitrage Bounds for Volatility Surface",
        os.path.join(vis_path, "bounds_comparison.png")
    )
    
    # 6. Volatility smile plots
    print("Generating volatility smile plots...")
    
    # For arbitrage-free surface
    plot_volatility_smile(
        arbitrage_free_surface,
        maturities_to_show=[0.33, 1.0, 2.0],
        title="Volatility Smiles from Arbitrage-Free Surface",
        save_path=os.path.join(vis_path, "arbitrage_free_smiles.png")
    )
    
    # For surface with arbitrage
    plot_volatility_smile(
        arbitrage_surface,
        maturities_to_show=[0.33, 1.0, 2.0],
        title="Volatility Smiles from Surface with Arbitrage",
        save_path=os.path.join(vis_path, "arbitrage_smiles.png")
    )
    
    # 7. Arbitrage summary
    print("Generating arbitrage summary...")
    plot_arbitrage_summary(
        arbitrage_opportunities,
        "Summary of Detected Arbitrage Opportunities",
        os.path.join(vis_path, "arbitrage_summary.png")
    )
    
    print(f"All visualizations saved to {vis_path}")

if __name__ == "__main__":
    main() 