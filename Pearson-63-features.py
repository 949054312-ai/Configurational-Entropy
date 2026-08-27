# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set Arial font with bold style and increased font sizes
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'  # Set font to bold
plt.rcParams['font.size'] = 22  # Increased by 4 points from typical 10

def plot_pearson_correlation():    
    # 1. Read Excel file
    print("Reading Excel file...")
    try:
        # Read Excel file, skipping the first row (header row)
        df = pd.read_excel('Database-1.xlsx', header=None)
        print(f"File loaded successfully! Data shape: {df.shape}")
        
        # Extract columns 1-63 (indices 0-62 in pandas)
        # Start from the second row (skip first row which contains parameter names)
        data = df.iloc[1:, 0:63]  # 1: means from second row, 0:63 means first 63 columns
        print(f"Extracted data shape: {data.shape}")
        
        # Ensure data is numeric
        data = data.apply(pd.to_numeric, errors='coerce')
        
        # Check for missing values
        if data.isnull().any().any():
            print("Warning: Data contains missing values, filling with column mean")
            data = data.fillna(data.mean())
        
        # 2. Calculate Pearson correlation matrix (absolute values)
        print("Calculating Pearson correlation coefficients...")
        corr_matrix = data.corr(method='pearson')
        corr_matrix_abs = np.abs(corr_matrix)  # Take absolute values
        
        # 3. Create figure with high DPI
        print("Creating high-resolution heatmap...")
        fig, ax = plt.subplots(figsize=(14, 12), dpi=100)  # Adjusted figure size for 63 features
        
        # Use rainbow colormap
        im = ax.imshow(corr_matrix_abs, cmap='rainbow', vmin=0, vmax=1, aspect='auto')
        
        # 4. Set axes with increased font sizes
        # Set ticks from 1 to 63
        tick_positions = np.arange(0, 63, 5)  # Show tick every 5 features
        tick_labels = [str(i+1) for i in tick_positions]
        
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontname='Arial', fontweight='bold', fontsize=14)  # Increased font size
        ax.set_yticklabels(tick_labels, fontname='Arial', fontweight='bold', fontsize=14)  # Increased font size
        
        # Set axis labels with increased font sizes
        ax.set_xlabel('Descriptor Serial Number', fontname='Arial', fontweight='bold', fontsize=16)
        ax.set_ylabel('Descriptor Serial Number', fontname='Arial', fontweight='bold', fontsize=16)
        
        # Set title with increased font size
        ax.set_title('Absolute Pearson Correlation Coefficient Matrix (Descriptors 1-63)', 
                    fontname='Arial', fontweight='bold', fontsize=18, pad=25)
        
        # 5. Add colorbar with increased font size
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Absolute Correlation Coefficient', fontname='Arial', fontweight='bold', fontsize=15)
        
        # Set colorbar ticks from 0 to 1 with 0.2 interval
        cbar.set_ticks(np.arange(0, 1.1, 0.2))
        cbar.ax.tick_params(labelsize=13)  # Increased tick label size
        cbar.set_ticklabels([f'{i:.1f}' for i in np.arange(0, 1.1, 0.2)], fontweight='bold')
        
        # 6. Add grid lines (optional)
        ax.set_xticks(np.arange(-0.5, 63, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 63, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5, alpha=0.3)
        
        # 7. Adjust layout
        plt.tight_layout()
        
        # 8. Save figure with 300 DPI resolution
        output_filename = 'pearson_correlation_heatmap_highres.png'
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')  # 300 DPI
        print(f"High-resolution heatmap saved as: {output_filename}")
        
        # Display figure
        plt.show()
        
        # 9. Output statistics
        print("\n=== Statistics ===")
        print(f"Correlation matrix shape: {corr_matrix_abs.shape}")
        print(f"Mean correlation coefficient: {corr_matrix_abs.values.mean():.4f}")
        print(f"Maximum correlation coefficient: {corr_matrix_abs.values.max():.4f}")
        print(f"Minimum correlation coefficient: {corr_matrix_abs.values.min():.4f}")
        print(f"Standard deviation: {corr_matrix_abs.values.std():.4f}")
        
        # Save correlation matrix to CSV file
        corr_matrix_abs.to_csv('correlation_matrix_absolute.csv')
        print("Correlation matrix saved as: correlation_matrix_absolute.csv")
        
        # Display high correlation pairs
        print("\n=== High Correlation Pairs (|r| > 0.8) ===")
        high_corr_count = 0
        for i in range(63):
            for j in range(i+1, 63):
                if corr_matrix_abs.iloc[i, j] > 0.8:
                    if high_corr_count < 10:  # Show only first 10
                        print(f"Descriptor {i+1:3d} and Descriptor {j+1:3d}: {corr_matrix_abs.iloc[i, j]:.4f}")
                    high_corr_count += 1
        
        if high_corr_count == 0:
            print("No descriptor pairs with |r| > 0.8 found")
        elif high_corr_count > 10:
            print(f"... and {high_corr_count - 10} more high-correlation pairs")
        
    except FileNotFoundError:
        print("Error: File 'Database-1.xlsx' not found")
        print("Please ensure the file is in the current directory or provide the correct path")
    except Exception as e:
        print(f"Error occurred: {e}")

def plot_pearson_correlation_alternative():
    """
    Alternative version with different layout and bold fonts
    """
    
    print("Reading Excel file for alternative visualization...")
    try:
        # Read Excel file
        df = pd.read_excel('Database-1.xlsx', header=None)
        
        # Extract columns 1-63, starting from second row
        data = df.iloc[1:, 0:63]
        
        # Convert to numeric
        data = data.apply(pd.to_numeric, errors='coerce')
        data = data.fillna(data.mean())
        
        print(f"Data shape: {data.shape}")
        
        # Calculate correlation matrix
        print("Calculating Pearson correlation coefficients...")
        corr_matrix = data.corr(method='pearson')
        corr_matrix_abs = np.abs(corr_matrix)
        
        # Create figure with high DPI
        fig, ax = plt.subplots(figsize=(16, 14), dpi=100)
        
        # Plot heatmap with rainbow colormap
        im = ax.imshow(corr_matrix_abs, cmap='rainbow', vmin=0, vmax=1, aspect='auto')
        
        # Set ticks and labels with bold fonts
        ax.set_xticks(np.arange(0, 63, 10))
        ax.set_yticks(np.arange(0, 63, 10))
        ax.set_xticklabels([str(i+1) for i in np.arange(0, 63, 10)], 
                          fontname='Arial', fontweight='bold', fontsize=15)
        ax.set_yticklabels([str(i+1) for i in np.arange(0, 63, 10)], 
                          fontname='Arial', fontweight='bold', fontsize=15)
        
        # Set labels with bold fonts and increased size
        ax.set_xlabel('Descriptor Serial Number', fontname='Arial', 
                     fontweight='bold', fontsize=18)
        ax.set_ylabel('Descriptor Serial Number', fontname='Arial', 
                     fontweight='bold', fontsize=18)
        
        # Set title with bold font and increased size
        ax.set_title('Pearson Correlation Heatmap (Descriptors 1-63)\nAbsolute Values', 
                    fontname='Arial', fontweight='bold', fontsize=20, pad=20)
        
        # Add colorbar with bold font
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Absolute Correlation Coefficient', 
                      fontname='Arial', fontweight='bold', fontsize=16)
        
        # Set colorbar ticks
        cbar.set_ticks(np.arange(0, 1.1, 0.2))
        cbar.ax.tick_params(labelsize=14)  # Increased size for colorbar ticks
        cbar.set_ticklabels([f'{i:.1f}' for i in np.arange(0, 1.1, 0.2)], fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save with 300 DPI
        output_filename = 'pearson_correlation_alternative.png'
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Alternative heatmap saved as: {output_filename}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error occurred: {e}")

# Simple version with minimal settings
def plot_pearson_correlation_minimal():
    """
    Minimal version focusing on essential elements with bold fonts
    """
    
    print("Reading Excel file for minimal visualization...")
    try:
        # Read data
        df = pd.read_excel('Database-1.xlsx', header=None)
        data = df.iloc[1:, 0:63].apply(pd.to_numeric, errors='coerce')
        data = data.fillna(data.mean())
        
        # Calculate correlation
        corr_matrix_abs = np.abs(data.corr(method='pearson'))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10), dpi=100)
        
        # Plot heatmap
        im = ax.imshow(corr_matrix_abs, cmap='rainbow', vmin=0, vmax=1)
        
        # Set labels and ticks with bold fonts
        ax.set_xticks(np.arange(0, 63, 10))
        ax.set_yticks(np.arange(0, 63, 10))
        ax.set_xticklabels([str(i+1) for i in np.arange(0, 63, 10)], 
                          fontname='Arial', fontweight='bold', fontsize=12)
        ax.set_yticklabels([str(i+1) for i in np.arange(0, 63, 10)], 
                          fontname='Arial', fontweight='bold', fontsize=12)
        
        ax.set_xlabel('Descriptor Serial Number', fontname='Arial', 
                     fontweight='bold', fontsize=14)
        ax.set_ylabel('Descriptor Serial Number', fontname='Arial', 
                     fontweight='bold', fontsize=14)
        ax.set_title('Absolute Pearson Correlation (Descriptors 1-63)', 
                    fontname='Arial', fontweight='bold', fontsize=16)
        
        # Add colorbar with bold font
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('|r|', fontname='Arial', fontweight='bold', fontsize=13)
        cbar.set_ticks(np.arange(0, 1.1, 0.2))
        cbar.ax.tick_params(labelsize=11)
        cbar.set_ticklabels([f'{i:.1f}' for i in np.arange(0, 1.1, 0.2)], fontweight='bold')
        
        plt.tight_layout()
        
        # Save and show
        plt.savefig('pearson_correlation_minimal.png', dpi=300, bbox_inches='tight')
        print("Minimal heatmap saved as: pearson_correlation_minimal.png")
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")

# Main execution
if __name__ == "__main__":
    print("=" * 70)
    print("PEARSON CORRELATION HEATMAP GENERATOR")
    print("=" * 70)
    print("Configuration:")
    print("- Font: Arial Bold")
    print("- Font sizes: Increased by 4 points")
    print("- Resolution: 300 DPI")
    print("- Color range: 0 to 1 with 0.2 intervals")
    print("- Colormap: Rainbow")
    print("- Data: Descriptors 1-63 from Database-1.xlsx")
    print("=" * 70)
    
    # Run the main function
    plot_pearson_correlation()
    
    # Uncomment to run alternative versions:
    # plot_pearson_correlation_alternative()
    # plot_pearson_correlation_minimal()
    
    print("\nScript execution completed!")

def create_multiple_formats():
    """
    Create multiple image formats for different use cases
    """
    print("\nCreating multiple image formats...")
    
    try:
        # Read data
        df = pd.read_excel('Database-1.xlsx', header=None)
        data = df.iloc[1:, 0:63].apply(pd.to_numeric, errors='coerce')
        data = data.fillna(data.mean())
        
        # Calculate correlation
        corr_matrix_abs = np.abs(data.corr(method='pearson'))
        
        # Create different DPI versions
        dpi_values = [150, 300, 600]
        for dpi in dpi_values:
            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(corr_matrix_abs, cmap='rainbow', vmin=0, vmax=1)
            
            # Set all text to bold Arial
            ax.set_xticks(np.arange(0, 63, 10))
            ax.set_yticks(np.arange(0, 63, 10))
            ax.set_xticklabels([str(i+1) for i in np.arange(0, 63, 10)], 
                              fontname='Arial', fontweight='bold', fontsize=12)
            ax.set_yticklabels([str(i+1) for i in np.arange(0, 63, 10)], 
                              fontname='Arial', fontweight='bold', fontsize=12)
            
            ax.set_xlabel('Descriptor Serial Number', fontname='Arial', 
                         fontweight='bold', fontsize=14)
            ax.set_ylabel('Descriptor Serial Number', fontname='Arial', 
                         fontweight='bold', fontsize=14)
            ax.set_title(f'Absolute Pearson Correlation (Descriptors 1-63, DPI: {dpi})', 
                        fontname='Arial', fontweight='bold', fontsize=16)
            
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('|r|', fontname='Arial', fontweight='bold', fontsize=12)
            cbar.set_ticks(np.arange(0, 1.1, 0.2))
            cbar.ax.tick_params(labelsize=10)
            
            plt.tight_layout()
            plt.savefig(f'pearson_correlation_dpi_{dpi}.png', dpi=dpi, bbox_inches='tight')
            plt.close()
            print(f"Saved: pearson_correlation_dpi_{dpi}.png")
    
    except Exception as e:
        print(f"Error creating multiple formats: {e}")