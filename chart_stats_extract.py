from alphagenome.data import genome
from alphagenome.models import dna_client
from alphagenome.visualization import plot_components
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Initialize API
API_KEY = 'ENTER_API_KEY' #Enter your Alphagenome API key here
model = dna_client.create(API_KEY)

# Define your interval and variant
interval = genome.Interval(chromosome='chr22', start=35677410, end=36725986)
variant = genome.Variant(
    chromosome='chr22',
    position=36201698,
    reference_bases='A',
    alternate_bases='C',
)

# Get predictions with multiple output types
outputs = model.predict_variant(
    interval=interval,
    variant=variant,
    ontology_terms=['UBERON:0001157'],  # tissue type
    requested_outputs=[
        dna_client.OutputType.RNA_SEQ,
        dna_client.OutputType.CAGE,  # if available
        dna_client.OutputType.DNASE,  # if available
        # Add other output types as needed
    ],
)

# Extract numerical data from RNA-seq
def extract_numerical_data(rna_seq_data, label):
    """Extract numerical values from RNA-seq data"""

    # Get the actual values
    values = rna_seq_data.values  # Don't flatten here, handle dimensions later
    positions = np.arange(rna_seq_data.interval.start,
                         rna_seq_data.interval.end,
                         step=1)  # Assuming 1bp resolution

    # Calculate statistics
    # If values is multi-dimensional, calculate stats for the first dimension (usually the relevant track)
    if values.ndim > 1:
      values_1d = values[:, 0]
    else:
      values_1d = values

    peak_indices = np.where(values_1d > np.percentile(values_1d, 95))[0]
    stats = {
        'label': label,
        'max_value': np.max(values_1d),
        'min_value': np.min(values_1d),
        'mean_value': np.mean(values_1d),
        'median_value': np.median(values_1d),
        'std_value': np.std(values_1d),
        'total_sum': np.sum(values_1d),
        'peak_positions': positions[peak_indices],
        'peak_values': values_1d[peak_indices]
    }

    return values, positions, stats # Return original values for DataFrame creation

# Extract data for both reference and alternate
ref_values_full, ref_positions, ref_stats = extract_numerical_data(outputs.reference.rna_seq, 'Reference')
alt_values_full, alt_positions, alt_stats = extract_numerical_data(outputs.alternate.rna_seq, 'Alternate')

# Ensure values are 1D for calculations and DataFrame creation
if ref_values_full.ndim > 1:
  ref_values = ref_values_full[:, 0]
else:
  ref_values = ref_values_full

if alt_values_full.ndim > 1:
  alt_values = alt_values_full[:, 0]
else:
  alt_values = alt_values_full


# Create comprehensive data analysis
def analyze_variant_impact():
    """Analyze the impact of the variant on gene expression"""

    # Calculate difference between alt and ref
    difference = alt_values - ref_values
    fold_change = np.where(ref_values != 0, alt_values / ref_values, 0)

    impact_stats = {
        'max_difference': np.max(difference),
        'min_difference': np.min(difference),
        'mean_difference': np.mean(difference),
        'max_fold_change': np.max(fold_change[np.isfinite(fold_change)]),
        'min_fold_change': np.min(fold_change[np.isfinite(fold_change)]),
        'significantly_changed_positions': np.sum(np.abs(difference) > np.std(difference)),
        'upregulated_positions': np.sum(difference > 0),
        'downregulated_positions': np.sum(difference < 0)
    }

    return difference, fold_change, impact_stats

difference, fold_change, impact_stats = analyze_variant_impact()

# Print all numerical data
print("= REFERENCE SEQUENCE STATISTICS =")
for key, value in ref_stats.items():
    if isinstance(value, np.ndarray):
        print(f"{key}: {len(value)} values (showing first 10: {value[:10]})")
    else:
        print(f"{key}: {value}")

print("\n= ALTERNATE SEQUENCE STATISTICS =")
for key, value in alt_stats.items():
    if isinstance(value, np.ndarray):
        print(f"{key}: {len(value)} values (showing first 10: {value[:10]})")
    else:
        print(f"{key}: {value}")

print("\n= VARIANT IMPACT ANALYSIS =")
for key, value in impact_stats.items():
    print(f"{key}: {value}")

# Create detailed DataFrame for export
detailed_data = pd.DataFrame({
    'Position': ref_positions,
    'Reference_Expression': ref_values,
    'Alternate_Expression': alt_values,
    'Difference': difference,
    'Fold_Change': fold_change
})


# Add significance flags
detailed_data['Significant_Change'] = np.abs(detailed_data['Difference']) > np.std(difference)
detailed_data['Change_Direction'] = np.where(detailed_data['Difference'] > 0, 'Upregulated',
                                   np.where(detailed_data['Difference'] < 0, 'Downregulated', 'No_Change'))

# Save detailed data
detailed_data.to_csv('variant_analysis_detailed.csv', index=False)
print(f"\n= DETAILED DATA EXPORTED =")
print(f"Saved {len(detailed_data)} data points to 'variant_analysis_detailed.csv'")

# Create enhanced visualization with value annotations
fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# Plot 1: Original overlaid tracks with peak annotations
axes[0].plot(ref_positions, ref_values, color='dimgrey', label='Reference', alpha=0.7)
axes[0].plot(alt_positions, alt_values, color='red', label='Alternate', alpha=0.7)

# Annotate peaks
ref_peak_idx = np.argmax(ref_values)
alt_peak_idx = np.argmax(alt_values)
axes[0].annotate(f'REF Peak: {ref_values[ref_peak_idx]:.4f}',
                xy=(ref_positions[ref_peak_idx], ref_values[ref_peak_idx]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))
axes[0].annotate(f'ALT Peak: {alt_values[alt_peak_idx]:.4f}',
                xy=(alt_positions[alt_peak_idx], alt_values[alt_peak_idx]),
                xytext=(10, -20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='orange', alpha=0.7))

axes[0].axvline(x=variant.position, color='blue', linestyle='--', alpha=0.8, label='Variant Position')
axes[0].set_title('RNA Expression: Reference vs Alternate')
axes[0].set_ylabel('Expression Level')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Difference plot
axes[1].plot(ref_positions, difference, color='purple', alpha=0.7)
axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
axes[1].axvline(x=variant.position, color='blue', linestyle='--', alpha=0.8)
axes[1].set_title('Expression Difference (Alternate - Reference)')
axes[1].set_ylabel('Difference')
axes[1].grid(True, alpha=0.3)

# Annotate significant changes
significant_indices = np.where(np.abs(difference) > np.std(difference))[0]
for idx in significant_indices[:5]:  # Show first 5 significant changes
    axes[1].annotate(f'{difference[idx]:.4f}',
                    xy=(ref_positions[idx], difference[idx]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.7)

# Plot 3: Fold change (log scale)
valid_fc = fold_change[np.isfinite(fold_change)]
valid_pos = ref_positions[np.isfinite(fold_change)]
axes[2].plot(valid_pos, np.log2(valid_fc), color='green', alpha=0.7)
axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
axes[2].axvline(x=variant.position, color='blue', linestyle='--', alpha=0.8)
axes[2].set_title('Fold Change (Log2 Scale)')
axes[2].set_ylabel('Log2(Fold Change)')
axes[2].set_xlabel('Chromosome Position')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comprehensive_variant_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Create summary statistics table
summary_df = pd.DataFrame({
    'Metric': ['Max Expression', 'Mean Expression', 'Peak Count (>95th percentile)',
               'Max Difference', 'Mean Absolute Difference', 'Positions with Significant Change'],
    'Reference': [ref_stats['max_value'], ref_stats['mean_value'], len(ref_stats['peak_values']),
                 '-', '-', '-'],
    'Alternate': [alt_stats['max_value'], alt_stats['mean_value'], len(alt_stats['peak_values']),
                 '-', '-', '-'],
    'Impact': ['-', '-', '-', impact_stats['max_difference'],
              np.mean(np.abs(difference)), impact_stats['significantly_changed_positions']]
})

print("\n= SUMMARY STATISTICS TABLE =")
print(summary_df.to_string(index=False))

# Save summary
summary_df.to_csv('variant_summary_statistics.csv', index=False)

print("\n= DETAILED DATA DATAFRAME =")
display(detailed_data.head())
