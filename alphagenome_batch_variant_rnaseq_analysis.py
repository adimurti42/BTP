from alphagenome.data import genome
from alphagenome.models import dna_client
from alphagenome.visualization import plot_components
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from typing import List, Dict

class VariantAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = dna_client.create(api_key)
        self.results = []

    def read_variants_from_file(self, file_path: str, file_format: str = 'csv'):
        """
        Read variants from file. Expected columns:
        - chromosome, position, reference_bases, alternate_bases
        - Optional: gene_name, variant_id
        """
        if file_format.lower() == 'csv':
            df = pd.read_csv(file_path)
        elif file_format.lower() in ['xlsx', 'excel']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Supported formats: csv, xlsx, excel")

        # Validate required columns
        required_cols = ['chromosome', 'position', 'reference_bases', 'alternate_bases']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"File must contain columns: {required_cols}")

        variants = []
        for _, row in df.iterrows():
            variant = {
                'chromosome': str(row['chromosome']),
                'position': int(row['position']),
                'reference_bases': str(row['reference_bases']),
                'alternate_bases': str(row['alternate_bases']),
                'gene_name': row.get('gene_name', 'Unknown'),
                'variant_id': row.get('variant_id', f"{row['chromosome']}:{row['position']}")
            }
            variants.append(variant)

        print(f"Loaded {len(variants)} variants from {file_path}")
        return variants

    def analyze_variant_batch(self, variants: List[Dict],
                            ontology_terms: List[str] = ['UBERON:0001157'],
                            interval_size: int = dna_client.SEQUENCE_LENGTH_100KB): # Changed interval_size to a supported length
        """Process multiple variants in batch"""

        all_results = []
        failed_variants = []

        for i, variant_info in enumerate(variants):
            try:
                print(f"Processing variant {i+1}/{len(variants)}: {variant_info['variant_id']}")

                # Create genome objects
                variant = genome.Variant(
                    chromosome=variant_info['chromosome'],
                    position=variant_info['position'],
                    reference_bases=variant_info['reference_bases'],
                    alternate_bases=variant_info['alternate_bases']
                )

                # Define interval around variant using resize
                interval = variant.reference_interval.resize(interval_size) # Use resize method

                # Get predictions
                outputs = self.model.predict_variant(
                    interval=interval,
                    variant=variant,
                    ontology_terms=ontology_terms,
                    requested_outputs=[dna_client.OutputType.RNA_SEQ],
                )

                # Extract data
                result = self.extract_variant_data(outputs, variant_info)
                all_results.append(result)

                # Plot the variant results
                self.plot_variant_results(outputs, variant, variant_info)


                # Small delay to avoid rate limiting
                time.sleep(0.1)

            except Exception as e:
                print(f"Failed to process {variant_info['variant_id']}: {str(e)}")
                failed_variants.append(variant_info)
                continue

        print(f"Successfully processed: {len(all_results)}")
        print(f"Failed variants: {len(failed_variants)}")

        return all_results, failed_variants


    def extract_variant_data(self, outputs, variant_info):
        """Extract numerical data from a single variant prediction"""

        ref_values = outputs.reference.rna_seq.values
        alt_values = outputs.alternate.rna_seq.values

        # Calculate statistics
        difference = alt_values - ref_values
        fold_change = np.where(ref_values != 0, alt_values / ref_values, 0)

        result = {
            'variant_id': variant_info['variant_id'],
            'gene_name': variant_info['gene_name'],
            'chromosome': variant_info['chromosome'],
            'position': variant_info['position'],
            'ref_bases': variant_info['reference_bases'],
            'alt_bases': variant_info['alternate_bases'],

            # Reference statistics
            'ref_max': np.max(ref_values),
            'ref_mean': np.mean(ref_values),
            'ref_sum': np.sum(ref_values),

            # Alternate statistics
            'alt_max': np.max(alt_values),
            'alt_mean': np.mean(alt_values),
            'alt_sum': np.sum(alt_values),

            # Impact metrics
            'max_difference': np.max(np.abs(difference)),
            'mean_difference': np.mean(difference),
            'max_fold_change': np.max(fold_change[np.isfinite(fold_change)]) if np.any(np.isfinite(fold_change)) else 0,
            'upregulated_positions': np.sum(difference > 0),
            'downregulated_positions': np.sum(difference < 0),
            'significant_changes': np.sum(np.abs(difference) > np.std(difference)),

            # Raw data for detailed analysis
            'ref_values': ref_values,
            'alt_values': alt_values,
            'positions': np.arange(outputs.reference.rna_seq.interval.start,
                                 outputs.reference.rna_seq.interval.end)
        }

        return result

    def plot_variant_results(self, outputs, variant, variant_info):
        """Plots the RNA-seq predictions for a single variant."""
        try:
            print(f"  - Generating plot for {variant_info['variant_id']}")
            plot_components.plot(
                [
                    plot_components.OverlaidTracks(
                        tdata={
                            'REF': outputs.reference.rna_seq,
                            'ALT': outputs.alternate.rna_seq,
                        },
                        colors={'REF': 'dimgrey', 'ALT': 'red'},
                    ),
                ],
                interval=outputs.reference.rna_seq.interval.resize(2**15), # Zoom in for better visualization
                annotations=[plot_components.VariantAnnotation([variant], alpha=0.8)],
            )
            plt.suptitle(f"Variant Analysis: {variant_info['variant_id']} ({variant_info.get('gene_name', 'Unknown Gene')})", y=1.02)
            plt.show()
        except Exception as e:
            print(f"  - Error generating plot for {variant_info['variant_id']}: {e}")
            # Continue processing other variants even if plotting fails


# Usage example
API_KEY = 'YOUR_API_KEY' #Enter your Alphagenome API key here
analyzer = VariantAnalyzer(API_KEY)

# Process variants from file
variants = analyzer.read_variants_from_file('my_mutations2.csv')
results, failed = analyzer.analyze_variant_batch(variants)

# Convert results to DataFrame and save
if results:
    results_df = pd.DataFrame([{k: v for k, v in result.items()
                              if not isinstance(v, np.ndarray)} for result in results])
    results_df.to_csv('batch_variant_analysis.csv', index=False)
    print("Results saved to batch_variant_analysis.csv")
