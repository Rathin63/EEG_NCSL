"""
Script to load two EDF files and print the channels they have in common.
Also shows channels unique to each file.
"""

import mne
import os
import numpy as np
from typing import List, Set, Tuple

def clean_channel_name(channel_name: str) -> str:
    """
    Clean channel name by removing 'POL' prefix if present.

    Parameters
    ----------
    channel_name : str
        Raw channel name from EDF file

    Returns
    -------
    str
        Cleaned channel name
    """
    if 'POL' in channel_name:
        return channel_name.split("POL")[1].strip()
    return channel_name.strip()

def load_edf_channels(edf_path: str, verbose: bool = True) -> Tuple[List[str], List[str]]:
    """
    Load EDF file and extract channel names.

    Parameters
    ----------
    edf_path : str
        Path to EDF file
    verbose : bool
        Whether to print loading information

    Returns
    -------
    raw_channels : List[str]
        Original channel names from file
    clean_channels : List[str]
        Cleaned channel names
    """
    if not os.path.exists(edf_path):
        raise FileNotFoundError(f"EDF file not found: {edf_path}")

    if verbose:
        print(f"Loading: {edf_path}")

    # Load EDF without preloading data (just to get channel info)
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)

    # Get raw channel names
    raw_channels = raw.ch_names.copy()

    # Clean channel names
    clean_channels = [clean_channel_name(ch) for ch in raw_channels]

    if verbose:
        print(f"  - Found {len(clean_channels)} channels")
        print(f"  - Sampling rate: {raw.info['sfreq']} Hz")
        print(f"  - Duration: {raw.n_times / raw.info['sfreq']:.1f} seconds")

    return raw_channels, clean_channels

def compare_edf_channels(file1_path: str, file2_path: str, use_clean_names: bool = True) -> dict:
    """
    Compare channels between two EDF files.

    Parameters
    ----------
    file1_path : str
        Path to first EDF file
    file2_path : str
        Path to second EDF file
    use_clean_names : bool
        Whether to use cleaned channel names (remove 'POL' prefix) for comparison

    Returns
    -------
    dict
        Dictionary containing comparison results
    """
    print("=" * 60)
    print("EDF CHANNEL COMPARISON")
    print("=" * 60)

    # Extract filenames for display
    file1_name = os.path.basename(file1_path)
    file2_name = os.path.basename(file2_path)

    print(f"\nFile 1: {file1_name}")
    raw_channels1, clean_channels1 = load_edf_channels(file1_path)

    print(f"\nFile 2: {file2_name}")
    raw_channels2, clean_channels2 = load_edf_channels(file2_path)

    # Choose which channel names to use for comparison
    if use_clean_names:
        channels1 = clean_channels1
        channels2 = clean_channels2
        print("\n[Using cleaned channel names for comparison]")
    else:
        channels1 = raw_channels1
        channels2 = raw_channels2
        print("\n[Using raw channel names for comparison]")

    # Convert to sets for comparison
    set1 = set(channels1)
    set2 = set(channels2)

    # Find common channels
    common_channels = sorted(list(set1.intersection(set2)))

    # Find unique channels
    unique_to_file1 = sorted(list(set1 - set2))
    unique_to_file2 = sorted(list(set2 - set1))

    # Print results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    print(f"\n📊 Summary:")
    print(f"  - File 1 channels: {len(channels1)}")
    print(f"  - File 2 channels: {len(channels2)}")
    print(f"  - Common channels: {len(common_channels)}")
    print(f"  - Unique to File 1: {len(unique_to_file1)}")
    print(f"  - Unique to File 2: {len(unique_to_file2)}")

    print(f"\n✓ Common Channels ({len(common_channels)}):")
    if common_channels:
        # Print in columns for better readability
        n_cols = 4
        for i in range(0, len(common_channels), n_cols):
            row = common_channels[i:i+n_cols]
            print("  " + "  ".join(f"{ch:15s}" for ch in row))
    else:
        print("  [No common channels found]")

    if unique_to_file1:
        print(f"\n⚠ Unique to {file1_name} ({len(unique_to_file1)}):")
        for i in range(0, len(unique_to_file1), 4):
            row = unique_to_file1[i:i+4]
            print("  " + "  ".join(f"{ch:15s}" for ch in row))

    if unique_to_file2:
        print(f"\n⚠ Unique to {file2_name} ({len(unique_to_file2)}):")
        for i in range(0, len(unique_to_file2), 4):
            row = unique_to_file2[i:i+4]
            print("  " + "  ".join(f"{ch:15s}" for ch in row))

    # Return results dictionary
    results = {
        'file1_name': file1_name,
        'file2_name': file2_name,
        'file1_channels': channels1,
        'file2_channels': channels2,
        'common_channels': common_channels,
        'unique_to_file1': unique_to_file1,
        'unique_to_file2': unique_to_file2,
        'n_common': len(common_channels),
        'n_file1': len(channels1),
        'n_file2': len(channels2)
    }

    return results

def check_channel_order(file1_path: str, file2_path: str, use_clean_names: bool = True):
    """
    Check if common channels appear in the same order in both files.

    Parameters
    ----------
    file1_path : str
        Path to first EDF file
    file2_path : str
        Path to second EDF file
    use_clean_names : bool
        Whether to use cleaned channel names
    """
    print("\n" + "=" * 60)
    print("CHANNEL ORDER CHECK")
    print("=" * 60)

    # Load channels
    raw_channels1, clean_channels1 = load_edf_channels(file1_path, verbose=False)
    raw_channels2, clean_channels2 = load_edf_channels(file2_path, verbose=False)

    if use_clean_names:
        channels1 = clean_channels1
        channels2 = clean_channels2
    else:
        channels1 = raw_channels1
        channels2 = raw_channels2

    # Find common channels
    common = sorted(list(set(channels1).intersection(set(channels2))))

    if not common:
        print("No common channels to check order.")
        return

    # Check order for common channels
    print(f"\nChecking order of {len(common)} common channels:")

    order_preserved = True
    for ch in common:
        idx1 = channels1.index(ch)
        idx2 = channels2.index(ch)
        if idx1 != idx2:
            order_preserved = False
            print(f"  ⚠ '{ch}': position {idx1} in file1, position {idx2} in file2")

    if order_preserved:
        print("  ✓ All common channels appear in the same order in both files")
    else:
        print("\n  Note: Channel order differs between files")

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Configure your file paths here
    PATIENT_ID = 'PY23N009'
    EEG_DIR = f"D:\\{PATIENT_ID}\\3_segmented"

    # Specify the two EDF files to compare
    FILE1 = "R1_2"  # First file (without .edf extension)
    FILE2 = "R1_3"  # Second file (without .edf extension)

    # Build full paths
    file1_path = os.path.join(EEG_DIR, f"{FILE1}.edf")
    file2_path = os.path.join(EEG_DIR, f"{FILE2}.edf")

    try:
        # Compare channels
        results = compare_edf_channels(
            file1_path,
            file2_path,
            use_clean_names=True  # Set to False to use raw channel names
        )

        # Check channel order
        check_channel_order(file1_path, file2_path, use_clean_names=True)

        # Optional: Save results to file
        save_results = False  # Set to True to save results
        if save_results:
            import json
            output_file = f"channel_comparison_{FILE1}_vs_{FILE2}.json"

            # Convert results to JSON-serializable format
            save_data = {
                'comparison_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'file1': results['file1_name'],
                'file2': results['file2_name'],
                'statistics': {
                    'n_channels_file1': results['n_file1'],
                    'n_channels_file2': results['n_file2'],
                    'n_common_channels': results['n_common']
                },
                'common_channels': results['common_channels'],
                'unique_to_file1': results['unique_to_file1'],
                'unique_to_file2': results['unique_to_file2']
            }

            with open(output_file, 'w') as f:
                json.dump(save_data, f, indent=4)
            print(f"\n✓ Results saved to: {output_file}")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Please check that the file paths are correct.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# ALTERNATIVE: Batch comparison for multiple file pairs
# ============================================================================

def batch_compare_channels(eeg_dir: str, file_pairs: List[Tuple[str, str]], use_clean_names: bool = True):
    """
    Compare channels for multiple pairs of EDF files.

    Parameters
    ----------
    eeg_dir : str
        Directory containing EDF files
    file_pairs : List[Tuple[str, str]]
        List of file pairs to compare (without .edf extension)
    use_clean_names : bool
        Whether to use cleaned channel names
    """
    print("\n" + "=" * 60)
    print("BATCH CHANNEL COMPARISON")
    print("=" * 60)

    all_results = []

    for file1, file2 in file_pairs:
        print(f"\n\nComparing {file1} vs {file2}...")
        print("-" * 40)

        file1_path = os.path.join(eeg_dir, f"{file1}.edf")
        file2_path = os.path.join(eeg_dir, f"{file2}.edf")

        try:
            results = compare_edf_channels(file1_path, file2_path, use_clean_names)
            all_results.append(results)
        except Exception as e:
            print(f"Error comparing {file1} vs {file2}: {e}")
            continue

    # Summary of all comparisons
    print("\n" + "=" * 60)
    print("BATCH SUMMARY")
    print("=" * 60)

    for result in all_results:
        print(f"\n{result['file1_name']} vs {result['file2_name']}:")
        print(f"  Common: {result['n_common']}/{result['n_file1']} and {result['n_common']}/{result['n_file2']}")

    return all_results

# Example batch usage (uncomment to use):
if __name__ == "__main__":
    PATIENT_ID = 'PY23N017'
    EEG_DIR = f"D:\\{PATIENT_ID}\\3_segmented"

    # Define pairs of files to compare
    file_pairs = [
        ("F5_3", "FO_2"),
    ]

    batch_results = batch_compare_channels(EEG_DIR, file_pairs, use_clean_names=True)
