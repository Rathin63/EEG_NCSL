import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import seaborn as sns
import pandas as pd
import mne
from tqdm.auto import tqdm
import copy
import pickle
import os, sys
from datetime import datetime
from scipy.signal import iirnotch, butter, filtfilt

# Add parent directory to path to access utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import from utils
from utils import (
    estimateA_subject,
    identifySS,
    create_folders,
    viz_eeg,
    _xlogx,
    compute_spectrogram,
    tukeys_method,
    sec2time,
    time_sum
    # ... any other functions you need from utils
)

class EEGAnalysisPipeline:
    """Main pipeline class for EEG analysis."""

    def __init__(self, config):
        """Initialize pipeline with configuration."""
        self.config = config
        self.data = None
        self.labels = None
        self.fs = None
        self.fs0 = None
        self.A_hat = None
        self.xhat = None
        self.correlation_results = None
        self.sink_indices = None
        self.metadata_path = config.get('metadata_path', None)
        self.metadata = None

        # Load processing parameters from metadata and merge with config
        self._load_processing_parameters_from_metadata()

        self.enable_caching = config.get('enable_caching', True)  # Enable caching by default
        self.force_recompute = config.get('force_recompute', False)  # Force recomputation if True

        if config["verbose"]:
            self.print_header("EEG ANALYSIS PIPELINE")
            self.print_config()

    def _get_file_path(self, feature_type):
        """Get the file path for a specific feature type."""
        filename = self.config['eeg_filename']

        paths = {
            'a_matrices': f"{self.config['a_matrices_dir']}\\{filename}_A.mat",
            'sink_indices': f"{self.config['sink_indices_dir']}\\{filename}_SI.mat",
            'reconstruction': f"{self.config['reconstruction_dir']}\\{filename}_recon.mat",
            'entropy': f"{self.config['entropy_dir']}\\{filename}_entropy.mat",
            'entropy_pickle': f"{self.config['entropy_dir']}\\{filename}_entropy.pkl",
            'energy': f"{self.config['energy_dir']}\\{filename}_energy.mat",
            'energy_pickle': f"{self.config['energy_dir']}\\{filename}_energy.pkl",
            'outliers': f"{self.config['outlier_dir']}\\{filename}_outliers.pkl",
            'correlation': f"{self.config['reconstruction_dir']}\\correlation_{self.config['patient']}_{filename}.pkl"
        }

        return paths.get(feature_type)

    def _load_processing_parameters_from_metadata(self):
        """
        Load processing parameters from metadata Excel file.

        Returns
        -------
        dict
            Dictionary containing processing parameters
        """
        import pandas as pd

        if not self.metadata_path or not os.path.exists(self.metadata_path):
            if self.config.get("verbose", True):
                print(f"  Warning: Metadata file not found at {self.metadata_path}. Using config defaults.")
            return {}

        try:
            df = pd.read_excel(self.metadata_path)
            patient_row = df[df['patient_no'] == self.config['patient']]

            if patient_row.empty:
                if self.config.get("verbose", True):
                    print(f"  Warning: Patient {self.config['patient']} not found in metadata. Using config defaults.")
                return {}

            # Extract parameters if they exist
            params = {}
            for param in ['l_freq', 'h_freq', 'fs', 'downsample_factor']:
                if param in df.columns and pd.notna(patient_row[param].iloc[0]):
                    params[param] = patient_row[param].iloc[0]

            if self.config.get("verbose", True) and params:
                print(f"  Loaded processing parameters from metadata: {params}")

            # Update config with metadata parameters (metadata takes precedence)
            for param, value in params.items():
                if param in ['l_freq', 'h_freq']:
                    self.config[param] = float(value)
                elif param in ['downsample_factor']:
                    if value >= 0:
                        self.config[param] = int(value)  # e.g. 2, 4
                    else:
                        self.config[param] = float(value)  # e.g. 0.5
                elif param == 'fs':
                    # Store as mat_fs for MAT file loading
                    # self.config['eeg_params']['fs'] = int(value)
                    self.config['mat_fs'] = int(value)
            return params

        except Exception as e:
            if self.config.get("verbose", True):
                print(f"  Warning: Failed to load processing parameters from metadata: {e}")
            return {}

    def _feature_exists(self, feature_type):
        """Check if a feature file exists."""
        if not self.enable_caching or self.force_recompute:
            return False

        file_path = self._get_file_path(feature_type)
        if file_path and os.path.exists(file_path):
            if self.config.get("verbose", True):
                print(f"Found existing {feature_type} file: {file_path}")
            return True
        return False

    def _load_a_matrices(self):
        """Load existing A matrices."""
        file_path = self._get_file_path('a_matrices')
        try:
            a_data = sio.loadmat(file_path)
            self.A_hat = a_data['A_hat']

            # newly added
            self.a_win_size = float(a_data['win_size'][0, 0]) if a_data['win_size'].ndim > 0 else float(a_data['win_size'])
            self.a_n_windows = self.A_hat.shape[2]
            self.a_time_axis = np.arange(self.a_n_windows) * self.a_win_size

            self.fs = float(a_data['fs'][0, 0]) if a_data['fs'].ndim > 0 else float(a_data['fs'])
            self.fs0 = float(a_data['fs0'][0, 0]) if a_data['fs0'].ndim > 0 else float(a_data['fs0'])

            # Load labels - handle different formats
            if 'labels' in a_data:
                labels = a_data['labels']
                if isinstance(labels[0], np.ndarray):
                    self.a_labels = [str(lab[0]) if isinstance(lab, np.ndarray) and lab.size > 0 else str(lab)
                                     for lab in labels]
                else:
                    self.a_labels = [str(lab) for lab in labels]

                # Also update the general labels if not already set
                if self.labels is None:
                    self.labels = self.a_labels

            print(f"Loaded A matrices from cache: shape {self.A_hat.shape}")
            return True
        except Exception as e:
            print(f"Error loading A matrices: {e}")
            return False

    def _load_sink_indices(self):
        """Load existing sink indices."""
        file_path = self._get_file_path('sink_indices')
        try:
            si_data = sio.loadmat(file_path)

            self.sink_indices = {
                "sink_wins": si_data['sink_wins'],
                "source_wins": si_data['source_wins'],
                "row_ranks": si_data['row_ranks'],
                "col_ranks": si_data['col_ranks'],
                "SI_overall": si_data.get('SI_overall', np.array([]))
            }

            # Load labels if we don't have them
            if 'labels' in si_data and self.labels is None:
                labels = si_data['labels']
                if isinstance(labels[0], np.ndarray):
                    self.labels = [str(lab[0]) if isinstance(lab, np.ndarray) else str(lab) for lab in labels]
                else:
                    self.labels = [str(lab) for lab in labels]

            print(f"Loaded sink indices from cache: shape {self.sink_indices['sink_wins'].shape}")
            return True
        except Exception as e:
            print(f"Error loading sink indices: {e}")
            return False

    def _load_reconstruction(self):
        """Load existing reconstruction."""
        file_path = self._get_file_path('reconstruction')
        try:
            recon_data = sio.loadmat(file_path)

            self.data = recon_data['data']
            self.xhat = recon_data['data_recon']
            self.fs = float(recon_data['fs'][0, 0]) if recon_data['fs'].ndim > 0 else float(recon_data['fs'])
            self.fs0 = float(recon_data['fs0'][0, 0]) if recon_data['fs0'].ndim > 0 else float(recon_data['fs0'])

            # Load labels if we don't have them
            if 'labels' in recon_data and self.labels is None:
                labels = recon_data['labels']
                if isinstance(labels[0], np.ndarray):
                    self.labels = [str(lab[0]) if isinstance(lab, np.ndarray) else str(lab) for lab in labels]
                else:
                    self.labels = [str(lab) for lab in labels]

            print(f"Loaded reconstruction from cache: shape {self.data.shape}")
            return True
        except Exception as e:
            print(f"Error loading reconstruction: {e}")
            return False

    def _load_entropy(self):
        """Load existing entropy data."""
        # Try pickle first (more reliable), then MAT
        pkl_path = self._get_file_path('entropy_pickle')
        mat_path = self._get_file_path('entropy')

        try:
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as f:
                    entropy_data = pickle.load(f)
                print(f"Loaded entropy from pickle cache")
            elif os.path.exists(mat_path):
                entropy_data = sio.loadmat(mat_path)
                print(f"Loaded entropy from MAT cache")
            else:
                return False

            self.entropy_sink = entropy_data['entropy_sink']
            self.entropy_source = entropy_data['entropy_source']

            # Load z-scored entropy if available
            if 'entropy_sink_zscore' in entropy_data:
                self.entropy_sink_zscore = entropy_data['entropy_sink_zscore']
                self.entropy_source_zscore = entropy_data['entropy_source_zscore']

            # Load labels if we don't have them
            if 'labels' in entropy_data and self.labels is None:
                self.labels = entropy_data['labels']

            print(f"Loaded entropy data: sink shape {self.entropy_sink.shape}")
            return True
        except Exception as e:
            print(f"Error loading entropy: {e}")
            return False

    def _load_energy(self):
        """Load existing energy data."""
        # Try pickle first, then MAT
        pkl_path = self._get_file_path('energy_pickle')
        mat_path = self._get_file_path('energy')

        try:
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as f:
                    energy_data = pickle.load(f)
                print(f"Loaded energy from pickle cache")
            elif os.path.exists(mat_path):
                energy_data = sio.loadmat(mat_path)
                print(f"Loaded energy from MAT cache")
            else:
                return False

            self.energy = energy_data['energy']
            self.energy_normalized = energy_data['energy_normalized']
            self.energy_time_axis = energy_data['energy_time_axis']

            print(f"Loaded energy data: shape {self.energy.shape}")
            return True
        except Exception as e:
            print(f"Error loading energy: {e}")
            return False

    def _load_outliers(self):
        """Load existing outlier results."""
        file_path = self._get_file_path('outliers')
        try:
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    self.outlier_results = pickle.load(f)
                print(f"Loaded outlier results from cache")
                return True
            return False
        except Exception as e:
            print(f"Error loading outliers: {e}")
            return False

    def _load_correlation(self):
        """Load existing correlation results."""
        file_path = self._get_file_path('correlation')
        try:
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    self.correlation_results = pickle.load(f)
                print(f"Loaded correlation results from cache")
                return True
            return False
        except Exception as e:
            print(f"Error loading correlation: {e}")
            return False

    def print_header(self, text):
        """Print formatted header."""
        print("\n" + "="*60)
        print(f"{text:^60}")
        print("="*60)

    def print_step(self, step_num, text):
        """Print formatted step."""
        print(f"\n{'='*60}")
        print(f"STEP {step_num}: {text}")
        print("="*60)

    def print_config(self):
        """Print configuration summary."""
        print("\nConfiguration Summary:")
        print("-" * 40)
        print(f"Patient: {self.config.get('patient', None)}")
        print(f"EEG File: {self.config.get('eeg_filename', None)}")
        print(f"Filters: {self.config.get('l_freq', None)}-{self.config.get('h_freq', None)} Hz")
        print(f"Notch: {self.config.get('notch_freq', None)} Hz")
        print(f"Downsampling: {self.config.get('downsample_factor', None)}x")
        print(f"Original FS: {self.config.get('mat_fs', 'auto')} Hz")
        print(f"Window size: {self.config.get('win_size', None)} seconds")
        if self.metadata_path:
            print(f"Metadata: {os.path.basename(self.metadata_path)}")
        print("-" * 40)

    # ------------------------------------------------------------------------
    # STEP 1: Load and Preprocess Data
    # ------------------------------------------------------------------------

    def load_and_preprocess(self):
        """Load EDF or MAT file and apply preprocessing."""
        self.print_step(1, "LOAD AND PREPROCESS DATA")

        # Check for EDF file first
        edf_path = f"{self.config['data_location']}\\{self.config['eeg_filename']}.edf"
        mat_path = f"{self.config['data_location']}\\{self.config['eeg_filename']}.mat"
        eeg_path = f"{self.config['data_location']}\\{self.config['eeg_filename']}.eeg"

        if os.path.exists(edf_path):
            # ====== LOAD EDF FILE ======
            print(f"Loading EDF: {edf_path}")

            raw = mne.io.read_raw_edf(edf_path, preload=True)
            self.fs0 = int(raw.info['sfreq'])

            # Get labels and clean them
            self.labels = copy.deepcopy(raw.ch_names)
            for i, label in enumerate(self.labels):
                if 'POL' in label:
                    self.labels[i] = label.split("POL")[1].strip()

            print(f"Channels: {len(self.labels)}")
            print(f"Original sampling rate: {self.fs0} Hz")

            # Apply filters using MNE
            print(f"Applying notch filter at {self.config['notch_freq']} Hz...")
            raw.notch_filter(freqs=self.config['notch_freq'])

            print(f"Applying bandpass filter: {self.config['l_freq']}-{self.config['h_freq']} Hz...")
            raw.filter(l_freq=self.config['l_freq'], h_freq=self.config['h_freq'])

            # Get data and resample
            data_full = raw.get_data()

            if self.config['downsample_factor'] < 1:
                # UPSAMPLING
                upsample_factor = int(1 / self.config['downsample_factor'])
                print(f"Upsampling by factor of {upsample_factor}...")

                from scipy import signal
                self.data = signal.resample_poly(data_full, upsample_factor, 1, axis=1)
                self.fs = int(self.fs0 * upsample_factor)
                print(f"Upsampled to: {self.fs} Hz")

            elif self.config['downsample_factor'] > 1:
                # DOWNSAMPLING
                self.data = data_full[:, ::self.config['downsample_factor']]
                self.fs = self.fs0 // self.config['downsample_factor']
                print(f"Downsampled to: {self.fs} Hz")

            else:  # downsample_factor == 1
                # NO RESAMPLING
                self.data = data_full
                self.fs = self.fs0

        elif os.path.exists(mat_path):
            # ====== LOAD MAT FILE ======
            print(f"Loading MAT: {mat_path}")

            # Load data from MAT file
            mat_data = sio.loadmat(mat_path)
            if 'pt_data' in mat_data.keys():
                self.data = mat_data['pt_data'][0]["data_filt"][0]
            elif 'data_filt' in mat_data.keys():
                self.data = mat_data['data_filt'][0][0]
            elif 'EEG' in mat_data.keys():
                if mat_data['EEG'].shape[1] > mat_data['EEG'].shape[0]:
                    self.data = mat_data['EEG']
                else:
                    self.data = mat_data['EEG'].T
            # Get sampling frequency from config or default
            self.fs0 = self.config.get('mat_fs', 1000)  # Default to 1000 Hz if not specified
            print(f"Using sampling rate: {self.fs0} Hz")

            # Get labels from MAT file or generate default ones
            if 'labels' in mat_data.keys():
                if len(mat_data['labels']) > 1:
                    self.labels = [str(label).strip() for label in mat_data['labels']]
                elif len(mat_data['labels'][0]) > 1:
                    self.labels = [str(label).strip() for label in mat_data['labels'][0]]
                elif len(mat_data['labels'][0][0]) > 1:
                    self.labels = [str(label).strip() for label in mat_data['labels'][0][0]]
                else:
                    self.labels = [f"CH{i:02d}" for i in range(self.data.shape[0])]
                    print(f"Generated random channel labels for: {self.config['eeg_filename']}")
            elif 'pt_data' in mat_data.keys():
                self.labels = [chi_i[0][0] for chi_i in mat_data['pt_data'][0][0][0]]
            else:
                self.labels = [f"CH{i:02d}" for i in range(self.data.shape[0])]
                print(f"Generated random channel labels for: {self.config['eeg_filename']}")

            print(f"Channels: {len(self.labels)}")
            print(f"Data shape before filtering: {self.data.shape}")

            # Apply custom filter function
            # from scipy.signal import iirnotch, butter, filtfilt

            def filter_data(X, l_freq, h_freq, sfreq):
                """Apply notch and bandpass filters to data."""
                # Apply notch filter at 60 Hz
                notch_freq = self.config['notch_freq']
                quality_factor = 30
                b_notch, a_notch = iirnotch(notch_freq, quality_factor, sfreq)
                X_notched = filtfilt(b_notch, a_notch, X, axis=1)

                # Design and apply band-pass filter
                nyquist = 0.5 * sfreq
                low = l_freq / nyquist
                high = h_freq / nyquist
                b_band, a_band = butter(N=4, Wn=[low, high], btype='band')
                X_filtered = filtfilt(b_band, a_band, X_notched, axis=1)

                return X_filtered

            print(f"Applying filters: {self.config['l_freq']}-{self.config['h_freq']} Hz...")
            self.data = filter_data(
                self.data,
                self.config['l_freq'],
                self.config['h_freq'],
                self.fs0
            )

            # Resample if needed
            if self.config['downsample_factor'] < 1:
                # UPSAMPLING
                upsample_factor = int(1 / self.config['downsample_factor'])
                print(f"Upsampling by factor of {upsample_factor}...")

                from scipy import signal
                self.data = signal.resample_poly(self.data, upsample_factor, 1, axis=1)
                self.fs = int(self.fs0 * upsample_factor)
                print(f"Upsampled to: {self.fs} Hz")

            elif self.config['downsample_factor'] > 1:
                # DOWNSAMPLING
                self.data = self.data[:, ::self.config['downsample_factor']]
                self.fs = self.fs0 // self.config['downsample_factor']
                print(f"Downsampled to: {self.fs} Hz")

            else:  # downsample_factor == 1
                # NO RESAMPLING
                self.fs = self.fs0

        elif os.path.exists(eeg_path):
            # ====== LOAD EEG FILE ======
            print(f"Loading EEG: {eeg_path}")

            # Try to load with MNE - it will auto-detect the format
            try:
                # First try as BrainVision format (most common for .eeg)
                # BrainVision requires .vhdr file
                vhdr_path = eeg_path.replace('.eeg', '.vhdr')
                if os.path.exists(vhdr_path):
                    print(f"Found BrainVision header file: {vhdr_path}")
                    raw = mne.io.read_raw_brainvision(vhdr_path, preload=True)
                else:
                    # Try generic MNE reader
                    print(f"Attempting to load with generic MNE reader...")
                    raw = mne.io.read_raw(eeg_path, preload=True)

            except Exception as e:
                print(f"MNE couldn't load the file directly: {e}")
                print("Attempting fallback to custom loader...")

                # Fallback: Try to load as generic binary/text
                # You may need to specify parameters based on your file format
                try:
                    # Try as EGI format
                    raw = mne.io.read_raw_egi(eeg_path, preload=True)
                except:
                    try:
                        # Try as Neuroscan format
                        raw = mne.io.read_raw_cnt(eeg_path, preload=True)
                    except:
                        # Last resort: load as array and create Raw object
                        print("Loading as generic array...")
                        # This assumes a simple binary format - adjust as needed
                        data = np.fromfile(eeg_path, dtype=np.float32)

                        # Estimate channels and samples (you may need to adjust this)
                        n_channels = 32  # Default assumption - adjust based on your data
                        n_samples = len(data) // n_channels
                        data = data.reshape(n_channels, n_samples)

                        # Create MNE Raw object from array
                        sfreq = self.config.get('mat_fs', 1000)
                        ch_names = [f'CH{i:02d}' for i in range(n_channels)]
                        ch_types = ['eeg'] * n_channels
                        info = mne.create_info(ch_names=ch_names,
                                               sfreq=sfreq,
                                               ch_types=ch_types)
                        raw = mne.io.RawArray(data, info)

            # Now process the loaded raw object (same as EDF processing)
            self.fs0 = int(raw.info['sfreq'])

            # Get labels and clean them
            self.labels = copy.deepcopy(raw.ch_names)
            for i, label in enumerate(self.labels):
                if 'POL' in label:
                    self.labels[i] = label.split("POL")[1].strip()

            print(f"Channels: {len(self.labels)}")
            print(f"Original sampling rate: {self.fs0} Hz")

            # Apply filters using MNE
            print(f"Applying notch filter at {self.config['notch_freq']} Hz...")
            raw.notch_filter(freqs=self.config['notch_freq'])

            print(f"Applying bandpass filter: {self.config['l_freq']}-{self.config['h_freq']} Hz...")
            raw.filter(l_freq=self.config['l_freq'], h_freq=self.config['h_freq'])

            # Get data and resample
            data_full = raw.get_data()

            if self.config['downsample_factor'] < 1:
                # UPSAMPLING
                upsample_factor = int(1 / self.config['downsample_factor'])
                print(f"Upsampling by factor of {upsample_factor}...")

                from scipy import signal
                self.data = signal.resample_poly(data_full, upsample_factor, 1, axis=1)
                self.fs = int(self.fs0 * upsample_factor)
                print(f"Upsampled to: {self.fs} Hz")

            elif self.config['downsample_factor'] > 1:
                # DOWNSAMPLING
                self.data = data_full[:, ::self.config['downsample_factor']]
                self.fs = self.fs0 // self.config['downsample_factor']
                print(f"Downsampled to: {self.fs} Hz")

            else:  # downsample_factor == 1
                # NO RESAMPLING
                self.data = data_full
                self.fs = self.fs0

        else:
            raise FileNotFoundError(f"No EDF or MAT file found for: {self.config['eeg_filename']}")

        print(f"Final data shape: {self.data.shape}")
        print(f"Duration: {self.data.shape[1] / self.fs:.1f} seconds")

        return self.data, self.labels, self.fs


    # ------------------------------------------------------------------------
    # STEP 2: Estimate A Matrices
    # ------------------------------------------------------------------------

    def estimate_a_matrices(self):
        """Estimate state-space model A matrices."""
        self.print_step(2, "ESTIMATE A MATRICES")

        # Check if A matrices already exist
        if self._feature_exists('a_matrices'):
            if self._load_a_matrices():
                return self.A_hat
            else:
                print("Failed to load cached A matrices, recomputing...")

        print(f"Window size: {self.config['win_size']} seconds")
        print(f"Samples per window: {int(self.config['win_size'] * self.fs)}")

        self.A_hat = estimateA_subject(
            self.data,
            fs=self.fs,
            winsize=self.config['win_size']
        )

        print(f"A matrices shape: {self.A_hat.shape}")
        print(f"Number of windows: {self.A_hat.shape[2]}")

        # Save A matrices if requested
        if self.config["save_results"]:
            self.save_a_matrices()

        return self.A_hat

    def save_a_matrices(self):
        """Save A matrices to file."""
        create_folders(self.config["a_matrices_dir"])

        A_data = {
            "patient": self.config["patient"],
            "EEGFileName": f"{self.config['eeg_filename']}.edf",
            "A_hat": self.A_hat,
            "win_size": self.config["win_size"],
            "factor": self.config["downsample_factor"],
            "fs": self.fs,
            "fs0": self.fs0,
            "labels": self.labels,
            "h_freq": self.config["h_freq"],
            "l_freq": self.config["l_freq"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        save_path = f"{self.config['a_matrices_dir']}\\{self.config['eeg_filename']}_A.mat"
        sio.savemat(save_path, A_data)
        print(f"Saved A matrices to: {save_path}")

    # ------------------------------------------------------------------------
    # STEP 3: Reconstruct Signal
    # ------------------------------------------------------------------------

    def reconstruct_signal(self):
        """Reconstruct signal using estimated A matrices."""
        self.print_step(3, "RECONSTRUCT SIGNAL")

        # Check if reconstruction already exists
        if self._feature_exists('reconstruction'):
            if self._load_reconstruction():
                return self.xhat
            else:
                print("Failed to load cached reconstruction, recomputing...")

        nWin = self.A_hat.shape[2]
        nCH = self.A_hat.shape[0]
        signal_length = self.data.shape[1]

        # Initialize reconstructed signal
        self.xhat = np.zeros((nCH, signal_length))
        self.xhat[:, 0] = self.data[:, 0]

        # Reconstruction loop
        progress_bar = tqdm(range(nWin), desc="Reconstructing")

        for i in progress_bar:
            progress_bar.set_description(f'Window {i+1}/{nWin}')
            A = self.A_hat[:, :, i]

            for j in range(1, int(self.config["win_size"] * self.fs)):
                idx = int(i * self.fs * self.config["win_size"]) + j
                if idx < signal_length:
                    self.xhat[:, idx] = A @ self.xhat[:, idx - 1]

            # Reset at window boundary
            if i < nWin - 1:
                next_idx = int((i + 1) * self.fs * self.config["win_size"])
                if next_idx < signal_length:
                    self.xhat[:, next_idx] = self.data[:, next_idx]

        print(f"Reconstruction complete!")

        # Save reconstruction if requested
        if self.config["save_results"]:
            self.save_reconstruction()

        # Create plots if requested
        if self.config["create_plots"]:
            self.plot_reconstruction_example()

        return self.xhat

    def save_reconstruction(self):
        """Save reconstructed signal."""
        create_folders(self.config["reconstruction_dir"])

        recon_data = {
            "patient": self.config["patient"],
            "EEGFileName": f"{self.config['eeg_filename']}.edf",
            "data": self.data,
            "data_recon": self.xhat,
            "win_size": self.config["win_size"],
            "factor": self.config["downsample_factor"],
            "fs": self.fs,
            "fs0": self.fs0,
            "labels": self.labels,
            "h_freq": self.config["h_freq"],
            "l_freq": self.config["l_freq"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        save_path = f"{self.config['reconstruction_dir']}\\{self.config['eeg_filename']}_recon.mat"
        sio.savemat(save_path, recon_data)
        print(f"Saved reconstruction to: {save_path}")

    def plot_reconstruction_example(self):
        """Plot example of reconstructed signal."""
        fig, ax = plt.subplots(1, figsize=(18, 5))

        start_idx = self.config["plot_start_sec"] * self.fs
        end_idx = self.config["plot_end_sec"] * self.fs
        channel_idx = min(15, self.data.shape[0] - 1)  # Use channel 15 or last available

        ax.plot(self.data[channel_idx, start_idx:end_idx] * 1e6,
                label='Original', alpha=0.7)
        ax.plot(self.xhat[channel_idx, start_idx:end_idx] * 1e6,
                label='Reconstructed', alpha=0.7)

        ylim0 = (self.data[channel_idx, start_idx:end_idx].mean() - self.data[channel_idx, start_idx:end_idx].std()) * 1e6
        ylim1 = (self.data[channel_idx, start_idx:end_idx].mean() + self.data[channel_idx, start_idx:end_idx].std()) \
                * 1e6
        ax.set_ylim([ylim0, ylim1])
        ax.set_xlabel('Sample')
        ax.set_ylabel('Amplitude (μV)')
        ax.set_title(
            f"Reconstruction Example - Channel {self.labels[channel_idx]}\n"
            f"Duration: {self.config['plot_end_sec'] - self.config['plot_start_sec']} seconds | "
            f"Window size: {self.config['win_size']} s | "
            f"Sampling: {self.fs} Hz"
        )
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------------
    # STEP 4: Correlation Analysis
    # ------------------------------------------------------------------------

    def analyze_correlations(self):
        """Perform windowed correlation analysis."""
        self.print_step(4, "CORRELATION ANALYSIS")

        # Check if correlation results already exist
        if self._feature_exists('correlation'):
            if self._load_correlation():
                return self.correlation_results
            else:
                print("Failed to load cached correlation, recomputing...")

        # Import the analysis function (assuming it's in utils or defined elsewhere)
        from utils import analyze_correlations

        self.correlation_results = analyze_correlations(
            self.data,
            self.xhat,
            self.labels,
            fs=self.fs,
            window_duration=self.config["corr_win_duration"],
            overlap=self.config["corr_overlap"]
        )

        # Save correlation results
        if self.config["save_results"]:
            self.save_correlation_results()

        return self.correlation_results

    def save_correlation_results(self):
        """Save correlation analysis results."""
        create_folders(self.config["reconstruction_dir"])

        # Save pickle file with all results
        pkl_path = f"{self.config['reconstruction_dir']}\\correlation_{self.config['patient']}_{self.config['eeg_filename']}.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(self.correlation_results, f)

        # Save CSV summary
        csv_path = f"{self.config['reconstruction_dir']}\\correlation_summary_{self.config['patient']}_{self.config['eeg_filename']}.csv"
        self.correlation_results['summary'].to_csv(csv_path, index=False)

        print(f"Saved correlation results to: {pkl_path}")
        print(f"Saved correlation summary to: {csv_path}")

    # ------------------------------------------------------------------------
    # STEP 5: Compute Sink Indices
    # ------------------------------------------------------------------------

    def compute_sink_indices(self):
        """Compute sink and source indices."""
        self.print_step(5, "COMPUTE SINK INDICES")

        # Check if sink indices already exist
        if self._feature_exists('sink_indices'):
            if self._load_sink_indices():
                return self.sink_indices
            else:
                print("Failed to load cached sink indices, recomputing...")

        nWin = self.A_hat.shape[2]
        nCh = self.A_hat.shape[0]

        # Initialize arrays
        sink_wins = np.zeros((nCh, nWin))
        source_wins = np.zeros((nCh, nWin))
        row_ranks = np.zeros((nCh, nWin))
        col_ranks = np.zeros((nCh, nWin))

        # Compute for each window
        print("Computing sink/source indices for each window...")
        for iW in tqdm(range(nWin)):
            A_win = self.A_hat[:, :, iW]
            sink_wins[:, iW], source_wins[:, iW], row_ranks[:, iW], col_ranks[:, iW] = identifySS(A_win)

        self.sink_indices = {
            "sink_wins": sink_wins,
            "source_wins": source_wins,
            "row_ranks": row_ranks,
            "col_ranks": col_ranks
        }

        # Compute overall indices
        A_mean = np.mean(self.A_hat, axis=2)
        SI_overall, _, _, _ = identifySS(A_mean)
        self.sink_indices["SI_overall"] = SI_overall

        print(f"Sink indices shape: {sink_wins.shape}")
        print(f"Top sink channel: {self.labels[np.argmax(SI_overall)]}")
        print(f"Top source channel: {self.labels[np.argmax(source_wins.mean(axis=1))]}")

        # Save sink indices
        if self.config["save_results"]:
            self.save_sink_indices()

        # Plot heatmap
        if self.config["create_plots"]:
            self.plot_sink_heatmap()

        return self.sink_indices

    def save_sink_indices(self):
        """Save sink indices."""
        create_folders(self.config["sink_indices_dir"])

        SI_data = {
            "patient": self.config["patient"],
            "EEGFileName": f"{self.config['eeg_filename']}.edf",
            "sink_wins": self.sink_indices["sink_wins"],
            "source_wins": self.sink_indices["source_wins"],
            "row_ranks": self.sink_indices["row_ranks"],
            "col_ranks": self.sink_indices["col_ranks"],
            "SI_overall": self.sink_indices["SI_overall"],
            "win_size": self.config["win_size"],
            "factor": self.config["downsample_factor"],
            "fs": self.fs,
            "fs0": self.fs0,
            "labels": self.labels,
            "h_freq": self.config["h_freq"],
            "l_freq": self.config["l_freq"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        save_path = f"{self.config['sink_indices_dir']}\\{self.config['eeg_filename']}_SI.mat"
        sio.savemat(save_path, SI_data)
        print(f"Saved sink indices to: {save_path}")

    def plot_sink_heatmap(self):
        """Plot sink index heatmap."""
        # Sort by overall sink index
        # SI_sort_idx = np.argsort(self.sink_indices["SI_overall"])[::-1]
        SI_sort_idx = np.argsort( self.sink_indices["sink_wins"].mean(axis=1) )[::-1]
        SI_wins_sorted = self.sink_indices["sink_wins"][SI_sort_idx, :]
        labels_sorted = [self.labels[i] for i in SI_sort_idx]

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(20, 12))

        sns.heatmap(
            SI_wins_sorted,
            yticklabels=labels_sorted,
            cmap=sns.color_palette("rainbow", as_cmap=True),
            cbar_kws={"pad": 0.01}
        )

        ax.set_title(
            f"Sink Index Heatmap - {self.config['patient']} - {self.config['eeg_filename']}\n"
            f"Window size: {self.config['win_size']}s | Channels sorted by mean sink index",
            fontsize=14
        )
        ax.set_xlabel("Window Index")
        ax.set_ylabel("Channel")

        plt.tight_layout()

        # Save the heatmap if save_results is enabled
        if self.config["save_results"]:
            # Create the heatmaps directory if it doesn't exist
            create_folders(self.config["heatmaps_dir"])

            # Define the save path
            heatmap_path = f"{self.config['heatmaps_dir']}\\{self.config['eeg_filename']}_sink_heatmap.png"

            # Save the figure
            plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
            print(f"Saved sink heatmap to: {heatmap_path}")

        plt.show()

    # ------------------------------------------------------------------------
    # STEP 6: Visualize EEG
    # ------------------------------------------------------------------------

    def visualize_eeg(self):
        """Create EEG visualization with reconstruction overlay."""
        self.print_step(6, "VISUALIZE EEG")

        print(f"Creating EEG plots...")
        print(f"Window length: {self.config['viz_win_len']} seconds")
        print(f"Window step: {self.config['viz_win_step']} seconds")

        viz_eeg(
            self.data * 0.1**self.config["eeg_zoom_out_factor"],
            labels=self.labels,
            fs=self.fs,
            WIN_LEN_SEC=self.config["viz_win_len"],
            WIN_STEP_SEC=self.config["viz_win_step"],
            fig_out_dir=self.config["reconstruction_dir"],
            fig_name=self.config["eeg_filename"],
            plot_title=f"{self.config['patient']} - {self.config['eeg_filename']}",
            start_time='0:0:0',
            data_recon=self.xhat * 0.1**self.config["eeg_zoom_out_factor"]
        )

        print(f"Saved EEG plots to: {self.config['reconstruction_dir']}")

    # ------------------------------------------------------------------------
    # STEP 7: Compute Spectral Entropy
    # ------------------------------------------------------------------------

    def _find_robust_baseline(self, entropy_channel, min_threshold=0.8, max_threshold=1.0,
                              default_mean=0.8, default_std=0.003):
        """
        Find robust baseline by looking at entropy values in expected range.

        Parameters:
        -----------
        entropy_channel : np.ndarray
            Entropy values for a single channel
        min_threshold : float
            Minimum expected baseline value
        max_threshold : float
            Maximum expected baseline value
        default_mean : float
            Default mean if insufficient baseline data
        default_std : float
            Default std if insufficient baseline data

        Returns:
        --------
        mean_baseline, std_baseline : float
            Robust baseline statistics
        """
        # Find values in expected baseline range
        baseline_mask = (entropy_channel >= min_threshold) & (entropy_channel <= max_threshold)
        baseline_values = entropy_channel[baseline_mask]

        # Need at least 10 samples for reliable statistics
        if len(baseline_values) >= 10:
            # Use median for robustness against outliers
            mean_baseline = np.median(baseline_values)
            # Use MAD (Median Absolute Deviation) for robust std estimation
            mad = np.median(np.abs(baseline_values - mean_baseline))
            std_baseline = 1.4826 * mad  # Scale MAD to approximate std

            # Sanity check on std
            if std_baseline < 1e-6:  # Too small, use default
                std_baseline = default_std
        else:
            # Not enough baseline data, use defaults
            mean_baseline = default_mean
            std_baseline = default_std

        return mean_baseline, std_baseline

    def compute_spectral_entropy_nonCausal(self):
        """Compute spectral entropy of sink and source indices using centered windows."""
        self.print_step(7, "COMPUTE SPECTRAL ENTROPY (CENTERED WINDOW)")

        # Check if entropy already exists
        if self._feature_exists('entropy') or self._feature_exists('entropy_pickle'):
            if self._load_entropy():
                return self.entropy_sink, self.entropy_source
            else:
                print("Failed to load cached entropy, recomputing...")

        if self.sink_indices is None:
            raise ValueError("Sink indices must be computed first. Run compute_sink_indices().")

        if self.metadata_path and os.path.isfile(self.metadata_path):
            import pandas as pd
            self.metadata_df = pd.read_excel(self.metadata_path)
            print(f"Loaded metadata from {self.metadata_path}")

        # Get dimensions
        nCH = self.sink_indices["sink_wins"].shape[0]
        nWin = self.sink_indices["sink_wins"].shape[1]

        # Calculate window parameters for entropy
        win_sec = self.config["win_size"]
        minu = self.config["entropy_window_minutes"]
        n = int(minu * 60 / win_sec)  # Number of windows for entropy computation
        half_n = n // 2  # Half window for centering

        print(f"Computing spectral entropy with centered windows...")
        print(f"  - Window for entropy: {minu} minutes ({n} windows)")
        print(f"  - Half window: {half_n} windows")
        print(f"  - Total windows: {nWin}")
        print(f"  - Channels: {nCH}")

        # Initialize entropy arrays
        self.entropy_sink = np.zeros((nCH, nWin))
        self.entropy_source = np.zeros((nCH, nWin))

        # Initialize z-scored entropy arrays if requested
        if self.config.get("compute_zscore_entropy", False):
            self.entropy_sink_zscore = np.zeros((nCH, nWin))
            self.entropy_source_zscore = np.zeros((nCH, nWin))

        # Progress bar
        pbar = tqdm(total=nCH, desc='Computing entropy for channels')

        # Compute entropy for each channel
        for chi in range(nCH):
            pbar.set_postfix({'Channel': self.labels[chi]})

            # Process SINK indices
            SI_channel = self.sink_indices["sink_wins"][chi, :]

            # PAD FOR CENTERED WINDOWS: pad with half_n on each side
            # Use mirror padding to maintain signal characteristics at edges
            SI_channel_padded = np.pad(SI_channel, (half_n, half_n), mode='reflect')

            # Compute spectrogram ONCE for the entire padded signal
            time_axis, frequency_axis, power_spectrogram = compute_spectrogram(
                input_vector=SI_channel_padded,
                sample_rate=1 / win_sec,
                window_size=n,
                hop_size=1,  # Slide by 1 sample
                n_fft=n,
                window_type="boxcar"
            )

            # The spectrogram now has centered windows for each position
            # Remove extra columns if any (due to padding/windowing)
            power_spectrogram = power_spectrogram[:, :nWin]
            psd = power_spectrogram[1:, :]  # Exclude DC

            # Normalize PSD and compute entropy
            with np.errstate(divide='ignore', invalid='ignore'):
                psd_norm = psd / psd.sum(axis=0, keepdims=True)
                psd_norm[psd_norm == np.inf] = 0
                psd_norm[np.isnan(psd_norm)] = 0

            # Compute spectral entropy
            se = -_xlogx(psd_norm).sum(axis=0)
            se /= np.log2(psd_norm.shape[0])
            self.entropy_sink[chi, :] = se

            # Process SOURCE indices (same process)
            SO_channel = self.sink_indices["source_wins"][chi, :]
            SO_channel_padded = np.pad(SO_channel, (half_n, half_n), mode='reflect')

            # Compute spectrogram for source
            time_axis, frequency_axis, power_spectrogram = compute_spectrogram(
                input_vector=SO_channel_padded,
                sample_rate=1 / win_sec,
                window_size=n,
                hop_size=1,
                n_fft=n,
                window_type="boxcar"
            )

            power_spectrogram = power_spectrogram[:, :nWin]
            psd = power_spectrogram[1:, :]

            with np.errstate(divide='ignore', invalid='ignore'):
                psd_norm = psd / psd.sum(axis=0, keepdims=True)
                psd_norm[psd_norm == np.inf] = 0
                psd_norm[np.isnan(psd_norm)] = 0

            se = -_xlogx(psd_norm).sum(axis=0)
            se /= np.log2(psd_norm.shape[0])
            self.entropy_source[chi, :] = se

            pbar.update(1)

        pbar.close()

        # ARTIFACT FIXING for edge windows
        print(f"Fixing edge artifacts in first and last {half_n} windows...")
        for chi in range(nCH):
            if nWin > n:
                # Fix beginning: use mean of windows just after the edge region
                mean_sink_start = np.mean(self.entropy_sink[chi, half_n:half_n + half_n])
                mean_source_start = np.mean(self.entropy_source[chi, half_n:half_n + half_n])

                # Fix end: use mean of windows just before the edge region
                mean_sink_end = np.mean(self.entropy_sink[chi, -half_n - half_n:-half_n])
                mean_source_end = np.mean(self.entropy_source[chi, -half_n - half_n:-half_n])

                # Replace edge windows
                self.entropy_sink[chi, :half_n] = mean_sink_start
                self.entropy_source[chi, :half_n] = mean_source_start
                self.entropy_sink[chi, -half_n:] = mean_sink_end
                self.entropy_source[chi, -half_n:] = mean_source_end
            else:
                # If we have very few windows, use overall mean
                mean_sink = np.mean(self.entropy_sink[chi, :])
                mean_source = np.mean(self.entropy_source[chi, :])

                self.entropy_sink[chi, :half_n] = mean_sink
                self.entropy_source[chi, :half_n] = mean_source
                self.entropy_sink[chi, -half_n:] = mean_sink
                self.entropy_source[chi, -half_n:] = mean_source

                if chi == 0:  # Print warning only once
                    print(f"  Warning: Total windows ({nWin}) ≤ entropy window ({n})")

        if nWin > n:
            print(f"  ✓ First and last {half_n} windows replaced with adjacent means")
        else:
            print(f"  ✓ Edge windows replaced with overall mean (edge case)")

        # Compute z-scored entropy if requested
        if self.config.get("compute_zscore_entropy", False):
            print(f"\nComputing z-scored entropy...")

            # Check if we should use metadata for z-scoring
            use_metadata_baseline = self.config.get("use_metadata_baseline", False)
            metadata_found = False

            if use_metadata_baseline and self.metadata_path and os.path.exists(self.metadata_path):
                import pandas as pd
                import ast

                # Load metadata
                df = pd.read_excel(self.metadata_path)
                if self.config['patient'] in df['patient_no'].values:
                    patient_row = df[df['patient_no'] == self.config['patient']].iloc[0]

                    if pd.notna(patient_row.get('ents_med')) and pd.notna(patient_row.get('ents_std')):
                        # Use metadata values
                        ents_med_dict = ast.literal_eval(patient_row['ents_med'])
                        ents_std_dict = ast.literal_eval(patient_row['ents_std'])

                        print(f"  - Using metadata baseline from patient {self.config['patient']}")

                        for chi in range(nCH):
                            channel_name = self.labels[chi]
                            if channel_name in ents_med_dict and channel_name in ents_std_dict:
                                mean_baseline = ents_med_dict[channel_name]
                                std_baseline = ents_std_dict[channel_name]

                                # Compute z-scores using metadata baseline
                                if std_baseline > 0:
                                    self.entropy_sink_zscore[chi, :] = (self.entropy_sink[chi, :] - mean_baseline) / std_baseline
                                    self.entropy_source_zscore[chi, :] = (self.entropy_source[chi, :] - mean_baseline) / std_baseline
                                else:
                                    self.entropy_sink_zscore[chi, :] = 0
                                    self.entropy_source_zscore[chi, :] = 0

                        metadata_found = True
                        print(f"  ✓ Z-scored entropy computed using metadata baseline")

            # If metadata not found but was requested, compute and save it
            if use_metadata_baseline and not metadata_found:
                print(f"  Warning: Metadata baseline not found, computing and saving...")

                ents_med, ents_std = {}, {}
                for chi in range(nCH):
                    # Use robust baseline even for metadata
                    mean_sink, std_sink = self._find_robust_baseline(
                        self.entropy_sink[chi, half_n:],
                        min_threshold=self.config.get("baseline_range", [0.8, 1.0])[0],
                        max_threshold=self.config.get("baseline_range", [0.8, 1.0])[1],
                        default_mean=self.config.get("baseline_defaults", {"mean": 0.8})["mean"],
                        default_std=self.config.get("baseline_defaults", {"std": 0.003})["std"]
                    )

                    ents_med[self.labels[chi]] = float(mean_sink)
                    ents_std[self.labels[chi]] = float(std_sink)

                # Save to metadata (rest of the code stays the same)
                import pandas as pd
                if os.path.exists(self.metadata_path):
                    df = pd.read_excel(self.metadata_path)
                else:
                    df = pd.DataFrame(columns=['patient_no', 'ents_med', 'ents_std'])

            # # If metadata not found but was requested, compute and save it
            # if use_metadata_baseline and not metadata_found:
            #     print(f"  Warning: Metadata baseline not found, computing and saving...")
            #
            #     # Compute baseline statistics
            #     baseline_minutes = self.config.get("zscore_baseline_minutes", 1)
            #     baseline_windows = int(baseline_minutes * 60 / win_sec)
            #
            #     # START AFTER half_n WINDOWS
            #     baseline_start = half_n
            #     baseline_end = min(baseline_start + baseline_windows, nWin)
            #
            #     if baseline_end <= baseline_start:
            #         print(f"  Warning: Not enough windows for baseline after edge region. Using all available.")
            #         baseline_start = 0
            #         baseline_end = nWin
            #
            #     print(f"  - Baseline: {baseline_minutes} min ({baseline_windows} windows)")
            #     print(f"  - Using windows {baseline_start} to {baseline_end} for baseline")
            #
            #     ents_med, ents_std = {}, {}
            #     for chi in range(nCH):
            #         # Compute baseline statistics for sink entropy
            #         baseline_sink = self.entropy_sink[chi, baseline_start:baseline_end]
            #         mean_sink = np.mean(baseline_sink)
            #         std_sink = np.std(baseline_sink)
            #
            #         ents_med[self.labels[chi]] = float(mean_sink)
            #         ents_std[self.labels[chi]] = float(std_sink)
            #
            #     # Save to metadata
            #     import pandas as pd
            #     if os.path.exists(self.metadata_path):
            #         df = pd.read_excel(self.metadata_path)
            #     else:
            #         df = pd.DataFrame(columns=['patient_no', 'ents_med', 'ents_std'])

                # Update or add row
                patient_mask = df['patient_no'] == self.config['patient']
                if patient_mask.any():
                    df.loc[patient_mask, 'ents_med'] = str(ents_med)
                    df.loc[patient_mask, 'ents_std'] = str(ents_std)
                else:
                    new_row = pd.DataFrame([{'patient_no': self.config['patient'], 'ents_med': str(ents_med), 'ents_std': str(ents_std)}])
                    df = pd.concat([df, new_row], ignore_index=True)

                # Save
                df.to_excel(self.metadata_path, index=False)
                print(f"  ✓ Saved baseline statistics to metadata for patient {self.config['patient']}")

                # Now use these values for z-scoring
                for chi in range(nCH):
                    channel_name = self.labels[chi]
                    mean_baseline = ents_med[channel_name]
                    std_baseline = ents_std[channel_name]

                    if std_baseline > 0:
                        self.entropy_sink_zscore[chi, :] = (self.entropy_sink[chi, :] - mean_baseline) / std_baseline
                        self.entropy_source_zscore[chi, :] = (self.entropy_source[chi, :] - mean_baseline) / std_baseline
                    else:
                        self.entropy_sink_zscore[chi, :] = 0
                        self.entropy_source_zscore[chi, :] = 0

                print(f"  ✓ Z-scored entropy computed using newly saved baseline")

            # If metadata not requested, use window-based baseline
            if not use_metadata_baseline:
                use_robust = self.config.get("robust_baseline", True)
                if use_robust:
                    baseline_range = self.config.get("baseline_range", [0.8, 1.0])
                    defaults = self.config.get("baseline_defaults", {"mean": 0.8, "std": 0.003})

                    print(f"  - Using robust baseline estimation (range: {baseline_range[0]}-{baseline_range[1]})")

                    baseline_stats = []
                    for chi in range(nCH):
                        # Find robust baseline for sink entropy
                        mean_sink, std_sink = self._find_robust_baseline(
                            self.entropy_sink[chi, half_n:],
                            min_threshold=baseline_range[0],
                            max_threshold=baseline_range[1],
                            default_mean=defaults["mean"],
                            default_std=defaults["std"]
                        )

                        # Find robust baseline for source entropy
                        mean_source, std_source = self._find_robust_baseline(
                            self.entropy_source[chi, half_n:],
                            min_threshold=baseline_range[0],
                            max_threshold=baseline_range[1],
                            default_mean=defaults["mean"],
                            default_std=defaults["std"]
                        )

                        baseline_stats.append((mean_sink, std_sink))

                        # Compute z-scores
                        if std_sink > 0:
                            self.entropy_sink_zscore[chi, :] = (self.entropy_sink[chi, :] - mean_sink) / std_sink
                        else:
                            self.entropy_sink_zscore[chi, :] = 0

                        if std_source > 0:
                            self.entropy_source_zscore[chi, :] = (self.entropy_source[chi, :] - mean_source) / std_source
                        else:
                            self.entropy_source_zscore[chi, :] = 0

                    # Report statistics
                    mean_baselines = [s[0] for s in baseline_stats]
                    std_baselines = [s[1] for s in baseline_stats]
                    n_defaults = sum(1 for m in mean_baselines if abs(m - defaults["mean"]) < 1e-6)

                    print(f"  ✓ Robust baseline computed")
                    print(f"    - Mean of baseline means: {np.mean(mean_baselines):.3f} ± {np.std(mean_baselines):.3f}")
                    print(f"    - Mean of baseline stds: {np.mean(std_baselines):.4f} ± {np.std(std_baselines):.4f}")
                    print(f"    - Channels using default: {n_defaults}/{nCH}")

                else:
                    baseline_minutes = self.config.get("zscore_baseline_minutes", 1)
                    baseline_windows = int(baseline_minutes * 60 / win_sec)

                    # START AFTER half_n WINDOWS
                    baseline_start = half_n
                    baseline_end = min(baseline_start + baseline_windows, nWin)

                    if baseline_end <= baseline_start:
                        print(f"  Warning: Not enough windows for baseline after edge region. Using all available.")
                        baseline_start = 0
                        baseline_end = nWin

                    print(f"  - Baseline: {baseline_minutes} min ({baseline_windows} windows)")
                    print(f"  - Using windows {baseline_start} to {baseline_end} for baseline")

                    for chi in range(nCH):
                        # Compute baseline statistics for sink entropy
                        baseline_sink = self.entropy_sink[chi, baseline_start:baseline_end]
                        mean_sink = np.mean(baseline_sink)
                        std_sink = np.std(baseline_sink)

                        # Compute baseline statistics for source entropy
                        baseline_source = self.entropy_source[chi, baseline_start:baseline_end]
                        mean_source = np.mean(baseline_source)
                        std_source = np.std(baseline_source)

                        # Avoid division by zero
                        if std_sink > 0:
                            self.entropy_sink_zscore[chi, :] = (self.entropy_sink[chi, :] - mean_sink) / std_sink
                        else:
                            self.entropy_sink_zscore[chi, :] = 0

                        if std_source > 0:
                            self.entropy_source_zscore[chi, :] = (self.entropy_source[chi, :] - mean_source) / std_source
                        else:
                            self.entropy_source_zscore[chi, :] = 0

                    print(f"  ✓ Z-scored entropy computed using baseline from windows {baseline_start}-{baseline_end}")

        print(f"Spectral entropy computation complete!")
        print(f"  - Sink entropy shape: {self.entropy_sink.shape}")
        print(f"  - Source entropy shape: {self.entropy_source.shape}")

        # Save entropy results
        if self.config["save_results"]:
            self.save_entropy_results()

        # Create plots if requested
        if self.config["create_plots"]:
            self.plot_spectral_entropy()
            self.plot_mean_entropy()
            if self.config.get("compute_zscore_entropy", False):
                self.plot_zscore_entropy()

        return self.entropy_sink, self.entropy_source

    def compute_spectral_entropy(self):
        """Compute spectral entropy of sink and source indices using causal windows."""
        self.print_step(7, "COMPUTE SPECTRAL ENTROPY (CAUSAL)")

        # Check if entropy already exists
        if self._feature_exists('entropy') or self._feature_exists('entropy_pickle'):
            if self._load_entropy():
                return self.entropy_sink, self.entropy_source
            else:
                print("Failed to load cached entropy, recomputing...")

        if self.sink_indices is None:
            raise ValueError("Sink indices must be computed first. Run compute_sink_indices().")

        if self.metadata_path and os.path.isfile(self.metadata_path):
            import pandas as pd
            self.metadata_df = pd.read_excel(self.metadata_path)
            print(f"Loaded metadata from {self.metadata_path}")

        # Get dimensions
        nCH = self.sink_indices["sink_wins"].shape[0]
        nWin = self.sink_indices["sink_wins"].shape[1]

        # Calculate window parameters for entropy
        win_sec = self.config["win_size"]
        minu = self.config["entropy_window_minutes"]
        n = int(minu * 60 / win_sec)  # Number of windows for entropy computation

        print(f"Computing spectral entropy with causal windows...")
        print(f"  - Window for entropy: {minu} minutes ({n} windows)")
        print(f"  - Total windows: {nWin}")
        print(f"  - Channels: {nCH}")

        # Initialize entropy arrays
        self.entropy_sink = np.zeros((nCH, nWin))
        self.entropy_source = np.zeros((nCH, nWin))

        # Initialize z-scored entropy arrays if requested
        if self.config.get("compute_zscore_entropy", False):
            self.entropy_sink_zscore = np.zeros((nCH, nWin))
            self.entropy_source_zscore = np.zeros((nCH, nWin))

        # Progress bar
        pbar = tqdm(total=nCH, desc='Computing entropy for channels')

        # Compute entropy for each channel
        for chi in range(nCH):
            pbar.set_postfix({'Channel': self.labels[chi]})

            # Process SINK indices
            SI_channel = self.sink_indices["sink_wins"][chi, :]

            # CAUSAL PADDING: pad only at the beginning with n samples
            # Use the first value to pad (constant padding) or mirror padding
            # SI_channel_padded = np.pad(SI_channel, (n, 0), mode='constant', constant_values=SI_channel[0])

            # Alternative: use mirror padding for potentially better spectral properties
            SI_channel_padded = np.pad(SI_channel, (n, 0), mode='reflect')

            # Compute spectrogram for the entire padded signal
            # Use right-aligned windows: each window ends at the current time point
            time_axis, frequency_axis, power_spectrogram = compute_spectrogram(
                input_vector=SI_channel_padded,
                sample_rate=1 / win_sec,
                window_size=n,
                hop_size=1,  # Slide by 1 sample
                n_fft=n,
                window_type="boxcar"
            )

            # Extract only the windows corresponding to our original signal
            # The padding allows us to have full windows even for the first samples
            # power_spectrogram = power_spectrogram[:, n:n + nWin]  # Skip the padded part
            psd = power_spectrogram[1:, :]  # Exclude DC

            # Normalize PSD and compute entropy
            with np.errstate(divide='ignore', invalid='ignore'):
                psd_norm = psd / psd.sum(axis=0, keepdims=True)
                psd_norm[psd_norm == np.inf] = 0
                psd_norm[np.isnan(psd_norm)] = 0

            # Compute spectral entropy
            se = -_xlogx(psd_norm).sum(axis=0)
            se /= np.log2(psd_norm.shape[0])

            # Replace first n entropy values with mean of remaining values
            # Replace first n entropy values with mean of remaining values
            if np.sum(se>0.82) > n//2:
                mean_entropy_sink = np.mean(se[se > 0.82])  # Mean of causal part
                se[:n] = mean_entropy_sink
            else:
                if nWin > n:
                    mean_entropy_sink = np.mean(se[n:])  # Mean of causal part
                    se[:n] = mean_entropy_sink
                else:
                    # If we have very few windows, use overall mean
                    mean_entropy_sink = np.mean(se)
                    se[:min(n, nWin)] = mean_entropy_sink


            self.entropy_sink[chi, :] = se[:nWin]

            # Process SOURCE indices (same process)
            SO_channel = self.sink_indices["source_wins"][chi, :]
            SO_channel_padded = np.pad(SO_channel, (n, 0), mode='constant', constant_values=SO_channel[0])

            # Compute spectrogram for source
            time_axis, frequency_axis, power_spectrogram = compute_spectrogram(
                input_vector=SO_channel_padded,
                sample_rate=1 / win_sec,
                window_size=n,
                hop_size=1,
                n_fft=n,
                window_type="boxcar"
            )

            # power_spectrogram = power_spectrogram[:, n:n + nWin]  # Skip the padded part
            psd = power_spectrogram[1:, :]

            with np.errstate(divide='ignore', invalid='ignore'):
                psd_norm = psd / psd.sum(axis=0, keepdims=True)
                psd_norm[psd_norm == np.inf] = 0
                psd_norm[np.isnan(psd_norm)] = 0

            se = -_xlogx(psd_norm).sum(axis=0)
            se /= np.log2(psd_norm.shape[0])

            # Replace first n entropy values with mean of remaining values
            if np.sum(se>0.82) > n//2:
                mean_entropy_source = np.mean(se[se > 0.82])  # Mean of causal part
                se[:n] = mean_entropy_source
            else:
                if nWin > n:
                    mean_entropy_source = np.mean(se[n:])  # Mean of causal part
                    se[:n] = mean_entropy_source
                else:
                    # If we have very few windows, use overall mean
                    mean_entropy_source = np.mean(se)
                    se[:min(n, nWin)] = mean_entropy_source

            self.entropy_source[chi, :] = se[:nWin]

            pbar.update(1)

        pbar.close()

        print(f"Causal entropy computation complete!")
        print(f"  - First {min(n, nWin)} windows replaced with channel mean for causality")
        print(f"  - Sink entropy shape: {self.entropy_sink.shape}")
        print(f"  - Source entropy shape: {self.entropy_source.shape}")

        # Compute z-scored entropy if requested
        if self.config.get("compute_zscore_entropy", False):
            print(f"\nComputing z-scored entropy...")

            # Check if we should use metadata for z-scoring
            use_metadata_baseline = self.config.get("use_metadata_baseline", False)
            metadata_found = False

            if use_metadata_baseline and self.metadata_path and os.path.exists(self.metadata_path):
                import pandas as pd
                import ast

                # Load metadata
                df = pd.read_excel(self.metadata_path)
                if self.config['patient'] in df['patient_no'].values:
                    patient_row = df[df['patient_no'] == self.config['patient']].iloc[0]

                    if pd.notna(patient_row.get('ents_med')) and pd.notna(patient_row.get('ents_std')):
                        # Use metadata values
                        ents_med_dict = ast.literal_eval(patient_row['ents_med'])
                        ents_std_dict = ast.literal_eval(patient_row['ents_std'])

                        print(f"  - Using metadata baseline from patient {self.config['patient']}")

                        for chi in range(nCH):
                            channel_name = self.labels[chi]
                            if channel_name in ents_med_dict and channel_name in ents_std_dict:
                                mean_baseline = ents_med_dict[channel_name]
                                std_baseline = ents_std_dict[channel_name]

                                # Compute z-scores using metadata baseline
                                if std_baseline > 0:
                                    self.entropy_sink_zscore[chi, :] = (self.entropy_sink[chi, :] - mean_baseline) / std_baseline
                                    self.entropy_source_zscore[chi, :] = (self.entropy_source[chi, :] - mean_baseline) / std_baseline
                                else:
                                    self.entropy_sink_zscore[chi, :] = 0
                                    self.entropy_source_zscore[chi, :] = 0

                        metadata_found = True
                        print(f"  ✓ Z-scored entropy computed using metadata baseline")

            # If metadata not found but was requested, compute and save it
            if use_metadata_baseline and not metadata_found:
                print(f"  Warning: Metadata baseline not found, computing and saving...")

                ents_med, ents_std = {}, {}
                for chi in range(nCH):
                    # Use robust baseline even for metadata - start from window n to avoid padded values
                    mean_sink, std_sink = self._find_robust_baseline(
                        self.entropy_sink[chi, n:],  # Start from window n
                        min_threshold=self.config.get("baseline_range", [0.8, 1.0])[0],
                        max_threshold=self.config.get("baseline_range", [0.8, 1.0])[1],
                        default_mean=self.config.get("baseline_defaults", {"mean": 0.8})["mean"],
                        default_std=self.config.get("baseline_defaults", {"std": 0.003})["std"]
                    )

                    ents_med[self.labels[chi]] = float(mean_sink)
                    ents_std[self.labels[chi]] = float(std_sink)

                # Save to metadata
                import pandas as pd
                if os.path.exists(self.metadata_path):
                    df = pd.read_excel(self.metadata_path)
                else:
                    df = pd.DataFrame(columns=['patient_no', 'ents_med', 'ents_std'])

                # Update or add row
                patient_mask = df['patient_no'] == self.config['patient']
                if patient_mask.any():
                    df.loc[patient_mask, 'ents_med'] = str(ents_med)
                    df.loc[patient_mask, 'ents_std'] = str(ents_std)
                else:
                    new_row = pd.DataFrame([{'patient_no': self.config['patient'], 'ents_med': str(ents_med), 'ents_std': str(ents_std)}])
                    df = pd.concat([df, new_row], ignore_index=True)

                # Save
                df.to_excel(self.metadata_path, index=False)
                print(f"  ✓ Saved baseline statistics to metadata for patient {self.config['patient']}")

                # Now use these values for z-scoring
                for chi in range(nCH):
                    channel_name = self.labels[chi]
                    mean_baseline = ents_med[channel_name]
                    std_baseline = ents_std[channel_name]

                    if std_baseline > 0:
                        self.entropy_sink_zscore[chi, :] = (self.entropy_sink[chi, :] - mean_baseline) / std_baseline
                        self.entropy_source_zscore[chi, :] = (self.entropy_source[chi, :] - mean_baseline) / std_baseline
                    else:
                        self.entropy_sink_zscore[chi, :] = 0
                        self.entropy_source_zscore[chi, :] = 0

                print(f"  ✓ Z-scored entropy computed using newly saved baseline")

            # If metadata not requested, use window-based baseline
            if not use_metadata_baseline:
                use_robust = self.config.get("robust_baseline", True)
                if use_robust:
                    baseline_range = self.config.get("baseline_range", [0.8, 1.0])
                    defaults = self.config.get("baseline_defaults", {"mean": 0.8, "std": 0.003})

                    print(f"  - Using robust baseline estimation (range: {baseline_range[0]}-{baseline_range[1]})")

                    baseline_stats = []
                    for chi in range(nCH):
                        # Find robust baseline for sink entropy - start from window n to avoid padded values
                        mean_sink, std_sink = self._find_robust_baseline(
                            self.entropy_sink[chi, n:],  # Start from window n
                            min_threshold=baseline_range[0],
                            max_threshold=baseline_range[1],
                            default_mean=defaults["mean"],
                            default_std=defaults["std"]
                        )

                        # Find robust baseline for source entropy
                        mean_source, std_source = self._find_robust_baseline(
                            self.entropy_source[chi, n:],  # Start from window n
                            min_threshold=baseline_range[0],
                            max_threshold=baseline_range[1],
                            default_mean=defaults["mean"],
                            default_std=defaults["std"]
                        )

                        baseline_stats.append((mean_sink, std_sink))

                        # Compute z-scores
                        if std_sink > 0:
                            self.entropy_sink_zscore[chi, :] = (self.entropy_sink[chi, :] - mean_sink) / std_sink
                        else:
                            self.entropy_sink_zscore[chi, :] = 0

                        if std_source > 0:
                            self.entropy_source_zscore[chi, :] = (self.entropy_source[chi, :] - mean_source) / std_source
                        else:
                            self.entropy_source_zscore[chi, :] = 0

                    # Report statistics
                    mean_baselines = [s[0] for s in baseline_stats]
                    std_baselines = [s[1] for s in baseline_stats]
                    n_defaults = sum(1 for m in mean_baselines if abs(m - defaults["mean"]) < 1e-6)

                    print(f"  ✓ Robust baseline computed")
                    print(f"    - Mean of baseline means: {np.mean(mean_baselines):.3f} ± {np.std(mean_baselines):.3f}")
                    print(f"    - Mean of baseline stds: {np.mean(std_baselines):.4f} ± {np.std(std_baselines):.4f}")
                    print(f"    - Channels using default: {n_defaults}/{nCH}")

                else:
                    baseline_minutes = self.config.get("zscore_baseline_minutes", 1)
                    baseline_windows = int(baseline_minutes * 60 / win_sec)

                    # START AFTER n WINDOWS (to avoid padded values)
                    baseline_start = n
                    baseline_end = min(baseline_start + baseline_windows, nWin)

                    if baseline_end <= baseline_start:
                        print(f"  Warning: Not enough windows for baseline after padding. Using all available.")
                        baseline_start = n
                        baseline_end = nWin

                    print(f"  - Baseline: {baseline_minutes} min ({baseline_windows} windows)")
                    print(f"  - Using windows {baseline_start} to {baseline_end} for baseline")

                    for chi in range(nCH):
                        # Compute baseline statistics for sink entropy
                        baseline_sink = self.entropy_sink[chi, baseline_start:baseline_end]
                        mean_sink = np.mean(baseline_sink)
                        std_sink = np.std(baseline_sink)

                        # Compute baseline statistics for source entropy
                        baseline_source = self.entropy_source[chi, baseline_start:baseline_end]
                        mean_source = np.mean(baseline_source)
                        std_source = np.std(baseline_source)

                        # Avoid division by zero
                        if std_sink > 0:
                            self.entropy_sink_zscore[chi, :] = (self.entropy_sink[chi, :] - mean_sink) / std_sink
                        else:
                            self.entropy_sink_zscore[chi, :] = 0

                        if std_source > 0:
                            self.entropy_source_zscore[chi, :] = (self.entropy_source[chi, :] - mean_source) / std_source
                        else:
                            self.entropy_source_zscore[chi, :] = 0

                    print(f"  ✓ Z-scored entropy computed using baseline from windows {baseline_start}-{baseline_end}")

        # Save entropy results
        if self.config["save_results"]:
            self.save_entropy_results()

        # Create plots if requested
        if self.config["create_plots"]:
            self.plot_spectral_entropy()
            self.plot_mean_entropy()
            if self.config.get("compute_zscore_entropy", False):
                self.plot_zscore_entropy()

        return self.entropy_sink, self.entropy_source

    def save_entropy_results(self):
        """Save spectral entropy results."""
        create_folders(self.config["entropy_dir"])

        entropy_data = {
            "patient": self.config["patient"],
            "EEGFileName": f"{self.config['eeg_filename']}.edf",
            "entropy_sink": self.entropy_sink,
            "entropy_source": self.entropy_source,
            "labels": self.labels,
            "win_size": self.config["win_size"],
            "entropy_window_minutes": self.config["entropy_window_minutes"],
            "factor": self.config["downsample_factor"],
            "fs": self.fs,
            "fs0": self.fs0,
            "h_freq": self.config["h_freq"],
            "l_freq": self.config["l_freq"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Add z-scored data if computed
        if hasattr(self, 'entropy_sink_zscore'):
            entropy_data["entropy_sink_zscore"] = self.entropy_sink_zscore
            entropy_data["entropy_source_zscore"] = self.entropy_source_zscore
            entropy_data["zscore_baseline_minutes"] = self.config.get("zscore_baseline_minutes", 1)
            entropy_data["robust_baseline"] = self.config.get("robust_baseline", False)
            entropy_data["baseline_range"] = self.config.get("baseline_range", [0.8, 1.0])
            entropy_data["baseline_defaults"] = self.config.get("baseline_defaults", {"mean": 0.8, "std": 0.003})

        # Save as .mat file
        mat_path = f"{self.config['entropy_dir']}\\{self.config['eeg_filename']}_entropy.mat"
        sio.savemat(mat_path, entropy_data)
        print(f"Saved entropy results to: {mat_path}")

        # Save as pickle for easy Python access
        pkl_path = f"{self.config['entropy_dir']}\\{self.config['eeg_filename']}_entropy.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(entropy_data, f)
        print(f"Saved entropy results (pickle) to: {pkl_path}")

        # Save z-scored entropy separately if computed
        if hasattr(self, 'entropy_sink_zscore'):
            zscore_data = {
                "patient": self.config["patient"],
                "EEGFileName": f"{self.config['eeg_filename']}.edf",
                "entropy_sink_zscore": self.entropy_sink_zscore,
                "entropy_source_zscore": self.entropy_source_zscore,
                "labels": self.labels,
                "zscore_baseline_minutes": self.config.get("zscore_baseline_minutes", 1),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            # Save z-scored data separately
            zscore_mat_path = f"{self.config['entropy_dir']}\\{self.config['eeg_filename']}_entropy_zscore.mat"
            sio.savemat(zscore_mat_path, zscore_data)
            print(f"Saved z-scored entropy to: {zscore_mat_path}")

    def plot_spectral_entropy(self):
        """Plot spectral entropy for all channels."""
        nCH = self.entropy_sink.shape[0]
        nWin = self.entropy_sink.shape[1]

        # Create time axis in seconds
        time_axis = np.arange(nWin) * self.config["win_size"]

        # Create figure with two subplots (sink and source)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

        # Plot SINK entropy
        for i, label in enumerate(self.labels):
            # Offset each channel for visualization
            offset = i * self.config["entropy_plot_channel_height"]
            ax1.plot(time_axis, self.entropy_sink[i, :] + offset,
                     linewidth=0.8, alpha=0.8)

        ax1.set_title(
            f"Spectral Entropy of Sink Indices - {self.config['patient']} - {self.config['eeg_filename']}",
            fontsize=14)
        ax1.set_xlabel("Time (seconds)")
        ax1.set_ylabel("Channel")
        ax1.set_yticks(np.arange(nCH) * self.config["entropy_plot_channel_height"])
        ax1.set_yticklabels(self.labels, fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Plot SOURCE entropy
        for i, label in enumerate(self.labels):
            offset = i * self.config["entropy_plot_channel_height"]
            ax2.plot(time_axis, self.entropy_source[i, :] + offset,
                     linewidth=0.8, alpha=0.8)

        ax2.set_title(
            f"Spectral Entropy of Source Indices - {self.config['patient']} - {self.config['eeg_filename']}",
            fontsize=14)
        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("Channel")
        ax2.set_yticks(np.arange(nCH) * self.config["entropy_plot_channel_height"])
        ax2.set_yticklabels(self.labels, fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        if self.config["save_results"]:
            fig_path = f"{self.config['entropy_dir']}\\{self.config['eeg_filename']}_entropy_channels.png"
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            print(f"Saved entropy plot to: {fig_path}")

        plt.show()

    def plot_mean_entropy(self):
        """Plot mean entropy across all channels."""
        # Calculate mean entropy
        mean_entropy_sink = np.mean(self.entropy_sink, axis=0)
        mean_entropy_source = np.mean(self.entropy_source, axis=0)

        # Calculate standard deviation for error bars
        std_entropy_sink = np.std(self.entropy_sink, axis=0)
        std_entropy_source = np.std(self.entropy_source, axis=0)

        # Create time axis
        time_axis = np.arange(self.entropy_sink.shape[1]) * self.config["win_size"]

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(16, 6))

        # Plot mean entropy with error bands
        ax.plot(time_axis, mean_entropy_source, 'r-', label='Mean Source Entropy', linewidth=0.5)
        ax.fill_between(time_axis,
                        mean_entropy_source - std_entropy_source,
                        mean_entropy_source + std_entropy_source,
                        alpha=0.05, color='red', label='±1 STD (Source)')


        ax.plot(time_axis, mean_entropy_sink, 'b-', label='Mean Sink Entropy', linewidth=1)
        ax.fill_between(time_axis,
                        mean_entropy_sink - std_entropy_sink,
                        mean_entropy_sink + std_entropy_sink,
                        alpha=0.05, color='blue', label='±1 STD (Sink)')


        ax.set_title(f"Mean Spectral Entropy - {self.config['patient']} - {self.config['eeg_filename']}\n"
                     f"Window: {self.config['entropy_window_minutes']} min | "
                     f"Averaged across {len(self.labels)} channels",
                     fontsize=14)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Spectral Entropy")
        ax.set_ylim([0.5, 1])
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Add statistics text
        stats_text = (f"Sink: μ={mean_entropy_sink.mean():.3f}±{mean_entropy_sink.std():.3f}\n"
                      f"Source: μ={mean_entropy_source.mean():.3f}±{mean_entropy_source.std():.3f}")
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # Save figure
        if self.config["save_results"]:
            fig_path = f"{self.config['entropy_dir']}\\{self.config['eeg_filename']}_entropy_mean.png"
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            print(f"Saved mean entropy plot to: {fig_path}")

        plt.show()

    def plot_zscore_entropy(self):
        """Plot z-scored spectral entropy."""
        if not hasattr(self, 'entropy_sink_zscore'):
            print("Z-scored entropy not computed. Set compute_zscore_entropy=True")
            return

        # Calculate mean z-scored entropy
        mean_zscore_sink = np.mean(self.entropy_sink_zscore, axis=0)
        mean_zscore_source = np.mean(self.entropy_source_zscore, axis=0)

        # Create time axis
        time_axis = np.arange(self.entropy_sink_zscore.shape[1]) * self.config["win_size"]

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(16, 6))

        # Plot mean z-scored entropy
        ax.plot(time_axis, mean_zscore_source, 'r-', label='Mean Z-scored Source', linewidth=0.5)
        ax.plot(time_axis, mean_zscore_sink, 'b-', label='Mean Z-scored Sink', linewidth=1)

        # Add zero line
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

        ax.set_title(f"Mean Z-scored Spectral Entropy - {self.config['patient']} - {self.config['eeg_filename']}\n"
                     f"Baseline: {self.config.get('zscore_baseline_minutes', 1)} min | "
                     f"Averaged across {len(self.labels)} channels",
                     fontsize=14)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Z-score")
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-10, 3])

        plt.tight_layout()

        # Save figure
        if self.config["save_results"]:
            fig_path = f"{self.config['entropy_dir']}\\{self.config['eeg_filename']}_entropy_zscore.png"
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            print(f"Saved z-scored entropy plot to: {fig_path}")

        plt.show()

    # ------------------------------------------------------------------------
    # STEP 8: Outlier Channel Detection
    # ------------------------------------------------------------------------

    def detect_outlier_channels(self):
        """Detect outlier channels based on spectral entropy."""
        self.print_step(8, "DETECT OUTLIER CHANNELS")

        # Check if outlier results already exist
        if self._feature_exists('outliers'):
            if self._load_outliers():
                return self.outlier_results
            else:
                print("Failed to load cached outliers, recomputing...")

        if not hasattr(self, 'entropy_sink') or not hasattr(self, 'entropy_source'):
            raise ValueError("Spectral entropy must be computed first. Run compute_spectral_entropy().")

        nCH = self.entropy_sink.shape[0]
        nWin = self.entropy_sink.shape[1]

        print(f"Detecting outlier channels based on spectral entropy...")
        print(f"  - Method: Tukey's method (IQR-based)")
        print(f"  - Entropy type: {self.config['outlier_entropy_type']}")

        # Select which entropy to use for outlier detection
        if self.config['outlier_entropy_type'] == 'sink':
            entropy_data = self.entropy_sink
        elif self.config['outlier_entropy_type'] == 'source':
            entropy_data = self.entropy_source
        else:  # 'both' - use average of sink and source
            entropy_data = (self.entropy_sink + self.entropy_source) / 2

        # Detect outliers at each time window
        outlier_matrix = np.zeros((nCH, nWin))

        for win_idx in range(nWin):
            # Get entropy values for all channels at this time window
            entropy_values = entropy_data[:, win_idx]

            # Skip if all values are NaN or zero
            if np.all(np.isnan(entropy_values)) or np.all(entropy_values == 0):
                continue

            # Apply Tukey's method
            probable_outliers, possible_outliers = tukeys_method(
                entropy_values,
                q1_percentile=self.config['outlier_q1_percentile'],
                q3_percentile=self.config['outlier_q3_percentile'],
                inner_fence_multiplier=self.config['outlier_inner_fence'],
                outer_fence_multiplier=self.config['outlier_outer_fence']
            )

            # Mark outliers: 2 for probable, 1 for possible, 0 for normal
            outlier_matrix[probable_outliers, win_idx] = 2
            outlier_matrix[possible_outliers, win_idx] = 1

        # Calculate average outlier score for each channel
        outlier_scores = np.mean(outlier_matrix > 0, axis=1)  # Fraction of time channel is outlier
        outlier_severity = np.mean(outlier_matrix, axis=1)  # Average severity score

        # Apply Tukey's method on the outlier scores to identify consistently outlier channels
        probable_outlier_channels, possible_outlier_channels = tukeys_method(
            outlier_scores,
            q1_percentile=self.config['outlier_q1_percentile'],
            q3_percentile=self.config['outlier_q3_percentile'],
            inner_fence_multiplier=self.config['outlier_inner_fence'],
            outer_fence_multiplier=self.config['outlier_outer_fence']
        )

        # Also consider channels that are outliers more than threshold of the time
        high_outlier_channels = np.where(outlier_scores > self.config['outlier_threshold'])[0]

        # Combine criteria for final recommendations
        channels_to_remove = np.unique(np.concatenate([
            probable_outlier_channels,
            high_outlier_channels
        ]))

        channels_possibly_remove = np.setdiff1d(possible_outlier_channels, channels_to_remove)

        # Store results
        self.outlier_results = {
            "outlier_matrix": outlier_matrix,
            "outlier_scores": outlier_scores,
            "outlier_severity": outlier_severity,
            "probable_outlier_channels": probable_outlier_channels,
            "possible_outlier_channels": possible_outlier_channels,
            "channels_to_remove": channels_to_remove,
            "channels_possibly_remove": channels_possibly_remove,
            "channel_labels_to_remove": [self.labels[i] for i in channels_to_remove],
            "channel_labels_possibly_remove": [self.labels[i] for i in channels_possibly_remove]
        }

        # Print summary
        print(f"\nOutlier Detection Results:")
        print(f"  - Channels recommended for removal ({len(channels_to_remove)}):")
        if len(channels_to_remove) > 0:
            for idx in channels_to_remove:
                print(f"    • {self.labels[idx]} (CH{idx}) - Score: {outlier_scores[idx]:.3f}")
        else:
            print("    • None")

        print(f"\n  - Channels possibly noisy ({len(channels_possibly_remove)}):")
        if len(channels_possibly_remove) > 0:
            for idx in channels_possibly_remove:
                print(f"    • {self.labels[idx]} (CH{idx}) - Score: {outlier_scores[idx]:.3f}")
        else:
            print("    • None")

        # Save results
        if self.config["save_results"]:
            self.save_outlier_results()

        # Create visualization
        if self.config["create_plots"]:
            self.plot_outlier_channels()

        return self.outlier_results

    def save_outlier_results(self):
        """Save outlier detection results."""
        create_folders(self.config["outlier_dir"])

        # Save as .mat file
        mat_path = f"{self.config['outlier_dir']}\\{self.config['eeg_filename']}_outliers.mat"
        sio.savemat(mat_path, self.outlier_results)
        print(f"\nSaved outlier results to: {mat_path}")

        # Save as pickle
        pkl_path = f"{self.config['outlier_dir']}\\{self.config['eeg_filename']}_outliers.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(self.outlier_results, f)

        # Save summary as text file
        txt_path = f"{self.config['outlier_dir']}\\{self.config['eeg_filename']}_outliers_summary.txt"
        with open(txt_path, 'w') as f:
            f.write(f"Outlier Channel Detection Summary\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Patient: {self.config['patient']}\n")
            f.write(f"EEG File: {self.config['eeg_filename']}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"\nParameters:\n")
            f.write(f"  - Entropy type: {self.config['outlier_entropy_type']}\n")
            f.write(
                f"  - IQR percentiles: {self.config['outlier_q1_percentile']}-{self.config['outlier_q3_percentile']}\n")
            f.write(f"  - Inner fence: {self.config['outlier_inner_fence']}×IQR\n")
            f.write(f"  - Outer fence: {self.config['outlier_outer_fence']}×IQR\n")
            f.write(f"\nChannels Recommended for Removal:\n")
            if len(self.outlier_results['channels_to_remove']) > 0:
                for idx in self.outlier_results['channels_to_remove']:
                    f.write(
                        f"  - {self.labels[idx]} (index: {idx}, score: {self.outlier_results['outlier_scores'][idx]:.3f})\n")
            else:
                f.write("  - None\n")
            f.write(f"\nChannels Possibly Noisy:\n")
            if len(self.outlier_results['channels_possibly_remove']) > 0:
                for idx in self.outlier_results['channels_possibly_remove']:
                    f.write(
                        f"  - {self.labels[idx]} (index: {idx}, score: {self.outlier_results['outlier_scores'][idx]:.3f})\n")
            else:
                f.write("  - None\n")

        print(f"Saved outlier summary to: {txt_path}")

    def plot_outlier_channels(self):
        """Create visualization of outlier channel detection."""
        nCH = len(self.labels)
        outlier_scores = self.outlier_results['outlier_scores']
        probable_channels = self.outlier_results['probable_outlier_channels']
        possible_channels = self.outlier_results['possible_outlier_channels']

        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), height_ratios=[3, 1])

        # Prepare data for bar plot
        colors = ['green'] * nCH
        for idx in possible_channels:
            colors[idx] = '#E8EC1E'
        for idx in probable_channels:
            colors[idx] = '#ff4d50'

        # Plot 1: Bar chart of outlier scores
        bars = ax1.bar(range(nCH), outlier_scores, color=colors, edgecolor='black', linewidth=0.5)

        # Add scatter points for emphasis
        ax1.scatter(range(nCH), outlier_scores, s=25, c='darkgreen', zorder=5, label='Normal channels')
        if len(possible_channels) > 0:
            ax1.scatter(possible_channels, outlier_scores[possible_channels],
                        s=50, c='#D5D934', zorder=6, label='Possibly noisy')
        if len(probable_channels) > 0:
            ax1.scatter(probable_channels, outlier_scores[probable_channels],
                        s=50, c='#C33F42', zorder=7, label='Probably noisy')

        # Add threshold line
        ax1.axhline(y=self.config['outlier_threshold'], color='red', linestyle='--',
                    alpha=0.5, label=f'Threshold ({self.config["outlier_threshold"]})')

        # Formatting
        ax1.set_xlabel('Channel', fontsize=12)
        ax1.set_ylabel('Outlier Score\n(Fraction of time as outlier)', fontsize=12)
        ax1.set_title(f'Outlier Channel Detection - {self.config["patient"]} - {self.config["eeg_filename"]}\n'
                      f'Based on {self.config["outlier_entropy_type"]} entropy | '
                      f'Duration: {self.entropy_sink.shape[1] * self.config["win_size"] / 60:.1f} min',
                      fontsize=14)
        ax1.set_xticks(range(nCH))
        ax1.set_xticklabels(self.labels, rotation=45, ha='right', fontsize=9)
        ax1.set_ylim([0, max(1, np.max(outlier_scores) * 1.1)])
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3, axis='y')

        # Color the x-tick labels
        for idx in possible_channels:
            ax1.get_xticklabels()[idx].set_color("#B2700B")
        for idx in probable_channels:
            ax1.get_xticklabels()[idx].set_color("#C33F42")

        # Plot 2: Heatmap of outlier matrix over time
        im = ax2.imshow(self.outlier_results['outlier_matrix'],
                        aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=2)
        ax2.set_xlabel('Time Window', fontsize=12)
        ax2.set_ylabel('Channel', fontsize=12)
        ax2.set_title('Outlier Status Over Time (0=Normal, 1=Possible, 2=Probable)', fontsize=12)
        ax2.set_yticks(range(0, nCH, max(1, nCH // 20)))
        ax2.set_yticklabels([self.labels[i] for i in range(0, nCH, max(1, nCH // 20))], fontsize=8)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2, orientation='horizontal', pad=0.1, fraction=0.05)
        cbar.set_label('Outlier Status', fontsize=10)

        # Add summary text
        summary_text = (f"Channels to remove: {len(self.outlier_results['channels_to_remove'])}\n"
                        f"Possibly noisy: {len(self.outlier_results['channels_possibly_remove'])}\n"
                        f"Normal: {nCH - len(self.outlier_results['channels_to_remove']) - len(self.outlier_results['channels_possibly_remove'])}")
        ax1.text(0.02, 0.98, summary_text, transform=ax1.transAxes,
                 fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        plt.tight_layout()

        # Save figure
        if self.config["save_results"]:
            fig_path = f"{self.config['outlier_dir']}\\{self.config['eeg_filename']}_outlier_channels.png"
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            print(f"Saved outlier plot to: {fig_path}")

        plt.show()

    def get_clean_channels(self):
        """Get list of clean channel indices (non-outlier channels)."""
        if not hasattr(self, 'outlier_results'):
            raise ValueError("Outlier detection must be run first.")

        all_channels = np.arange(len(self.labels))
        outlier_channels = np.concatenate([
            self.outlier_results['channels_to_remove'],
            self.outlier_results['channels_possibly_remove']
        ])
        clean_channels = np.setdiff1d(all_channels, outlier_channels)

        return clean_channels

    # ------------------------------------------------------------------------
    # STEP 9: Compute Signal Energy
    # ------------------------------------------------------------------------

    def compute_signal_energy(self):
        """Compute windowed energy of the EEG signal."""
        self.print_step(9, "COMPUTE SIGNAL ENERGY")

        # Check if energy already exists
        if self._feature_exists('energy') or self._feature_exists('energy_pickle'):
            if self._load_energy():
                return self.energy, self.energy_normalized
            else:
                print("Failed to load cached energy, recomputing...")

        if self.data is None:
            raise ValueError("EEG data must be loaded first. Run load_and_preprocess().")

        print(f"Computing signal energy...")
        print(f"  - Window size: {self.config['energy_window_sec']} seconds")
        print(f"  - Overlap: {self.config['energy_overlap'] * 100}%")
        print(f"  - Channels: {len(self.labels)}")

        # Import the energy computation function
        from utils import compute_signal_energy

        # Compute energy
        self.energy, self.energy_normalized, self.energy_time_axis = compute_signal_energy(
            data=self.data,
            fs=self.fs,
            window_size_sec=self.config['energy_window_sec'],
            overlap=self.config['energy_overlap']
        )

        print(f"Energy computation complete!")
        print(f"  - Energy matrix shape: {self.energy.shape}")
        print(f"  - Number of windows: {self.energy.shape[1]}")
        print(f"  - Duration covered: {self.energy_time_axis[-1]:.1f} seconds")

        # Compute statistics
        mean_energy = np.mean(self.energy, axis=1)
        std_energy = np.std(self.energy, axis=1)

        # Find channels with highest and lowest average energy
        high_energy_idx = np.argmax(mean_energy)
        low_energy_idx = np.argmin(mean_energy)

        print(f"\nEnergy Statistics:")
        print(f"  - Highest energy channel: {self.labels[high_energy_idx]} "
              f"(mean: {mean_energy[high_energy_idx]:.2e})")
        print(f"  - Lowest energy channel: {self.labels[low_energy_idx]} "
              f"(mean: {mean_energy[low_energy_idx]:.2e})")

        # Save energy results
        if self.config["save_results"]:
            self.save_energy_results()

        # Create plots if requested
        if self.config["create_plots"]:
            self.plot_energy_heatmap()
            self.plot_mean_energy()

        return self.energy, self.energy_normalized

    def save_energy_results(self):
        """Save energy computation results."""
        create_folders(self.config["energy_dir"])

        energy_data = {
            "patient": self.config["patient"],
            "EEGFileName": f"{self.config['eeg_filename']}.edf",
            "energy": self.energy,
            "energy_normalized": self.energy_normalized,
            "energy_time_axis": self.energy_time_axis,
            "labels": self.labels,
            "energy_window_sec": self.config["energy_window_sec"],
            "energy_overlap": self.config["energy_overlap"],
            "fs": self.fs,
            "fs0": self.fs0,
            "h_freq": self.config["h_freq"],
            "l_freq": self.config["l_freq"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Save as .mat file
        mat_path = f"{self.config['energy_dir']}\\{self.config['eeg_filename']}_energy.mat"
        sio.savemat(mat_path, energy_data)
        print(f"Saved energy results to: {mat_path}")

        # Save as pickle for easy Python access
        pkl_path = f"{self.config['energy_dir']}\\{self.config['eeg_filename']}_energy.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(energy_data, f)
        print(f"Saved energy results (pickle) to: {pkl_path}")

        # Save summary statistics as CSV
        summary_df = pd.DataFrame({
            'Channel': self.labels,
            'Mean_Energy': np.mean(self.energy, axis=1),
            'Std_Energy': np.std(self.energy, axis=1),
            'Max_Energy': np.max(self.energy, axis=1),
            'Min_Energy': np.min(self.energy, axis=1),
            'Mean_Normalized': np.mean(self.energy_normalized, axis=1),
            'Std_Normalized': np.std(self.energy_normalized, axis=1)
        })

        csv_path = f"{self.config['energy_dir']}\\{self.config['eeg_filename']}_energy_summary.csv"
        summary_df.to_csv(csv_path, index=False)
        print(f"Saved energy summary to: {csv_path}")

    def plot_energy_heatmap(self):
        """Plot energy heatmap for all channels."""
        # Sort channels by mean energy
        mean_energy = np.mean(self.energy_normalized, axis=1)
        sort_idx = np.argsort(mean_energy)[::-1]
        energy_sorted = self.energy_normalized[sort_idx, :]
        labels_sorted = [self.labels[i] for i in sort_idx]

        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))

        # Plot 1: Normalized energy heatmap
        sns.heatmap(
            energy_sorted,
            yticklabels=labels_sorted,
            cmap='hot',
            cbar_kws={"pad": 0.01, "label": "Normalized Energy"},
            ax=ax1
        )

        ax1.set_title(
            f"Normalized Energy Heatmap - {self.config['patient']} - {self.config['eeg_filename']}\n"
            f"Window: {self.config['energy_window_sec']}s | Overlap: {self.config['energy_overlap'] * 100}% | "
            f"Channels sorted by mean energy",
            fontsize=14
        )
        ax1.set_xlabel("Window Index")
        ax1.set_ylabel("Channel")

        # Plot 2: Raw energy heatmap (log scale)
        energy_log = np.log10(self.energy[sort_idx, :] + 1e-10)  # Add small value to avoid log(0)

        sns.heatmap(
            energy_log,
            yticklabels=labels_sorted,
            cmap='viridis',
            cbar_kws={"pad": 0.01, "label": "Log10(Energy)"},
            ax=ax2
        )

        ax2.set_title(
            f"Raw Energy Heatmap (Log Scale) - {self.config['patient']} - {self.config['eeg_filename']}",
            fontsize=14
        )
        ax2.set_xlabel("Window Index")
        ax2.set_ylabel("Channel")

        plt.tight_layout()

        # Save the heatmap
        if self.config["save_results"]:
            heatmap_path = f"{self.config['energy_dir']}\\{self.config['eeg_filename']}_energy_heatmap.png"
            plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
            print(f"Saved energy heatmap to: {heatmap_path}")

        plt.show()

    def plot_mean_energy(self):
        """Plot mean energy across channels over time."""
        # Calculate mean and std across channels
        mean_energy_time = np.mean(self.energy_normalized, axis=0)
        std_energy_time = np.std(self.energy_normalized, axis=0)

        # Also calculate for raw energy
        mean_raw_energy = np.mean(self.energy, axis=0)
        std_raw_energy = np.std(self.energy, axis=0)

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

        # Plot 1: Normalized energy
        ax1.plot(self.energy_time_axis, mean_energy_time, 'b-', label='Mean', linewidth=2)
        ax1.fill_between(self.energy_time_axis,
                         mean_energy_time - std_energy_time,
                         mean_energy_time + std_energy_time,
                         alpha=0.3, color='blue', label='±1 STD')

        ax1.set_title(f"Mean Normalized Energy - {self.config['patient']} - {self.config['eeg_filename']}\n"
                      f"Window: {self.config['energy_window_sec']}s | "
                      f"Averaged across {len(self.labels)} channels",
                      fontsize=14)
        ax1.set_xlabel("Time (seconds)")
        ax1.set_ylabel("Normalized Energy")
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Raw energy
        ax2.plot(self.energy_time_axis, mean_raw_energy, 'g-', label='Mean', linewidth=2)
        ax2.fill_between(self.energy_time_axis,
                         mean_raw_energy - std_raw_energy,
                         mean_raw_energy + std_raw_energy,
                         alpha=0.3, color='green', label='±1 STD')

        ax2.set_title(f"Mean Raw Energy", fontsize=14)
        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("Energy (V²)")
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

        # Add statistics text
        stats_text = (f"Normalized: μ={mean_energy_time.mean():.3f}±{mean_energy_time.std():.3f}\n"
                      f"Raw: μ={mean_raw_energy.mean():.2e}±{std_raw_energy.mean():.2e}")
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # Save figure
        if self.config["save_results"]:
            fig_path = f"{self.config['energy_dir']}\\{self.config['eeg_filename']}_energy_mean.png"
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            print(f"Saved mean energy plot to: {fig_path}")

        plt.show()

    # ------------------------------------------------------------------------
    # Run Complete Pipeline
    # ------------------------------------------------------------------------

    def run_pipeline(self):
        """Run the complete analysis pipeline."""
        self.print_header("STARTING ANALYSIS")
        start_time = datetime.now()

        try:
            # Step 1: Load and preprocess
            self.load_and_preprocess()

            # Step 2: Estimate A matrices
            self.estimate_a_matrices()

            # Step 3: Reconstruct signal
            self.reconstruct_signal()

            # Step 4: Correlation analysis
            self.analyze_correlations()

            # Step 5: Compute sink indices
            self.compute_sink_indices()

            # Step 6: Visualize EEG (optional)
            self.visualize_eeg() if self.config.get("create_plots", False) else None

            # Step 7: Compute spectral entropy (optional)
            self.compute_spectral_entropy() if self.config.get("compute_entropy", False) else None

            # Step 8: Detect outlier channels (requires entropy)
            self.detect_outlier_channels() if self.config.get("compute_entropy", False) and self.config.get(
                "detect_outliers", False) else None

            # Step 9: Compute signal energy (optional)
            if self.config.get("compute_energy", False):
                self.compute_signal_energy()

            # Complete
            duration = (datetime.now() - start_time).total_seconds()
            self.print_header("ANALYSIS COMPLETE")
            print(f"Total time: {duration:.1f} seconds")
            print(f"Results saved to: {self.config['output_base']}")

        except Exception as e:
            self.print_header("ERROR OCCURRED")
            print(f"Error: {str(e)}")
            raise