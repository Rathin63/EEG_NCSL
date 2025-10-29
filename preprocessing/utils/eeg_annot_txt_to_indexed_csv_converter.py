import os
import pandas as pd

# Define helper functions
def sec2time(sec):
    hrs = int(sec // 3600)
    mins = int((sec % 3600) // 60)
    secs = sec % 60
    return f"{hrs:02d}:{mins:02d}:{secs:05.2f}"

def time_sum(time1, time2):
    h1, m1, s1 = map(float, time1.split(':'))
    h2, m2, s2 = map(float, time2.split(':'))
    total_sec = (h1 + h2) * 3600 + (m1 + m2) * 60 + s1 + s2
    return sec2time(total_sec)

def create_folder(directory):
    from os import makedirs
    from os.path import exists

    """
    Create a folder if it does not already exist.

    Args:
        directory (str): The path of the directory to be created.
    """
    if not exists(directory):
        makedirs(directory)
        print(f"Directory {directory} created successfully.")
    else:
        print(f"Directory {directory} already exists.")

# Define variables
patient_examNo = 'PY24N013'
name = 'TK4' # SPECIFY THIS (above)
rec_start = "17:2:34"
segment = 3  # SPECIFY THIS (above)

# Adding paths
os.makedirs(os.path.join('D:\\', patient_examNo, 'annot'), exist_ok=True)
os.makedirs(os.path.join('D:\\', patient_examNo, '4_a_matrices'), exist_ok=True)

# Events to find in the annotation file
annot_sz = ['EEG Change', 'SZ', 'PB', 'AURA', 'ELECTROGRAPHIC', 'Sleeping', 'BURST', 'BURST.', '"BURST', 'BURST"', 'CLINICAL', 'sz', 'bathroom', 'MARK', 'SUBCLINICAL', 'Z ', 'EVENT', 'start']

fs = 2000  # in hertz
winSize_sec = 0.5  # in seconds
nWin = 28800  # Next step is to tune this

# Adjust the recording time with the segment number
seg_start = (segment - 1) * nWin * winSize_sec
seg_start = sec2time(seg_start)
rec_start = time_sum(rec_start, seg_start)

# Read annotations
annot_table = pd.read_csv(f'D:\\{patient_examNo}\\annot\\{name}_annot.txt', delimiter='\t')
# Split the OneColumn into two separate columns
annot_table[['Onset', 'Annotation']] = annot_table["Onset,Annotation"].str.split(',', n=1, expand=True)
# Convert Onset to float
annot_table['Onset'] = annot_table['Onset'].astype(float)
# Drop the original OneColumn
annot_table = annot_table[['Onset', 'Annotation']]

annot_table.columns = ['Onset', 'Annotation']

# Convert annot.Onset to decimal hours
t = annot_table['Onset']
hours = (t // 3600).astype(int)
t = t - hours * 3600
mins = (t // 60).astype(int)
secs = t - mins * 60
h = hours + (mins / 60) + (secs / 3600)
annot_table['DecimalHour'] = h

# Find and store the rows in the event file for each event of interest
# Create a regex pattern from the list
pattern = '|'.join(annot_sz).lower()
# Select rows where 'Annotation' column contains any of the words from annot_sz
annot_table = annot_table[annot_table['Annotation'].str.lower().str.contains(pattern, case=False, regex=True)]


# Calculate Event Times
Time = []
for i in range(len(annot_table)):
    EV_sec = annot_table.iloc[i]['Onset']
    EV_time = sec2time(EV_sec)
    EV_time = time_sum(EV_time, rec_start)
    Time.append(EV_time)
annot_table['Time'] = Time

# Sort and prepare columns
annot_table = annot_table.sort_values(by='Onset')
annots_col = annot_table['Annotation']
Sec_col = annot_table['Onset']
decimalHour_col = annot_table['DecimalHour']
iW_col = (Sec_col // winSize_sec).astype(int) + 1

# Mapping seizure onset from seconds to index of A matrix
SZ_segment = (iW_col // nWin) + 1
nSZ = SZ_segment.shape[0]
SZ_iW_new, SZ_sec_new, SZ_decimalHour_new, SZ_annot, SZ_time, EMU_time = [], [], [], [], [], []

for i in range(nSZ):
    if SZ_segment.iloc[i] == segment:
        SZ_iW_new.append(iW_col.iloc[i] % nWin)
        SZ_sec_new.append((iW_col.iloc[i] % nWin) * winSize_sec)
        SZ_decimalHour_new.append(sec2time((iW_col.iloc[i] % nWin) * winSize_sec))
        SZ_annot.append(annots_col.iloc[i])
        Time = sec2time((iW_col.iloc[i] % nWin) * winSize_sec)
        Time = time_sum(Time, rec_start)
        SZ_time.append(Time)
        EMU_time.append(rec_start)

if nSZ:
    iW_col = SZ_iW_new
    Sec_col = SZ_sec_new
    decimalHour_col = SZ_decimalHour_new

# Store annotations in a table
tab = pd.DataFrame({
    'SegmentName': [f"{name}_{segment}"] * len(Sec_col),
    'RecStartTime': [rec_start] * len(Sec_col),
    'SegNumber': [segment] * len(Sec_col),
    'EventSec': Sec_col,
    'Annot': SZ_annot,
    'Time': SZ_time,
    'DecimalHour': decimalHour_col,
    'Aidx_iW': iW_col,
    'EMU_time': EMU_time
})

# Delete duplicate events
tab = tab.drop_duplicates()

# Save annotations
create_folder(f'C:\\Users\\adaraie\\Desktop\\NCSL_Desk\\Prediction\\Codes\\Python\\Outputs')
create_folder(f'C:\\Users\\adaraie\\Desktop\\NCSL_Desk\\Prediction\\Codes\\Python\\Outputs\\{patient_examNo}')
create_folder(f'C:\\Users\\adaraie\\Desktop\\NCSL_Desk\\Prediction\\Codes\\Python\\Outputs\\{patient_examNo}\\7_annot_indices')
tab.to_csv(f'C:\\Users\\adaraie\\Desktop\\NCSL_Desk\\Prediction\\Codes\\Python\\Outputs\\{patient_examNo}\\7_annot_indices\\{name}_{segment}_annot.csv', index=False)
