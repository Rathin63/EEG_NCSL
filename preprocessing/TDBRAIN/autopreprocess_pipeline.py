#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 09:45:43 2019

@author: hannekevandijk

copyright: Research Institute Brainclinics, Brainclinics Foundation, Nijmegen, the Netherlands

"""

from autopreprocessing import dataset as ds
from inout import FilepathFinder as FF
import os
from pathlib import Path
import numpy as np
import copy

def autopreprocess_standard(varargsin, subject=None, startsubj=0):
    """Standard autopreprocessing pipeline."""
    import os
    import numpy as np
    import copy
    from pathlib import Path
    import logging
    from datetime import datetime
    from autopreprocessing import dataset as ds
    from inout import FilepathFinder as FF

    # ---------------- Path setup ----------------
    if 'sourcepath' not in varargsin:
        raise ValueError('sourcepath not defined, where is your data?')
    if 'preprocpath' not in varargsin:
        raise ValueError('preprocpath not defined')

    sourcepath = varargsin['sourcepath']
    preprocpath = varargsin['preprocpath']
    
    # Set up logging
    subject_id = subject if subject is not None else "all"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(preprocpath, f"preprocessing_log_sub-{subject_id}_{timestamp}.log")

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logging.info(f"Source path: {sourcepath}")
    logging.info(f"Preprocessing path: {preprocpath}")

    # ---------------- File discovery ----------------
    csv = FF('eeg.csv', sourcepath)
    csv.get_filenames()
    if len(csv.files) < 1:
        print(f"[WARNING]: No files found matching 'eeg.csv' in {sourcepath}. Continuing anyway.")

    # ---------------- Config defaults ----------------
    reqconds = varargsin.get('condition', ['EO', 'EC'])
    varargsin.setdefault('exclude', [])
    rawreport = 'yes'

    # ---------------- Subject detection ----------------
    subs = [s for s in os.listdir(sourcepath)
            if os.path.isdir(os.path.join(sourcepath, s))
            and not any(e in s for e in ['preprocessed', 'results', 'DS'])]
    subs = np.sort(subs)

    flat_mode = False
    if len(subs) == 0:
        csv_files = [os.path.splitext(f)[0] for f in os.listdir(sourcepath)
                     if f.endswith('.csv')]
        subs = np.sort(csv_files)
        flat_mode = True
        print(f"No subject folders found — treating {len(subs)} CSV files as subjects.")
    else:
        print(f"{len(subs)} subject folders found.")

    # ---------------- Subject selection ----------------
    k = startsubj
    if subject is None:
        subarray = range(k, len(subs))
    elif isinstance(subject, int):
        subarray = [subject]
    elif isinstance(subject, str):
        subarray = np.array([np.where(subs == subject)[0]][0])
    else:
        subarray = []

    # Safety fallback
    if len(subs) == 0 or subarray is None:
        print("[WARNING]: No valid subjects found. Exiting.")
        return

    # ---------------- Main processing loop ----------------
    sp = k
    for s in subarray:
        print(f"[INFO]: processing subject: {sp+1} of {len(subs)}")

        if flat_mode:
            inname = os.path.join(sourcepath, subs[s] + '.csv')
            if not os.path.exists(inname):
                print(f"[SKIP]: Missing file {inname}")
                continue

            tmpdat = ds(inname)
            tmpdat.loaddata()
            tmpdat.bipolarEOG()
            tmpdat.apply_filters()
            tmpdat.correct_EOG()
            tmpdat.detect_emg()
            tmpdat.detect_jumps()
            tmpdat.detect_kurtosis()
            tmpdat.detect_extremevoltswing()
            tmpdat.residual_eyeblinks()
            tmpdat.define_artifacts()

            sesspath = os.path.join(preprocpath, subs[s])
            Path(sesspath).mkdir(parents=True, exist_ok=True)

            npy = copy.deepcopy(tmpdat)
            npy.segment(trllength='all', remove_artifact='no')
            npy.save(sesspath)

            if rawreport == 'yes':
                pdf = copy.deepcopy(tmpdat)
                pdf.segment(trllength=10, remove_artifact='no')
                pdf.save_pdfs(sesspath)

        else:
            sessions = [session for session in os.listdir(os.path.join(sourcepath, subs[s]))
                        if not any(e in session for e in ['preprocessed', 'results', 'DS'])]
            for sess in sessions:
                conditions = [conds for conds in os.listdir(os.path.join(sourcepath, subs[s], sess, 'eeg'))
                              if any(a.upper() in conds for a in reqconds)]

                for cond in conditions:
                    inname = os.path.join(sourcepath, subs[s], sess, 'eeg', cond)
                    tmpdat = ds(inname)
                    tmpdat.loaddata()
                    tmpdat.bipolarEOG()
                    tmpdat.apply_filters()
                    tmpdat.correct_EOG()
                    tmpdat.detect_emg()
                    tmpdat.detect_jumps()
                    tmpdat.detect_kurtosis()
                    tmpdat.detect_extremevoltswing()
                    tmpdat.residual_eyeblinks()
                    tmpdat.define_artifacts()

                    sesspath = os.path.join(preprocpath, subs[s], sess, 'eeg')
                    Path(sesspath).mkdir(parents=True, exist_ok=True)

                    npy = copy.deepcopy(tmpdat)
                    npy.segment(trllength='all', remove_artifact='no')
                    npy.save(sesspath)

                    if rawreport == 'yes':
                        pdf = copy.deepcopy(tmpdat)
                        pdf.segment(trllength=10, remove_artifact='no')
                        pdf.save_pdfs(sesspath)
        def summary(self):
            info_lines = [
                f"Data shape: {getattr(self, 'data', np.array([])).shape}",
                f"Number of channels: {len(getattr(self, 'labels', []))}",
                f"Artifacts detected: {self.info.get('repaired channels', 'N/A')}",
                f"Data quality: {self.info.get('data quality', 'Unknown')}",
            ]
            return " | ".join(info_lines)

        logging.info(f"Subject {subs[s]} info:")
        logging.info(tmpdat.info)
        logging.info(f"Subject {subs[s]} summary:")
        logging.info(tmpdat.summary())
        sp += 1
