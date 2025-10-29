from autopreprocess_pipeline import autopreprocess_standard

varargsin = {
    "sourcepath": r"E:\JHU_Postdoc\Research\TDBrain\TD_BRAIN_code\BRAIN_code\Sample\diff_data",
    "preprocpath": r"E:\JHU_Postdoc\Research\TDBrain\TD_BRAIN_code\BRAIN_code\Sample\p_diff_data",
    "condition": ["EO","EC"],  # or ['rest','task']
    "exclude": []               # optional
}

# Run pipeline on all subjects
autopreprocess_standard(varargsin,subject=3)
