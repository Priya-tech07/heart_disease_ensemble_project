import shap
import pandas as pd
import numpy as np

def get_local_shap_plot(xgb_model, X_train_df, patient_df):
    """
    Returns SHAP values for one patient and the figure (waterfall plot)
    """
    explainer = shap.Explainer(xgb_model, X_train_df)

    shap_values = explainer(patient_df)

    # Waterfall plot figure
    fig = shap.plots.waterfall(shap_values[0], show=False)
    return shap_values, fig
