import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, classification_report)


import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_bucket_trends(bucket_analysis, bucket_names, yt_palette, 
                       x_col, y_col,
                       show_axes=True, figsize_scatter=(25,5), figsize_chain=(5,5)):
    """
    Plots scatter + regression for each bucket and chained unit trend lines.
    
    Parameters:
    - bucket_analysis : pd.DataFrame containing x_col, y_col, and 'Log Video Views Bucket'
    - bucket_names : list of bucket names to plot
    - yt_palette : dict of custom colors
    - x_col : str, column name for x-axis
    - y_col : str, column name for y-axis
    - show_axes : bool, if True shows x and y axes, otherwise hides them
    - figsize_scatter : tuple, size of the scatter plots figure
    - figsize_chain : tuple, size of the chained unit vectors figure
    """
    
    custom_colors = list(yt_palette.values())
    
    # --- Scatter + regression plots ---
    fig, axes = plt.subplots(1, len(bucket_names), figsize=figsize_scatter, sharey=False)
    if len(bucket_names) == 1:
        axes = [axes]
    
    slopes = []
    intercepts = []
    
    for i, bucket in enumerate(bucket_names):
        subset = bucket_analysis[bucket_analysis["Log Video Views Bucket"] == bucket]
        
        # Scatter
        sns.scatterplot(
            data=subset,
            x=x_col,
            y=y_col,
            color=custom_colors[i % len(custom_colors)],
            alpha=0.7,
            ax=axes[i],
            legend=False
        )
        
        # Linear regression
        x = subset[x_col].values
        y = subset[y_col].values
        if len(x) > 1:
            slope, intercept = np.polyfit(x, y, 1)
            slopes.append(slope)
            intercepts.append(intercept)
            axes[i].plot(x, slope * x + intercept, color="black", linewidth=2, linestyle='--',alpha=0.7) 
            
            eq_text = f"y = {slope:.2f}x + {intercept:.2f}"
            axes[i].text(0.05, 0.9, eq_text, transform=axes[i].transAxes, 
                         fontsize=10, color="black", fontweight="bold")
        else:
            slopes.append(np.nan)
            intercepts.append(np.nan)
        
        axes[i].set_title(f"{bucket}", fontsize=12, fontweight="bold")
        
        if show_axes:
            axes[i].set_xlabel(x_col, fontsize=10)
            axes[i].set_ylabel(y_col, fontsize=10)
        else:
            axes[i].set_xlabel("")
            axes[i].set_ylabel("")
            axes[i].tick_params(axis='both', which='both', length=0)
    
    fig.suptitle(f"{y_col} vs {x_col} by Log Video Views Buckets", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()
    
    # --- Chained unit trend lines ---
    directions = []
    valid_buckets = []
    valid_slopes = []
    valid_intercepts = []
    
    for slope, intercept, bucket in zip(slopes, intercepts, bucket_names):
        if not np.isnan(slope):
            vec = np.array([1, slope])
            vec = vec / np.linalg.norm(vec)
            directions.append(vec)
            valid_buckets.append(bucket)
            valid_slopes.append(slope)
            valid_intercepts.append(intercept)
    
    points = [(0,0)]
    for vec in directions:
        last_point = np.array(points[-1])
        new_point = last_point + vec
        points.append(tuple(new_point))
    
    points = np.array(points)
    plt.figure(figsize=figsize_chain)
    for i, bucket in enumerate(valid_buckets):
        plt.plot(points[i:i+2,0], points[i:i+2,1], marker="o",
                 label=f"{bucket} | y={valid_slopes[i]:.2f}x + {valid_intercepts[i]:.2f}")
    
    plt.axis("equal")
    if not show_axes:
        plt.xticks([])
        plt.yticks([])
    plt.title(f"Chained Unit Trend Lines ({y_col} vs {x_col})")
    plt.grid(True)
    plt.show()


def model_evaluation(model, X_test, y_test):
    # Predictions and probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics

    report_dict = classification_report(y_test, y_pred, output_dict=True)

    # Convert classification report to dataframe
    report_df = pd.DataFrame(report_dict).T.round(2)

    # Subplots 2x2
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # --- 4. Classification Report as Table ---
    axes[0,0].axis("off")  # no axes
    table = axes[0,0].table(cellText=report_df.values,
                            rowLabels=report_df.index,
                            colLabels=report_df.columns,
                            cellLoc='center',
                            loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(0.7, 1.2)
    axes[0,0].set_title("Classification Report")


    # --- 1. Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred 0', 'Pred 1'],
                yticklabels=['True 0', 'True 1'],
                ax=axes[0,1])
    axes[0,1].set_title("Confusion Matrix")
    axes[0,1].set_xlabel("Predicted")
    axes[0,1].set_ylabel("Actual")

    # --- 2. ROC Curve ---
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    axes[1,1].plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    axes[1,1].plot([1, 1], [1, 1], 'k--', linewidth=1)
    axes[1,1].set_title("ROC Curve")
    axes[1,1].set_xlabel("False Positive Rate")
    axes[1,1].set_ylabel("True Positive Rate")
    axes[1,1].legend(loc="lower right")
    axes[1,1].grid(True)

    # --- 3. Precision-Recall Curve ---
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    axes[1,0].plot(recall, precision)
    axes[1,0].set_title("Precision-Recall Curve")
    axes[1,0].set_xlabel("Recall")
    axes[1,0].set_ylabel("Precision")
    axes[1,0].grid(True)



    plt.tight_layout()
    plt.show()



def compare_calibration_curves(model, X_train, y_train, X_test, y_test, n_bins=20):
    """
    Compare calibration of original, Platt scaling, and Isotonic regression models.

    Parameters
    ----------
    model : sklearn estimator
        The base model (must have predict_proba).
    X_train, y_train : array-like
        Training data.
    X_test, y_test : array-like
        Test data.
    n_bins : int, default=20
        Number of bins for calibration curve.

    Returns
    -------
    dict
        Dictionary with Brier scores for Original, Platt, and Isotonic.
    """

    # --- Original model probabilities ---
    probs_original = model.predict_proba(X_test)[:, 1]

    # --- Platt Scaling (sigmoid) ---
    calibrated_clf_platt = CalibratedClassifierCV(model, method='sigmoid', cv=7)
    calibrated_clf_platt.fit(X_train, y_train)
    probs_platt = calibrated_clf_platt.predict_proba(X_test)[:, 1]

    # --- Isotonic Regression ---
    calibrated_clf_iso = CalibratedClassifierCV(model, method='isotonic', cv=7)
    calibrated_clf_iso.fit(X_train, y_train)
    probs_iso = calibrated_clf_iso.predict_proba(X_test)[:, 1]

    # --- Compute Brier scores ---
    scores = {
        "Original": brier_score_loss(y_test, probs_original),
        "Platt": brier_score_loss(y_test, probs_platt),
        "Isotonic": brier_score_loss(y_test, probs_iso)
    }

    # --- Calibration curves ---
    prob_true_orig, prob_pred_orig = calibration_curve(y_test, probs_original, n_bins=n_bins)
    prob_true_platt, prob_pred_platt = calibration_curve(y_test, probs_platt, n_bins=n_bins)
    prob_true_iso, prob_pred_iso = calibration_curve(y_test, probs_iso, n_bins=n_bins)

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Original
    axes[0].plot(prob_pred_orig, prob_true_orig, marker='o', label='Original')
    axes[0].plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    axes[0].set_title(f'Original (Brier: {scores["Original"]:.3f})')
    axes[0].set_xlabel('Predicted probability')
    axes[0].set_ylabel('True probability')
    axes[0].legend()
    axes[0].grid(True)

    # Platt
    axes[1].plot(prob_pred_platt, prob_true_platt, marker='o', label='Platt')
    axes[1].plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    axes[1].set_title(f'Platt (Brier: {scores["Platt"]:.3f})')
    axes[1].set_xlabel('Predicted probability')
    axes[1].set_ylabel('True probability')
    axes[1].legend()
    axes[1].grid(True)

    # Isotonic
    axes[2].plot(prob_pred_iso, prob_true_iso, marker='o', label='Isotonic')
    axes[2].plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    axes[2].set_title(f'Isotonic (Brier: {scores["Isotonic"]:.3f})')
    axes[2].set_xlabel('Predicted probability')
    axes[2].set_ylabel('True probability')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

    return calibrated_clf_platt, calibrated_clf_iso
