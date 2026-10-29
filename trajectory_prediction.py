from __future__ import annotations
import argparse
import json
import math
import os
import platform
import sys
import warnings
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
os.environ.setdefault('PYTHONHASHSEED', '123')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, log_loss, matthews_corrcoef, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
try:
    import lightgbm as lgb
except ImportError as exc:
    raise ImportError('Missing lightgbm. Install with: pip install lightgbm') from exc
try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError('Missing xgboost. Install with: pip install xgboost') from exc
try:
    import shap
except ImportError as exc:
    raise ImportError('Missing shap. Install with: pip install shap') from exc

@dataclass
class AnalysisConfig:
    data_path: str = 'data.sav'
    outdir: str = 'chb_trajectory_prediction_results'
    target_column: str = 'Classification'
    predictor_columns: list[str] = field(default_factory=lambda: ['Gen', 'OC', 'PROA', 'REAA', 'ANXRS', 'ANGRS', 'BAT', 'VB', 'SECRDM', 'SECRS', 'SECSA', 'SECSM', 'CDI', 'SEXCL', 'HABT', 'LL', 'SRUM', 'ARUM', 'PSB', 'SDGO', 'ISS', 'CNAR', 'BC', 'PCOS', 'CPMD', 'CMAL', 'PSYA', 'PPOPS', 'PSOCS', 'FQUA', 'DPEE', 'SSB', 'LIKT', 'CCRI'])
    categorical_columns: list[str] = field(default_factory=lambda: ['Gen', 'OC'])
    class_names: dict[int, str] = field(default_factory=lambda: {0: 'Non-involved', 1: 'Bully-victim', 2: 'Victim'})
    random_seed: int = 123
    test_size: float = 0.3
    smote_k_neighbors: int = 5
    sampling_strategy: str = 'not majority'
    search_cv_splits: int = 10
    search_n_iter: int = 50
    selection_metric: str = 'roc_auc_ovr_macro'
    n_jobs: int = -1
    bootstrap_iterations: int = 2000
    bootstrap_alpha: float = 0.05
    figure_dpi: int = 600
    calibration_bins: int = 10
    shap_max_samples: int = 1200
    shap_background_samples: int = 200
    shap_top_n: int = 20
    shap_dependence_top_n: int = 9
    shap_dependence_smooth_window: int = 101
    shap_dependence_min_points: int = 50
    shap_dependence_show_iqr: bool = True
    shap_zero_eps: float = 0.001
    quick_mode: bool = False

    @property
    def numeric_columns(self) -> list[str]:
        return [c for c in self.predictor_columns if c not in self.categorical_columns]

    def finalise(self) -> None:
        if len(self.predictor_columns) != 34:
            raise ValueError(f'Exactly 34 predictors are required, but {len(self.predictor_columns)} were configured.')
        if len(set(self.predictor_columns)) != 34:
            raise ValueError('Duplicate variable names were found in predictor_columns.')
        if not set(self.categorical_columns).issubset(self.predictor_columns):
            raise ValueError('All categorical_columns must also appear in predictor_columns.')
        if len(self.numeric_columns) != 32:
            raise ValueError(f'Exactly 32 numeric predictors are required, but {len(self.numeric_columns)} were configured.')
        if self.quick_mode:
            self.search_cv_splits = 3
            self.search_n_iter = 1
            self.n_jobs = 1
            self.bootstrap_iterations = 30
            self.figure_dpi = 150
            self.shap_max_samples = 100
            self.shap_background_samples = 50
            self.shap_top_n = min(self.shap_top_n, 10)
            self.shap_dependence_top_n = min(self.shap_dependence_top_n, 2)
            self.shap_dependence_smooth_window = 31

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_json(obj: Any, path: Path) -> None:

    def convert(x: Any) -> Any:
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, Mapping):
            return {str(k): convert(v) for k, v in x.items()}
        if isinstance(x, (list, tuple, set)):
            return [convert(v) for v in x]
        return x
    with path.open('w', encoding='utf-8') as f:
        json.dump(convert(obj), f, ensure_ascii=False, indent=2)

def save_figure(fig: plt.Figure, prefix: Path, dpi: int) -> None:
    fig.savefig(prefix.with_suffix('.png'), dpi=dpi, bbox_inches='tight')
    fig.savefig(prefix.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)

def slug(text: str) -> str:
    return text.lower().replace(' ', '_').replace('-', '_').replace('/', '_').replace('(', '').replace(')', '')

def package_versions() -> dict[str, str]:
    packages = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'imbalanced-learn', 'xgboost', 'lightgbm', 'shap', 'matplotlib', 'joblib', 'pyreadstat']
    versions: dict[str, str] = {'python': sys.version.replace('\n', ' '), 'platform': platform.platform()}
    for pkg in packages:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = 'not installed'
    return versions

def read_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == '.sav':
        try:
            return pd.read_spss(path)
        except ImportError as exc:
            raise ImportError('Reading .sav files requires pyreadstat. Install with: pip install pyreadstat') from exc
    if suffix == '.csv':
        return pd.read_csv(path)
    if suffix in {'.xlsx', '.xls'}:
        return pd.read_excel(path)
    if suffix == '.parquet':
        return pd.read_parquet(path)
    raise ValueError(f'Unsupported data format: {suffix}')

def validate_and_prepare_data(data: pd.DataFrame, cfg: AnalysisConfig, outdir: Path) -> tuple[pd.DataFrame, pd.Series]:
    if cfg.target_column not in data.columns:
        raise KeyError(f"Target column '{cfg.target_column}' was not found.")
    missing_predictors = [c for c in cfg.predictor_columns if c not in data.columns]
    if missing_predictors:
        raise KeyError(f'The following required predictors were not found in the input data: {missing_predictors}')
    data = data.copy()
    data['__original_row_index__'] = data.index.astype(str)
    n_missing_target = int(data[cfg.target_column].isna().sum())
    if n_missing_target:
        warnings.warn(f'Dropping {n_missing_target} rows with missing target labels.')
        data = data.loc[data[cfg.target_column].notna()].copy()
    y = pd.to_numeric(data[cfg.target_column], errors='raise').astype(int)
    expected = set(cfg.class_names)
    observed = set(y.unique())
    if observed != expected:
        raise ValueError(f'Observed target labels {sorted(observed)} do not exactly match configured labels {sorted(expected)}.')
    X_raw = data.loc[:, cfg.predictor_columns].copy()
    for col in X_raw.columns:
        X_raw[col] = pd.to_numeric(X_raw[col], errors='raise')
    for col in cfg.categorical_columns:
        valid_values = set(pd.Series(X_raw[col]).dropna().unique().tolist())
        if not valid_values.issubset({0, 1, 0.0, 1.0}):
            raise ValueError(f"Categorical predictor '{col}' must be coded only as 0/1; observed values were {sorted(valid_values)}.")
    if X_raw.columns.duplicated().any():
        dupes = X_raw.columns[X_raw.columns.duplicated()].tolist()
        raise ValueError(f'Duplicate predictor names found: {dupes}')
    ignored_columns = [c for c in data.columns if c not in set(cfg.predictor_columns + [cfg.target_column, '__original_row_index__'])]
    save_json({'predictor_count': len(cfg.predictor_columns), 'predictors_used': cfg.predictor_columns, 'categorical_predictors': cfg.categorical_columns, 'numeric_predictors': cfg.numeric_columns, 'ignored_input_columns': ignored_columns}, outdir / 'predictor_allowlist_manifest.json')
    distribution = y.value_counts().sort_index().rename('n').to_frame().assign(proportion=lambda d: d['n'] / d['n'].sum())
    distribution['class_name'] = [cfg.class_names[int(i)] for i in distribution.index]
    distribution.to_csv(outdir / 'class_distribution_full_sample.csv', encoding='utf-8-sig')
    missing_summary = pd.DataFrame({'feature': X_raw.columns, 'n_missing': X_raw.isna().sum().values, 'proportion_missing': X_raw.isna().mean().values}).sort_values('proportion_missing', ascending=False)
    missing_summary.to_csv(outdir / 'predictor_missingness.csv', index=False, encoding='utf-8-sig')
    if X_raw.isna().any().any():
        missing_cols = X_raw.columns[X_raw.isna().any()].tolist()
        raise ValueError(f'This workflow expects the item-level-imputed predictor dataset to be complete. Missing values were found in: {missing_cols}.')
    row_ids = data['__original_row_index__'].copy()
    X_raw.index = row_ids
    y.index = row_ids
    return (X_raw, y)

def safe_smote_k(y_train: pd.Series, cv_splits: int, requested_k: int) -> int:
    min_count = int(y_train.value_counts().min())
    inner_train_min = math.floor(min_count * (cv_splits - 1) / cv_splits)
    if inner_train_min < 2:
        raise ValueError('Too few minority-class observations for SMOTENC within CV folds.')
    return max(1, min(requested_k, inner_train_min - 1))

def model_feature_order(cfg: AnalysisConfig) -> list[str]:
    return list(cfg.categorical_columns) + list(cfg.numeric_columns)

def build_preprocessor(cfg: AnalysisConfig) -> ColumnTransformer:
    return ColumnTransformer(transformers=[('categorical', 'passthrough', cfg.categorical_columns), ('numeric', StandardScaler(), cfg.numeric_columns)], remainder='drop', sparse_threshold=0.0, verbose_feature_names_out=False)

def build_pipeline(estimator: BaseEstimator, cfg: AnalysisConfig, safe_k: int) -> Pipeline:
    feature_order = model_feature_order(cfg)
    categorical_indices = [feature_order.index(c) for c in cfg.categorical_columns]
    sampler = SMOTENC(categorical_features=categorical_indices, sampling_strategy=cfg.sampling_strategy, random_state=cfg.random_seed, k_neighbors=safe_k)
    return Pipeline(steps=[('preprocessor', build_preprocessor(cfg)), ('sampler', sampler), ('clf', estimator)])

def transformed_model_dataframe(fitted_pipeline: Pipeline, X_raw: pd.DataFrame, cfg: AnalysisConfig) -> pd.DataFrame:
    transformed = fitted_pipeline.named_steps['preprocessor'].transform(X_raw)
    return pd.DataFrame(np.asarray(transformed, dtype=float), columns=model_feature_order(cfg), index=X_raw.index)

def save_selected_standardization_parameters(selected_pipeline: Pipeline, cfg: AnalysisConfig, outdir: Path) -> None:
    preprocessor = selected_pipeline.named_steps['preprocessor']
    scaler = preprocessor.named_transformers_['numeric']
    pd.DataFrame({'Feature': cfg.numeric_columns, 'Mean': scaler.mean_, 'Variance': scaler.var_, 'Scale_SD_ddof0': scaler.scale_}).to_csv(outdir / 'numeric_standardization_parameters.csv', index=False, encoding='utf-8-sig')
    save_json({'method': 'StandardScaler', 'fit_scope_during_tuning': 'fitted separately within each training fold of StratifiedKFold via the pipeline', 'fit_scope_final_model': 'refitted on the full original 70% training split after hyperparameter selection', 'test_used_to_fit_scaler': False, 'standardized_numeric_predictors': cfg.numeric_columns, 'unstandardized_categorical_predictors': cfg.categorical_columns, 'n_numeric': len(cfg.numeric_columns), 'n_categorical': len(cfg.categorical_columns)}, outdir / 'standardization_manifest.json')

def summarize_final_training_resample(selected_pipeline: Pipeline, X_train_raw: pd.DataFrame, y_train: pd.Series, cfg: AnalysisConfig, safe_k: int, outdir: Path) -> None:
    X_train_model = transformed_model_dataframe(selected_pipeline, X_train_raw, cfg)
    feature_order = model_feature_order(cfg)
    categorical_indices = [feature_order.index(c) for c in cfg.categorical_columns]
    sampler = SMOTENC(categorical_features=categorical_indices, sampling_strategy=cfg.sampling_strategy, random_state=cfg.random_seed, k_neighbors=safe_k)
    _, y_resampled = sampler.fit_resample(X_train_model, y_train)
    n_original_train = len(X_train_raw)
    n_synthetic_train = len(y_resampled) - n_original_train
    distribution = pd.Series(y_resampled).value_counts().sort_index().rename('n').to_frame().assign(proportion=lambda d: d['n'] / d['n'].sum())
    distribution['class_name'] = [cfg.class_names[int(i)] for i in distribution.index]
    distribution.to_csv(outdir / 'class_distribution_training_after_SMOTENC.csv', encoding='utf-8-sig')
    save_json({'resampling_method': 'SMOTENC', 'resampling_timing_during_tuning': 'inside each CV training fold after fold-specific numeric standardization', 'resampling_timing_final_refit': 'on the full original training split after training-fitted numeric standardization', 'test_oversampled': False, 'validation_folds_oversampled': False, 'k_neighbors': int(safe_k), 'sampling_strategy': cfg.sampling_strategy, 'original_training_n': int(n_original_train), 'synthetic_training_n_in_final_refit': int(n_synthetic_train), 'resampled_training_n_in_final_refit': int(len(y_resampled))}, outdir / 'post_split_SMOTENC_manifest.json')

def model_definitions(seed: int) -> dict[str, tuple[BaseEstimator, dict[str, Any]]]:
    return {'Logistic Regression': (LogisticRegression(solver='lbfgs', max_iter=8000, random_state=seed), {'clf__C': loguniform(0.001, 100.0), 'clf__tol': loguniform(1e-06, 0.01)}), 'Decision Tree': (DecisionTreeClassifier(random_state=seed), {'clf__criterion': ['gini', 'entropy', 'log_loss'], 'clf__max_depth': [None, 3, 5, 8, 12, 16, 24], 'clf__min_samples_split': randint(2, 30), 'clf__min_samples_leaf': randint(1, 20), 'clf__max_features': [None, 'sqrt', 'log2']}), 'K-Nearest Neighbors': (KNeighborsClassifier(), {'clf__n_neighbors': randint(3, 31), 'clf__weights': ['uniform', 'distance'], 'clf__p': [1, 2], 'clf__leaf_size': randint(15, 61)}), 'Random Forest': (RandomForestClassifier(random_state=seed, n_jobs=1), {'clf__n_estimators': randint(300, 1001), 'clf__max_depth': [None, 5, 8, 12, 16, 24], 'clf__min_samples_split': randint(2, 20), 'clf__min_samples_leaf': randint(1, 12), 'clf__max_features': ['sqrt', 'log2', 0.5, 0.8], 'clf__bootstrap': [True, False]}), 'Support Vector Machine': (SVC(kernel='rbf', probability=True, random_state=seed), {'clf__C': loguniform(0.01, 1000.0), 'clf__gamma': loguniform(0.0001, 1.0)}), 'Multilayer Perceptron': (MLPClassifier(max_iter=1500, early_stopping=True, n_iter_no_change=30, random_state=seed), {'clf__hidden_layer_sizes': [(32,), (64,), (128,), (64, 32), (128, 64), (128, 64, 32)], 'clf__activation': ['relu', 'tanh'], 'clf__alpha': loguniform(1e-06, 0.1), 'clf__learning_rate_init': loguniform(0.0001, 0.01), 'clf__batch_size': [32, 64, 128, 'auto']}), 'Light Gradient Boosting': (lgb.LGBMClassifier(objective='multiclass', random_state=seed, n_jobs=1, verbosity=-1, subsample_freq=1), {'clf__n_estimators': randint(200, 1001), 'clf__learning_rate': loguniform(0.001, 0.2), 'clf__num_leaves': randint(15, 128), 'clf__max_depth': [-1, 3, 5, 8, 12], 'clf__min_child_samples': randint(5, 80), 'clf__subsample': [0.8, 0.9, 1.0], 'clf__colsample_bytree': uniform(0.6, 0.4), 'clf__reg_alpha': loguniform(1e-06, 10), 'clf__reg_lambda': loguniform(1e-06, 10)}), 'eXtreme Gradient Boosting': (XGBClassifier(objective='multi:softprob', num_class=3, eval_metric='mlogloss', tree_method='hist', random_state=seed, n_jobs=1), {'clf__n_estimators': randint(200, 1001), 'clf__learning_rate': loguniform(0.001, 0.2), 'clf__max_depth': randint(2, 11), 'clf__min_child_weight': loguniform(0.1, 30), 'clf__subsample': uniform(0.6, 0.4), 'clf__colsample_bytree': uniform(0.6, 0.4), 'clf__gamma': loguniform(1e-06, 10), 'clf__reg_alpha': loguniform(1e-06, 10), 'clf__reg_lambda': loguniform(0.0001, 30)})}

def scoring_dictionary() -> dict[str, str]:
    return {'accuracy': 'accuracy', 'balanced_accuracy': 'balanced_accuracy', 'precision_macro': 'precision_macro', 'recall_macro': 'recall_macro', 'f1_macro': 'f1_macro', 'roc_auc_ovr_macro': 'roc_auc_ovr'}

def describe_search_parameter(value: Any) -> dict[str, Any]:
    if isinstance(value, (list, tuple, set)):
        return {'distribution_type': 'categorical/discrete choices', 'values': list(value)}
    dist = getattr(value, 'dist', None)
    dist_name = getattr(dist, 'name', None)
    args = list(getattr(value, 'args', ()))
    kwds = dict(getattr(value, 'kwds', {}))
    if dist_name == 'loguniform' and len(args) >= 2:
        return {'distribution_type': 'log-uniform', 'lower': float(args[0]), 'upper': float(args[1])}
    if dist_name == 'uniform' and len(args) >= 2:
        loc, scale = (float(args[0]), float(args[1]))
        return {'distribution_type': 'continuous uniform', 'lower': loc, 'upper': loc + scale, 'scipy_loc': loc, 'scipy_scale': scale}
    if dist_name == 'randint' and len(args) >= 2:
        return {'distribution_type': 'discrete integer uniform', 'lower_inclusive': int(args[0]), 'upper_exclusive': int(args[1]), 'upper_inclusive': int(args[1]) - 1}
    if dist_name is not None:
        return {'distribution_type': str(dist_name), 'args': args, 'kwargs': kwds}
    return {'distribution_type': 'fixed/object', 'value': repr(value)}

def search_space_manifest(definitions: Mapping[str, tuple[BaseEstimator, dict[str, Any]]]) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    for name, (estimator, space) in definitions.items():
        manifest[name] = {'estimator': repr(estimator), 'optimization_algorithm': 'RandomizedSearchCV', 'parameter_space': {key: describe_search_parameter(value) for key, value in space.items()}}
    return manifest

def multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray, labels: Sequence[int]) -> float:
    y_bin = label_binarize(y_true, classes=list(labels))
    return float(np.mean(np.sum((y_prob - y_bin) ** 2, axis=1)))

def evaluate_predictions(y_true: pd.Series | np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, labels: Sequence[int], class_names: Mapping[int, str]) -> tuple[dict[str, float], pd.DataFrame, np.ndarray]:
    y_true_arr = np.asarray(y_true, dtype=int)
    labels_arr = np.asarray(labels, dtype=int)
    y_bin = label_binarize(y_true_arr, classes=labels_arr)
    overall: dict[str, float] = {'Accuracy': float(accuracy_score(y_true_arr, y_pred)), 'Balanced_Accuracy': float(balanced_accuracy_score(y_true_arr, y_pred)), 'Precision_Macro': float(precision_score(y_true_arr, y_pred, average='macro', zero_division=0)), 'Recall_Macro': float(recall_score(y_true_arr, y_pred, average='macro', zero_division=0)), 'F1_Macro': float(f1_score(y_true_arr, y_pred, average='macro', zero_division=0)), 'Precision_Weighted': float(precision_score(y_true_arr, y_pred, average='weighted', zero_division=0)), 'Recall_Weighted': float(recall_score(y_true_arr, y_pred, average='weighted', zero_division=0)), 'F1_Weighted': float(f1_score(y_true_arr, y_pred, average='weighted', zero_division=0)), 'Matthews_Correlation': float(matthews_corrcoef(y_true_arr, y_pred)), 'ROC_AUC_OvR_Macro': float(roc_auc_score(y_true_arr, y_prob, labels=labels_arr, multi_class='ovr', average='macro')), 'AP_OvR_Macro': float(np.mean([average_precision_score(y_bin[:, j], y_prob[:, j]) for j in range(len(labels_arr))])), 'Log_Loss': float(log_loss(y_true_arr, y_prob, labels=labels_arr)), 'Brier_Multiclass': multiclass_brier_score(y_true_arr, y_prob, labels_arr)}
    cm = confusion_matrix(y_true_arr, y_pred, labels=labels_arr)
    rows: list[dict[str, Any]] = []
    for j, label in enumerate(labels_arr):
        true_bin = (y_true_arr == label).astype(int)
        pred_bin = (y_pred == label).astype(int)
        tn, fp, fn, tp = confusion_matrix(true_bin, pred_bin, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if tn + fp else np.nan
        rows.append({'Class': int(label), 'Class_Name': class_names[int(label)], 'Support': int(true_bin.sum()), 'Precision': float(precision_score(true_bin, pred_bin, zero_division=0)), 'Recall_Sensitivity': float(recall_score(true_bin, pred_bin, zero_division=0)), 'Specificity': float(specificity), 'F1': float(f1_score(true_bin, pred_bin, zero_division=0)), 'ROC_AUC_OvR': float(roc_auc_score(true_bin, y_prob[:, j])), 'AP_OvR': float(average_precision_score(true_bin, y_prob[:, j])), 'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn)})
    return (overall, pd.DataFrame(rows), cm)

def bootstrap_test_intervals(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, labels: Sequence[int], class_names: Mapping[int, str], iterations: int, alpha: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    records: list[dict[str, float]] = []
    for _ in range(iterations):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if set(np.unique(yt)) != set(labels):
            continue
        yp = y_pred[idx]
        pr = y_prob[idx]
        try:
            overall, per_class, _ = evaluate_predictions(yt, yp, pr, labels, class_names)
        except ValueError:
            continue
        row = {'Accuracy': overall['Accuracy'], 'Balanced_Accuracy': overall['Balanced_Accuracy'], 'F1_Macro': overall['F1_Macro'], 'Recall_Macro': overall['Recall_Macro'], 'Precision_Macro': overall['Precision_Macro'], 'ROC_AUC_OvR_Macro': overall['ROC_AUC_OvR_Macro'], 'AP_OvR_Macro': overall['AP_OvR_Macro']}
        for _, r in per_class.iterrows():
            label = int(r['Class'])
            row[f'Class{label}_Recall'] = float(r['Recall_Sensitivity'])
            row[f'Class{label}_Precision'] = float(r['Precision'])
            row[f'Class{label}_F1'] = float(r['F1'])
            row[f'Class{label}_ROC_AUC'] = float(r['ROC_AUC_OvR'])
            row[f'Class{label}_AP'] = float(r['AP_OvR'])
        records.append(row)
    if not records:
        raise RuntimeError('No valid bootstrap samples were generated.')
    boot = pd.DataFrame(records)
    lower_q = alpha / 2
    upper_q = 1 - alpha / 2
    summary = pd.DataFrame({'Metric': boot.columns, 'Bootstrap_Mean': boot.mean().values, 'CI_Lower': boot.quantile(lower_q).values, 'CI_Upper': boot.quantile(upper_q).values, 'Valid_Replicates': len(boot)})
    return summary

def plot_cv_model_comparison(cv_table: pd.DataFrame, outdir: Path, dpi: int) -> None:
    df = cv_table.sort_values('CV_ROC_AUC_OvR_Macro_Mean', ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(df['Model'], df['CV_ROC_AUC_OvR_Macro_Mean'], xerr=df['CV_ROC_AUC_OvR_Macro_SD'], capsize=3)
    ax.set_xlabel('Training CV macro ROC-AUC (OvR)')
    ax.set_ylabel('Algorithm')
    ax.set_title('Model selection based on training-set cross-validation')
    ax.set_xlim(0, 1)
    fig.tight_layout()
    save_figure(fig, outdir / 'Figure_model_selection_cv_macro_roc_auc', dpi)

def plot_test_model_comparison(test_table: pd.DataFrame, outdir: Path, dpi: int) -> None:
    df = test_table.sort_values('F1_Macro', ascending=False).reset_index(drop=True)
    x = np.arange(len(df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, df['F1_Macro'], width, label='Macro-F1')
    ax.bar(x, df['ROC_AUC_OvR_Macro'], width, label='Macro ROC-AUC')
    ax.bar(x + width, df['AP_OvR_Macro'], width, label='Macro AP')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Model'], rotation=35, ha='right')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Test-set performance')
    ax.set_title('Tuned algorithms on the test set\n(not used to select the final algorithm)')
    ax.legend()
    fig.tight_layout()
    save_figure(fig, outdir / 'Figure_all_models_test_metrics', dpi)

def plot_confusion(cm: np.ndarray, labels: Sequence[int], class_names: Mapping[int, str], outdir: Path, dpi: int) -> None:
    names = [class_names[int(c)] for c in labels]
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm, interpolation='nearest')
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_yticklabels(names)
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')
    ax.set_title('Selected model: test confusion matrix')
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > threshold else 'black')
    fig.tight_layout()
    save_figure(fig, outdir / 'Figure_selected_model_confusion_matrix', dpi)

def plot_roc_pr_calibration(y_true: np.ndarray, y_prob: np.ndarray, labels: Sequence[int], class_names: Mapping[int, str], outdir: Path, dpi: int, calibration_bins: int) -> None:
    y_bin = label_binarize(y_true, classes=list(labels))
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    ax_roc.plot([0, 1], [0, 1], linestyle='--', linewidth=1, label='Chance')
    for j, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_bin[:, j], y_prob[:, j])
        auc_val = roc_auc_score(y_bin[:, j], y_prob[:, j])
        ax_roc.plot(fpr, tpr, linewidth=2, label=f'{class_names[int(label)]} (AUC={auc_val:.3f})')
    ax_roc.set_xlabel('False positive rate')
    ax_roc.set_ylabel('True positive rate')
    ax_roc.set_title('Selected model: one-vs-rest ROC curves')
    ax_roc.legend(loc='lower right')
    fig_roc.tight_layout()
    save_figure(fig_roc, outdir / 'Figure_selected_model_ROC', dpi)
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
    for j, label in enumerate(labels):
        precision, recall, _ = precision_recall_curve(y_bin[:, j], y_prob[:, j])
        ap = average_precision_score(y_bin[:, j], y_prob[:, j])
        prevalence = y_bin[:, j].mean()
        ax_pr.plot(recall, precision, linewidth=2, label=f'{class_names[int(label)]} (AP={ap:.3f})')
        ax_pr.axhline(prevalence, linestyle=':', linewidth=0.8)
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Selected model: one-vs-rest precision-recall curves')
    ax_pr.legend(loc='lower left')
    fig_pr.tight_layout()
    save_figure(fig_pr, outdir / 'Figure_selected_model_precision_recall', dpi)
    fig_cal, axes = plt.subplots(1, len(labels), figsize=(5.2 * len(labels), 4.5), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for j, label in enumerate(labels):
        frac_pos, mean_pred = calibration_curve(y_bin[:, j], y_prob[:, j], n_bins=calibration_bins, strategy='quantile')
        axes[j].plot([0, 1], [0, 1], linestyle='--', linewidth=1)
        axes[j].plot(mean_pred, frac_pos, marker='o', linewidth=2)
        axes[j].set_title(class_names[int(label)])
        axes[j].set_xlabel('Mean predicted probability')
        if j == 0:
            axes[j].set_ylabel('Observed proportion')
        axes[j].set_xlim(0, 1)
        axes[j].set_ylim(0, 1)
    fig_cal.suptitle('Selected model: one-vs-rest calibration', y=1.02)
    fig_cal.tight_layout()
    save_figure(fig_cal, outdir / 'Figure_selected_model_calibration', dpi)

def is_tree_estimator(estimator: BaseEstimator) -> bool:
    name = estimator.__class__.__name__.lower()
    tokens = ('tree', 'forest', 'xgb', 'lgbm', 'gradientboost')
    return any((token in name for token in tokens))

def normalise_shap_values(values: Any, n_samples: int, n_features: int, n_classes: int) -> np.ndarray:
    if hasattr(values, 'values'):
        values = values.values
    if isinstance(values, list):
        arr = np.stack([np.asarray(v) for v in values], axis=-1)
    else:
        arr = np.asarray(values)
    if arr.ndim == 2:
        if n_classes != 1:
            raise ValueError(f'Received 2D SHAP values for {n_classes} classes: {arr.shape}')
        arr = arr[:, :, None]
    elif arr.ndim == 3:
        if arr.shape == (n_samples, n_features, n_classes):
            pass
        elif arr.shape == (n_classes, n_samples, n_features):
            arr = np.transpose(arr, (1, 2, 0))
        elif arr.shape == (n_samples, n_classes, n_features):
            arr = np.transpose(arr, (0, 2, 1))
        else:
            raise ValueError(f'Unrecognised multiclass SHAP shape: {arr.shape}')
    else:
        raise ValueError(f'Unsupported SHAP array dimensions: {arr.shape}')
    return arr

def compute_shap_values(pipeline: Pipeline, X_train_raw: pd.DataFrame, X_test_raw: pd.DataFrame, cfg: AnalysisConfig) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(cfg.random_seed)
    feature_order = model_feature_order(cfg)
    X_train_model = transformed_model_dataframe(pipeline, X_train_raw, cfg)
    X_test_model = transformed_model_dataframe(pipeline, X_test_raw, cfg)
    X_test_raw_ordered = X_test_raw.loc[:, feature_order].copy()
    if len(X_test_model) > cfg.shap_max_samples:
        chosen = rng.choice(X_test_model.index.to_numpy(), size=cfg.shap_max_samples, replace=False)
        X_explain_model = X_test_model.loc[chosen].copy()
        X_explain_raw = X_test_raw_ordered.loc[chosen].copy()
    else:
        X_explain_model = X_test_model.copy()
        X_explain_raw = X_test_raw_ordered.loc[X_explain_model.index].copy()
    clf = pipeline.named_steps['clf']
    classes = np.asarray(clf.classes_, dtype=int)
    if is_tree_estimator(clf):
        explainer = shap.TreeExplainer(clf)
        try:
            explanation = explainer(X_explain_model, check_additivity=False)
            values = explanation.values
        except Exception:
            values = explainer.shap_values(X_explain_model, check_additivity=False)
    else:
        if len(X_train_model) > cfg.shap_background_samples:
            bg_idx = rng.choice(X_train_model.index.to_numpy(), size=cfg.shap_background_samples, replace=False)
            background = X_train_model.loc[bg_idx]
        else:
            background = X_train_model
        explainer = shap.Explainer(clf.predict_proba, background, feature_names=feature_order, output_names=[cfg.class_names[int(c)] for c in classes], seed=cfg.random_seed)
        explanation = explainer(X_explain_model)
        values = explanation.values
    shap_3d = normalise_shap_values(values, n_samples=len(X_explain_model), n_features=X_explain_model.shape[1], n_classes=len(classes))
    return (X_explain_raw, shap_3d)

def plot_shap_outputs(X_raw: pd.DataFrame, shap_3d: np.ndarray, classes: Sequence[int], cfg: AnalysisConfig, outdir: Path) -> tuple[dict[int, list[str]], pd.DataFrame]:
    class_top_features: dict[int, list[str]] = {}
    importance_rows: list[dict[str, Any]] = []
    fig_bar, axes = plt.subplots(1, len(classes), figsize=(6 * len(classes), 8), squeeze=False)
    for class_pos, label in enumerate(classes):
        values = shap_3d[:, :, class_pos]
        mean_abs = np.mean(np.abs(values), axis=0)
        order = np.argsort(mean_abs)[::-1]
        top_idx = order[:min(cfg.shap_top_n, len(order))]
        top_features = X_raw.columns[top_idx].tolist()
        class_top_features[int(label)] = top_features
        for rank, idx in enumerate(order, start=1):
            importance_rows.append({'Class': int(label), 'Class_Name': cfg.class_names[int(label)], 'Feature': X_raw.columns[idx], 'Mean_Abs_SHAP': float(mean_abs[idx]), 'Rank': rank})
        ax = axes[0, class_pos]
        y_pos = np.arange(len(top_idx))[::-1]
        ax.barh(y_pos, mean_abs[top_idx][::-1])
        ax.set_yticks(y_pos)
        ax.set_yticklabels([X_raw.columns[i] for i in top_idx][::-1])
        ax.set_xlabel('Mean |SHAP value|')
        ax.set_title(cfg.class_names[int(label)])
        plt.figure(figsize=(9, 9))
        shap.summary_plot(values, X_raw, max_display=min(cfg.shap_top_n, X_raw.shape[1]), show=False, plot_size=None)
        fig_bee = plt.gcf()
        fig_bee.suptitle(f'SHAP summary: {cfg.class_names[int(label)]}', y=1.01)
        fig_bee.tight_layout()
        save_figure(fig_bee, outdir / f'Figure_SHAP_beeswarm_{slug(cfg.class_names[int(label)])}', cfg.figure_dpi)
    fig_bar.suptitle('Class-specific feature importance from the selected model', y=1.01)
    fig_bar.tight_layout()
    save_figure(fig_bar, outdir / 'Figure_SHAP_class_specific_importance', cfg.figure_dpi)
    importance_df = pd.DataFrame(importance_rows)
    importance_df.to_csv(outdir / 'SHAP_class_specific_importance.csv', index=False, encoding='utf-8-sig')
    global_df = importance_df.groupby('Feature', as_index=False)['Mean_Abs_SHAP'].mean().sort_values('Mean_Abs_SHAP', ascending=False).reset_index(drop=True)
    global_df['Rank'] = np.arange(1, len(global_df) + 1)
    global_df.to_csv(outdir / 'SHAP_global_importance.csv', index=False, encoding='utf-8-sig')
    fig_global, ax_global = plt.subplots(figsize=(8, 9))
    top = global_df.head(min(cfg.shap_top_n, len(global_df))).sort_values('Mean_Abs_SHAP')
    ax_global.barh(top['Feature'], top['Mean_Abs_SHAP'])
    ax_global.set_xlabel('Mean |SHAP value| averaged across classes')
    ax_global.set_title('Global feature importance')
    fig_global.tight_layout()
    save_figure(fig_global, outdir / 'Figure_SHAP_global_importance', cfg.figure_dpi)
    return (class_top_features, importance_df)

def feature_is_categorical_or_discrete(series: pd.Series, configured_categorical: set[str]) -> bool:
    if series.name in configured_categorical:
        return True
    values = pd.to_numeric(series, errors='coerce').dropna().unique()
    if len(values) <= 10 and np.allclose(values, np.round(values)):
        return True
    return False

def report_shap_plot_types(X_raw: pd.DataFrame, cfg: AnalysisConfig, outdir: Path) -> tuple[list[str], list[str]]:
    categorical_set = set(cfg.categorical_columns)
    discrete = [feature for feature in X_raw.columns if feature_is_categorical_or_discrete(X_raw[feature], categorical_set)]
    continuous = [feature for feature in X_raw.columns if feature not in discrete]
    save_json({'rule': 'configured categorical OR <=10 unique integer values', 'discrete_plot_variables': discrete, 'continuous_dependence_plot_variables': continuous, 'continuous_display': 'raw-score scatter + rolling mean + rolling IQR band'}, outdir / 'SHAP_plot_variable_types.json')
    return (discrete, continuous)

def _safe_rolling_window(window: int, n_points: int) -> int:
    if n_points < 5:
        return max(1, n_points)
    window = max(5, min(int(window), int(n_points)))
    if window % 2 == 0:
        window += 1
    if window > n_points:
        window = n_points if n_points % 2 == 1 else n_points - 1
    return max(5, window)

def _summarize_x_near_shap_zero(feature: str, x_raw: pd.Series, shap_values: np.ndarray, zero_eps: float, discrete: bool) -> dict[str, Any]:
    df = pd.DataFrame({'Raw_X': pd.to_numeric(x_raw, errors='coerce').to_numpy(), 'SHAP_Value': np.asarray(shap_values, dtype=float)}).dropna()
    near = df.loc[np.abs(df['SHAP_Value']) <= float(zero_eps)].copy()
    record: dict[str, Any] = {'Feature': feature, 'N_Near_Zero': int(len(near)), 'Zero_Epsilon': float(zero_eps), 'Plot_Type': 'discrete' if discrete else 'continuous'}
    if near.empty:
        return record
    if discrete:
        record['Level_Counts'] = {str(k): int(v) for k, v in near['Raw_X'].value_counts().sort_index().items()}
    else:
        qs = near['Raw_X'].quantile([0, 0.25, 0.5, 0.75, 1.0])
        record.update({'Raw_X_Min': float(qs.loc[0.0]), 'Raw_X_Q1': float(qs.loc[0.25]), 'Raw_X_Median': float(qs.loc[0.5]), 'Raw_X_Q3': float(qs.loc[0.75]), 'Raw_X_Max': float(qs.loc[1.0])})
    return record

def _find_all_shap_zero_crossings(observed_x: np.ndarray, smooth_x: np.ndarray, smooth_y: np.ndarray) -> list[tuple[float, float]]:
    xs = np.asarray(smooth_x, dtype=float)
    ys = np.asarray(smooth_y, dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[mask]
    ys = ys[mask]
    observed_x = np.asarray(observed_x, dtype=float)
    observed_x = observed_x[np.isfinite(observed_x)]
    if xs.size < 2 or observed_x.size == 0:
        return []
    crossings: list[tuple[float, float]] = []
    for idx in range(len(ys) - 1):
        y1, y2 = (ys[idx], ys[idx + 1])
        x1, x2 = (xs[idx], xs[idx + 1])
        x_theory: float | None = None
        if np.isclose(y1, 0.0):
            x_theory = float(x1)
        elif y1 * y2 < 0 and (not np.isclose(y2 - y1, 0.0)):
            x_theory = float(x1 - y1 * (x2 - x1) / (y2 - y1))
        if x_theory is None or not np.isfinite(x_theory):
            continue
        x_real = float(observed_x[np.argmin(np.abs(observed_x - x_theory))])
        crossings.append((x_theory, x_real))
    crossings.sort(key=lambda item: item[0])
    unique: list[tuple[float, float]] = []
    for x_theory, x_real in crossings:
        if not unique or not np.isclose(x_theory, unique[-1][0], atol=1e-08, rtol=0):
            unique.append((x_theory, x_real))
    return unique

def plot_shap_dependence_grids(X_raw: pd.DataFrame, shap_3d: np.ndarray, classes: Sequence[int], class_top_features: Mapping[int, Sequence[str]], cfg: AnalysisConfig, outdir: Path) -> None:
    categorical_set = set(cfg.categorical_columns)
    all_crossings: list[dict[str, Any]] = []
    all_zero_summaries: list[dict[str, Any]] = []
    for label in list(classes):
        class_pos = list(classes).index(label)
        class_name = cfg.class_names[int(label)]
        values = shap_3d[:, :, class_pos]
        features = list(class_top_features[int(label)])[:cfg.shap_dependence_top_n]
        fig, axes = plt.subplots(3, 3, figsize=(16.2, 12.6))
        axes = np.asarray(axes).reshape(-1)
        class_crossings: list[dict[str, Any]] = []
        class_zero_summaries: list[dict[str, Any]] = []
        class_categorical_zero: list[dict[str, Any]] = []
        for panel_idx, feature in enumerate(features):
            ax = axes[panel_idx]
            feature_idx = X_raw.columns.get_loc(feature)
            x_series = pd.to_numeric(X_raw[feature], errors='coerce')
            sv = np.asarray(values[:, feature_idx], dtype=float)
            df_xy = pd.DataFrame({'x': x_series.to_numpy(), 'y': sv}).dropna()
            discrete = feature_is_categorical_or_discrete(X_raw[feature], categorical_set)
            zero_summary = _summarize_x_near_shap_zero(feature, X_raw[feature], sv, zero_eps=cfg.shap_zero_eps, discrete=discrete)
            zero_summary.update({'Class': int(label), 'Class_Name': class_name})
            class_zero_summaries.append(zero_summary)
            all_zero_summaries.append(zero_summary)
            if discrete and 'Level_Counts' in zero_summary:
                class_categorical_zero.append(zero_summary)
            if len(df_xy) < cfg.shap_dependence_min_points:
                ax.axis('off')
                continue
            if discrete:
                levels = np.sort(df_xy['x'].unique())
                positions = np.arange(len(levels), dtype=float)
                rng = np.random.default_rng(cfg.random_seed + int(label) * 100 + panel_idx)
                means: list[float] = []
                q1s: list[float] = []
                q3s: list[float] = []
                for pos, level in zip(positions, levels):
                    vals = df_xy.loc[np.isclose(df_xy['x'], level), 'y'].to_numpy()
                    jitter = rng.uniform(-0.12, 0.12, size=len(vals))
                    ax.scatter(np.full(len(vals), pos) + jitter, vals, s=14, alpha=0.35, c='lightgreen', edgecolors='none')
                    means.append(float(np.mean(vals)))
                    q1s.append(float(np.quantile(vals, 0.25)))
                    q3s.append(float(np.quantile(vals, 0.75)))
                if cfg.shap_dependence_show_iqr and len(levels) >= 2:
                    ax.fill_between(positions, q1s, q3s, alpha=0.18, linewidth=0)
                ax.plot(positions, means, linewidth=2.2, c='red', marker='o', markersize=4)
                ax.set_xticks(positions)
                ax.set_xticklabels([str(int(round(level))) if np.isclose(level, round(level)) else f'{level:.2f}'.rstrip('0').rstrip('.') for level in levels])
            else:
                ax.scatter(df_xy['x'].to_numpy(), df_xy['y'].to_numpy(), s=14, alpha=0.35, c='lightgreen', edgecolors='none')
                df_sorted = df_xy.sort_values('x').reset_index(drop=True)
                window = _safe_rolling_window(cfg.shap_dependence_smooth_window, len(df_sorted))
                min_periods = max(5, window // 3)
                x_smooth = df_sorted['x'].rolling(window=window, center=True, min_periods=min_periods).mean()
                y_smooth = df_sorted['y'].rolling(window=window, center=True, min_periods=min_periods).mean()
                q1 = df_sorted['y'].rolling(window=window, center=True, min_periods=min_periods).quantile(0.25)
                q3 = df_sorted['y'].rolling(window=window, center=True, min_periods=min_periods).quantile(0.75)
                line_mask = x_smooth.notna() & y_smooth.notna()
                if cfg.shap_dependence_show_iqr:
                    band_mask = x_smooth.notna() & q1.notna() & q3.notna()
                    if int(band_mask.sum()) >= 2:
                        ax.fill_between(x_smooth.to_numpy()[band_mask.to_numpy()], q1.to_numpy()[band_mask.to_numpy()], q3.to_numpy()[band_mask.to_numpy()], alpha=0.18, linewidth=0)
                if int(line_mask.sum()) >= 2:
                    xs = x_smooth.to_numpy()[line_mask.to_numpy()]
                    ys = y_smooth.to_numpy()[line_mask.to_numpy()]
                    ax.plot(xs, ys, linewidth=2.2, c='red')
                    crossings = _find_all_shap_zero_crossings(df_sorted['x'].to_numpy(), xs, ys)
                    for order_idx, (x_theory, x_real) in enumerate(crossings, start=1):
                        ax.axvline(x_theory, linestyle=':', linewidth=1.2, alpha=0.8)
                        record = {'Class': int(label), 'Class_Name': class_name, 'Feature': feature, 'Order': int(order_idx), 'X_Theory': float(x_theory), 'X_Real': float(x_real)}
                        class_crossings.append(record)
                        all_crossings.append(record)
            ax.axhline(0, linestyle='--', linewidth=1, alpha=0.5)
            ax.set_xlabel(feature)
            ax.set_ylabel(f'SHAP value ({class_name})')
        for extra_idx in range(len(features), len(axes)):
            axes[extra_idx].axis('off')
        fig.suptitle(f'SHAP dependence plots (Top {len(features)}): {class_name}', y=1.01)
        fig.tight_layout()
        prefix = outdir / f'Figure_SHAP_dependence_{slug(class_name)}'
        save_figure(fig, prefix, cfg.figure_dpi)
        if class_crossings:
            pd.DataFrame(class_crossings).to_csv(outdir / f'SHAP_dependence_{slug(class_name)}_shap0_crossings.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(class_zero_summaries).drop(columns=['Level_Counts'], errors='ignore').to_csv(outdir / f'SHAP_dependence_{slug(class_name)}_shap0_near_zero_summary.csv', index=False, encoding='utf-8-sig')
        if class_categorical_zero:
            save_json(class_categorical_zero, outdir / f'SHAP_dependence_{slug(class_name)}_categorical_shap0.json')
    if all_crossings:
        pd.DataFrame(all_crossings).to_csv(outdir / 'SHAP_dependence_shap0_crossings_all_classes.csv', index=False, encoding='utf-8-sig')
    if all_zero_summaries:
        flat_rows: list[dict[str, Any]] = []
        for record in all_zero_summaries:
            flat = {k: v for k, v in record.items() if k != 'Level_Counts'}
            if 'Level_Counts' in record:
                flat['Level_Counts_JSON'] = json.dumps(record['Level_Counts'], ensure_ascii=False)
            flat_rows.append(flat)
        pd.DataFrame(flat_rows).to_csv(outdir / 'SHAP_dependence_shap0_near_zero_summary_all_classes.csv', index=False, encoding='utf-8-sig')


def run_analysis(cfg: AnalysisConfig) -> None:
    cfg.finalise()
    np.random.seed(cfg.random_seed)
    outdir = ensure_dir(Path(cfg.outdir))
    search_dir = ensure_dir(outdir / 'search_results')
    save_json(asdict(cfg), outdir / 'analysis_config.json')
    save_json(package_versions(), outdir / 'software_versions.json')
    print('[1/8] Loading, validating, and locking the 34 predictors...')
    data_path = Path(cfg.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f'Data file not found: {data_path.resolve()}')
    data = read_dataset(data_path)
    X_raw_original, y_original = validate_and_prepare_data(data, cfg, outdir)
    print('[2/8] Performing the original-data 70/30 stratified train/test split...')
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw_original, y_original, test_size=cfg.test_size, stratify=y_original, random_state=cfg.random_seed)
    split_labels = pd.DataFrame({'Split': ['train'] * len(y_train) + ['test'] * len(y_test), 'Class': list(y_train.values) + list(y_test.values)})
    split_labels['Class_Name'] = split_labels['Class'].map(cfg.class_names)
    split_distribution = split_labels.groupby(['Split', 'Class', 'Class_Name'], as_index=False).size().rename(columns={'size': 'n'})
    split_distribution['proportion_within_split'] = split_distribution.groupby('Split')['n'].transform(lambda s: s / s.sum())
    split_distribution.to_csv(outdir / 'class_distribution_train_test.csv', index=False, encoding='utf-8-sig')
    safe_k = safe_smote_k(y_train, cfg.search_cv_splits, cfg.smote_k_neighbors)
    print('[3/8] Defining cross-validation pipelines and hyperparameter search spaces...')
    definitions = model_definitions(cfg.random_seed)
    search_manifest = search_space_manifest(definitions)
    search_manifest['_preprocessing_configuration'] = {'pipeline_order': ['ColumnTransformer: passthrough Gen/OC + StandardScaler on 32 numeric predictors', 'SMOTENC on training folds only', 'classifier'], 'standardization_scope': 'fit independently inside each CV training fold; refit on full training split', 'oversampling_scope': 'training folds only; validation and test observations are never oversampled', 'categorical_predictors': cfg.categorical_columns, 'numeric_predictors': cfg.numeric_columns, 'smotenc_k_neighbors': safe_k, 'sampling_strategy': cfg.sampling_strategy}
    search_manifest['_search_configuration'] = {'algorithm': 'RandomizedSearchCV', 'n_iter_per_algorithm': cfg.search_n_iter, 'cv': f'StratifiedKFold(n_splits={cfg.search_cv_splits}, shuffle=True, random_state={cfg.random_seed})', 'selection_metric': cfg.selection_metric, 'refit_metric': cfg.selection_metric}
    save_json(search_manifest, outdir / 'hyperparameter_search_spaces.json')
    cv = StratifiedKFold(n_splits=cfg.search_cv_splits, shuffle=True, random_state=cfg.random_seed)
    scorers = scoring_dictionary()
    print('[4/8] Fold-wise scaling, SMOTENC, hyperparameter tuning, and model selection on training data...')
    searches: dict[str, RandomizedSearchCV] = {}
    cv_rows: list[dict[str, Any]] = []
    for model_number, (name, (estimator, search_space)) in enumerate(definitions.items(), start=1):
        print(f'      [{model_number}/{len(definitions)}] {name}')
        pipeline = build_pipeline(estimator, cfg, safe_k)
        search = RandomizedSearchCV(estimator=pipeline, param_distributions=search_space, n_iter=cfg.search_n_iter, scoring=scorers, refit=cfg.selection_metric, cv=cv, random_state=cfg.random_seed, n_jobs=cfg.n_jobs, verbose=0, return_train_score=False, error_score='raise')
        search.fit(X_train_raw, y_train)
        searches[name] = search
        results = pd.DataFrame(search.cv_results_)
        results.to_csv(search_dir / f'randomized_search_{slug(name)}.csv', index=False, encoding='utf-8-sig')
        idx = int(search.best_index_)
        cv_rows.append({'Model': name, 'Best_CV_Rank': int(results.loc[idx, f'rank_test_{cfg.selection_metric}']), 'CV_F1_Macro_Mean': float(results.loc[idx, 'mean_test_f1_macro']), 'CV_F1_Macro_SD': float(results.loc[idx, 'std_test_f1_macro']), 'CV_Balanced_Accuracy_Mean': float(results.loc[idx, 'mean_test_balanced_accuracy']), 'CV_Balanced_Accuracy_SD': float(results.loc[idx, 'std_test_balanced_accuracy']), 'CV_Precision_Macro_Mean': float(results.loc[idx, 'mean_test_precision_macro']), 'CV_Recall_Macro_Mean': float(results.loc[idx, 'mean_test_recall_macro']), 'CV_ROC_AUC_OvR_Macro_Mean': float(results.loc[idx, 'mean_test_roc_auc_ovr_macro']), 'CV_ROC_AUC_OvR_Macro_SD': float(results.loc[idx, 'std_test_roc_auc_ovr_macro']), 'Best_Params': json.dumps(search.best_params_, ensure_ascii=False, default=str)})
        save_json(search.best_params_, search_dir / f'best_params_{slug(name)}.json')
    cv_table = pd.DataFrame(cv_rows).sort_values(['CV_ROC_AUC_OvR_Macro_Mean', 'CV_F1_Macro_Mean'], ascending=False).reset_index(drop=True)
    cv_table.to_csv(outdir / 'model_selection_training_CV.csv', index=False, encoding='utf-8-sig')
    plot_cv_model_comparison(cv_table, outdir, cfg.figure_dpi)
    selected_name = str(cv_table.loc[0, 'Model'])
    selected_pipeline = searches[selected_name].best_estimator_
    print(f'      Selected by training CV: {selected_name}')
    save_json({'selected_model': selected_name, 'selection_metric': cfg.selection_metric, 'selection_rule': 'highest training-CV macro ROC-AUC (OvR); macro-F1 used as table-order tie-break', 'best_params': searches[selected_name].best_params_, 'test_used_for_model_selection': False}, outdir / 'selected_model_manifest.json')
    save_selected_standardization_parameters(selected_pipeline, cfg, outdir)
    summarize_final_training_resample(selected_pipeline, X_train_raw, y_train, cfg, safe_k, outdir)
    print('[5/8] Evaluating tuned pipelines on the untouched original test set...')
    test_rows: list[dict[str, Any]] = []
    per_model_outputs: dict[str, tuple[np.ndarray, np.ndarray, dict[str, float], pd.DataFrame, np.ndarray]] = {}
    labels = np.asarray(sorted(cfg.class_names), dtype=int)
    for name, search in searches.items():
        estimator = search.best_estimator_
        pred = estimator.predict(X_test_raw)
        prob = estimator.predict_proba(X_test_raw)
        overall, per_class, cm = evaluate_predictions(y_test, pred, prob, labels, cfg.class_names)
        test_rows.append({'Model': name, **overall, 'Selected_By_CV': name == selected_name})
        per_model_outputs[name] = (pred, prob, overall, per_class, cm)
        per_class.to_csv(outdir / f'test_class_specific_metrics_{slug(name)}.csv', index=False, encoding='utf-8-sig')
        report = classification_report(y_test, pred, labels=labels, target_names=[cfg.class_names[int(c)] for c in labels], output_dict=True, zero_division=0)
        save_json(report, outdir / f'classification_report_{slug(name)}.json')
    test_table = pd.DataFrame(test_rows).sort_values('F1_Macro', ascending=False)
    test_table.to_csv(outdir / 'all_tuned_models_test_metrics.csv', index=False, encoding='utf-8-sig')
    plot_test_model_comparison(test_table, outdir, cfg.figure_dpi)
    selected_pred, selected_prob, selected_overall, selected_per_class, selected_cm = per_model_outputs[selected_name]
    pd.DataFrame([{'Model': selected_name, **selected_overall}]).to_csv(outdir / 'selected_model_test_overall_metrics.csv', index=False, encoding='utf-8-sig')
    selected_per_class.to_csv(outdir / 'selected_model_test_class_specific_metrics.csv', index=False, encoding='utf-8-sig')
    print('[6/8] Bootstrap CIs and publication-quality diagnostic figures...')
    bootstrap_ci = bootstrap_test_intervals(np.asarray(y_test), selected_pred, selected_prob, labels, cfg.class_names, iterations=cfg.bootstrap_iterations, alpha=cfg.bootstrap_alpha, seed=cfg.random_seed + 1001)
    bootstrap_ci.to_csv(outdir / 'selected_model_test_bootstrap_95CI.csv', index=False, encoding='utf-8-sig')
    plot_confusion(selected_cm, labels, cfg.class_names, outdir, cfg.figure_dpi)
    plot_roc_pr_calibration(np.asarray(y_test), selected_prob, labels, cfg.class_names, outdir, cfg.figure_dpi, cfg.calibration_bins)
    print('[7/8] SHAP analysis using original test observations and raw-scale feature values...')
    X_shap_raw, shap_3d = compute_shap_values(selected_pipeline, X_train_raw, X_test_raw, cfg)
    report_shap_plot_types(X_shap_raw, cfg, outdir)
    selected_classes = np.asarray(selected_pipeline.named_steps['clf'].classes_, dtype=int)
    class_top_features, _ = plot_shap_outputs(X_shap_raw, shap_3d, selected_classes, cfg, outdir)
    plot_shap_dependence_grids(X_shap_raw, shap_3d, selected_classes, class_top_features, cfg, outdir)
    print('[8/8] Analysis complete.')
    print(f'      Results: {outdir.resolve()}')
    print(f'      Selected model: {selected_name}')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Multiclass trajectory prediction with a 70/30 stratified split, cross-validated model selection, SMOTENC, held-out evaluation, and SHAP interpretation.')
    parser.add_argument('--data', default='data.sav', help='Input .sav/.csv/.xlsx/.parquet file')
    parser.add_argument('--outdir', default='chb_trajectory_prediction_results', help='Output directory')
    parser.add_argument('--target', default='Classification', help='Target column')
    parser.add_argument('--quick', action='store_true', help='Quick smoke-test settings')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    cfg = AnalysisConfig(data_path=args.data, outdir=args.outdir, target_column=args.target, quick_mode=args.quick)
    run_analysis(cfg)
if __name__ == '__main__':
    main()
