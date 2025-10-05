#!/usr/bin/env python3
"""
📊 K2 VISUALIZATION UTILITIES
=============================
Utilidades para visualización y reportes del sistema K2.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from pathlib import Path

from config.k2_config import EvalConfig, LogConfig

# Configurar logging
logging.basicConfig(level=getattr(logging, LogConfig.LOG_LEVEL))
logger = logging.getLogger(__name__)

# Configurar estilo
plt.style.use('default')
sns.set_palette("husl")

class K2Visualizer:
    """Generador de visualizaciones para el sistema K2"""

    def __init__(self, figsize=(12, 8), dpi=300):
        self.figsize = figsize
        self.dpi = dpi
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd'
        }

    def plot_confusion_matrices(self, y_true_list, y_pred_list, model_names, save_path=None):
        """Genera matrices de confusión comparativas"""
        n_models = len(model_names)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))

        if n_models == 1:
            axes = [axes]

        for i, (y_true, y_pred, name) in enumerate(zip(y_true_list, y_pred_list, model_names)):
            cm = confusion_matrix(y_true, y_pred)

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       ax=axes[i], cbar=False)
            axes[i].set_title(f'{name}\nConfusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"💾 Matrices de confusión guardadas: {save_path}")

        plt.show()
        return fig

    def plot_roc_curves(self, y_true_list, y_prob_list, model_names, save_path=None):
        """Genera curvas ROC comparativas"""
        plt.figure(figsize=self.figsize)

        for y_true, y_prob, name in zip(y_true_list, y_prob_list, model_names):
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_score = auc(fpr, tpr)

            plt.plot(fpr, tpr, linewidth=2,
                    label=f'{name} (AUC = {auc_score:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"💾 Curvas ROC guardadas: {save_path}")

        plt.show()
        return plt.gcf()

    def plot_precision_recall_curves(self, y_true_list, y_prob_list, model_names, save_path=None):
        """Genera curvas Precision-Recall"""
        plt.figure(figsize=self.figsize)

        for y_true, y_prob, name in zip(y_true_list, y_prob_list, model_names):
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            auc_pr = auc(recall, precision)

            plt.plot(recall, precision, linewidth=2,
                    label=f'{name} (AUC-PR = {auc_pr:.3f})')

        # Línea base (proporción de positivos)
        baseline = sum(y_true_list[0]) / len(y_true_list[0])
        plt.axhline(y=baseline, color='k', linestyle='--', alpha=0.5,
                   label=f'Baseline ({baseline:.3f})')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"💾 Curvas P-R guardadas: {save_path}")

        plt.show()
        return plt.gcf()

    def plot_model_comparison(self, results_dict, save_path=None):
        """Gráfico de barras comparativo de métricas"""
        models = list(results_dict.keys())
        metrics = ['precision', 'recall', 'f1_score']

        fig, ax = plt.subplots(figsize=self.figsize)

        x = np.arange(len(models))
        width = 0.25

        for i, metric in enumerate(metrics):
            values = [results_dict[model].get(metric, 0) for model in models]
            ax.bar(x + i*width, values, width, label=metric.title(), alpha=0.8)

        # Líneas objetivo
        ax.axhline(y=EvalConfig.TARGET_PRECISION, color='red', linestyle='--',
                  alpha=0.7, label=f'Target Precision ({EvalConfig.TARGET_PRECISION})')
        ax.axhline(y=EvalConfig.TARGET_RECALL, color='green', linestyle='--',
                  alpha=0.7, label=f'Target Recall ({EvalConfig.TARGET_RECALL})')
        ax.axhline(y=EvalConfig.TARGET_F1, color='blue', linestyle='--',
                  alpha=0.7, label=f'Target F1 ({EvalConfig.TARGET_F1})')

        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"💾 Comparación guardada: {save_path}")

        plt.show()
        return fig

    def plot_feature_importance(self, importance_dict, top_n=15, save_path=None):
        """Gráfico de importancia de características"""
        if isinstance(importance_dict, dict):
            features = list(importance_dict.keys())
            importance = list(importance_dict.values())
        else:
            features = [f'Feature_{i}' for i in range(len(importance_dict))]
            importance = importance_dict

        # Ordenar por importancia
        sorted_idx = np.argsort(importance)[-top_n:]
        sorted_features = [features[i] for i in sorted_idx]
        sorted_importance = [importance[i] for i in sorted_idx]

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(sorted_features)), sorted_importance, alpha=0.8)
        plt.yticks(range(len(sorted_features)), sorted_features)
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importance')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"💾 Importancia guardada: {save_path}")

        plt.show()
        return plt.gcf()

    def create_ensemble_dashboard(self, ensemble_results, save_path=None):
        """Dashboard interactivo del sistema ensemble"""

        # Crear subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Distribution', 'Performance Metrics',
                          'Confusion Matrix', 'Confidence Distribution'),
            specs=[[{'type': 'pie'}, {'type': 'bar'}],
                   [{'type': 'heatmap'}, {'type': 'histogram'}]]
        )

        # 1. Distribución de modelos
        model_dist = ensemble_results.get('model_distribution', {})
        if model_dist:
            fig.add_trace(
                go.Pie(labels=['RandomForest', 'TensorFlow'],
                      values=[model_dist.get('rf_cases', 0), model_dist.get('tf_cases', 0)],
                      name="Model Usage"),
                row=1, col=1
            )

        # 2. Métricas de rendimiento
        metrics = ensemble_results.get('metrics', {})
        if metrics:
            fig.add_trace(
                go.Bar(x=['Precision', 'Recall', 'F1-Score'],
                      y=[metrics.get('precision', 0), metrics.get('recall', 0), metrics.get('f1_score', 0)],
                      name="Performance"),
                row=1, col=2
            )

        # 3. Matriz de confusión
        cm = ensemble_results.get('confusion_matrix', [[0,0],[0,0]])
        fig.add_trace(
            go.Heatmap(z=cm, colorscale='Blues', showscale=False,
                      text=cm, texttemplate="%{text}", textfont={"size":16}),
            row=2, col=1
        )

        # 4. Distribución de confianza
        confidences = ensemble_results.get('confidences', [])
        if len(confidences) > 0:
            fig.add_trace(
                go.Histogram(x=confidences, name="Confidence Distribution"),
                row=2, col=2
            )

        fig.update_layout(
            title="K2 Ensemble System Dashboard",
            showlegend=False,
            height=800
        )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"💾 Dashboard guardado: {save_path}")

        fig.show()
        return fig

class K2Reporter:
    """Generador de reportes del sistema K2"""

    def __init__(self):
        self.visualizer = K2Visualizer()

    def generate_model_report(self, model_name, y_true, y_pred, y_prob=None,
                            feature_importance=None, save_dir=None):
        """Genera reporte completo de un modelo"""
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        # Calcular métricas
        metrics = self._calculate_metrics(y_true, y_pred, y_prob)

        # Generar visualizaciones
        figs = {}

        # Matriz de confusión
        cm_path = save_dir / f"{model_name}_confusion_matrix.png" if save_dir else None
        figs['confusion_matrix'] = self.visualizer.plot_confusion_matrices(
            [y_true], [y_pred], [model_name], cm_path
        )

        # Importancia de características
        if feature_importance is not None:
            fi_path = save_dir / f"{model_name}_feature_importance.png" if save_dir else None
            figs['feature_importance'] = self.visualizer.plot_feature_importance(
                feature_importance, save_path=fi_path
            )

        # Curvas ROC y P-R si hay probabilidades
        if y_prob is not None:
            roc_path = save_dir / f"{model_name}_roc_curve.png" if save_dir else None
            figs['roc_curve'] = self.visualizer.plot_roc_curves(
                [y_true], [y_prob], [model_name], roc_path
            )

            pr_path = save_dir / f"{model_name}_pr_curve.png" if save_dir else None
            figs['pr_curve'] = self.visualizer.plot_precision_recall_curves(
                [y_true], [y_prob], [model_name], pr_path
            )

        # Generar reporte HTML
        html_report = self._generate_html_report(model_name, metrics, figs)

        if save_dir:
            html_path = save_dir / f"{model_name}_report.html"
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_report)
            logger.info(f"📄 Reporte HTML guardado: {html_path}")

        return {
            'metrics': metrics,
            'figures': figs,
            'html_report': html_report
        }

    def _calculate_metrics(self, y_true, y_pred, y_prob=None):
        """Calcula métricas comprehensivas"""
        metrics = {
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred)
        }

        if y_prob is not None:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob)

        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics.update({
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'accuracy': (tp + tn) / (tp + tn + fp + fn)
        })

        return metrics

    def _generate_html_report(self, model_name, metrics, figures):
        """Genera reporte HTML"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>K2 Model Report - {model_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px;
                          border: 1px solid #ddd; border-radius: 5px; }}
                .good {{ background-color: #d4edda; }}
                .warning {{ background-color: #fff3cd; }}
                .bad {{ background-color: #f8d7da; }}
            </style>
        </head>
        <body>
            <h1>K2 Model Report: {model_name}</h1>
            <h2>Performance Metrics</h2>
        """

        # Añadir métricas con colores
        for metric, value in metrics.items():
            if metric in ['precision', 'recall', 'f1_score']:
                if value >= 0.8:
                    css_class = "good"
                elif value >= 0.6:
                    css_class = "warning"
                else:
                    css_class = "bad"

                html += f'<div class="metric {css_class}"><strong>{metric.title()}:</strong> {value:.3f}</div>'

        html += """
            <h2>Detailed Metrics</h2>
            <table border="1" style="border-collapse: collapse;">
                <tr><th>Metric</th><th>Value</th></tr>
        """

        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                formatted_value = f"{value:.3f}" if isinstance(value, float) else str(value)
                html += f"<tr><td>{metric.replace('_', ' ').title()}</td><td>{formatted_value}</td></tr>"

        html += """
            </table>
            </body>
            </html>
        """

        return html

if __name__ == "__main__":
    print("📊 K2 Visualization Utilities")
    print("Herramientas para visualización y reportes")

    # Ejemplo de uso con datos sintéticos
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.3, 1000)
    y_pred = np.random.binomial(1, 0.3, 1000)
    y_prob = np.random.random(1000)

    visualizer = K2Visualizer()
    print("Generando visualizaciones de ejemplo...")