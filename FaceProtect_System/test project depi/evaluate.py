import sys
import json
import os
import numpy as np
from utils.config_loader import load_config
from utils.logger import get_logger
from embeddings.vector_cache import VectorCache
from evaluation.metrics import generate_pairs, compute_distances
from evaluation.threshold_optimizer import ThresholdOptimizer
from evaluation.roc_plot import plot_roc_curve
from evaluation.confusion_matrix_plot import plot_confusion_matrix

def main():
    config = load_config("config.yaml")
    logger = get_logger("Evaluate", log_level=config['logging']['log_level'], log_dir=config['logging']['log_dir'])
    
    logger.info("--- Starting Evaluation ---")
    
    emb_dir = config['embeddings']['output_dir']
    cache = VectorCache(emb_dir)
    
    if not cache.load():
        logger.error(f"Could not load embeddings from {emb_dir}. Please run extract_embeddings.py first.")
        sys.exit(1)
        
    embeddings, metadata = cache.get_data()
    
    logger.info("Generating evaluation pairs...")
    pos_idx1, pos_idx2, neg_idx1, neg_idx2 = generate_pairs(metadata)
    
    if not pos_idx1 or not neg_idx1:
        logger.error("Not enough data to generate evaluation pairs.")
        sys.exit(1)
        
    logger.info(f"Generated {len(pos_idx1)} positive pairs and {len(neg_idx1)} negative pairs.")
    
    metric = config['evaluation']['distance_metric']
    logger.info(f"Computing distances using {metric} metric...")
    
    emb_pos1 = embeddings[pos_idx1]
    emb_pos2 = embeddings[pos_idx2]
    pos_dists = compute_distances(emb_pos1, emb_pos2, metric=metric)
    if metric == "cosine":
        pos_dists = np.diag(pos_dists) 
    else:
        pos_dists = np.linalg.norm(emb_pos1 - emb_pos2, axis=1) 
        
    emb_neg1 = embeddings[neg_idx1]
    emb_neg2 = embeddings[neg_idx2]
    neg_dists = compute_distances(emb_neg1, emb_neg2, metric=metric)
    if metric == "cosine":
        neg_dists = np.diag(neg_dists)
    else:
        neg_dists = np.linalg.norm(emb_neg1 - emb_neg2, axis=1)
        
    all_dists = np.concatenate([pos_dists, neg_dists])
    all_labels = np.concatenate([np.ones(len(pos_dists)), np.zeros(len(neg_dists))])
    
    logger.info("Optimizing threshold...")
    optimizer = ThresholdOptimizer(all_dists, all_labels)
    best_metrics, fpr, tpr, roc_auc = optimizer.optimize()
    
    plots_dir = config['evaluation']['output_plots_dir']
    plot_roc_curve(fpr, tpr, roc_auc, plots_dir)
    
    optimal_thresh = best_metrics['threshold']
    preds = (all_dists <= optimal_thresh).astype(int)
    plot_confusion_matrix(all_labels, preds, plots_dir)
    
    reports_dir = config['evaluation']['output_reports_dir']
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, "evaluation_report.json")
    
    with open(report_path, "w") as f:
        json.dump(best_metrics, f, indent=4)
        
    logger.info(f"Saved evaluation report to {report_path}")
    logger.info("--- Evaluation Completed ---")

if __name__ == "__main__":
    main()
