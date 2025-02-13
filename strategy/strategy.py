import logging
from typing import Dict, List, Optional, Tuple, Union
import flwr as fl
from flwr.common import EvaluateRes, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from prometheus_client import Gauge
from sklearn.metrics import roc_auc_score  # For AUC calculation

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module


class FedCustom(fl.server.strategy.FedAvg):
    def __init__(
        self, 
        accuracy_gauge: Gauge = None, 
        loss_gauge: Gauge = None, 
        f1_gauge: Gauge = None,
        auc_gauge: Gauge = None,  # Add AUC gauge
        mse_gauge: Gauge = None,  # Add MSE gauge
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.accuracy_gauge = accuracy_gauge
        self.loss_gauge = loss_gauge
        self.f1_gauge = f1_gauge
        self.auc_gauge = auc_gauge  # Initialize AUC gauge
        self.mse_gauge = mse_gauge  # Initialize MSE gauge

    def __repr__(self) -> str:
        return "FedCustom"

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses, accuracy, F1 score, and AUC using weighted average."""

        if not results:
            return None, {}

        # Calculate weighted average for loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Calculate weighted average for accuracy
        accuracies = [
            evaluate_res.metrics["accuracy"] * evaluate_res.num_examples
            for _, evaluate_res in results
        ]
        examples = [evaluate_res.num_examples for _, evaluate_res in results]
        accuracy_aggregated = (
            sum(accuracies) / sum(examples) if sum(examples) != 0 else 0
        )

        # Calculate weighted average for f1 score
        f1_scores = [
            evaluate_res.metrics["f1"] * evaluate_res.num_examples
            for _, evaluate_res in results
        ]
        f1_scores_aggregated = (
            sum(f1_scores) / sum(examples) if sum(examples) != 0 else 0
        )

        # Calculate weighted average for AUC
        auc_scores = [
            evaluate_res.metrics["auc"] * evaluate_res.num_examples
            for _, evaluate_res in results
        ]


        auc_aggregated = (
            sum(auc_scores) / sum(examples) if sum(examples) != 0 else 0
        )

        # Calculate weighted average for MSE
        mse_values = [
        evaluate_res.metrics["mse"] * evaluate_res.num_examples
        for _, evaluate_res in results
        ]

        mse_aggregated = (
            sum(mse_values) / sum(examples) if sum(examples) != 0 else 0
        )



        # Update the Prometheus gauges with the latest aggregated values
        self.accuracy_gauge.set(accuracy_aggregated)
        self.loss_gauge.set(loss_aggregated)
        self.f1_gauge.set(f1_scores_aggregated)
        self.auc_gauge.set(auc_aggregated)
        self.mse_gauge.set(mse_aggregated)  

        metrics_aggregated = {
            "loss": loss_aggregated, 
            "accuracy": accuracy_aggregated, 
            "f1": f1_scores_aggregated,
            "auc": auc_aggregated,
            "mse": mse_aggregated  
        }

        return loss_aggregated, metrics_aggregated