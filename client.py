import argparse
import logging
import os

import numpy as np
import flwr as fl
import tensorflow as tf
from sklearn.metrics import f1_score, roc_auc_score
from helpers.load_data import load_data

from model.model import Model

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Parse command line arguments
parser = argparse.ArgumentParser(description="Flower client")

parser.add_argument(
    "--server_address", type=str, default="server:8080", help="Address of the server"
)
parser.add_argument(
    "--batch_size", type=int, default=32, help="Batch size for training"
)
parser.add_argument(
    "--learning_rate", type=float, default=0.1, help="Learning rate for the optimizer"
)
parser.add_argument("--client_id", type=int, default=1, help="Unique ID for the client")
parser.add_argument(
    "--total_clients", type=int, default=2, help="Total number of clients"
)
parser.add_argument(
    "--data_percentage", type=float, default=0.5, help="Portion of client data to use"
)

args = parser.parse_args()

# Create an instance of the model and pass the learning rate as an argument
model = Model(learning_rate=args.learning_rate)

# Compile the model
model.compile()



class Client(fl.client.NumPyClient):
    def __init__(self, args):
        self.args = args

        logger.info("Preparing data...")
        (x_train, y_train), (x_test, y_test) = load_data(
            data_sampling_percentage=self.args.data_percentage,
            client_id=self.args.client_id,
            total_clients=self.args.total_clients,
        )

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def get_parameters(self, config):
        # Return the parameters of the model
        return model.get_model().get_weights()

    def fit(self, parameters, config):
        # Set the weights of the model
        model.get_model().set_weights(parameters)

        # Train the model
        history = model.get_model().fit(
            self.x_train, self.y_train, batch_size=self.args.batch_size
        )

        #########################################################################
        # y_pred_train = model.get_model().predict(self.x_train, verbose=False)
        # y_pred_train = np.argmax(y_pred_train, axis=1).reshape(-1, 1)
        # mse_train = np.mean((self.y_train - y_pred_train) ** 2)
        #########################################################################


        # Calculate evaluation metric
        results = {
            "accuracy": float(history.history["accuracy"][-1])
        }

        # Get the parameters after training
        parameters_prime = model.get_model().get_weights()

        # Directly return the parameters and the number of examples trained on
        return parameters_prime, len(self.x_train), results

    def evaluate(self, parameters, config):
        # Set the weights of the model
        model.get_model().set_weights(parameters)

        # Evaluate the model and get the loss and accuracy
        loss, accuracy = model.get_model().evaluate(
            self.x_test, self.y_test, batch_size=self.args.batch_size
        )
        
        # Get predicted probabilities for AUC calculation (use softmax for multi-class classification)
        y_pred_prob = model.get_model().predict(self.x_test, verbose=False)
        
        # Calculate AUC (using true labels and predicted probabilities)
        auc = roc_auc_score(self.y_test, y_pred_prob, multi_class="ovr", average="macro")

        # Convert predictions to class labels for F1 score
        y_pred = np.argmax(y_pred_prob, axis=1).reshape(-1, 1)

        # Calculate F1 score
        f1 = f1_score(self.y_test, y_pred, average="micro")
        mse = np.mean((self.y_test - y_pred) ** 2)  # Compute Mean Square Error  ##################################
        metrics = {
        "accuracy": float(accuracy),
        "f1": f1,
        "auc":auc,
        "mse":mse ####################
        }

        # Return the loss, the number of examples evaluated on, and the metrics (accuracy, f1, auc)
        return float(loss), len(self.x_test),metrics



# Function to Start the Client
def start_fl_client():
    try:
        client = Client(args).to_client()
        fl.client.start_client(server_address=args.server_address, client=client)
    except Exception as e:
        logger.error("Error starting FL client: %s", e)
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    # Call the function to start the client
    start_fl_client()