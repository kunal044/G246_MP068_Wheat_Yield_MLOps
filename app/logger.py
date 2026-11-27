import json
import logging
import sys


# Convert prediction records into JSON format
class PredictionFormatter(logging.Formatter):

    def format(self, record):
        return json.dumps(
            record.prediction_fields,
            allow_nan=False
        )


# Configure the prediction logger
def get_prediction_logger():

    logger = logging.getLogger("mp068.prediction")
    logger.setLevel(logging.INFO)

    # Prevent duplicate logs
    logger.propagate = False

    if not logger.handlers:

        # Write logs to stdout
        handler = logging.StreamHandler(sys.stdout)

        # Apply the JSON formatter
        handler.setFormatter(PredictionFormatter())

        logger.addHandler(handler)

    return logger