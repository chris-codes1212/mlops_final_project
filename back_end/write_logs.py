import boto3
from datetime import datetime
from decimal import Decimal

# Connect to DynamoDB table
dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
table = dynamodb.Table("toxicity_app")


# Create a function to handle writing JSON logs to DynamoDB
def write_log(input_data, pred_labels, pred_proba_dict, latency, labels=None):
    # Convert pred_proba to decimal values
    decimal_pred_proba = {k: Decimal(str(v)) for k, v in pred_proba_dict.items()}

    # Send log to DynamoDB
    table.put_item(
        Item={
            "timestamp": datetime.now().astimezone().isoformat(),
            "comment": input_data.comment,
            "prediction_labels": pred_labels,
            "predicted_proba": decimal_pred_proba,
            "latency_seconds": Decimal(str(latency)),
        }
    )

    return
