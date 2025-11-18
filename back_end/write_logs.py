import boto3
from datetime import datetime
from decimal import Decimal

# connect to DynamoDB table
dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
table = dynamodb.Table("toxicity_app")
    
# create a funciton to handle writing json logs to DyanmoDB
def write_log(input_data, pred_labels, pred_proba_dict, labels=None):
    # convert pred_proba to decimal values
    decimal_pred_proba = {k: Decimal(str(v)) for k, v in pred_proba_dict.items()}
    # send log to DynamoDB
    table.put_item(Item={"timestamp": datetime.now().astimezone().isoformat(), 
                         "comment": input_data.comment,
                         "prediction_labels": pred_labels,
                         "predicted_proba": decimal_pred_proba
                         })
    
    return