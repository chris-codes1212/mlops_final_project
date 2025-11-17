import boto3
from datetime import datetime

# connect to DynamoDB table
dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
table = dynamodb.Table("toxicity_app")
    
# create a funciton to handle writing json logs to DyanmoDB
def write_log(input_data, pred_labels, pred_proba_dict, labels=None):

    # send log to DynamoDB
    table.put_item(Item={"timestamp": datetime.now().astimezone().isoformat(), 
                         "comment": input_data.comment,
                         "predicted_labels": pred_proba_dict,
                         "prediction_proba": pred_labels
                         })
    
    return
    
