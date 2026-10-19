import json
import time

import boto3


def lambda_handler(event, context):
    t_summarize_start = time.time()
    s3_client = boto3.client('s3')

    items_bucket = event['body']['S3_BUCKET']
    items_key = event['body']['S3_KEY']
    obj = s3_client.get_object(Bucket=items_bucket, Key=items_key)
    items = json.loads(obj['Body'].read().decode('utf-8'))

    data_bucket = items[0]['DATA_BUCKET']
    prefix = items[0]['RESULT_PATH'].rstrip('/')
    metrics_key = f'{prefix}/aggregate_metrics.json'
    try:
        obj = s3_client.get_object(Bucket=data_bucket, Key=metrics_key)
    except s3_client.exceptions.NoSuchKey:
        return {
            'statusCode': 404,
            'body': json.dumps(f'{metrics_key} not found; rank 0 did not complete the gather')
        }
    aggregate_metrics = json.loads(obj['Body'].read().decode('utf-8'))

    world_size = int(items[0]['WORLD_SIZE'])
    if aggregate_metrics['ranks_aggregated'] != world_size:
        return {
            'statusCode': 500,
            'body': json.dumps(
                f'ranks_aggregated={aggregate_metrics["ranks_aggregated"]} != world_size={world_size} '
                f'(result_path={prefix})'
            )
        }

    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': f'Combined data at {prefix}/combined_data.json',
            **aggregate_metrics,
            'summarize_s': time.time() - t_summarize_start,
        })
    }
