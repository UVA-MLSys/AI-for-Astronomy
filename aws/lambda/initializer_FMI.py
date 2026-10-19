import json
import logging
import uuid

import boto3

from initializer import assign_partitions, get_file_list

logger = logging.getLogger()
logger.setLevel(logging.INFO)
s3_client = boto3.client('s3')

REQUIRED_FIELDS = (
    'bucket', 'object_type', 'script', 'S3_object_name', 'result_path', 'file_limit',
    'batch_size', 'data_bucket', 'data_prefix',
    'fmi_channel_type', 'fmi_options', 'fmi_max_timeout', 'rendezvous_host', 'rendezvous_port',
)


class InvalidFMIEvent(ValueError):
    pass


def validate_event(event):
    missing = [field for field in REQUIRED_FIELDS if event.get(field) in (None, '')]
    if missing:
        raise InvalidFMIEvent(f'Arm B execution input is missing {missing} (result_path={event.get("result_path")})')


def lambda_handler(event, context):
    validate_event(event)

    bucket = event['bucket']
    result_path = event['result_path']
    file_limit = int(event['file_limit'])
    world_size = int(event.get('world_size', file_limit))
    event['world_size'] = world_size
    comm_name = event.get('comm_name') or f'cosmic-fmi-{uuid.uuid4().hex}'
    event['comm_name'] = comm_name

    filenames = get_file_list(bucket=event['data_bucket'], prefix=event['data_prefix'])
    if len(filenames) == 0:
        return {
            'statusCode': 404,
            'body': 'No files found in the specified S3 bucket or prefix.'
        }
    if file_limit > len(filenames):
        file_limit = len(filenames)
        event['file_limit'] = file_limit
        logger.info(f'File limit is larger than the number of files. Set to {file_limit}.')
    filenames = filenames[:file_limit]

    run_id = str(uuid.uuid4())
    payload_key = f'temp-results/{run_id}-payload.json'
    event['data_map'] = assign_partitions(filenames, world_size)
    s3_client.put_object(
        Bucket=bucket, Key=payload_key,
        Body=json.dumps(event, indent=4),
        ContentType='application/json'
    )

    items = [
        {
            'S3_BUCKET': bucket,
            'S3_OBJECT_NAME': event['S3_object_name'],
            'SCRIPT': event['script'],
            'S3_OBJECT_TYPE': event['object_type'],
            'WORLD_SIZE': str(world_size),
            'RANK': str(rank),
            'DATA_BUCKET': event['data_bucket'],
            'DATA_PREFIX': event['data_prefix'],
            'RESULT_PATH': result_path,
            'BATCH_SIZE': int(event['batch_size']),
            'PAYLOAD_KEY': payload_key,
            'FMI_COMM_NAME': comm_name,
            'FMI_CHANNEL_TYPE': event['fmi_channel_type'],
            'FMI_OPTIONS': event['fmi_options'],
            'FMI_MAX_TIMEOUT': str(event['fmi_max_timeout']),
            'RENDEZVOUS_HOST': event['rendezvous_host'],
            'RENDEZVOUS_PORT': str(event['rendezvous_port']),
        }
        for rank in range(world_size)
    ]
    items_key = f'temp-results/{run_id}.json'
    s3_client.put_object(
        Bucket=bucket, Key=items_key,
        Body=json.dumps(items, indent=4),
        ContentType='application/json'
    )

    return {
        'statusCode': 200,
        'body': {
            'S3_BUCKET': bucket,
            'S3_KEY': items_key
        }
    }
