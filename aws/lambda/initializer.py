import json, boto3
import os, logging
import uuid
import time

logger = logging.getLogger()
logger.setLevel(logging.INFO)
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')


def get_ith_filename(i):
    return f'{i + 1}.pt'


def get_file_list(bucket, prefix):
    paginator = s3_client.get_paginator("list_objects_v2")
    contents = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        contents.extend(page.get("Contents", []))

    if not contents:
        logger.error("Inference: No files found in the specified S3 bucket or prefix.")
        return []

    filenames = []
    # Loop over each object across all pages
    for item in contents:
        file_key = item["Key"]

        # Only process pt files
        if file_key.endswith(".pt"):
            filenames.append(file_key)

    return filenames


def ceil(a, b):
    return (a + b - 1) // b


def assign_partitions(filenames, world_size):
    data_map = {}
    start_index = 0
    total_files = len(filenames)

    # some ranks may get multiple files
    if total_files >= world_size:
        for rank in range(0, int(world_size)):
            # num files left / num of worlds left
            step_size = ceil(total_files - start_index, world_size - rank)

            if step_size == 1:
                data_path = filenames[start_index]
            else:
                data_path = filenames[start_index:start_index + step_size]
            start_index += step_size

            data_map[rank] = data_path
    else:
        for rank in range(0, int(world_size)):
            data_map[rank] = filenames[rank % total_files]

    return data_map


def lambda_handler(event, context):
    bucket = event["bucket"]
    object_type = event["object_type"]

    script = event["script"]
    S3_object_name = event["S3_object_name"]
    result_path = event['result_path']
    file_limit = int(event['file_limit'])
    batch_size = int(event['batch_size'])

    # if you want one task to handle multiple files
    if "world_size" in event:
        world_size = int(event["world_size"])
    else:
        world_size = file_limit
        event['world_size'] = world_size

    # partitioned data are physically located here
    data_bucket = event['data_bucket']  # 'cosmicai-data'
    data_bucket_prefix = event['data_prefix']  # ''
    filenames = get_file_list(bucket=data_bucket, prefix=data_bucket_prefix)

    if len(filenames) == 0:
        return {
            'statusCode': 404,
            'body': f'No files found in the specified S3 bucket or prefix.'
        }

    if file_limit > len(filenames):
        file_limit = len(filenames)
        event['file_limit'] = file_limit
        logger.info(f'File limit is larger than the number of files. Set to {file_limit}.')

    filenames = filenames[:file_limit]
    # logging.info(f'Files {filenames}')

    data_map = assign_partitions(filenames, world_size)
    result = [
        {
            "S3_BUCKET": bucket,
            "S3_OBJECT_NAME": S3_object_name,
            "SCRIPT": script,
            "S3_OBJECT_TYPE": object_type,
            "WORLD_SIZE": str(world_size),
            "RANK": str(rank),
            "DATA_BUCKET": data_bucket,
            "DATA_PREFIX": data_bucket_prefix,
            "DATA_PATH": data_map[rank],
            "RESULT_PATH": result_path,
            "BATCH_SIZE": batch_size
        }
        for rank in range(world_size)
    ]

    event['data_map'] = data_map
    # used by the container to know world settings
    s3_client.put_object(
        Bucket=bucket, Key='payload.json',
        Body=json.dumps(event, indent=4),
        ContentType="application/json"
    )

    # Generate unique key for S3 storage
    body_key = str(uuid.uuid4())
    result_s3_key = f'temp-results/{body_key}.json'

    # Store result in S3
    s3_client.put_object(
        Bucket=bucket,
        Key=result_s3_key,
        Body=json.dumps(result, indent=4),
        ContentType="application/json"
    )

    return {
        'statusCode': 404 if filenames is None else 200,
        'body': {
            'S3_BUCKET': bucket,
            'S3_KEY': result_s3_key
        }
    }