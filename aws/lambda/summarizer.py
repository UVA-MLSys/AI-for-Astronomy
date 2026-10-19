import json
import time
import boto3, logging


def lambda_handler(event, context):
    bucket_name = "cosmicai-data-cylon"  # replace with your bucket name
    # prefix = "results" # replace with your folder path within the bucket if needed

    t_aggregate_start = time.time()

    s3_client = boto3.client('s3')

    # Get the payload data from S3
    payload_bucket = event['body']['S3_BUCKET']
    payload_key = event['body']['S3_KEY']

    # Read the payload data to get RESULT_PATH
    obj = s3_client.get_object(Bucket=payload_bucket, Key=payload_key)
    payload_data = json.loads(obj["Body"].read().decode("utf-8"))

    prefix = payload_data[0]['RESULT_PATH'].rstrip('/')

    logging.info(f'Combined result will be saved in {prefix}')

    # List all JSON files in the specified bucket and prefix, across all pages
    paginator = s3_client.get_paginator("list_objects_v2")
    contents = []
    for page in paginator.paginate(Bucket=bucket_name, Prefix=f"{prefix}/"):
        contents.extend(page.get("Contents", []))

    # Check if files exist in the specified location
    if not contents:
        return {
            "statusCode": 404,
            "body": json.dumps("Output/No files found in the specified S3 bucket or prefix.")
        }

    output_file_key = f"{prefix}/combined_data.json"  # Path for the output file
    metrics_file_key = f"{prefix}/aggregate_metrics.json"
    all_data = []
    ranks_aggregated = 0
    bytes_aggregated = 0
    # Loop over each object across all pages
    for item in contents:
        file_key = item["Key"]

        # Only process JSON files
        if file_key.endswith(".json") and file_key not in (output_file_key, metrics_file_key):
            # Retrieve the file content
            obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
            body_bytes = obj["Body"].read()
            bytes_aggregated += len(body_bytes)
            file_content = body_bytes.decode("utf-8")

            # Load JSON data and extend all_data list
            try:
                data = json.loads(file_content)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    all_data.append(data)
                ranks_aggregated += 1
            except json.JSONDecodeError:
                return {
                    "statusCode": 500,
                    "body": json.dumps(f"Error decoding JSON in file {file_key}")
                }

    # Convert concatenated data to JSON string
    concatenated_json = json.dumps(all_data)

    # Upload the combined JSON data back to the same folder in S3
    s3_client.put_object(
        Bucket=bucket_name,
        Key=output_file_key,
        Body=concatenated_json,
        ContentType="application/json"
    )

    aggregate_metrics = {
        "aggregate_s": time.time() - t_aggregate_start,
        "ranks_aggregated": ranks_aggregated,
        "bytes_aggregated": bytes_aggregated,
    }
    s3_client.put_object(
        Bucket=bucket_name,
        Key=metrics_file_key,
        Body=json.dumps(aggregate_metrics),
        ContentType="application/json",
    )

    # Return the S3 path of the concatenated JSON file
    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": f"Combined data uploaded to {output_file_key}",
            **aggregate_metrics,
        })
    }
