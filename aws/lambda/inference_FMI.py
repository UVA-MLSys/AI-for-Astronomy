import argparse
import json
import logging
import math
import os
import sys
import time

import pyarrow as pa

from communicator.fmi_bridge import FMIBridge
from inference import (
    STAGE_FIELDS,
    environ_or_required,
    load_partition_and_model,
    run_inference,
    s3_client,
    startup_timings,
)

PAYLOAD_ROOT = 0

GATHERED_FIELDS = tuple(f for f in STAGE_FIELDS if f not in ('publish_s', 'total_s')) + (
    'num_samples',
    'num_batches',
    'total_cpu_time (seconds)',
    'total_cpu_memory (MB)',
    'execution_time (seconds/batch)',
    'throughput_bps',
    'sample_persec',
)

TRAILING_FIELDS = ('publish_s', 'total_s')

PARTITION_MAP_SCHEMA = pa.schema([('rank', pa.int32()), ('partition_index', pa.int64())])

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FMIStageError(RuntimeError):
    def __init__(self, message, rank, world_size, comm_name):
        super().__init__(f'{message} (rank={rank}, world_size={world_size}, comm_name={comm_name})')


def partition_key(data_prefix, index):
    return f'{data_prefix.rstrip("/")}/{index}.pt'


def partition_index(data_prefix, key):
    directory, filename = os.path.split(key)
    stem, extension = os.path.splitext(filename)
    if directory != data_prefix.rstrip('/') or extension != '.pt' or not stem.isdigit():
        raise ValueError(f'partition key {key!r} is not of the form {data_prefix}/<n>.pt')
    return int(stem)


def encode_data_map(data_map, world_size, data_prefix):
    ranks, indices = [], []
    for rank in range(world_size):
        paths = data_map[str(rank)]
        for path in paths if isinstance(paths, list) else [paths]:
            ranks.append(rank)
            indices.append(partition_index(data_prefix, path))
    return pa.table([pa.array(ranks, pa.int32()), pa.array(indices, pa.int64())], schema=PARTITION_MAP_SCHEMA)


def decode_rank_paths(partition_map, rank, data_prefix):
    columns = partition_map.to_pydict()
    paths = [
        partition_key(data_prefix, index)
        for r, index in zip(columns['rank'], columns['partition_index'])
        if r == rank
    ]
    if not paths:
        raise ValueError(f'partition map assigns no partition to rank {rank}')
    return paths[0] if len(paths) == 1 else paths


def broadcast_table(bridge, table, root):
    if not bridge.available:
        return table
    from pycylon import Table
    return bridge.broadcast(Table.from_arrow(bridge.context, table), root).to_arrow()


def gather_table(bridge, table, root):
    if not bridge.available:
        return [table]
    from pycylon import Table
    return [t.to_arrow() for t in bridge.gather(Table.from_arrow(bridge.context, table), root)]


def broadcast_partition_map(bridge, rank, world_size, args):
    partition_map = PARTITION_MAP_SCHEMA.empty_table()
    if rank == PAYLOAD_ROOT:
        obj = s3_client.get_object(Bucket=args.payload_bucket, Key=args.payload_key)
        payload = json.loads(obj['Body'].read().decode('utf-8'))
        partition_map = encode_data_map(payload['data_map'], world_size, args.data_prefix)
    return broadcast_table(bridge, partition_map, PAYLOAD_ROOT)


def to_wire(value):
    return math.nan if value is None else float(value)


def from_wire(value):
    return None if value is None or math.isnan(value) else value


def row_table(rank, fields, record):
    columns = {'rank': pa.array([rank], pa.int32())}
    columns.update({field: pa.array([to_wire(record.get(field))], pa.float64()) for field in fields})
    return pa.table(columns)


def rows_by_rank(tables, fields):
    rows = {}
    for table in tables:
        for row in table.to_pylist():
            rows[row['rank']] = {field: from_wire(row[field]) for field in fields}
    return rows


def timed_barrier(bridge):
    t_start = time.time()
    bridge.barrier()
    return time.time() - t_start


def connect(args):
    t_start = time.time()
    bridge = FMIBridge(
        world_size=args.world_size,
        rank=args.rank,
        channel_type=args.fmi_channel_type,
        rendezvous_host=args.rendezvous_host,
        rendezvous_port=args.rendezvous_port,
        comm_name=args.comm_name,
        maxtimeout=args.fmi_max_timeout_ms,
        nonblocking=args.fmi_options != 'blocking',
    )
    comm_init_s = time.time() - t_start
    if args.world_size > 1 and not bridge.available:
        raise FMIStageError('FMIBridge did not initialise', args.rank, args.world_size, args.comm_name)
    if bridge.available and bridge.rank != args.rank:
        raise FMIStageError(f'communicator assigned rank {bridge.rank}', args.rank, args.world_size, args.comm_name)
    return bridge, comm_init_s


def run_rank(args, process_start_ts):
    rank, world_size = args.rank, args.world_size
    stage_timings = startup_timings(process_start_ts)

    bridge, stage_timings['comm_init_s'] = connect(args)
    try:
        barrier_s = timed_barrier(bridge)

        t_payload_fetch_start = time.time()
        partition_map = broadcast_partition_map(bridge, rank, world_size, args)
        stage_timings['payload_fetch_s'] = time.time() - t_payload_fetch_start
        args.data_path = decode_rank_paths(partition_map, rank, args.data_prefix)

        dataloader, model = load_partition_and_model(args, stage_timings)
        execution_info = run_inference(
            model, dataloader, args.device, args.batch_size,
            rank, args.result_path, args.data_path,
        )

        barrier_s += timed_barrier(bridge)
        stage_timings['barrier_s'] = barrier_s
        stage_timings['inference_s'] = execution_info['inference_s']
        record = {**execution_info, **stage_timings}

        t_publish_start = time.time()
        gathered = gather_table(bridge, row_table(rank, GATHERED_FIELDS, record), PAYLOAD_ROOT)
        publish_s = time.time() - t_publish_start
        total_s = time.time() - process_start_ts

        trailing = gather_table(
            bridge, row_table(rank, TRAILING_FIELDS, {'publish_s': publish_s, 'total_s': total_s}), PAYLOAD_ROOT,
        )

        if rank == PAYLOAD_ROOT:
            write_combined_result(args, partition_map, gathered, trailing, t_publish_start)
    finally:
        bridge.finalize()


def write_combined_result(args, partition_map, gathered, trailing, t_aggregate_start):
    world_size = args.world_size
    stages = rows_by_rank(gathered, GATHERED_FIELDS)
    tails = rows_by_rank(trailing, TRAILING_FIELDS)
    if sorted(stages) != list(range(world_size)) or sorted(tails) != list(range(world_size)):
        raise FMIStageError(
            f'gather returned ranks {sorted(stages)} and {sorted(tails)}',
            args.rank, world_size, args.comm_name,
        )

    records = []
    for rank in range(world_size):
        record = {**stages[rank], **tails[rank]}
        record['rank'] = rank
        record['batch_size'] = args.batch_size
        record['device'] = args.device
        record['result_path'] = args.result_path
        record['data_path'] = decode_rank_paths(partition_map, rank, args.data_prefix)
        records.append(record)

    s3_client.put_object(
        Bucket=args.data_bucket,
        Key=f'{args.result_path}/combined_data.json',
        Body=json.dumps(records),
        ContentType='application/json',
    )
    aggregate_metrics = {
        'aggregate_s': time.time() - t_aggregate_start,
        'ranks_aggregated': len(records),
        'bytes_aggregated': sum(t.nbytes for t in gathered) + sum(t.nbytes for t in trailing),
    }
    s3_client.put_object(
        Bucket=args.data_bucket,
        Key=f'{args.result_path}/aggregate_metrics.json',
        Body=json.dumps(aggregate_metrics),
        ContentType='application/json',
    )
    logging.info(f'Rank: {args.rank}. aggregate metrics: {aggregate_metrics}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path', type=str,
        default=os.path.join(PROJECT_DIR, 'Fine_Tune_Model', 'Mixed_Inception_z_VITAE_Base_Img_Full_New_Full.pt'),
    )
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--rank', type=int, **environ_or_required('RANK'))
    parser.add_argument('--world_size', type=int, **environ_or_required('WORLD_SIZE'))
    parser.add_argument('--batch_size', type=int, **environ_or_required('BATCH_SIZE'))
    parser.add_argument('--data_bucket', type=str, **environ_or_required('DATA_BUCKET'))
    parser.add_argument('--data_prefix', type=str, **environ_or_required('DATA_PREFIX'))
    parser.add_argument('--result_path', type=str, **environ_or_required('RESULT_PATH'))
    parser.add_argument('--payload_bucket', type=str, **environ_or_required('S3_BUCKET'))
    parser.add_argument('--payload_key', type=str, **environ_or_required('PAYLOAD_KEY'))
    parser.add_argument('--comm_name', type=str, **environ_or_required('FMI_COMM_NAME'))
    parser.add_argument('--fmi_channel_type', type=str, **environ_or_required('FMI_CHANNEL_TYPE'))
    parser.add_argument('--fmi_options', type=str, choices=['nonblocking', 'blocking'],
                        **environ_or_required('FMI_OPTIONS'))
    parser.add_argument('--fmi_max_timeout_ms', type=int, **environ_or_required('FMI_MAX_TIMEOUT'))
    parser.add_argument('--rendezvous_host', type=str, **environ_or_required('RENDEZVOUS_HOST'))
    parser.add_argument('--rendezvous_port', type=int, **environ_or_required('RENDEZVOUS_PORT'))
    return parser.parse_args()


if __name__ == '__main__':
    process_start_ts = time.time()
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    try:
        run_rank(args, process_start_ts)
    except Exception as e:
        logging.exception(str(FMIStageError(f'Arm B rank failed: {e}', args.rank, args.world_size, args.comm_name)))
        sys.exit(1)
