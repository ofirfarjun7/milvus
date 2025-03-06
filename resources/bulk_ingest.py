"""
Bulk insert data into a Milvus collection. 

Assume that the batches of data have been uploaded to MinIO.
"""

import dataclasses
import os
import time
from pprint import pprint
from typing import Optional

import minio
import pymilvus

# Params
N_ROWS = int(os.getenv("BENCH_N_ROWS", 100000))
BATCH_SIZE = int(os.getenv("BENCH_BATCH_SIZE", 10000))
# BATCH_SIZE = int(os.getenv("BENCH_BATCH_SIZE", 200000))
DIM = int(os.getenv("BENCH_DIM", 1024))

MINIO_PORT = os.getenv("MINIO_PORT", "3334")
MINIO_ADDRESS = "1.1.60.20:{}".format(MINIO_PORT)
MINIO_SECRET_KEY = "minioadmin"
MINIO_ACCESS_KEY = "minioadmin"

# TODO: preferrably, the following params are not hard-coded.

# This should be the same as `minio.bucketName` in the milvus custom values yaml
BUCKET_NAME = "a-bucket"
REMOTE_DATA_PATH = "milvus_bulkinsert"
COLLECTION_NAME = "VectorDBBenchCollection"

# NOTE: it only works if the column name matches the file name
DATA_COL_NAME = "data"

# Number of shards in the collection
N_SHARDS = 2


assert N_ROWS % BATCH_SIZE == 0, "N_ROWS must be divisible by BATCH_SIZE"


_minio_client = None


def get_minio_client():
    global _minio_client
    if _minio_client is None:
        _minio_client = minio.Minio(
            endpoint=MINIO_ADDRESS,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False,
        )
    return _minio_client


def get_data_files() -> list[str]:
    """Return MinIO data file paths that have been uploaded.

    NOTE: We assume that each file contains a single batch of data (i.e. BATCH_SIZE rows and DIM dimensions).
    """
    minio_client = get_minio_client()
    data_files = [
        obj.object_name
        for obj in minio_client.list_objects(
            BUCKET_NAME, prefix=REMOTE_DATA_PATH, recursive=True
        )
    ]
    n_batches = N_ROWS // BATCH_SIZE
    if len(data_files) < n_batches:
        raise ValueError(
            "Insufficient data files uploaded. Expected at least {} files, got {}.".format(
                n_batches, len(data_files)
            )
        )
    return data_files[:n_batches]


def initialize_collection() -> pymilvus.Collection:
    pymilvus.connections.connect()

    print("Prepare Milvus collection")
    collection_list = pymilvus.list_collections()
    print(pymilvus.list_collections())
    if COLLECTION_NAME in collection_list:
        print(COLLECTION_NAME, "already exists. Dropping it.")
        coll = pymilvus.Collection(COLLECTION_NAME)
        coll.release()
        coll.drop()
        print("dropped")

    print("Create collection")
    fields = [
        pymilvus.FieldSchema(
            name="id", dtype=pymilvus.DataType.INT64, is_primary=True, auto_id=True
        ),
        pymilvus.FieldSchema(
            name=DATA_COL_NAME, dtype=pymilvus.DataType.FLOAT_VECTOR, dim=DIM
        ),
    ]
    schema = pymilvus.CollectionSchema(fields)
    coll = pymilvus.Collection(COLLECTION_NAME, schema, shards_num=N_SHARDS)
    return coll


@dataclasses.dataclass
class Stats:
    n_rows: int
    batch_size: int
    dim: int
    bulk_insert_time: Optional[float] = None


def main():
    stats = Stats(n_rows=N_ROWS, batch_size=BATCH_SIZE, dim=DIM)

    coll = initialize_collection()
    data_files = get_data_files()
    print(data_files)

    insert_t = time.time()
    task_ids = [
        pymilvus.utility.do_bulk_insert(
            collection_name=COLLECTION_NAME, files=[remote_path]
        )
        for remote_path in data_files
    ]
    remaining_tasks = set(task_ids)
    while remaining_tasks:
        print("Remaining tasks:", len(remaining_tasks), flush=True)
        for tid in [*remaining_tasks]:
            task = pymilvus.utility.get_bulk_insert_state(task_id=tid)
            if task.state in (
                pymilvus.BulkInsertState.ImportFailed,
                pymilvus.BulkInsertState.ImportFailedAndCleaned,
            ):
                print("Task failed:")
                pprint(task)
                print("--------------------------------------------")
                remaining_tasks.remove(tid)
            elif task.state == pymilvus.BulkInsertState.ImportCompleted:
                print("Task completed:")
                pprint(task)
                print("--------------------------------------------")
                print("Total number of entities inserted:", coll.num_entities)
                remaining_tasks.remove(tid)
        time.sleep(1)

    coll.flush()
    stats.bulk_insert_time = time.time() - insert_t
    print("bulk insert time:", stats.bulk_insert_time)

    pprint(dataclasses.asdict(stats))

    if coll.num_entities == stats.n_rows:
        print("Successfully inserted all {} entities".format(coll.num_entities))
    else:
        print("Bulk insert failed.")
        print("Total number of entities inserted:", coll.num_entities)
        print("Expected number of entities:", stats.n_rows)


if __name__ == "__main__":
    main()
