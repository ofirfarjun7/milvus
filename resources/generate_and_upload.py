"""Upload generated files to Minio"""

import dataclasses
import os
import pathlib
import shutil
import time
from pprint import pprint
from typing import Iterator, Optional

import dask.array as da
import minio
import numpy as np
import pynvml
from dask.distributed import Client

# Params
N_ROWS = int(os.getenv("BENCH_N_ROWS", 100000))
# Large batch sizes can seem to make bulk insert faster
BATCH_SIZE = int(os.getenv("BENCH_BATCH_SIZE", 10000))
# BATCH_SIZE = int(os.getenv("BENCH_BATCH_SIZE", 200000))
DIM = int(os.getenv("BENCH_DIM", 1024))

MINIO_PORT = os.getenv("MINIO_PORT", "3334")
# MINIO_ADDRESS = "1.1.60.14:{}".format(MINIO_PORT)
# MINIO_ADDRESS = "0.0.0.0:{}".format(MINIO_PORT)
MINIO_ADDRESS = "1.1.60.20:{}".format(MINIO_PORT)
MINIO_SECRET_KEY = "minioadmin"
MINIO_ACCESS_KEY = "minioadmin"
# This should be the same as `minio.bucketName` in the milvus custom values yaml
DEFAULT_BUCKET_NAME = "a-bucket"

REMOTE_DATA_PATH = "milvus_bulkinsert"
COLLECTION_NAME = "VectorDBBenchCollection"
# TEMP_DATA_PATH = (pathlib.Path() / "blobs_npy_data").resolve()
# TEMP_DATA_PATH = pathlib.Path("/mnt/minio_storage/blobs_npy_data").resolve()
TEMP_DATA_PATH = pathlib.Path("/mnt/minio_storage/blobs_npy_data").resolve()
print("Temporarily data path:", TEMP_DATA_PATH)

# FIXME: it only works if the column name matches the file name
DATA_COL_NAME = "data"


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


def use_gpu():
    try:
        pynvml.nvmlInit()
        return True
    except Exception:
        return False


def generate_data_files(
    n_rows: int, batch_size: int, dim: int, n_centers: int
) -> Iterator[pathlib.Path]:
    # TODO: refactor this function

    if use_gpu():
        from cuml.dask.datasets import make_blobs as _make_blobs
        from dask_cuda import LocalCUDACluster as LocalCluster

        def make_blobs(**kwargs):
            _ = kwargs.pop("chunks", None)
            return _make_blobs(**kwargs)

        def make_blobs_with_centers(**kwargs):
            # A make_blobs function that can return centers
            return _make_blobs(**kwargs)

    else:
        from dask.distributed import LocalCluster
        from dask_ml.datasets import make_blobs
        from sklearn.datasets import make_blobs as make_blobs_with_centers

    cluster = LocalCluster(n_workers=1, threads_per_worker=1)
    client = Client(cluster)

    n_batches = int(np.ceil(n_rows / batch_size))
    print("n_batches:", n_batches)
    try:
        _, _, centers = make_blobs_with_centers(
            n_samples=min(batch_size, 50000),
            n_features=dim,
            centers=n_centers,
            return_centers=True,
        )

        for batch in range(n_batches):
            adjusted_batch_size = min(batch_size, n_rows - batch * batch_size)
            X, y = make_blobs(
                n_samples=adjusted_batch_size,
                n_features=dim,
                centers=centers,
                chunks=(adjusted_batch_size, -1),
            )
            X = X.astype(np.float32)
            # TODO: numpy stack causes Milvus to complain about "no corresponding field in collection".
            # This is likely due to schema column name mapping errors.
            data_path = TEMP_DATA_PATH / str(batch)
            data_path.mkdir(parents=True, exist_ok=True)
            da.to_npy_stack(TEMP_DATA_PATH / str(batch), X, axis=0)
            yield from data_path.rglob("*.npy")
    finally:
        client.close()
        cluster.close()


def upload_file(local_path, minio_path):
    minio_client = get_minio_client()
    minio_client.fput_object(
        bucket_name=DEFAULT_BUCKET_NAME,
        object_name=minio_path,
        file_path=local_path,
        part_size=100 * 1024 * 1024,
    )
    return minio_path


def initialize_upload():
    # Create the MinIO bucket if it doesn't exist
    minio_client = get_minio_client()
    found = minio_client.bucket_exists(DEFAULT_BUCKET_NAME)
    if not found:
        print(
            "MinIO bucket '{}' doesn't exist, creating...".format(DEFAULT_BUCKET_NAME)
        )
        minio_client.make_bucket(DEFAULT_BUCKET_NAME)

    # Delete the local data directory if it exists
    if TEMP_DATA_PATH.exists():
        print("Data file directory already exists locally. Deleting...")
        shutil.rmtree(TEMP_DATA_PATH)

    # Delete the remote data files if they exist
    remote_data_objects = list(
        minio_client.list_objects(
            DEFAULT_BUCKET_NAME, prefix=REMOTE_DATA_PATH, recursive=True
        )
    )
    if remote_data_objects:
        print(
            "Data files already exist in MinIO. Deleting {} files...".format(
                len(remote_data_objects)
            )
        )
        for obj in remote_data_objects:
            minio_client.remove_object(DEFAULT_BUCKET_NAME, obj.object_name)


@dataclasses.dataclass
class Stats:
    n_rows: int
    batch_size: int
    dim: int
    bucket: str
    remote_data_path: str
    data_gen_time: Optional[float] = None
    upload_time: Optional[float] = None


def main():
    stats = Stats(
        n_rows=N_ROWS,
        batch_size=BATCH_SIZE,
        dim=DIM,
        bucket=DEFAULT_BUCKET_NAME,
        remote_data_path=REMOTE_DATA_PATH,
    )

    initialize_upload()
    # Generate data files, then upload to MinIO
    n_centers = 1024

    upload_time = 0
    data_gen_time = 0
    data_gen_t = time.time()
    data_files = generate_data_files(
        n_rows=N_ROWS,
        batch_size=BATCH_SIZE,
        dim=DIM,
        n_centers=n_centers,
    )
    print(data_files)
    for local_path in data_files:
        data_gen_time += time.time() - data_gen_t

        # e.g. local/blobs_npy_data/0/0.npy -> remote/0/base.npy
        minio_path = (
            pathlib.Path(REMOTE_DATA_PATH)
            / local_path.relative_to(TEMP_DATA_PATH).with_suffix("")
            / f"{DATA_COL_NAME}.npy"
        )
        print(
            "Uploading data file {} to {}".format(local_path, minio_path),
            flush=True,
        )

        upload_t = time.time()
        upload_file(local_path=str(local_path), minio_path=str(minio_path))
        upload_time += time.time() - upload_t

        local_path.unlink()  # TODO: delete the local file after uploading?

        data_gen_t = time.time()

    stats.data_gen_time = data_gen_time
    stats.upload_time = upload_time
    print("Data generation time:", stats.data_gen_time)
    print("upload time", stats.upload_time)
    pprint(stats)


if __name__ == "__main__":
    main()
