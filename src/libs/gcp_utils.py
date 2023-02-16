"""
google service account 를 이용하여 google cloud storage 에 접근하는 함수들을 만들었습니다.

API key Google Cloud -> IAM & Admin -> Servie Accounts -> AIoT_rfactor2 -> Key
cloud storage client : https://cloud.google.com/storage/docs/reference/libraries#client-libraries-install-python

"""
import logging
import os
import sys
from io import BytesIO
from time import time

import pandas as pd
from google.cloud import storage

file_dir = os.path.realpath(os.path.dirname(__file__))
project_dir = file_dir.split("src")[0]
sys.path.append(project_dir)
import env_config.env_config_reader

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


class GoogleCloudStorage:
    def __init__(self, project_name="UMOS-FII", bucket_name="ufos-solution"):
        self.project_name = project_name
        self.bucket_name = bucket_name
        self.client = storage.Client(self.project_name)
        self.bucket = self.client.bucket(self.bucket_name)

    def check_blob_exists(self, blob_path):
        stats = self.bucket.blob(blob_path).exists()
        return stats

    def list_blobs_in_condition(
        self, prefix=None, delimiter=None, should_contain=None, should_delete=None
    ):
        blob_list = [
            x for x in self.bucket.list_blobs(prefix=prefix, delimiter=delimiter)
        ]
        if should_contain is not None:
            blob_list = [
                x for x in blob_list if any([word in x.name for word in should_contain])
            ]
        if should_delete is not None:
            blob_list = [
                x
                for x in blob_list
                if not any([word in x.name for word in should_delete])
            ]
        return blob_list

    def download_blob_to_file(
        self,
        from_dir=None,
        to_dir=None,
        delimiter=None,
        should_contain=[],
        should_delete=[],
    ):
        blob_list = self.list_blobs_in_condition(
            from_dir, delimiter, should_contain, should_delete
        )
        download_path = [f"{to_dir}/{x.name.split('/')[-1]}" for x in blob_list]
        t = time()
        for i, (blob, to_path) in enumerate(zip(blob_list, download_path)):
            blob.download_to_filename(to_path)
            if i % 100 == 1:
                logging.info(f"{i} downloaded :: {time()-t}[s]")
                t = time()

    def download_blob_in_memory(self, blob_path):
        blob = self.bucket.blob(blob_path)
        content = blob.download_as_string()
        df = pd.read_csv(BytesIO(content))
        return df

    def upload_blob(self, source_file_name, destination_blob_name):
        blob = self.bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        logging.info(f"File {source_file_name} uploaded to {destination_blob_name}.")
