"""
AccidentDetection_Lv2 repo에서 data/* 밑에 csv나 pickle 파일이 생성되면
1. gcs에 같은 위치에 파일을 올리고
2. .md 파일을 만든다

이 코드는 pre-commit 을 이용해 commit 할 때 자동으로 실행되도록 한다.
"""
import logging
import os
import sys
from pathlib import Path

repo_dir = Path(".").parent.absolute().__str__()
sys.path.append(repo_dir)

from src.libs.gcp_utils import GoogleCloudStorage

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

gcs = GoogleCloudStorage()
current_dir = os.path.realpath(".")
data_directory = os.path.join(current_dir, "data")
data_file_type_list = ["csv", "pickle"]


def get_datafile_name_without_md(data_dir: str, data_file_type: list) -> list:
    """

    :param data_dir: directory to check added files
    :param data_file_type: 'csv', 'pickle', etc.
    :return: data list that should be uploaded to gcs
    """
    total_file = [
        os.path.join(root, file)
        for root, _, files in os.walk(data_dir)
        for file in files
    ]
    md_path_list = [
        file.split(".")[0] for file in total_file if file.split(".")[-1] == "md"
    ]
    data_path_list = [
        file for file in total_file if file.split(".")[-1] in data_file_type
    ]
    return [data for data in data_path_list if data.split(".")[0] not in md_path_list]


should_upload_data_list = get_datafile_name_without_md(
    data_directory, data_file_type_list
)

for data_path in should_upload_data_list:
    added_file_name = f"data/{data_path.split('data/')[-1]}"
    gcs_save_path = f"accident-detection/{added_file_name}"
    if not gcs.check_blob_exists(gcs_save_path):
        gcs.upload_blob(
            source_file_name=added_file_name, destination_blob_name=gcs_save_path
        )
        md_file_name = f"{added_file_name.rsplit('.', 1)[0]}.md"
        with open(md_file_name, "w") as f:
            f.write("### google cloud storage location\n")
            f.write(f"{gcs_save_path}\n\n")
            f.write("### data description")
        logging.info(f"{md_file_name} created")
