import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
import os
import sys
from dotenv import load_dotenv
from time import time
from datetime import datetime
import hmac
import hashlib
import logging
import urllib
import traceback
import socket

# .../fii-accident-detection 절대 경로
file_dir = os.path.realpath(__file__)
project_dir = file_dir.split('s3_libs')[0]
sys.path.append(project_dir)

AWS_BUCKET_NAME = "42dot-umos-dev-solution"


logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.info(f"file_dir: {file_dir}")
logger.info(f"project_dir: {project_dir}")

class AmazonWebService:
    def __init__(self, aws_credential_path=None, bucket_name=AWS_BUCKET_NAME):
        """
        함수를 실행하기 전에 terminal에서 vault-login, vault-auth 를 먼저 진행해줘서 키를 발급해야 된다.
        자세한 내용은 https://42dot.atlassian.net/wiki/spaces/sec/pages/852197381/AWS 참고
        """
        hostname = str(socket.gethostname())
        if hostname.split('.')[-1] == 'local':
            absolute_env = os.path.join(project_dir, '.env.absolute')
            relative_env = os.path.join(project_dir, '.env.relative')
            load_dotenv(absolute_env)
            load_dotenv(relative_env)
            logger.debug(f"absolute_env: {absolute_env}")
            logger.debug(f"relative_env: {relative_env}")

            self.session = boto3.Session(profile_name='common-developer')
            self.bucket_name = os.getenv("AWS_BUCKET")
            self.aws_credential = self.session.get_credentials()

            logging.debug(f"READ CREDENTIAL FILE :: {self.aws_credential}")
            self.client = boto3.client('s3', 'ap-northeast-2', aws_access_key_id=self.aws_credential.access_key,
                                       aws_secret_access_key=self.aws_credential.secret_key,
                                       aws_session_token=self.aws_credential.token, config=Config(signature_version='s3v4'))
            self.s3 = boto3.resource('s3', 'ap-northeast-2', aws_access_key_id=self.aws_credential.access_key,
                                     aws_secret_access_key=self.aws_credential.secret_key,
                                     aws_session_token=self.aws_credential.token, config=Config(signature_version='s3v4'))

        else:  # if file is running at kubernetes
            self.client = boto3.client('s3', 'ap-northeast-2', config=Config(signature_version='s3v4'))
            self.s3 = boto3.resource('s3', 'ap-northeast-2', config=Config(signature_version='s3v4'))

        self.bucket_name = bucket_name
        self.bucket = self.s3.Bucket(self.bucket_name)

    def s3_list_objects(self, prefix='', delimiter='', should_contain=None, should_delete=None):
        paginator = self.client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self.bucket_name,
                                   Prefix=prefix,
                                   Delimiter=delimiter)
        try:
            object_list = [obj for page in pages for obj in page['Contents']]
        except KeyError:
            return []

        if should_contain is not None:
            object_list = [obj for obj in object_list if any(
                [word in obj['Key'] for word in should_contain])]
        if should_delete is not None:
            object_list = [obj for obj in object_list if not any(
                [word in obj['Key'] for word in should_delete])]
        object_name_list = [obj['Key'] for obj in object_list]
        return object_name_list

    def s3_upload_object(self, from_path, key, bucket=''):
        if not bucket:
            bucket = self.bucket_name
        self.s3.meta.client.upload_file(from_path, bucket, key)

    def s3_upload_objects(self, upload_files, keys, bucket=''):
        if not bucket:
            bucket = self.bucket_name
        t = time()
        assert len(upload_files) == len(
            keys), "upload_files, keys length should be same"
        length = len(upload_files)
        for i, (from_path, to_path) in enumerate(zip(upload_files, keys)):
            self.s3_upload_object(from_path, to_path, bucket)
            if not i % (length//10) or i == length-1:
                print(f"{i} 번째 uploaded :: {time() - t}")

    @staticmethod
    def sign(key, msg):
        return hmac.new(key, msg.encode('utf8'), hashlib.sha256).digest()

    def get_signature_key(self, key, date_stamp, region_name, service_name):
        kDate = self.sign(('AWS4' + key).encode('utf8'), date_stamp)
        kRegion = self.sign(kDate, region_name)
        kService = self.sign(kRegion, service_name)
        kSigning = self.sign(kService, 'aws4_request')
        return kSigning

    # def create_presigned_url(self, object_key, region='ap-northeast-2', expiration=3600):
    #     presigned_url = self.client.generate_presigned_url(
    #         ClientMethod='get_object',
    #         Params={'Bucket': self.bucket_name, 'Key': object_key, },
    #         ExpiresIn=expiration,
    #     )
    #     return presigned_url

    def create_cdn_url(self, object_key, region='ap-northeast-2', expiration=3600):
        return "https://static.tap-dev.umos.ai/ufos/" + object_key

    def get_object(self, key):
        return self.client.get_object(Bucket=self.bucket_name, Key=key)

    def download_object(self, from_key, to_path, bucket=''):
        if not bucket:
            bucket = self.bucket_name
        self.client.download_file(bucket, from_key, to_path)
