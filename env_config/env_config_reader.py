"""
Infra access 와 관련된 데이터를 저장한 .env.absolute 변수값을 읽어오는 함수.
"""
import os

from dotenv import load_dotenv


def read_absolute_env():
    absolute_env = os.path.join(os.path.dirname(__file__), ".env.absolute")
    load_dotenv(absolute_env)


def read_relative_env():
    relative_env = os.path.join(os.path.dirname(__file__), ".env.relative")
    load_dotenv(relative_env)


if __name__ != "__main__":
    read_relative_env()
    read_absolute_env()
