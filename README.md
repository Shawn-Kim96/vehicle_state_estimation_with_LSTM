# AccidentDetection_Lv2
- Accident Detection Lv.2 알고리즘, 모델을 연구하는 repo
---
## 관련 문서
- Jira: [[UF-1438]](https://42dot.atlassian.net/browse/UF-1438) Accident Detection Lv.2
- Confluence: TBD

## Directory Structure

```
├── README.md
│
├── data               <- Uploaded to Google Storage or AWS. Not uploaded to Gitlab (DataBricks usage should be discussed)
│   ├── external       <- Data from third party sources.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── models             <- Trained and serialized models (rule-base, ML, DL models included)
│                         Finalized model that can directly read by pytorch or tensorflow, e.g. .pth
│
├── results            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── src                <- Source code for use in this project.
    ├── __init__.py    <- Makes src a Python module
    │
    ├── libs           <- Scripts for common class & functions
    │   └── example_library.py
    │
    ├── data           <- Scripts to download or generate data
    │   └── make_dataset.py
    │
    ├── models         <- Scripts for main model predictions. Models are imported from Project/model
    │   ├── train_model.py
    │   ├── predict_model.py
    │   ├── model_script1.py
    │   └── model_script2.py
    │
    └── visualization  <- Scripts to create exploratory and results oriented visualizations
        └── visualize.py
```
## Env Guide
- 환경변수는 `env_config` 폴더 밑에 저장하고 관리한다.
  - `src/.env.relative`: 환경에 따라 바뀌는 값 저장 (e.g. aws_credential_key, key_path)
  - `src/.env.absolute`: 환경에 따라 변하지 않는 값 저장 (e.g. acc_threshold, aws_url)
  - `env_config_reader.py`: 환경변수를 읽어오는 python script

## Data Guide
- 실험을 하기 위해 데이터를 만들었으면, `./data/` 밑에 저장한다. (데이터는 `.csv`, `.pickle`로 저장한다.)
- `pre-commit-config.yaml` 에 의해 새로 저장된 데이터를 자동으로 google-cloud-storage 에 업로드 한다.
  - 데이터 업로드 코드는 `data_operation.py` 를 참고하면 된다.
-----

## Gitflow Guide
### Branch guide
- branch 는 main, develop, feature branches, hotfix branches, chore branches 로 구성한다.
- feature branch와 지라 이슈는 1:1 연결되도록 생성하며, merge는 develop 브랜치로 한다.
- branch 이름은 아래의 패턴을 유지한다.
    - `topic-type/{Jira issue #}-{:desc}`
        - e.g., `feature/UF-999-test_google_api`
    - topic types
        - feature : 기능 추가
        - hotfix : 버그 수정
        - chore : 나머지 (리팩토링, 환경변수 설정, 파일 삭제 등)
- 참고 : [Git branch management](https://42dot.atlassian.net/wiki/spaces/EN/pages/105414823/Git+branch+management)


### Commit note guide
- [Conventional Commit](https://www.conventionalcommits.org/en/v1.0.0/#summary) 을 따른다.
- commit note 는 아래의 패턴을 유진한다.
    - `{commit type}: {description}`
        - e.g., `feat: geolocation mapmatching API added`
        - commit이 task랑 연결되어 있으면 `feat: [UF-999] desc` 형식으로 적어도 된다.
    - commit types
        - `fix`: A bug fix. Correlates with PATCH in SemVer
        - `feat`: A new feature. Correlates with MINOR in SemVer
        - `docs`: Documentation only changes
        - `style`: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-color)
        - `refactor`: A code change that neither fixes a bug nor adds a feature
        - `perf`: A code change that improves performance
        - `test`: Adding missing or correcting existing tests
        - `build`: Changes that affect the build system or external dependencies (example scopes: pip, docker)
        - `ci`: Changes to our CI configuration files and scripts (example scopes: GitLabCI)
- commit 메세지 형식을 유지하기 위해 `commitizen`, `pre-commit` 플러그인을 사용한다.
    - `fii-accident-reconstruction` 디렉토리의 터미널에서 commit 해야 commitizen 적용 가능
    - 디렉토리 안에 있는 `.pre-commit-config.yaml` 파일을 기반으로 commit 메세지 형식 검토
    - 참고 : [Git 환경설정 컨플루언스 문서](https://42dot.atlassian.net/wiki/spaces/UFII/pages/2501869793/WIP+Git)

## Python Guide
### 가상환경 설정
- 기본적으로 `pyenv`와 `poetry` 사용한다.
  - `pyenv` : 다양한 버전의 파이썬을 다운받고 정리하는 역할
  - `poetry` :가상 환경 생성과 파이썬 패키지 매니징을
- `poetry`를 사용해서 가상환경 및 패키지를 솔루션 별로 관리한다.
  - [Poetry 설명 링크](https://python-poetry.org/docs/master/#installing-with-the-official-installer)
  - 주의할 점 : `poetry` 경로를 인식할 수 있도록 `~/.zshrc` 혹은 `~/.bash_profile` 등에 경로를 잘 저장한다.
- `poetry`는 디렉토리 별 `pyproject.toml` 파일로 관리된다.
  - 각 프로젝트(`accident-reconstruction`, `geolocation-insurance`, ..)의 디렉토리 별로 `pyproject.toml` 파일로 관리한다.
  - `pyproject.toml` 파일이 없으면 `poetry init`을 입력해 설정해준다.
- 새로운 패키지를 설치하고 싶으면 `pip install` 대신 `poetry add {library}`를 사용하면 된다.
- 설치된 라이브러리 / 종속관계를 보고싶으면 아래 명령어를 입력하면 된다.
  - 라이브러리 목록 : `poetry show`
  - 라이브러리 목록 / 관계 : `poetry show --tree`
- 자세한 내용은 [컨플루언스](https://42dot.atlassian.net/wiki/spaces/UFII/pages/2500591617/WIP+Python)를 참고하면 된다.
