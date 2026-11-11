# NueroBridgeEdge

# NeuroBridge Edge 개발 시작

상상하는 다양한 장면·사물·동작을 실시간 시각화하는 시스템을 목표로 개발합니다. 현재는 개발 머신 진단 도구만 구현되어 있으며, 전체 설계도 확인 후 시스템 구현을 진행합니다.

## 개발 머신 진단

Python 3.10 이상에서 외부 패키지 없이 실행합니다. Windows는 `python` 대신 `py -3` 명령을 사용할 수 있습니다.

```sh
python tools/doctor.py
python -m unittest discover -s tests -v
```

노트북에서 첫 명령을 실행한 JSON 결과로 운영체제, 아키텍처, 메모리와 디스크를 확인할 수 있습니다. 클라우드에서 실행하면 클라우드 머신 정보가 출력됩니다. 호스트명, 사용자명, 환경변수와 인증정보는 수집하지 않습니다.

[개발 및 검증 현황](docs/개발현황.md)을 참고하세요. 실제 EEG 해독과 장비 동작은 아직 검증하지 않았습니다.

## 합성 EEG 원본 기록·재생

```sh
python -m neurobridge
```

새 임시 폴더에 합성 ThinkGear 패킷 256개를 분할 수신하고 기록합니다. 원본 바이트와 파싱 EEG를 별도로 저장하고 SHA-256·재생 일치를 검사합니다. 출력 경로를 지정하려면 `python -m neurobridge --output reports/session-001`을 사용하세요. 기존 폴더는 덮어쓰지 않습니다.

현재 명령은 합성 데이터 전용이며 실제 Bluetooth 수신, 신호 분류, NPU 추론이나 시각화가 아닙니다. 저장 실패 후 추가 데이터 처리는 차단됩니다. 실시간 명령 경로와의 STOP 연동은 아직 구현되지 않았습니다.

기준 설계도 원본은 [v0.8 문서](docs/design/NeuroBridge_Edge_v0.8.txt)에 보존했습니다. 문서의 과거 승인 상태와 현재 대화에서 주신 개발 지시는 구분합니다. 미승인 성능값은 자동 확정하지 않습니다.

## CPU 합성 통합 실행

Python 3.11 이상에서 과학 계산 의존성을 설치합니다. GPU는 필요하지 않습니다.

```sh
python -m venv .venv
# Linux/macOS
.venv/bin/python -m pip install -r requirements-science.txt
.venv/bin/python -m neurobridge.simulation --output reports/demo-001
.venv/bin/python -m unittest discover -s tests -v
```

Windows에서는 `.venv/bin/python`을 `.venv\Scripts\python.exe`로 바꾸세요. 출력 폴더는 매 실행 새 이름을 사용합니다.

Godot 4.x 설치 후 다음 명령으로 3D 객체를 표시합니다. 명령·보고서 경로는 실제 절대 경로로 바꾸세요.

```sh
godot --path renderer -- --commands /절대경로/reports/demo-001/scene_commands.jsonl --report /절대경로/reports/demo-001/render.jsonl
```

창 없는 자동 검사에는 `--headless`를 Godot 옵션에 추가하고 마지막에 `--test`를 붙입니다. `--object sphere`로 표시 객체를 바꿀 수 있습니다. 제한된 클라우드에서는 `XDG_DATA_HOME`, `XDG_CACHE_HOME`, `XDG_CONFIG_HOME`을 쓰기 가능한 `reports/` 하위 절대 경로로 지정하세요.

현재 통합 경로는 **합성 정현파 + 사전 정의 명령**입니다. 특징으로 명령을 판별하지 않으며, 실시간 EEG 해독·NPU·독립 의도 판별을 시연하지 않습니다. Godot은 검증된 합성 명령 파일을 재생합니다. 헤드리스 검사는 실제 화면 출력·30fps 성능 검증과 구분합니다.

## 합성 모델 학습과 실시간 장면 연결

모델 경로는 ONNX/ONNX Runtime 추가 설치가 필요합니다.

```sh
.venv/bin/python -m pip install -r requirements-model.txt
.venv/bin/python -m neurobridge.training --output reports/model-001
.venv/bin/python -m neurobridge.simulation --model reports/model-001 --output reports/model-demo-001
.venv/bin/python -m neurobridge.live_demo --commands reports/model-demo-001/scene_commands.jsonl --output reports/live-001
```

마지막 명령은 Godot 창을 열고 PC 내부 TCP로 명령을 전송하며, 연결 종료 후 STOP 기록을 확인합니다. 창 없는 검증은 마지막에 `--headless`를 붙입니다. Linux 클라우드 의존성 재설치는 `bash tools/install.sh`로 할 수 있습니다.

모델은 6/10/24Hz **합성 주파수 클래스**를 학습합니다. 집중·시각상상·좌우 운동상상을 해독하는 모델이 아니며, 이 CPU 경로는 실제 DX-M1 필수 요구를 대체하지 않습니다. 합성 재생 타임스탬프는 원본 수집 시각과 구분합니다.

## 실제 장비 확보 후

[실기 검증 재개 절차](docs/실기검증_재개.md)에 따라 P1부터 수행합니다. 실제 Orange Pi에서 이미 페어링된 헤드셋의 RFCOMM 채널을 확인한 뒤 아래 수집 진입점을 사용할 수 있습니다. 현재 실기 수신은 **미검증**입니다.

```sh
python3 -m neurobridge.capture --address AA:BB:CC:DD:EE:FF --channel 1 --output reports/measured-001 --duration 300
```

예시 주소를 실제 장치 주소로 교체하세요. BlueZ 자동 페어링과 SDK/NPU 실행은 아직 구현되지 않았습니다. 원본 기록 파일과 모델 산출물은 `reports/`에 두고 GitHub에 업로드하지 마세요.

[전체 구현 현황](docs/status/전체현황.md)에서 최신 기능별 커밋, 합성 검증과 실기 보류 항목을 확인할 수 있습니다.
