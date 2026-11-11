"""진단 데이터와 실제 명령 실행을 검증한다."""

import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from tools import doctor


class DoctorTests(unittest.TestCase):
    def test_memory_failure_is_unknown(self):
        with patch.object(doctor.sys, "platform", "linux"), patch.object(
            doctor.os, "sysconf", side_effect=OSError("조회 실패"), create=True
        ):
            self.assertIsNone(doctor.memory_bytes())

    def test_report_does_not_claim_hardware_validation(self):
        report = doctor.collect_report()
        self.assertEqual(report["실제BCI검증"], "NOT TESTED")
        self.assertEqual(report["실제NPU검증"], "NOT TESTED")
        self.assertGreater(report["현재작업디스크여유바이트"], 0)

    def test_cli_outputs_valid_report(self):
        script = Path(__file__).resolve().parents[1] / "tools" / "doctor.py"
        result = subprocess.run(
            [sys.executable, str(script)], capture_output=True, encoding="utf-8",
            timeout=10, check=True,
        )
        report = json.loads(result.stdout)
        self.assertEqual(report["운영체제"], doctor.platform.system())
        self.assertEqual(report["Python버전"], doctor.platform.python_version())
        self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
