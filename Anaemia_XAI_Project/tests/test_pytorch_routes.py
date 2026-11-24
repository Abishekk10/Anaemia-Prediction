import importlib.util
import io
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from test_app_startup import APP_DIR, guard_legacy_artifacts


MANUAL_FORM = {
    "gender": "mAlE", "hemoglobin": "14.9", "mch": "22.7",
    "mchc": "29.1", "mcv": "83.7",
}


class PyTorchRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("torch") is None:
            raise unittest.SkipTest("PyTorch is unavailable")
        with patch.object(sys, "path", [str(APP_DIR), *sys.path]):
            import app
            import pytorch_inference
        cls.app_module = app
        cls.inference = pytorch_inference

    def setUp(self):
        self.client = self.app_module.app.test_client()
        self.inference._load_model_bundle.cache_clear()
        self.addCleanup(self.inference._load_model_bundle.cache_clear)

    def _predict(self):
        return self.client.post("/predict", data=MANUAL_FORM)

    def _assert_safe_failure(self, target, method):
        failure = f"internal diagnostic: {MANUAL_FORM}"
        with (
            patch.object(target, method, side_effect=RuntimeError(failure)) as failing_call,
            patch.object(self.app_module, "build_svm_prediction_result") as svm_predict,
            guard_legacy_artifacts() as attempts,
            self.assertLogs(self.app_module.app.logger, level="ERROR") as logs,
        ):
            response = self._predict()

        failing_call.assert_called_once()
        svm_predict.assert_not_called()
        self.assertEqual(attempts, [])
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"The prediction model is temporarily unavailable.", response.data)
        self.assertNotIn(failure.encode(), response.data)
        self.assertEqual(logs.output, ["ERROR:app:PyTorch form prediction failed (RuntimeError)"])
        self.assertNotIn(b'<div class="prediction-result ', response.data)

    def test_checkpoint_loading_failure_does_not_use_svm(self):
        self._assert_safe_failure(self.inference.torch, "load")

    def test_scaler_loading_failure_does_not_use_svm(self):
        self._assert_safe_failure(self.inference.joblib, "load")

    def test_inference_failure_does_not_use_svm(self):
        self._assert_safe_failure(self.inference.AnaemiaNN, "forward")

    def test_missing_or_malformed_measurement_returns_safe_error_without_svm(self):
        malformed_value = "private malformed measurement"
        missing = MANUAL_FORM.copy()
        del missing["hemoglobin"]
        malformed = {**MANUAL_FORM, "hemoglobin": malformed_value}

        for case, form in (("missing", missing), ("malformed", malformed)):
            with (
                self.subTest(case=case),
                patch.object(self.app_module, "predict_pytorch") as predict,
                patch.object(self.app_module, "build_svm_prediction_result") as svm_predict,
                guard_legacy_artifacts() as attempts,
                self.assertLogs(self.app_module.app.logger, level="ERROR"),
            ):
                response = self.client.post("/predict", data=form)

                predict.assert_not_called()
                svm_predict.assert_not_called()
                self.assertEqual(attempts, [])
                self.assertEqual(response.status_code, 200)
                self.assertIn(
                    b"Could not process this prediction. Please check your inputs and try again.",
                    response.data,
                )
                self.assertNotIn(malformed_value.encode(), response.data)
                self.assertNotIn(b'<div class="prediction-result ', response.data)

    def test_unexpected_error_hides_internal_details(self):
        internal_path = "/srv/private/prediction.json"
        detail = f"unexpected diagnostic for {MANUAL_FORM} at {internal_path}"
        console = io.StringIO()
        with (
            patch.object(self.app_module, "build_prediction_result", side_effect=RuntimeError(detail)),
            self.assertLogs(self.app_module.app.logger, level="ERROR") as logs,
            redirect_stdout(console),
            redirect_stderr(console),
        ):
            response = self._predict()

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Could not process this prediction. Please check your inputs and try again.", response.data)
        self.assertNotIn(b"unexpected diagnostic", response.data)
        self.assertNotIn(internal_path.encode(), response.data)
        self.assertEqual(logs.output, ["ERROR:app:Manual prediction request failed (RuntimeError)"])
        self.assertEqual(console.getvalue(), "")

    def test_predict_uses_real_checkpoint(self):
        console = io.StringIO()
        with (
            patch.object(self.inference.torch, "load", wraps=self.inference.torch.load) as load,
            patch.object(self.app_module, "predict_pytorch", wraps=self.inference.predict_pytorch) as predict,
            guard_legacy_artifacts() as attempts,
            self.assertNoLogs(self.app_module.app.logger, level="DEBUG"),
            redirect_stdout(console),
            redirect_stderr(console),
        ):
            response = self._predict()

        load.assert_called_once_with(self.inference.CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        predict.assert_called_once_with({
            "Gender": 0, "Hemoglobin": 14.9, "MCH": 22.7, "MCHC": 29.1, "MCV": 83.7,
        })
        self.assertEqual(attempts, [])
        self.assertEqual(console.getvalue(), "")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<div class="prediction-result ', response.data)
