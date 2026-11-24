import io
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


APP_DIR = Path(__file__).resolve().parents[1]
try:
    import pymupdf
except ImportError as exc:
    pymupdf = None
    PDF_UNAVAILABLE = str(exc)
else:
    PDF_UNAVAILABLE = ""

with patch.object(sys, "path", [str(APP_DIR), *sys.path]):
    import app as app_module


def make_pdf(text="", pages=1, encrypted=False):
    with pymupdf.open() as document:
        for _ in range(pages):
            page = document.new_page()
            if text:
                page.insert_text((72, 72), text)
        if encrypted:
            return document.tobytes(
                encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="test-password"
            )
        return document.tobytes()


class GenderValidationTests(unittest.TestCase):
    def setUp(self):
        self.client = app_module.app.test_client()
        self.prediction_inputs = []

        def fake_predict(features):
            self.prediction_inputs.append(features.copy())
            return {
                "prediction_index": 0,
                "prediction_proba": np.array([0.8, 0.2]),
                "features": np.zeros((1, 5)),
                "features_scaled": np.zeros((1, 5)),
            }

        self.predict_patcher = patch.object(app_module, "predict_pytorch", side_effect=fake_predict)
        self.predict_patcher.start()
        self.addCleanup(self.predict_patcher.stop)

    def _manual_request(self, gender_marker=...):
        data = {
            "hemoglobin": "14.9",
            "mcv": "83.7",
            "mch": "22.7",
            "mchc": "29.1",
        }
        if gender_marker is not ...:
            data["gender"] = gender_marker
        return self.client.post("/predict", data=data)

    def _pdf_request(self, report_gender=None, form_gender=...):
        lines = ["Hemoglobin 14.9 g/dL", "MCV 83.7 fL", "MCH 22.7 pg", "MCHC 29.1 g/dL"]
        if report_gender is not None:
            lines.insert(0, f"Age / Sex: 25 {report_gender}")
        data = {"pdf_file": (io.BytesIO(make_pdf("\n".join(lines))), "cbc.pdf")}
        if form_gender is not ...:
            data["gender"] = form_gender
        return self.client.post("/upload_pdf", data=data, content_type="multipart/form-data")

    def test_manual_accepts_male_and_female_case_insensitively(self):
        for gender, expected_code in (("mAlE", 0), ("FEMALE", 1)):
            with self.subTest(gender=gender):
                response = self._manual_request(gender)
                self.assertEqual(response.status_code, 200)
                self.assertIn(b"Prediction Results", response.data)
                self.assertEqual(self.prediction_inputs[-1]["Gender"], expected_code)

    def test_manual_rejects_missing_and_unsupported_gender(self):
        for marker in (..., "Other"):
            with self.subTest(gender=marker):
                response = self._manual_request(marker)
                self.assertEqual(response.status_code, 200)
                self.assertIn(b"Gender must be exactly Male or Female.", response.data)
        self.assertEqual(self.prediction_inputs, [])

    def test_shared_builder_rejects_unsupported_gender(self):
        with self.assertRaises(app_module.GenderValidationError):
            app_module.build_prediction_result("Other", 14.9, 83.7, 22.7, 29.1)
        self.assertEqual(self.prediction_inputs, [])

    @unittest.skipIf(pymupdf is None, PDF_UNAVAILABLE)
    def test_pdf_accepts_report_gender_and_form_gender(self):
        for report_gender, form_gender, expected_code in (
            ("Male", ..., 0),
            ("Female", ..., 1),
            (None, "fEmAlE", 1),
        ):
            with self.subTest(report_gender=report_gender, form_gender=form_gender):
                response = self._pdf_request(report_gender, form_gender)
                self.assertEqual(response.status_code, 200)
                self.assertIn(b"Prediction Results", response.data)
                self.assertEqual(self.prediction_inputs[-1], {
                    "Gender": expected_code, "Hemoglobin": 14.9,
                    "MCV": 83.7, "MCH": 22.7, "MCHC": 29.1,
                })

    @unittest.skipIf(pymupdf is None, PDF_UNAVAILABLE)
    def test_pdf_rejects_missing_and_unsupported_gender(self):
        for form_gender in (..., "Other"):
            with self.subTest(form_gender=form_gender):
                response = self._pdf_request(form_gender=form_gender)
                self.assertEqual(response.status_code, 200)
                self.assertIn(b"Gender must be exactly Male or Female.", response.data)
        self.assertEqual(self.prediction_inputs, [])

    def _upload(self, data, filename="cbc.pdf"):
        return self.client.post("/upload_pdf", data={"pdf_file": (io.BytesIO(data), filename)})

    def test_upload_rejects_missing_file_and_blank_filename(self):
        missing = self.client.post("/upload_pdf", data={"gender": "Male"}, content_type="multipart/form-data")
        blank = self._upload(b"%PDF-1.7", "")
        self.assertIn(b"No PDF file uploaded", missing.data)
        self.assertIn(b"No file selected", blank.data)
        self.assertFalse(self.prediction_inputs)

    def test_upload_rejects_empty_and_unsupported_inputs(self):
        for data, filename, message in (
            (b"", "empty.pdf", b"uploaded PDF is empty"),
            (b"not a PDF", "renamed.pdf", b"Invalid file format"),
            (b"%PDF-1.7", "wrong.txt", b"Invalid file format"),
        ):
            with self.subTest(filename=filename):
                self.assertIn(message, self._upload(data, filename).data)
        self.assertFalse(self.prediction_inputs)

    def test_upload_rejects_oversize_file_and_request(self):
        with patch.object(app_module, "PDF_MAX_BYTES", 1024):
            for size in (1025, 70 * 1024):
                with self.subTest(size=size):
                    response = self._upload(b"x" * size)
                    self.assertEqual(response.status_code, 413)
                    self.assertIn(b"no larger than", response.data)

    @unittest.skipIf(pymupdf is None, PDF_UNAVAILABLE)
    def test_real_pdf_error_cases(self):
        for data, status, message in (
            (b"%PDF-1.7\ncorrupt", 200, b"corrupt or unreadable"),
            (make_pdf(encrypted=True), 200, b"password-protected"),
            (make_pdf(), 200, b"no selectable text"),
            (make_pdf(pages=app_module.PDF_MAX_PAGES + 1), 413, b"at most 20 pages"),
        ):
            with self.subTest(message=message):
                response = self._upload(data)
                self.assertEqual(response.status_code, status)
                self.assertIn(message, response.data)
        self.assertFalse(self.prediction_inputs)

    def test_lab_labels_distinguish_mch_from_mchc(self):
        extract = app_module.extract_values_from_pdf
        for text in (
            "mChC: 31.2\nMcH: 24.6",
            "Mean Corpuscular Hemoglobin Concentration: 31.2\nMean Corpuscular Hemoglobin\n24.6",
        ):
            with self.subTest(text=text):
                self.assertEqual(extract(text), {"mchc": 31.2, "mch": 24.6})
        self.assertEqual(extract("MCH\nMCHC 31.2"), {"mchc": 31.2})
        self.assertEqual(extract("MCHC 31.2"), {"mchc": 31.2})

    def test_extraction_failure_closes_document_and_redacts_error(self):
        document = MagicMock(is_pdf=True, needs_pass=False, page_count=1)
        document.__enter__.return_value = document
        page = MagicMock()
        page.get_text.side_effect = RuntimeError("sensitive report data")
        document.__iter__.return_value = iter([page])
        parser = types.SimpleNamespace(open=MagicMock(return_value=document), FileDataError=ValueError)

        with patch.object(app_module, "fitz", parser), self.assertLogs(app_module.app.logger) as logs:
            response = self._upload(b"%PDF-1.7", "sensitive-filename.pdf")

        document.__exit__.assert_called_once()
        self.assertIn(b"Could not process this PDF", response.data)
        self.assertNotIn(b"sensitive", response.data)
        self.assertNotIn("sensitive", " ".join(logs.output))


if __name__ == "__main__":
    unittest.main()
