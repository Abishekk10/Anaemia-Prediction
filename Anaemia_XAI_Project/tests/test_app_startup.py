import builtins
import importlib.util
import io
import os
import runpy
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch


APP_DIR = Path(__file__).resolve().parents[1]
LEGACY_ARTIFACTS = {"svm_model.pkl", "scaler.pkl"}


@contextmanager
def guard_legacy_artifacts():
    attempts = []

    def guard_open(original_open):
        def guarded_open(file, *args, **kwargs):
            if not isinstance(file, int):
                name = Path(os.fsdecode(file)).name.lower()
                if name in LEGACY_ARTIFACTS:
                    attempts.append(name)
                    raise AssertionError(f"Unexpected legacy artifact access: {name}")
            return original_open(file, *args, **kwargs)

        return guarded_open

    with (
        patch("builtins.open", guard_open(builtins.open)),
        patch("io.open", guard_open(io.open)),
        patch("os.open", guard_open(os.open)),
    ):
        yield attempts


class AppStartupTests(unittest.TestCase):
    def test_render_uses_configured_session_key(self):
        with (
            patch.object(sys, "path", [str(APP_DIR), *sys.path]),
            patch.dict(os.environ, {"RENDER": "true", "FLASK_SECRET_KEY": "test-only-key"}),
        ):
            module = runpy.run_path(str(APP_DIR / "app.py"))

        self.assertEqual(module["app"].secret_key, "test-only-key")

    def test_render_requires_nonblank_session_key(self):
        for env in ({"RENDER": "true"}, {"RENDER": "true", "FLASK_SECRET_KEY": "   "}):
            with (
                self.subTest(configured="FLASK_SECRET_KEY" in env),
                patch.object(sys, "path", [str(APP_DIR), *sys.path]),
                patch.dict(os.environ, env, clear=True),
                self.assertRaisesRegex(RuntimeError, "FLASK_SECRET_KEY must be set"),
            ):
                runpy.run_path(str(APP_DIR / "app.py"))

    def test_local_startup_without_configured_key_can_sign_sessions(self):
        with (
            patch.object(sys, "path", [str(APP_DIR), *sys.path]),
            patch.dict(os.environ, {}, clear=True),
        ):
            app = runpy.run_path(str(APP_DIR / "app.py"))["app"]

        client = app.test_client()
        with client.session_transaction() as session:
            session["user_name"] = "Local test user"
        with client.session_transaction() as session:
            self.assertEqual(session["user_name"], "Local test user")

    def test_direct_run_disables_debug_mode(self):
        with (
            patch.object(sys, "path", [str(APP_DIR), *sys.path]),
            patch.dict(os.environ, {"FLASK_DEBUG": "1"}),
            patch("flask.Flask.run") as run,
        ):
            runpy.run_path(str(APP_DIR / "app.py"), run_name="__main__")

        run.assert_called_once_with(host="0.0.0.0", port=5000, debug=False)

    def test_import_does_not_open_legacy_artifacts(self):
        # Execute the module even if another test has already imported app.
        spec = importlib.util.spec_from_file_location("app", APP_DIR / "app.py")
        module = importlib.util.module_from_spec(spec)
        with (
            patch.object(sys, "path", [str(APP_DIR), *sys.path]),
            patch.dict(sys.modules, {"app": module}),
            guard_legacy_artifacts() as attempts,
        ):
            spec.loader.exec_module(module)
        self.assertEqual(attempts, [])
