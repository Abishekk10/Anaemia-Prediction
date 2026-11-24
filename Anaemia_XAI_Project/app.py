from flask import Flask, flash, redirect, render_template, request, session, url_for
import numpy as np
import os
import fitz  # PyMuPDF for PDF processing (import as fitz)
import re
import json
import hashlib
from pytorch_inference import PyTorchInferenceError, predict_pytorch
from werkzeug.exceptions import RequestEntityTooLarge

app = Flask(__name__)
app.secret_key = "anaemia-ai-local-auth"

USERS_FILE = os.path.join("data", "users.json")
PDF_MAX_BYTES = 5 * 1024 * 1024
PDF_MAX_PAGES = 20

def load_users():
    if not os.path.exists(USERS_FILE):
        return []
    with open(USERS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def save_users(users):
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    with open(USERS_FILE, "w", encoding="utf-8") as file:
        json.dump(users, file, indent=2)


def hash_password(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def find_user_by_email(email):
    normalized_email = email.strip().lower()
    return next((user for user in load_users() if user["email"] == normalized_email), None)

def extract_values_from_pdf(pdf_content):
    """Extract hemoglobin, MCV, MCH, and MCHC from row-style lab report text."""
    extracted_values = {}
    normalized_text = pdf_content.replace(",", ".").lower()
    normalized_text = re.sub(r"[ \t]+", " ", normalized_text)
    lines = [line.strip() for line in normalized_text.splitlines() if line.strip()]

    field_config = {
        "hemoglobin": {
            "aliases": ["haemoglobin", "hemoglobin", "hgb", "hb"],
            "range": (3.0, 25.0),
        },
        "mcv": {
            "aliases": ["mean corpuscular volume", "mcv"],
            "range": (50.0, 130.0),
        },
        "mch": {
            "aliases": ["mean corpuscular hemoglobin", "mch"],
            "range": (15.0, 45.0),
        },
        "mchc": {
            "aliases": ["mean corpuscular hemoglobin concentration", "mchc"],
            "range": (20.0, 45.0),
        },
    }

    skip_tokens = [
        "method",
        "specimen",
        "reference",
        "interval",
        "page",
        "department",
        "processed at",
        "collected at",
        "patient id",
        "registered date",
        "reported date",
        "age / sex",
        "investigation",
        "observed value",
        "units",
    ]

    def is_plausible(field, value):
        low, high = field_config[field]["range"]
        return low <= value <= high

    def extract_number(text):
        match = re.search(r"(?<![a-z])(-?\d+(?:\.\d+)?)(?![a-z])", text)
        return float(match.group(1)) if match else None

    labels = {
        alias: field
        for field, config in field_config.items()
        for alias in config["aliases"]
    }
    label_pattern = re.compile(
        r"^(" + "|".join(map(re.escape, sorted(labels, key=len, reverse=True))) + r")\b"
    )

    # First pass: handle row-style reports where the label and numeric value are separated by a few lines.
    for index, line in enumerate(lines):
        label = label_pattern.match(line)
        if label is None:
            continue
        field = labels[label.group()]
        if field in extracted_values:
            continue

        same_line_value = extract_number(line[label.end():])
        if same_line_value is not None and is_plausible(field, same_line_value):
            extracted_values[field] = same_line_value
            continue

        for candidate in lines[index + 1:index + 6]:
            if label_pattern.match(candidate):
                break
            if any(token in candidate for token in skip_tokens):
                continue
            value = extract_number(candidate)
            if value is not None and is_plausible(field, value):
                extracted_values[field] = value
                break

    # Fallback: tolerate label/value splits across whitespace in the whole document.
    fallback_patterns = {
        "hemoglobin": [
            r"(?<!mean corpuscular )\b(?:haemoglobin|hemoglobin|hgb|hb)\b(?:[\s:=-]|method|specimen)*(?P<value>\d+(?:\.\d+)?)",
        ],
        "mcv": [
            r"(?:^|\b)(?:mean corpuscular volume|mcv)\b(?:[\s:=-]|method|specimen)*(?P<value>\d+(?:\.\d+)?)",
        ],
        "mch": [
            r"(?:^|\b)(?:mean corpuscular hemoglobin|mch)\b(?:[\s:=-]|method|specimen)*(?P<value>\d+(?:\.\d+)?)",
        ],
        "mchc": [
            r"(?:^|\b)(?:mean corpuscular hemoglobin concentration|mchc)\b(?:[\s:=-]|method|specimen)*(?P<value>\d+(?:\.\d+)?)",
        ],
    }

    for field, patterns in fallback_patterns.items():
        if field in extracted_values:
            continue
        for pattern in patterns:
            match = re.search(pattern, normalized_text, re.IGNORECASE | re.DOTALL)
            if not match:
                continue
            value = float(match.group("value"))
            if is_plausible(field, value):
                extracted_values[field] = value
                break

    return extracted_values


def extract_gender_from_pdf(pdf_content):
    """Best-effort gender extraction from report text."""
    normalized_text = pdf_content.lower()
    match = re.search(r"age\s*/\s*sex\s*[:\-]?\s*[\d\s\w/]*\b(male|female)\b", normalized_text)
    if match:
        return match.group(1)
    return None


class GenderValidationError(ValueError):
    pass


def validate_gender(gender):
    if not isinstance(gender, str) or gender.casefold() not in {"male", "female"}:
        raise GenderValidationError("Gender must be exactly Male or Female.")
    return gender.casefold()


def calculate_anaemia_risk_index(gender, hemoglobin, mcv, mch, mchc):
    """Calculate Anaemia Risk Index (display-only, not passed to scaler)"""
    risk_score = 0
    
    # Hemoglobin scoring
    if gender.lower() == 'male':
        if hemoglobin < 13:
            risk_score += 30
        elif hemoglobin < 14:
            risk_score += 15
    else:
        if hemoglobin < 12:
            risk_score += 30
        elif hemoglobin < 13:
            risk_score += 15
    
    # MCV scoring
    if mcv < 80 or mcv > 100:
        risk_score += 20
    
    # MCH scoring
    if mch < 27 or mch > 33:
        risk_score += 20
    
    # MCHC scoring
    if mchc < 32 or mchc > 36:
        risk_score += 15
    
    return min(risk_score, 100)


def calculate_dynamic_confidence(prediction_proba, prediction, risk_index, gender, hemoglobin, mcv, mch, mchc):
    """Blend model certainty with clinical abnormality so confidence changes across inputs."""
    probability = float(np.max(prediction_proba))
    margin = float(abs(prediction_proba[1] - prediction_proba[0]))

    hb_target = 14.5 if gender.lower() == "male" else 13.0
    deviation_score = 0.0
    deviation_score += min(abs(hemoglobin - hb_target) / 4.0, 1.5)
    deviation_score += min(abs(mcv - 90.0) / 20.0, 1.5)
    deviation_score += min(abs(mch - 30.0) / 8.0, 1.2)
    deviation_score += min(abs(mchc - 34.0) / 6.0, 1.2)

    risk_factor = risk_index / 100.0

    if prediction == 1:
        confidence = (
            56.0
            + (probability * 18.0)
            + (margin * 14.0)
            + (risk_factor * 12.0)
            + (deviation_score * 4.5)
        )
    else:
        normality_bonus = max(0.0, 1.0 - deviation_score / 4.5)
        confidence = (
            58.0
            + (probability * 16.0)
            + (margin * 12.0)
            + (normality_bonus * 8.0)
            - (risk_factor * 8.0)
        )

    return round(max(min(confidence, 99.0), 52.0), 2)


def build_svm_prediction_result(gender, hemoglobin, mcv, mch, mchc, source=None, *, model, scaler):
    """Run legacy inference with an explicitly supplied SVM and scaler."""
    risk_index = calculate_anaemia_risk_index(gender, hemoglobin, mcv, mch, mchc)

    gender_encoded = 1 if gender.lower() == "male" else 0
    features = np.array([[gender_encoded, hemoglobin, mcv, mch, mchc]])
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    prediction_proba = model.predict_proba(features_scaled)[0]
    confidence = calculate_dynamic_confidence(
        prediction_proba, prediction, risk_index, gender, hemoglobin, mcv, mch, mchc
    )

    causes, recommendations, precautions = get_clinical_recommendations(
        prediction, confidence, risk_index, gender, hemoglobin, mcv, mch, mchc
    )

    input_data = {
        "gender": gender,
        "hemoglobin": hemoglobin,
        "mcv": mcv,
        "mch": mch,
        "mchc": mchc,
    }
    if source:
        input_data["source"] = source

    return {
        "prediction": "Anaemic" if prediction == 1 else "Non-Anaemic",
        "confidence": confidence,
        "risk_index": risk_index,
        "causes": causes,
        "recommendations": recommendations,
        "precautions": precautions,
        "input_data": input_data,
        "prediction_proba": prediction_proba,
        "features": features,
        "features_scaled": features_scaled,
    }


def build_prediction_result(gender, hemoglobin, mcv, mch, mchc, source=None):
    """Run the primary PyTorch model and package the existing response payload."""
    gender = validate_gender(gender)
    risk_index = calculate_anaemia_risk_index(gender, hemoglobin, mcv, mch, mchc)
    gender_encoded = 0 if gender.lower() == "male" else 1
    inference = predict_pytorch({
        "Gender": gender_encoded,
        "Hemoglobin": hemoglobin,
        "MCH": mch,
        "MCHC": mchc,
        "MCV": mcv,
    })
    prediction = inference["prediction_index"]
    prediction_proba = inference["prediction_proba"]
    confidence = calculate_dynamic_confidence(
        prediction_proba, prediction, risk_index, gender, hemoglobin, mcv, mch, mchc
    )

    causes, recommendations, precautions = get_clinical_recommendations(
        prediction, confidence, risk_index, gender, hemoglobin, mcv, mch, mchc
    )

    input_data = {
        "gender": gender,
        "hemoglobin": hemoglobin,
        "mcv": mcv,
        "mch": mch,
        "mchc": mchc,
    }
    if source:
        input_data["source"] = source

    return {
        "prediction": "Anaemic" if prediction == 1 else "Non-Anaemic",
        "confidence": confidence,
        "risk_index": risk_index,
        "causes": causes,
        "recommendations": recommendations,
        "precautions": precautions,
        "input_data": input_data,
        "prediction_proba": prediction_proba,
        "features": inference["features"],
        "features_scaled": inference["features_scaled"],
    }

def get_clinical_recommendations(prediction, confidence, risk_index, gender=None, hemoglobin=None, mcv=None, mch=None, mchc=None):
    """Generate clinical recommendations based on prediction and specific values"""
    if prediction == 1:  # Anaemic
        causes = []
        recommendations = []
        precautions = []

        if hemoglobin is not None:
            if hemoglobin < 8:
                causes.append("Severe anaemia pattern with markedly reduced hemoglobin")
                precautions.append("Urgent medical review is recommended for severe low hemoglobin values")
                precautions.append("Monitor for dizziness, chest discomfort, breathlessness, or near-syncope")
                recommendations.append("Consider emergency evaluation and transfusion assessment if symptomatic")
            elif hemoglobin < 10:
                causes.append("Moderate anaemia with clinically significant hemoglobin reduction")
                precautions.append("Monitor for fatigue, exertional dyspnea, palpitations, and reduced exercise tolerance")
                recommendations.append("Arrange prompt clinician follow-up with repeat CBC and symptom review")
            elif hemoglobin < 12:
                causes.append("Mild anaemia with early reduction in oxygen-carrying capacity")
                precautions.append("Track progressive fatigue, poor concentration, or worsening weakness")
                recommendations.append("Repeat blood counts after initial workup and correlate with symptoms")

        if mcv is not None:
            if mcv < 80:
                causes.append("Microcytic pattern suggests iron deficiency or chronic blood loss")
                recommendations.append("Order ferritin, serum iron, TIBC, and evaluate for occult blood loss")
            elif mcv > 100:
                causes.append("Macrocytic pattern raises concern for vitamin B12 or folate deficiency")
                recommendations.append("Check vitamin B12, folate, thyroid profile, and medication history")
            else:
                causes.append("Normocytic pattern may reflect chronic disease, mixed deficiency, or early anaemia")
                recommendations.append("Correlate with inflammatory markers, renal profile, and reticulocyte count")

        if mch is not None and mch < 27:
            causes.append("Low MCH indicates reduced hemoglobin content within red blood cells")
            recommendations.append("Review iron availability and red cell production status")
        elif mch is not None and mch > 33:
            causes.append("Elevated MCH may align with macrocytic red cell changes")

        if mchc is not None and mchc < 32:
            causes.append("Low MCHC supports a hypochromic picture, often seen with iron deficiency")
            precautions.append("Assess dietary iron intake and any ongoing blood loss history")
        elif mchc is not None and mchc > 36:
            causes.append("High MCHC may warrant review for red cell membrane or hemolytic processes")
            recommendations.append("Consider peripheral smear and hemolysis markers if clinically indicated")

        if risk_index > 70:
            recommendations.append("Perform urgent CBC with differential, peripheral smear, and reticulocyte count")
        elif risk_index > 40:
            recommendations.append("Complete focused anaemia workup with CBC trend review and deficiency screening")
        else:
            recommendations.append("Use short-interval follow-up testing to confirm persistence and direction of change")

        if not precautions:
            precautions.append("Monitor symptoms closely and escalate if weakness or breathlessness worsens")
            precautions.append("Avoid delaying follow-up if repeat values continue to fall")

    else:  # Non-anaemic
        causes = []
        recommendations = []
        precautions = []

        if hemoglobin is not None:
            causes.append("Current hemoglobin does not meet the model threshold for anaemia")
            normal_low = 13.0 if (gender or "").lower() == "male" else 12.0
            if hemoglobin < normal_low + 0.5:
                causes.append("Hemoglobin is near the lower range, so trend monitoring is still useful")
                recommendations.append("Repeat CBC in follow-up if symptoms persist or risk factors are present")
            elif hemoglobin > 16.5:
                causes.append("Hemoglobin is preserved and may reflect good oxygen-carrying capacity or dehydration")
                precautions.append("Interpret elevated hemoglobin alongside hydration status and clinical context")
            else:
                causes.append("Hemoglobin is within a relatively stable range for a non-anaemic result")

        if mcv is not None:
            if mcv < 80:
                causes.append("Low MCV without model-classified anaemia may suggest early iron deficiency")
                recommendations.append("Check iron studies even though the current prediction is non-anaemic")
            elif mcv > 100:
                causes.append("High MCV may indicate macrocytosis despite the non-anaemic prediction")
                recommendations.append("Review vitamin B12, folate status, liver function, and medications")
            else:
                causes.append("MCV is in the expected range, supporting stable red cell size")

        if mch is not None:
            if mch < 27:
                causes.append("Low MCH may reflect early hypochromic change before overt anaemia")
                precautions.append("Watch for developing fatigue or falling hemoglobin on future testing")
            elif mch > 33:
                causes.append("Higher MCH may align with larger red blood cells and should be trended")

        if mchc is not None:
            if mchc < 32:
                causes.append("Low MCHC can be seen in early iron-restricted erythropoiesis")
                recommendations.append("Support iron intake review and clinical correlation")
            elif mchc > 36:
                causes.append("High MCHC may justify smear review if other abnormalities are present")

        if risk_index >= 40:
            recommendations.append("Maintain closer follow-up because composite risk markers are not fully ideal")
            precautions.append("Do not ignore persistent symptoms even with a non-anaemic prediction")
        else:
            recommendations.append("Continue routine wellness screening and balanced nutritional intake")
            precautions.append("Maintain hydration, nutrition, and periodic preventive blood testing")

        if not recommendations:
            recommendations.append("Continue regular health check-ups and periodic CBC review")
        if not precautions:
            precautions.append("Report new fatigue, pallor, or exertional breathlessness if they develop")

    def unique_keep_order(items):
        seen = set()
        ordered = []
        for item in items:
            if item and item not in seen:
                seen.add(item)
                ordered.append(item)
        return ordered

    causes = unique_keep_order(causes)
    recommendations = unique_keep_order(recommendations)
    precautions = unique_keep_order(precautions)
    
    return causes, recommendations, precautions

# ---------------- HOME ----------------
@app.route("/")
def index():
    return render_template("index.html")


# ---------------- AUTH ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = find_user_by_email(email)
        if not user or user["password_hash"] != hash_password(password):
            flash("Invalid email or password.", "danger")
            return render_template("auth.html", auth_mode="login")

        session["user_email"] = user["email"]
        session["user_name"] = user["full_name"]
        flash(f"Welcome back, {user['full_name']}.", "success")
        return redirect(url_for("prediction"))

    return render_template("auth.html", auth_mode="login")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        contact_number = request.form.get("contact_number", "").strip()
        password = request.form.get("password", "")

        if not all([username, full_name, email, contact_number, password]):
            flash("Please fill in all sign-up fields.", "warning")
            return render_template("auth.html", auth_mode="signup")

        if find_user_by_email(email):
            flash("An account with this email already exists.", "warning")
            return render_template("auth.html", auth_mode="signup")

        users = load_users()
        users.append({
            "username": username,
            "full_name": full_name,
            "email": email,
            "contact_number": contact_number,
            "password_hash": hash_password(password),
        })
        save_users(users)

        session["user_email"] = email
        session["user_name"] = full_name
        flash("Account created successfully. You are now signed in.", "success")
        return redirect(url_for("prediction"))

    return render_template("auth.html", auth_mode="signup")


@app.route("/signup/google")
def signup_google():
    return redirect("https://accounts.google.com/signup/v2/createaccount?flowName=GlifWebSignIn&flowEntry=SignUp")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))

# ---------------- PREDICTION ----------------
@app.route("/prediction", methods=["GET"])
def prediction():
    return render_template("prediction.html")

@app.route("/upload_pdf", methods=["GET", "POST"])
def upload_pdf():
    if request.method == "GET":
        return render_template("prediction.html")
    
    # Reserve space for multipart headers and the gender field.
    request.max_content_length = PDF_MAX_BYTES + 64 * 1024
    try:
        file = request.files.get("pdf_file")
        if file is None:
            return render_template("result.html", error="No PDF file uploaded")
        if not file.filename:
            return render_template("result.html", error="No file selected")
        if not file.filename.lower().endswith(".pdf"):
            return render_template("result.html", error="Invalid file format. Please upload a PDF file.")

        data = file.read(PDF_MAX_BYTES + 1)
        if len(data) > PDF_MAX_BYTES:
            raise RequestEntityTooLarge()
        if not data:
            return render_template("result.html", error="The uploaded PDF is empty.")
        if b"%PDF-" not in data[:1024]:
            return render_template("result.html", error="Invalid file format. Please upload a PDF file.")

        with fitz.open(stream=data, filetype="pdf") as document:
            if not document.is_pdf:
                return render_template("result.html", error="Invalid file format. Please upload a PDF file.")
            if document.needs_pass:
                return render_template("result.html", error="This PDF is password-protected. Please upload an unlocked copy.")
            if document.page_count > PDF_MAX_PAGES:
                return render_template("result.html", error=f"Please upload a PDF with at most {PDF_MAX_PAGES} pages."), 413
            text = "\n".join(page.get_text() for page in document)

        if not text.strip():
            return render_template(
                "result.html",
                error="This PDF appears to contain no selectable text. If it is a scanned image report, please use a text-based PDF or enter the values manually.",
            )

        extracted = extract_values_from_pdf(text)
        required_fields = ("hemoglobin", "mcv", "mch", "mchc")
        missing_fields = [field.upper() for field in required_fields if field not in extracted]
        if not extracted:
            return render_template(
                "result.html",
                error="Could not extract Hb, MCV, MCH, or MCHC from this PDF. Please upload a clearer lab report or use manual entry.",
            )
        if missing_fields:
            return render_template(
                "result.html",
                error=f"PDF upload found some values, but these fields are still missing: {', '.join(missing_fields)}. Please use a clearer report or enter the values manually.",
            )

        gender = extract_gender_from_pdf(text)
        if gender is None:
            gender = request.form.get("gender")
        gender = validate_gender(gender)
        result = build_prediction_result(gender, **extracted, source="PDF Upload")
        result.pop("prediction_proba", None)
        result.pop("features", None)
        result.pop("features_scaled", None)
        return render_template("result.html", result=result)

    except RequestEntityTooLarge:
        return render_template("result.html", error=f"Please upload a PDF no larger than {PDF_MAX_BYTES // (1024 * 1024)} MB."), 413
    except fitz.FileDataError:
        app.logger.warning("PDF parser rejected the document")
        return render_template("result.html", error="This PDF is corrupt or unreadable. Please upload a valid PDF.")
    except GenderValidationError as exc:
        return render_template("result.html", error=str(exc))
    except PyTorchInferenceError:
        app.logger.error("PyTorch PDF prediction failed")
        return render_template("result.html", error="The prediction model is temporarily unavailable. Please try again later.")
    except Exception as exc:
        app.logger.error("PDF processing failed (%s)", type(exc).__name__)
        return render_template("result.html", error="Could not process this PDF. Please upload a valid PDF or enter the values manually.")


@app.route("/test")
def test_form():
    return render_template("test_form.html")

@app.route("/simple")
def simple_test():
    return render_template("simple_test.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        gender = validate_gender(request.form.get('gender'))
        hemoglobin = float(request.form['hemoglobin'])
        mcv = float(request.form['mcv'])
        mch = float(request.form['mch'])
        mchc = float(request.form['mchc'])

        result = build_prediction_result(gender, hemoglobin, mcv, mch, mchc)

        result.pop('prediction_proba', None)
        result.pop('features', None)
        result.pop('features_scaled', None)
        return render_template('result.html', result=result)

    except GenderValidationError as exc:
        return render_template('result.html', error=str(exc))
    except PyTorchInferenceError as exc:
        app.logger.error("PyTorch form prediction failed (%s)", type(exc.__cause__ or exc).__name__)
        return render_template('result.html', error="The prediction model is temporarily unavailable. Please try again later.")
    except Exception as exc:
        app.logger.error("Manual prediction request failed (%s)", type(exc).__name__)
        return render_template('result.html', error="Could not process this prediction. Please check your inputs and try again.")

# ---------------- DASHBOARD ----------------
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

# ---------------- EXPLAIN AI ----------------
@app.route("/explain")
def explain():
    return render_template("explain.html")

# ---------------- COUNTERFACTUAL ----------------
@app.route("/counterfactual")
def counterfactual():
    return render_template("counterfactual.html")

# ---------------- CLINICAL ----------------
@app.route("/clinical")
def clinical():
    return render_template("clinical.html")

# ---------------- INSIGHTS ----------------
@app.route("/insights")
def insights():
    return render_template("insights.html")

# ---------------- ABOUT ----------------
@app.route("/about")
def about():
    return render_template("about.html")

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
