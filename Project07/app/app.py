from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import json
from pathlib import Path


app = Flask(__name__)


# ==================================================
# PROJECT ROOT
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ==================================================
# LOAD TRAINED MODEL
# ==================================================

model = joblib.load(
    PROJECT_ROOT / "models" / "model.joblib"
)


# ==================================================
# LOAD EVALUATION RESULTS
# ==================================================

with open(
    PROJECT_ROOT / "reports" / "evaluation.json",
    "r",
    encoding="utf-8"
) as f:
    evaluation = json.load(f)


# ==================================================
# DECISION THRESHOLD
# ==================================================

# Automatically use the best threshold
# selected from the validation set
THRESHOLD = evaluation["validation"]["selected_threshold"]


# ==================================================
# REQUIRED FIELDS
# ==================================================

REQUIRED_FIELDS = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "campaign",
    "pdays",
    "previous",
    "poutcome"
]


# ==================================================
# VALID VALUES
# ==================================================

VALID_VALUES = {

    "job": [
        "admin.",
        "blue-collar",
        "entrepreneur",
        "housemaid",
        "management",
        "retired",
        "self-employed",
        "services",
        "student",
        "technician",
        "unemployed",
        "unknown"
    ],

    "marital": [
        "divorced",
        "married",
        "single"
    ],

    "education": [
        "primary",
        "secondary",
        "tertiary",
        "unknown"
    ],

    "default": [
        "no",
        "yes"
    ],

    "housing": [
        "no",
        "yes"
    ],

    "loan": [
        "no",
        "yes"
    ],

    "contact": [
        "cellular",
        "telephone",
        "unknown"
    ],

    "month": [
        "apr",
        "aug",
        "dec",
        "feb",
        "jan",
        "jul",
        "jun",
        "mar",
        "may",
        "nov",
        "oct",
        "sep"
    ],

    "poutcome": [
        "failure",
        "other",
        "success",
        "unknown"
    ]
}


# ==================================================
# NUMERIC FIELDS
# ==================================================

NUMERIC_FIELDS = [
    "age",
    "balance",
    "day",
    "campaign",
    "pdays",
    "previous"
]


# ==================================================
# HOME PAGE
# ==================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        threshold=THRESHOLD,
        evaluation=evaluation["test"]
    )


# ==================================================
# API: SCORE
# ==================================================

@app.route("/api/score", methods=["POST"])
def score():

    # --------------------------------------------------
    # Read JSON request
    # --------------------------------------------------

    data = request.get_json(
        silent=True
    )

    if data is None:

        return jsonify({
            "error": "Request body must be valid JSON."
        }), 400


    # --------------------------------------------------
    # Check missing fields
    # --------------------------------------------------

    missing_fields = [
        field
        for field in REQUIRED_FIELDS
        if field not in data
    ]

    if missing_fields:

        return jsonify({
            "error": "Missing required fields.",
            "missing_fields": missing_fields
        }), 400


    # --------------------------------------------------
    # Check extra fields
    # --------------------------------------------------

    extra_fields = [
        field
        for field in data
        if field not in REQUIRED_FIELDS
    ]

    if extra_fields:

        return jsonify({
            "error": "Unknown fields.",
            "extra_fields": extra_fields
        }), 400


    # --------------------------------------------------
    # Check numeric types
    # --------------------------------------------------

    for field in NUMERIC_FIELDS:

        if (
            not isinstance(
                data[field],
                (int, float)
            )
            or isinstance(
                data[field],
                bool
            )
        ):

            return jsonify({
                "error": (
                    f"Field '{field}' "
                    "must be a number."
                )
            }), 400


    # --------------------------------------------------
    # Check categorical values
    # --------------------------------------------------

    for field, valid_values in VALID_VALUES.items():

        if data[field] not in valid_values:

            return jsonify({
                "error": (
                    f"Invalid value for '{field}'."
                ),
                "allowed_values": valid_values
            }), 400


    # --------------------------------------------------
    # Check value ranges
    # --------------------------------------------------

    if not 18 <= data["age"] <= 100:

        return jsonify({
            "error": (
                "Field 'age' must be "
                "between 18 and 100."
            )
        }), 400


    if not 1 <= data["day"] <= 31:

        return jsonify({
            "error": (
                "Field 'day' must be "
                "between 1 and 31."
            )
        }), 400


    if data["campaign"] < 1:

        return jsonify({
            "error": (
                "Field 'campaign' "
                "must be >= 1."
            )
        }), 400


    if data["pdays"] < -1:

        return jsonify({
            "error": (
                "Field 'pdays' "
                "must be >= -1."
            )
        }), 400


    if data["previous"] < 0:

        return jsonify({
            "error": (
                "Field 'previous' "
                "must be >= 0."
            )
        }), 400


    # ==================================================
    # CREATE INPUT DATAFRAME
    # ==================================================

    X = pd.DataFrame([
        data
    ])


    # ==================================================
    # MODEL PREDICTION
    # ==================================================

    probability = model.predict_proba(
        X
    )[0][1]


    prediction = int(
        probability >= THRESHOLD
    )


    # ==================================================
    # API RESPONSE
    # ==================================================

    return jsonify({

        "probability": round(
            float(probability),
            4
        ),

        "threshold": THRESHOLD,

        "prediction": prediction,

        "decision": (
            "yes"
            if prediction == 1
            else "no"
        )
    })


# ==================================================
# RUN APPLICATION
# ==================================================

if __name__ == "__main__":

    print(
        f"Using threshold: {THRESHOLD:.2f}"
    )

    app.run(
        debug=True
    )