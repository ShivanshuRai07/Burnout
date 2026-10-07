"""Vercel serverless Flask backend for the Workforce Burnout Analytics dashboard."""
from __future__ import annotations

import io
import json
import os
import re
import sys
from pathlib import Path

# Make burnout_platform importable from repo root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, request, send_file, send_from_directory
import joblib
import numpy as np
import pandas as pd

# ─── Cross-version Scikit-Learn Unpickler Compatibility ──────────────────────
try:
    import sklearn._loss._loss as _sklearn_loss
    sys.modules["_loss"] = _sklearn_loss
except Exception:
    pass

_orig_find_class = joblib.numpy_pickle.NumpyUnpickler.find_class
def _compat_find_class(self, module, name):
    if module == "_loss":
        module = "sklearn._loss._loss"
    return _orig_find_class(self, module, name)
joblib.numpy_pickle.NumpyUnpickler.find_class = _compat_find_class

from burnout_platform.features import engineer_features

# ─── Constants ────────────────────────────────────────────────────────────────

FEATURES = [
    "caffeine_intake", "company_size", "deadlines_missed", "experience_years", "job_role",
    "job_satisfaction", "manager_support", "meetings_per_day", "overtime_hours",
    "physical_activity_days", "screen_time_hours", "sleep_hours", "social_support_score",
    "stress_level", "work_hours_per_week", "work_life_balance", "work_mode",
]
CORE_UPLOAD_FIELDS = {
    "work_hours_per_week", "overtime_hours", "meetings_per_day", "deadlines_missed",
    "experience_years", "job_role", "work_mode"
}
OPTIONAL_WORKPLACE_FIELDS = set(FEATURES) - CORE_UPLOAD_FIELDS

DURATION_BINS   = [0, 35, 40, 50, 60, 999]
DURATION_LABELS = ["Undertime (<35h)", "Standard (35–40h)", "Overtime (41–50h)", "Extended (51–60h)", "Extreme (>60h)"]

CITY_TO_STATE: dict[str, str] = {
    "new delhi": "Delhi", "delhi": "Delhi",
    "noida": "Uttar Pradesh", "ghaziabad": "Uttar Pradesh",
    "gurugram": "Haryana", "faridabad": "Haryana",
    "mumbai": "Maharashtra", "pune": "Maharashtra", "nagpur": "Maharashtra",
    "nashik": "Maharashtra", "thane": "Maharashtra", "aurangabad": "Maharashtra",
    "bengaluru": "Karnataka", "bangalore": "Karnataka", "hubballi": "Karnataka",
    "mysuru": "Karnataka", "mangaluru": "Karnataka", "belagavi": "Karnataka",
    "chennai": "Tamil Nadu", "coimbatore": "Tamil Nadu", "salem": "Tamil Nadu",
    "tiruchirappalli": "Tamil Nadu", "madurai": "Tamil Nadu",
    "hyderabad": "Telangana", "warangal": "Telangana", "nizamabad": "Telangana",
    "karimnagar": "Telangana",
    "ahmedabad": "Gujarat", "surat": "Gujarat", "vadodara": "Gujarat",
    "rajkot": "Gujarat", "gandhinagar": "Gujarat",
    "jaipur": "Rajasthan", "jodhpur": "Rajasthan", "udaipur": "Rajasthan",
    "kota": "Rajasthan", "ajmer": "Rajasthan", "bikaner": "Rajasthan",
    "lucknow": "Uttar Pradesh", "kanpur": "Uttar Pradesh", "agra": "Uttar Pradesh",
    "varanasi": "Uttar Pradesh", "meerut": "Uttar Pradesh", "allahabad": "Uttar Pradesh",
    "prayagraj": "Uttar Pradesh",
    "kolkata": "West Bengal", "howrah": "West Bengal", "durgapur": "West Bengal",
    "siliguri": "West Bengal", "asansol": "West Bengal",
    "panipat": "Haryana", "rohtak": "Haryana", "hisar": "Haryana", "ambala": "Haryana",
    "ludhiana": "Punjab", "amritsar": "Punjab", "jalandhar": "Punjab",
    "patiala": "Punjab", "chandigarh": "Punjab",
    "kochi": "Kerala", "thiruvananthapuram": "Kerala", "kozhikode": "Kerala",
    "thrissur": "Kerala", "kollam": "Kerala",
    "bhopal": "Madhya Pradesh", "indore": "Madhya Pradesh", "jabalpur": "Madhya Pradesh",
    "gwalior": "Madhya Pradesh", "ujjain": "Madhya Pradesh",
    "ranchi": "Jharkhand", "jamshedpur": "Jharkhand", "dhanbad": "Jharkhand",
    "bhubaneswar": "Odisha", "cuttack": "Odisha", "rourkela": "Odisha",
    "patna": "Bihar", "gaya": "Bihar", "muzaffarpur": "Bihar", "bhagalpur": "Bihar",
    "guwahati": "Assam", "silchar": "Assam", "dibrugarh": "Assam", "jorhat": "Assam",
    "dehradun": "Uttarakhand", "haridwar": "Uttarakhand", "haldwani": "Uttarakhand",
    "panaji": "Goa", "margao": "Goa", "vasco da gama": "Goa",
    "raipur": "Chhattisgarh", "bilaspur": "Chhattisgarh", "durg": "Chhattisgarh",
    "visakhapatnam": "Andhra Pradesh", "vijayawada": "Andhra Pradesh", "guntur": "Andhra Pradesh",
    "shimla": "Himachal Pradesh", "manali": "Himachal Pradesh",
    "srinagar": "Jammu & Kashmir", "jammu": "Jammu & Kashmir",
}

# ─── Lazy-loaded global state ─────────────────────────────────────────────────

_state: "DashboardState | None" = None


def get_state() -> "DashboardState":
    global _state
    if _state is None:
        artifacts = ROOT / "artifacts"
        pkl_path = artifacts / "master_workforce_dataset.pkl.gz"
        data_path = pkl_path if pkl_path.exists() else artifacts / "master_workforce_dataset.csv"
        _state = DashboardState(data_path, artifacts)
    return _state


# ─── Helper functions ─────────────────────────────────────────────────────────

def _parse_multipart(data: bytes, content_type: str):
    m = re.search(r'boundary=([^;\s]+)', content_type)
    if not m:
        return
    boundary = ("--" + m.group(1).strip('"')).encode()
    for chunk in data.split(boundary)[1:]:
        if chunk in (b'--', b'--\r\n', b''):
            continue
        if b'\r\n\r\n' not in chunk:
            continue
        header_block, body = chunk.split(b'\r\n\r\n', 1)
        body = body.rstrip(b'\r\n')
        header_text = header_block.decode('utf-8', errors='replace')
        name_m = re.search(r'name="([^"]*)"', header_text, re.IGNORECASE)
        file_m = re.search(r'filename="([^"]*)"', header_text, re.IGNORECASE)
        yield (name_m.group(1) if name_m else ''), (file_m.group(1) if file_m else ''), body


def _summarize_groups(aggregate: dict, limit: int = 20) -> list:
    result = [
        {"name": name, "mean_score": round(v["score_sum"] / v["count"], 2),
         "elevated_rate": round(v["elevated"] / v["count"] * 100, 1), "employees": v["count"]}
        for name, v in aggregate.items() if v["count"] > 0
    ]
    return sorted(result, key=lambda x: (-x["mean_score"], -x["employees"]))[:limit]


def _agg_chunk(series_vals: pd.Series, score: np.ndarray, is_elevated: np.ndarray, agg_dict: dict) -> None:
    tmp = pd.DataFrame({"g": series_vals.values, "s": score, "e": is_elevated})
    for grp, vals in tmp.groupby("g", sort=False).agg(ss=("s", "sum"), cnt=("s", "size"), el=("e", "sum")).iterrows():
        cur = agg_dict.setdefault(str(grp), {"score_sum": 0.0, "count": 0, "elevated": 0})
        cur["score_sum"] += float(vals["ss"]); cur["count"] += int(vals["cnt"]); cur["elevated"] += int(vals["el"])


# ─── Dashboard State ──────────────────────────────────────────────────────────

class DashboardState:
    def __init__(self, data_path: Path, artifacts: Path) -> None:
        if str(data_path).endswith((".pkl", ".pkl.gz")):
            self.df = pd.read_pickle(data_path)
        else:
            self.df = pd.read_csv(data_path)
        self.score_model = joblib.load(artifacts / "burnout_score_model.joblib")
        self.risk_model  = joblib.load(artifacts / "elevated_risk_model.joblib")
        self.threshold   = json.loads((artifacts / "metrics.json").read_text())["risk_model"]["threshold"]
        self.uploaded_employees: dict = {}
        self.employee_df: pd.DataFrame | None = None
        self.uploaded_raw_df: pd.DataFrame | None = None
        self._cached_summary: dict | None = None
        self._cached_defaults: dict | None = None

    def summary(self) -> dict:
        score = self.df["burnout_score"]
        return {
            "employees": int(len(self.df)),
            "mean_burnout": round(float(score.mean()), 2),
            "elevated_rate": round(float((score >= self.threshold).mean() * 100), 1),
            "mean_stress": round(float(self.df["stress_level"].mean()), 2),
            "threshold": self.threshold,
            "role_burnout": self.df.groupby("job_role")["burnout_score"].mean().sort_values(ascending=False).round(2).to_dict(),
            "mode_burnout": self.df.groupby("work_mode")["burnout_score"].mean().sort_values(ascending=False).round(2).to_dict(),
            "risk_bands": {
                "Low (< 2.5)": int((score < 2.5).sum()),
                "Moderate (2.5–3.7)": int(((score >= 2.5) & (score < self.threshold)).sum()),
                "Elevated (≥ 3.7)": int((score >= self.threshold).sum()),
            },
            "defaults": self.df[FEATURES].median(numeric_only=True).to_dict(),
        }

    def predict(self, payload: dict) -> dict:
        defaults = self.default_features()
        for name in FEATURES:
            if name in payload and payload[name] not in (None, ""):
                defaults[name] = payload[name]
        numeric = [x for x in FEATURES if x not in {"company_size", "job_role", "work_mode"}]
        for name in numeric:
            defaults[name] = float(defaults[name])
        row = pd.DataFrame([defaults])
        row_eng = engineer_features(row)
        score = float(self.score_model.predict(row_eng)[0])
        risk  = float(self.risk_model.predict_proba(row_eng)[0, 1])
        band  = "Elevated" if score >= self.threshold else ("Moderate" if score >= 2.5 else "Low")
        return {"burnout_score": round(score, 2), "elevated_risk_probability": round(risk * 100, 1),
                "risk_band": band, "threshold": self.threshold,
                "notice": "Decision-support only. Use supportive interventions, not adverse employment actions."}

    def default_features(self) -> dict:
        defaults = self.df[FEATURES].median(numeric_only=True).to_dict()
        for col in ["company_size", "job_role", "work_mode"]:
            defaults[col] = self.df[col].mode().iloc[0] if col in self.df and not self.df[col].mode().empty else "Unknown"
        return defaults

    def search_employee(self, emp_id: str) -> dict | None:
        emp_id = emp_id.strip()
        if emp_id in self.uploaded_employees:
            return self.uploaded_employees[emp_id]
        if self.uploaded_raw_df is not None:
            for col in ["employee_id", "employeeid", "emp_id", "empid", "EmployeeID"]:
                if col in self.uploaded_raw_df.columns:
                    match = self.uploaded_raw_df[self.uploaded_raw_df[col].astype(str).str.strip().str.lower() == emp_id.lower()]
                    if not match.empty:
                        res = self._build_employee_result(match.iloc[0])
                        self.uploaded_employees[emp_id] = res
                        return res
        for col in ["employee_id", "employeeid", "emp_id", "empid", "EmployeeID"]:
            if col in self.df.columns:
                match = self.df[self.df[col].astype(str).str.strip().str.lower() == emp_id.lower()]
                if not match.empty:
                    return self._build_employee_result(match.iloc[0])
        return None

    def _build_employee_result(self, row: pd.Series) -> dict:
        defaults = self.default_features()
        numeric  = [x for x in FEATURES if x not in {"company_size", "job_role", "work_mode"}]
        feat_row = {}
        for name in FEATURES:
            val = row.get(name, defaults.get(name))
            if not isinstance(val, str) and pd.isna(val):
                val = defaults.get(name)
            feat_row[name] = val
        for name in numeric:
            try:
                feat_row[name] = float(feat_row[name])
            except (TypeError, ValueError):
                feat_row[name] = float(defaults.get(name, 0))
        pred_df = pd.DataFrame([feat_row])
        pred_df_eng = engineer_features(pred_df)
        score = float(self.score_model.predict(pred_df_eng)[0])
        risk  = float(self.risk_model.predict_proba(pred_df_eng)[0, 1])
        band  = "Elevated" if score >= self.threshold else ("Moderate" if score >= 2.5 else "Low")
        meta  = {col: str(row[col]) for col in
                 ["employee_id","employeeid","emp_id","empid","EmployeeID","gender","sex","age",
                  "city","state","State","employee_location","location","job_role","work_mode","experience_years","company_size"]
                 if col in row.index and row[col] is not None and str(row[col]) not in ("nan","None","")}
        return {"features": {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in feat_row.items()},
                "metadata": meta, "burnout_score": round(score, 2),
                "elevated_risk_probability": round(risk * 100, 1), "risk_band": band, "threshold": self.threshold}

    def get_employee_table(self, page=0, page_size=25, search="", filter_band="",
                           filter_role="", filter_mode="", filter_city="",
                           sort_col="score", sort_dir="desc", export=False):
        empty = {"total": 0, "rows": [], "page": 0, "page_size": page_size, "unique_roles": [], "unique_modes": []}
        if self.employee_df is None or self.employee_df.empty:
            return pd.DataFrame() if export else empty
        df = self.employee_df
        mask = pd.Series(True, index=df.index)
        if search:
            mask &= df["id"].astype(str).str.contains(search, case=False, na=False)
        if filter_band in ("Low", "Moderate", "Elevated"):
            mask &= df["band"] == filter_band
        if filter_role:
            mask &= df["role"] == filter_role
        if filter_mode:
            mask &= df["mode"] == filter_mode
        if filter_city:
            mask &= df["city"].astype(str).str.contains(filter_city, case=False, na=False)
        df = df[mask]
        VALID = {"id","role","mode","city","state","gender","hours","stress","overtime","score","prob","band","experience"}
        sc = sort_col if sort_col in df.columns and sort_col in VALID else "score"
        df = df.sort_values(sc, ascending=(sort_dir == "asc"), na_position="last")
        total = len(df)
        if export:
            return df
        page_size = max(1, min(page_size, 200))
        start = page * page_size
        page_df = df.iloc[start: start + page_size]
        def _safe(v):
            if isinstance(v, float) and np.isnan(v): return None
            if isinstance(v, (np.integer,)): return int(v)
            if isinstance(v, (np.floating,)): return float(v)
            return v
        rows = [{col: _safe(r[col]) for col in page_df.columns} for _, r in page_df.iterrows()]
        return {
            "total": total, "page": page, "page_size": page_size, "rows": rows,
            "unique_roles": sorted(self.employee_df["role"].dropna().unique().tolist()),
            "unique_modes": sorted(self.employee_df["mode"].dropna().unique().tolist()),
        }

    def analyse_upload(self, file_obj, filename: str) -> dict:
        if not filename.lower().endswith(".csv"):
            raise ValueError("Only CSV files are accepted.")
        file_obj.seek(0, 2); size = file_obj.tell(); file_obj.seek(0)
        if size > 500 * 1024 * 1024:
            raise ValueError("File exceeds 500 MB limit.")
        defaults = self.default_features()
        numeric  = [x for x in FEATURES if x not in {"company_size", "job_role", "work_mode"}]
        rows = 0; score_total = 0.0; prob_total = 0.0
        bands_total = {"Low": 0, "Moderate": 0, "Elevated": 0}
        role_agg = {}; loc_agg = {}; gender_agg = {}; mode_agg = {}; state_agg = {}
        dur_agg = {lbl: {"score_sum": 0.0, "count": 0, "elevated": 0} for lbl in DURATION_LABELS}
        present = None; loc_field = gender_field = emp_id_field = state_field = None
        self.uploaded_employees = {}
        emp_chunks: list[pd.DataFrame] = []
        raw_chunks: list[pd.DataFrame] = []
        for chunk in pd.read_csv(file_obj, chunksize=100_000):
            chunk.columns = [str(c).strip() for c in chunk.columns]
            col_map = {c: c.lower().replace(" ", "_") for c in chunk.columns}
            cl = chunk.rename(columns=col_map)
            if present is None:
                present = set(cl.columns)
                miss = sorted(CORE_UPLOAD_FIELDS - present)
                if miss: raise ValueError("Missing required fields: " + ", ".join(miss))
                loc_field    = next((f for f in ["city","employee_location","location"] if f in present), None)
                state_field  = next((f for f in ["state","province","region"] if f in present), None)
                gender_field = next((f for f in ["gender","sex"] if f in present), None)
                emp_id_field = next((f for f in ["employeeid","employee_id","emp_id","empid"] if f in present), None)
            rows += len(cl)
            if rows > 1_000_000: raise ValueError("File exceeds 1,000,000 rows.")
            all_def = {col: (self.df[col].median() if self.df[col].dtype.kind in "bifc"
                             else (self.df[col].mode().iloc[0] if not self.df[col].mode().empty else ""))
                       for col in self.df.columns}
            batch = pd.DataFrame(index=cl.index)
            for col in self.df.columns:
                lc = col.lower().replace(" ", "_")
                batch[col] = cl[lc].values if lc in cl.columns else (cl[col].values if col in cl.columns else all_def[col])
            for col in numeric:
                if col in batch.columns:
                    batch[col] = pd.to_numeric(batch[col], errors="coerce").fillna(all_def.get(col, 0))
            for col in {"company_size", "job_role", "work_mode"}:
                if col in batch.columns:
                    batch[col] = batch[col].fillna(all_def.get(col, "")).astype(str)
            try:
                batch_eng = engineer_features(batch)
                score = np.clip(self.score_model.predict(batch_eng), 0, 10)
                prob  = self.risk_model.predict_proba(batch_eng)[:, 1]
            except Exception as exc:
                return {"error": f"Analysis failed: {exc}"}
            band     = np.where(score >= self.threshold, "Elevated", np.where(score >= 2.5, "Moderate", "Low"))
            elevated = score >= self.threshold
            score_total += float(score.sum()); prob_total += float(prob.sum())
            for b in bands_total: bands_total[b] += int((band == b).sum())
            if "job_role" in cl.columns: _agg_chunk(cl["job_role"].fillna("Unknown"), score, elevated, role_agg)
            if loc_field and loc_field in cl.columns: _agg_chunk(cl[loc_field].fillna("Unknown"), score, elevated, loc_agg)
            if gender_field and gender_field in cl.columns: _agg_chunk(cl[gender_field].fillna("Unknown"), score, elevated, gender_agg)
            if "work_mode" in cl.columns: _agg_chunk(cl["work_mode"].fillna("Unknown"), score, elevated, mode_agg)
            state_series = None
            if state_field and state_field in cl.columns:
                state_series = cl[state_field].astype(str)
            elif loc_field and loc_field in cl.columns:
                state_series = cl[loc_field].apply(lambda c: CITY_TO_STATE.get(str(c).lower().strip()))
            if state_series is not None:
                valid = state_series.notna() & (state_series != "None") & (state_series != "nan")
                if valid.any():
                    _agg_chunk(state_series[valid].fillna("Unknown"), score[valid.values], elevated[valid.values], state_agg)
            if "work_hours_per_week" in cl.columns:
                hrs = pd.to_numeric(cl["work_hours_per_week"], errors="coerce").fillna(40)
                dlbl = pd.cut(hrs, bins=DURATION_BINS, labels=DURATION_LABELS, right=True)
                for lbl, vals in pd.DataFrame({"g": dlbl, "s": score, "e": elevated}).groupby("g", observed=True).agg(ss=("s","sum"),cnt=("s","size"),el=("e","sum")).iterrows():
                    cur = dur_agg[str(lbl)]
                    cur["score_sum"] += float(vals["ss"]); cur["count"] += int(vals["cnt"]); cur["elevated"] += int(vals["el"])
            def _col(name):
                return cl[name] if name and name in cl.columns else pd.Series([None]*len(cl), index=cl.index, dtype=object)
            def _num_col(name):
                c = cl[name] if name and name in cl.columns else pd.Series([None]*len(cl), index=cl.index, dtype=object)
                return pd.to_numeric(c, errors="coerce").round(1)
            emp_chunk_df = pd.DataFrame({
                "id":         _col(emp_id_field).astype(str),
                "role":       _col("job_role").fillna("").astype(str),
                "mode":       _col("work_mode").fillna("").astype(str),
                "city":       _col(loc_field).fillna("").astype(str),
                "state":      (state_series.fillna("").astype(str) if state_series is not None else pd.Series([""] * len(cl), index=cl.index)),
                "gender":     _col(gender_field).fillna("").astype(str),
                "hours":      _num_col("work_hours_per_week"),
                "stress":     _num_col("stress_level"),
                "overtime":   _num_col("overtime_hours"),
                "experience": _num_col("experience_years"),
                "score":      pd.Series(score.round(2), index=cl.index),
                "prob":       pd.Series((prob * 100).round(1), index=cl.index),
                "band":       pd.Series(band, index=cl.index),
            }, index=cl.index)
            emp_chunks.append(emp_chunk_df)
            raw_chunks.append(cl)

        if not rows: raise ValueError("Uploaded file has no employee rows.")
        self.employee_df = pd.concat(emp_chunks, ignore_index=True) if emp_chunks else None
        self.uploaded_raw_df = pd.concat(raw_chunks, ignore_index=True) if raw_chunks else None
        return {
            "detected_fields": sorted(present & set(FEATURES)),
            "optional_fields_not_supplied": sorted(OPTIONAL_WORKPLACE_FIELDS - present),
            "has_gender": gender_field is not None,
            "has_location": loc_field is not None,
            "has_state": bool(state_agg),
            "has_emp_id": emp_id_field is not None,
            "has_employee_table": self.employee_df is not None and not self.employee_df.empty,
            "employee_table_count": len(self.employee_df) if self.employee_df is not None else 0,
            "mean_burnout_score": round(score_total / rows, 2),
            "elevated_review_rate": round(bands_total["Elevated"] / rows * 100, 1),
            "mean_elevated_probability": round(prob_total / rows * 100, 1),
            "risk_distribution": bands_total,
            "role_analysis": _summarize_groups(role_agg),
            "location_analysis": _summarize_groups(loc_agg),
            "gender_analysis": _summarize_groups(gender_agg),
            "work_mode_analysis": _summarize_groups(mode_agg),
            "duration_analysis": _summarize_groups({k: v for k, v in dur_agg.items() if v["count"] > 0}),
            "state_analysis": _summarize_groups(state_agg, limit=30),
            "upload_progress_percent": 100,
            "initial_table": self.get_employee_table(page=0, page_size=25),
            "notice": "The uploaded file is streamed in batches and is not retained by this service.",
        }


# ─── Flask App ────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder=str(ROOT), static_url_path="")


@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/", methods=["GET"])
def index():
    return send_from_directory(str(ROOT), "dashboard.html", mimetype="text/html; charset=utf-8")


@app.route("/api/summary", methods=["GET"])
def api_summary():
    try:
        return jsonify(get_state().summary())
    except Exception as exc:
        import traceback
        return jsonify({"error": str(exc), "traceback": traceback.format_exc()}), 500


@app.route("/api/predict", methods=["POST", "OPTIONS"])
def api_predict():
    if request.method == "OPTIONS":
        return "", 204
    try:
        payload = request.get_json(force=True) or {}
        return jsonify(get_state().predict(payload))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/search-employee", methods=["GET"])
def api_search_employee():
    emp_id = request.args.get("id", "").strip()
    if not emp_id:
        return jsonify({"error": "Employee ID is required."}), 400
    result = get_state().search_employee(emp_id)
    if result is None:
        return jsonify({"error": f"Employee '{emp_id}' not found."}), 404
    return jsonify(result)


@app.route("/api/employee-table", methods=["GET"])
def api_employee_table():
    state = get_state()
    export = request.args.get("export") == "1"
    try:
        page = int(request.args.get("page", "0"))
        page_size = min(int(request.args.get("page_size", "25")), 200)
    except ValueError:
        page, page_size = 0, 25
    result = state.get_employee_table(
        page=page, page_size=page_size,
        search=request.args.get("search", ""),
        filter_band=request.args.get("filter_band", ""),
        filter_role=request.args.get("filter_role", ""),
        filter_mode=request.args.get("filter_mode", ""),
        filter_city=request.args.get("filter_city", ""),
        sort_col=request.args.get("sort_col", "score"),
        sort_dir=request.args.get("sort_dir", "desc"),
        export=export,
    )
    if export:
        buf = io.StringIO()
        if isinstance(result, pd.DataFrame):
            result.to_csv(buf, index=False)
        return buf.getvalue(), 200, {
            "Content-Type": "text/csv",
            "Content-Disposition": 'attachment; filename="employee_burnout_report.csv"',
        }
    return jsonify(result)


@app.route("/api/analyse-upload", methods=["POST", "OPTIONS"])
def api_analyse_upload():
    if request.method == "OPTIONS":
        return "", 204
    try:
        if "file" not in request.files:
            return jsonify({"error": "No CSV file received."}), 400
        f = request.files["file"]
        return jsonify(get_state().analyse_upload(f.stream, f.filename or "upload.csv"))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


# Vercel uses the `app` object as the WSGI handler
