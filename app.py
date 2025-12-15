from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

CSV_FILE = "/mnt/data/expense_data_final.csv"
PLOTS_DIR = os.path.join(BASE_DIR, "static", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = "change_this_secret_key"


def ensure_csv_exists():
    if not os.path.exists(CSV_FILE):
        sample = pd.DataFrame([
            {
                "name": "divya",
                "date": (pd.Timestamp.now() - pd.Timedelta(days=40)).strftime("%Y-%m-%d"),
                "category": "food",
                "amount": 250.0,
                "monthly_income": 30000,
                "description": "",
            },
            {
                "name": "divya",
                "date": (pd.Timestamp.now() - pd.Timedelta(days=20)).strftime("%Y-%m-%d"),
                "category": "transport",
                "amount": 120.0,
                "monthly_income": 30000,
                "description": "",
            },
            {
                "name": "admin",
                "date": (pd.Timestamp.now() - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
                "category": "bills",
                "amount": 2000.0,
                "monthly_income": 50000,
                "description": "",
            },
        ])
        os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)
        sample.to_csv(CSV_FILE, index=False)


def load_data():
    ensure_csv_exists()
    df = pd.read_csv(CSV_FILE)

    # Normalize column names
    rename_map = {}
    if "name" in df.columns and "person" not in df.columns:
        rename_map["name"] = "person"
    if "user" in df.columns and "person" not in df.columns:
        rename_map["user"] = "person"
    if "username" in df.columns and "person" not in df.columns:
        rename_map["username"] = "person"

    if "Date" in df.columns and "date" not in df.columns:
        rename_map["Date"] = "date"
    if "Amount" in df.columns and "amount" not in df.columns:
        rename_map["Amount"] = "amount"
    if "Category" in df.columns and "category" not in df.columns:
        rename_map["Category"] = "category"
    if "Description" in df.columns and "description" not in df.columns:
        rename_map["Description"] = "description"

    if rename_map:
        df = df.rename(columns=rename_map)

    # Fill missing columns
    if "person" not in df.columns:
        df["person"] = session.get("username", "unknown")
    if "date" not in df.columns:
        df["date"] = pd.Timestamp.now().strftime("%Y-%m-%d")
    if "category" not in df.columns:
        df["category"] = "other"
    if "amount" not in df.columns:
        df["amount"] = 0.0
    if "monthly_income" not in df.columns:
        df["monthly_income"] = np.nan
    if "description" not in df.columns:
        df["description"] = ""

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

    cols = ["person", "date", "category", "amount", "monthly_income", "description"]
    other = [c for c in df.columns if c not in cols]
    df = df[cols + other]

    return df.reset_index(drop=True)


def save_data(df):
    df.to_csv(CSV_FILE, index=False)


USERS = {"divya": "1234", "admin": "1234"}


def logged_in():
    return session.get("logged_in", False)


def prepare_monthly(df, person, months=None):
    pdf = df[df["person"] == person].copy()
    if pdf.empty:
        return None, None
    pdf["date"] = pd.to_datetime(pdf["date"], errors="coerce")

    pdf["year_month"] = pdf["date"].dt.to_period("M").astype(str)
    monthly = pdf.groupby("year_month").agg(
        total_expense=("amount", "sum"),
        avg_daily_expense=("amount", "mean"),
        transactions=("amount", "count"),
        monthly_income=("monthly_income", "first"),
    ).reset_index()
    monthly["month_start"] = pd.to_datetime(monthly["year_month"] + "-01", errors="coerce")
    monthly = monthly.sort_values("month_start").reset_index(drop=True)
    if months and months != "all":
        months = int(months)
        monthly = monthly.tail(months)
    return monthly, pdf


def save_plots(person_df, monthly, person):
    plot_paths = {}

    # Trend
    if monthly is not None and len(monthly) > 0:
        plt.figure(figsize=(8, 4))
        plt.plot(monthly["month_start"], monthly["total_expense"], marker="o")
        plt.title(f"{person} — Monthly Expense Trend")
        plt.ylabel("Amount (₹)")
        plt.tight_layout()
        p1 = os.path.join(PLOTS_DIR, f"{person}_trend.png")
        plt.savefig(p1)
        plt.close()
        plot_paths["trend"] = os.path.basename(p1)
    else:
        plot_paths["trend"] = None

    # Regression
    if monthly is not None and len(monthly) > 2:
        m2 = monthly.copy()
        m2["t"] = range(len(m2))
        plt.figure(figsize=(8, 4))
        try:
            import seaborn as sns
            sns.regplot(x="t", y="total_expense", data=m2)
        except Exception:
            plt.scatter(m2["t"], m2["total_expense"])
            z = np.polyfit(m2["t"], m2["total_expense"], 1)
            p = np.poly1d(z)
            plt.plot(m2["t"], p(m2["t"]), "--")
        plt.title("Regression Fit (Monthly)")
        plt.tight_layout()
        p2 = os.path.join(PLOTS_DIR, f"{person}_regplot.png")
        plt.savefig(p2)
        plt.close()
        plot_paths["regplot"] = os.path.basename(p2)
    else:
        plot_paths["regplot"] = None

    # Histogram
    plt.figure(figsize=(6, 4))
    try:
        import seaborn as sns
        sns.histplot(person_df["amount"], kde=True)
    except Exception:
        plt.hist(person_df["amount"].dropna(), bins=20)
    plt.title("Daily Spending Distribution")
    plt.tight_layout()
    p3 = os.path.join(PLOTS_DIR, f"{person}_hist.png")
    plt.savefig(p3)
    plt.close()
    plot_paths["hist"] = os.path.basename(p3)

    # Box
    plt.figure(figsize=(10, 4))
    try:
        import seaborn as sns
        sns.boxplot(x="category", y="amount", data=person_df)
    except Exception:
        try:
            person_df.boxplot(column="amount", by="category", rot=45)
        except Exception:
            pass
    plt.xticks(rotation=45)
    plt.tight_layout()
    p4 = os.path.join(PLOTS_DIR, f"{person}_box.png")
    plt.savefig(p4)
    plt.close()
    plot_paths["box"] = os.path.basename(p4)

    # Violin
    plt.figure(figsize=(10, 4))
    try:
        import seaborn as sns
        sns.violinplot(x="category", y="amount", data=person_df)
    except Exception:
        pass
    plt.xticks(rotation=45)
    plt.tight_layout()
    p5 = os.path.join(PLOTS_DIR, f"{person}_violin.png")
    plt.savefig(p5)
    plt.close()
    plot_paths["violin"] = os.path.basename(p5)

    # Pie
    cat_sum = person_df.groupby("category")["amount"].sum()
    if not cat_sum.empty:
        plt.figure(figsize=(6, 6))
        plt.pie(cat_sum, labels=cat_sum.index, autopct="%1.1f%%")
        plt.title("Category Share")
        plt.tight_layout()
        p6 = os.path.join(PLOTS_DIR, f"{person}_pie.png")
        plt.savefig(p6)
        plt.close()
        plot_paths["pie"] = os.path.basename(p6)
    else:
        plot_paths["pie"] = None

    return plot_paths


def train_and_predict(monthly, model_choice="linear"):
    if monthly is None or len(monthly) == 0:
        return {"pred": 0.0, "mae": None, "rmse": None, "r2": None, "model_name": "none"}
    if len(monthly) < 3:
        return {
            "pred": round(monthly["total_expense"].mean(), 2),
            "mae": None,
            "rmse": None,
            "r2": None,
            "model_name": "mean",
        }

    m = monthly.copy().reset_index(drop=True)
    m["t"] = range(len(m))
    m["lag1"] = m["total_expense"].shift(1).fillna(method="bfill")
    m["lag2"] = m["total_expense"].shift(2).fillna(method="bfill")
    m["roll3"] = m["total_expense"].rolling(3, min_periods=1).mean()

    X = m[["t", "lag1", "lag2", "roll3", "monthly_income"]].fillna(method="bfill")
    y = m["total_expense"].values

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]

    model = LinearRegression()
    model_name = "LinearRegression"
    try:
        model.fit(X_train, y_train)
    except Exception:
        simple = LinearRegression()
        simple.fit(m[["t"]], m["total_expense"])
        model = simple
        model_name = "LinearRegression(simple)"

    if len(X_test) > 0:
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
    else:
        mae = rmse = r2 = None

    last = m.iloc[-1]
    X_next = np.array(
        [[last["t"] + 1, last["total_expense"], last["lag1"], m["roll3"].iloc[-1], last["monthly_income"]]]
    )
    pred_next = float(model.predict(X_next)[0])

    return {"pred": pred_next, "mae": mae, "rmse": rmse, "r2": r2, "model_name": model_name}


# ---------------- ROUTES ----------------

@app.route("/login", methods=["GET", "POST"])
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        if username in USERS and USERS[username] == password:
            session["logged_in"] = True
            session["username"] = username
            return redirect(url_for("index"))
        else:
            return render_template("login.html", error="Invalid username or password")
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        if not username or not password:
            return render_template("register.html", error="Username and password required")
        if username in USERS:
            return render_template("register.html", error="User exists")
        USERS[username] = password
        return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/index")
def index():
    if not logged_in():
        return redirect(url_for("login"))
    df = load_data()
    users = sorted(df["person"].unique())
    return render_template("index.html", users=users, results=None)


@app.route("/predict", methods=["POST"])
def predict():
    if not logged_in():
        return redirect(url_for("login"))
    person = request.form.get("name")
    months = request.form.get("months")
    df = load_data()
    monthly, person_df = prepare_monthly(df, person, months)
    if monthly is None:
        return render_template(
            "index.html",
            error="No data for person.",
            users=sorted(df["person"].unique()),
            results=None,
        )
    plots = save_plots(person_df, monthly, person)
    out = train_and_predict(monthly)
    pred = round(out["pred"], 2)
    income = (
        monthly["monthly_income"].iloc[-1]
        if pd.notna(monthly["monthly_income"].iloc[-1])
        else None
    )
    ratio = (pred / income) if income and income > 0 else None
    if ratio is None:
        rec = "No income data available to compute recommendation."
    else:
        if ratio > 0.8:
            rec = "⚠️ High spending. Reduce unnecessary expenses."
        elif ratio < 0.6:
            rec = "👍 Good saving trend."
        else:
            rec = "🙂 Stable but monitor spending."
    results = {
        "name": person,
        "predicted": pred,
        "mae": round(out["mae"], 2) if out["mae"] else None,
        "rmse": round(out["rmse"], 2) if out["rmse"] else None,
        "r2": round(out["r2"], 3) if out["r2"] else None,
        "recommendation": rec,
        "plots": plots,
        "model_name": out.get("model_name", ""),
    }
    return render_template("index.html", users=sorted(df["person"].unique()), results=results)




@app.route("/dashboard")
def dashboard():
    if not logged_in():
        return redirect(url_for("login"))

    df = load_data()
    user = request.args.get("user") or session.get("username")
    pdf = df[df["person"] == user].copy()
    pdf["date"] = pd.to_datetime(pdf["date"], errors="coerce")

    # Use the LATEST month present in user's data
    if not pdf.empty and pdf["date"].notna().any():
        month_periods = pdf["date"].dt.to_period("M")
        latest_month = month_periods.max()
        month_df = pdf[month_periods == latest_month]
    else:
        latest_month = None
        month_df = pd.DataFrame()

    total_current = month_df["amount"].sum() if not month_df.empty else 0.0
    avg_daily = month_df["amount"].mean() if not month_df.empty else 0.0

    top_cats = (
        pdf.groupby("category")["amount"].sum().sort_values(ascending=False).head(5).to_dict()
        if not pdf.empty
        else {}
    )

    
    latest_tx_df = pdf.sort_values("date", ascending=False)
    latest_tx_df = latest_tx_df.reset_index().rename(columns={"index": "row_id"})
    latest_tx = latest_tx_df.to_dict("records")

    # mini trend
    monthly, _ = prepare_monthly(df, user, months=12)
    tiny_trend = None
    if monthly is not None and len(monthly) > 0:
        plt.figure(figsize=(3, 1.2))
        plt.plot(monthly["month_start"], monthly["total_expense"], marker="o")
        plt.axis("off")
        p = os.path.join(PLOTS_DIR, f"{user}_mini.png")
        plt.savefig(p, bbox_inches="tight", pad_inches=0)
        plt.close()
        tiny_trend = os.path.basename(p)

    return render_template(
        "dashboard.html",
        users=sorted(df["person"].unique()),
        user_filter=user,
        total_current=round(total_current, 2),
        avg_daily=round(avg_daily, 2),
        top_cats=top_cats,
        latest_tx=latest_tx,
        tiny_trend=tiny_trend,
    )


@app.route("/add_expense", methods=["GET", "POST"])
def add_expense():
    if not logged_in():
        return redirect(url_for("login"))
    if request.method == "POST":
        df = load_data()
        name = request.form.get("name") or session.get("username")
        date = request.form.get("date") or datetime.now().strftime("%Y-%m-%d")
        category = request.form.get("category") or "other"
        amount = float(request.form.get("amount") or 0)
        monthly_income = request.form.get("monthly_income")
        monthly_income = (
            float(monthly_income) if monthly_income not in (None, "") else np.nan
        )
        description = request.form.get("description", "")
        new_row = {
            "person": name,
            "date": date,
            "category": category,
            "amount": amount,
            "monthly_income": monthly_income,
            "description": description,
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        save_data(df)
        flash("Expense added.", "success")
        return redirect(url_for("dashboard"))
    return render_template("add_expense.html", default_name=session.get("username"))


@app.route("/edit_expense/<int:idx>", methods=["GET", "POST"])
def edit_expense(idx):
    if not logged_in():
        return redirect(url_for("login"))
    df = load_data()
    if idx < 0 or idx >= len(df):
        flash("Invalid record index.", "danger")
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        df.at[idx, "person"] = request.form.get("name")
        df.at[idx, "date"] = request.form.get("date")
        df.at[idx, "category"] = request.form.get("category")
        df.at[idx, "amount"] = float(request.form.get("amount") or 0)
        mi = request.form.get("monthly_income")
        df.at[idx, "monthly_income"] = float(mi) if mi not in (None, "") else np.nan
        df.at[idx, "description"] = request.form.get("description", "")
        save_data(df)
        flash("Expense updated.", "success")
        return redirect(url_for("dashboard"))
    row = df.iloc[idx].to_dict()
    row["date"] = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
    return render_template("edit_expense.html", idx=idx, row=row)


@app.route("/delete_expense/<int:idx>", methods=["POST"])
def delete_expense(idx):
    if not logged_in():
        return redirect(url_for("login"))
    df = load_data()
    if idx < 0 or idx >= len(df):
        flash("Invalid record.", "danger")
        return redirect(url_for("dashboard"))
    df = df.drop(df.index[idx]).reset_index(drop=True)
    save_data(df)
    flash("Expense deleted.", "success")
    return redirect(url_for("dashboard"))


@app.route("/download_report", methods=["GET"])
def download_report():
    if not logged_in():
        return redirect(url_for("login"))
    person = request.args.get("person") or session.get("username")
    months = request.args.get("months") or "12"
    df = load_data()
    monthly, person_df = prepare_monthly(df, person, months)
    plots = save_plots(person_df, monthly, person) if monthly is not None else {}
    out = train_and_predict(monthly)
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()
    story.append(Paragraph(f"BudgetWise — Expense Report for {person}", styles["Title"]))
    story.append(Spacer(1, 12))
    data = [
        ["Metric", "Value"],
        ["Predicted next month expense (₹)", f"{round(out['pred'], 2)}"],
        ["Model used", out.get("model_name", "")],
        ["MAE", f"{round(out['mae'], 2) if out['mae'] else '—'}"],
        ["RMSE", f"{round(out['rmse'], 2) if out['rmse'] else '—'}"],
        ["R²", f"{round(out['r2'], 3) if out['r2'] else '—'}"],
    ]
    table = Table(data, hAlign="LEFT")
    story.append(table)
    story.append(Spacer(1, 12))
    for key in ["trend", "regplot", "hist", "box", "violin", "pie"]:
        fname = plots.get(key)
        if fname:
            full = os.path.join(PLOTS_DIR, fname)
            if os.path.exists(full):
                try:
                    img = RLImage(full, width=400, height=200)
                    story.append(Paragraph(key.title(), styles["Heading3"]))
                    story.append(img)
                    story.append(Spacer(1, 8))
                except Exception:
                    pass
    doc.build(story)
    buffer.seek(0)
    filename = f"BudgetWise_Report_{person}_{datetime.now().strftime('%Y%m%d')}.pdf"
    return send_file(
        buffer,
        as_attachment=True,
        download_name=filename,
        mimetype="application/pdf",
    )


@app.route("/upload_dataset", methods=["GET", "POST"])
def upload_dataset():
    if not logged_in():
        return redirect(url_for("login"))
    if request.method == "POST":
        f = request.files.get("dataset")
        if f and f.filename.endswith(".csv"):
            f.save(CSV_FILE)
            flash("Dataset uploaded and replaced.", "success")
            return redirect(url_for("index"))
        else:
            flash("Please upload a valid CSV file.", "danger")
    return render_template("upload.html")


if __name__ == "__main__":
    app.run(debug=True)
