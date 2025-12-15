BudgetWise — AI Expense Tracking & Forecasting System

BudgetWise is a web-based personal finance management system that allows users to record expenses, analyze spending patterns, and predict future monthly expenses using Machine Learning (Linear Regression).
It provides visual insights through interactive graphs and helps users make smarter budgeting decisions.

#Key Features

✔ Add, edit, delete, and track daily expenses

✔ Visual expense analysis using:

Line Chart (Trend)

Regression Plot

Histogram

Pie chart

Boxplot

Violin plot

✔ Machine learning–based future expense prediction

✔ Downloadable PDF financial report

✔ User login system with personalized dashboard

✔ CSV-based lightweight storage (no database required)

✔ Data stored in: expense_data_final.csv

#System Workflow

➤ User logs in / registers

➤ Expense data stored and processed using Pandas

➤ Charts generated using Matplotlib & Seaborn

➤ ML model (Linear Regression) predicts next month’s spending

➤ Dashboard displays insights + exportable report

#Tech Stack

| Component       | Technology                       |
| --------------- | -------------------------------- |
| Backend         | Flask (Python)                   |
| Frontend        | HTML, CSS                        |
| ML Model        | Scikit-Learn (Linear Regression) |
| Data Processing | Pandas, NumPy                    |
| Visualization   | Matplotlib, Seaborn              |
| Storage         | CSV file                         |

## Run
1. Make sure Python packages are installed:
   pip install flask pandas numpy matplotlib seaborn scikit-learn
2. Run:
   python app.py
3. Open http://127.0.0.1:5000/ or open browser and type localhost:5000
Default demo credentials: divya / 1234

#Output & Results

➤ Displays user’s monthly spending behavior

➤ Provides prediction accuracy metrics (MAE, RMSE, R² Score)

➤ Helps understand where money is spent and where it can be optimized

BudgetWise_Creative
│
├── app.py                         # Main Flask application
├── expense_data_final.csv         # Local dataset used for storing expenses
├── License.txt                    # Project license
├── README.md                      # Project documentation
│
├── static/
│   └── plots/ 
         example of 1 person plots            # Auto-generated graphs and visualizations
│       ├── Divya Yeddu_box.png
│       ├── Divya Yeddu_hist.png
│       ├── Divya Yeddu_pie.png
│       ├── Divya Yeddu_regplot.png
│       ├── Divya Yeddu_trend.png
│       └── Divya Yeddu_violin.png
│
└── templates/                      # Frontend HTML templates used by Flask
    ├── add_expense.html
    ├── base.html
    ├── dashboard.html
    ├── edit_expense.html
    ├── index.html
    ├── login.html
    ├── register.html
    └── upload.html

License
This project is licensed under the MIT License.See the LICENSE file for details.