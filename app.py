from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import uuid
from typing import Optional
from starlette.requests import Request

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class VisualizationOptions(BaseModel):
    chart_type: Optional[str] = None
    x_column: Optional[str] = None
    y_column: Optional[str] = None

def generate_plots(df: pd.DataFrame, options: VisualizationOptions = None) -> list:
    plots = []

    # Handle missing values
    df = df.fillna(df.select_dtypes(include=['float64', 'int64']).mean())
    df = df.fillna(df.select_dtypes(include=['object']).mode().iloc[0])

    # Get column types
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    datetime_cols = df.select_dtypes(include=['datetime64']).columns

    # User-specified plot
    if options and options.chart_type and options.x_column and options.y_column:
        if options.chart_type == "scatter" and options.x_column in df.columns and options.y_column in df.columns:
            fig = px.scatter(
                df,
                x=options.x_column,
                y=options.y_column,
                title=f"{options.y_column} vs {options.x_column}",
                hover_data=[df.index]
            )
            plots.append(fig.to_json())
        elif options.chart_type == "bar" and options.x_column in categorical_cols and options.y_column in numeric_cols:
            fig = px.bar(
                df,
                x=options.x_column,
                y=options.y_column,
                title=f"{options.y_column} by {options.x_column}"
            )
            plots.append(fig.to_json())

    # Default plots
    # Bar chart for categorical columns
    for cat_col in categorical_cols:
        if len(df[cat_col].unique()) < 20:
            value_counts = df[cat_col].value_counts()
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f'Distribution of {cat_col}',
                labels={'x': cat_col, 'y': 'Count'}
            )
            plots.append(fig.to_json())

    # Time-series for datetime columns
    if len(datetime_cols) > 0 and len(numeric_cols) > 0:
        fig = go.Figure()
        for col in numeric_cols[:3]:
            fig.add_trace(go.Scatter(
                x=df[datetime_cols[0]],
                y=df[col],
                mode='lines',
                name=col
            ))
        fig.update_layout(
            title=f'Trends over {datetime_cols[0]}',
            xaxis_title=datetime_cols[0],
            yaxis_title='Value'
        )
        plots.append(fig.to_json())

    # Histogram for numeric columns
    for num_col in numeric_cols:
        fig = px.histogram(
            df,
            x=num_col,
            title=f'Histogram of {num_col}',
            nbins=30
        )
        plots.append(fig.to_json())

    return plots

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    chart_type: Optional[str] = Form(None),
    x_column: Optional[str] = Form(None),
    y_column: Optional[str] = Form(None)
):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Invalid file type. Only CSV allowed.")

    filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    try:
        # Save file
        with open(file_path, "wb") as f:
            f.write(await file.read())
    
        # Read CSV
        df = pd.read_csv(file_path)
    
        # Parse datetime columns
        for col in df.columns:
            if df[col].dtype == "object":
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass
    
        # Generate plots
        options = VisualizationOptions(chart_type=chart_type, x_column=x_column, y_column=y_column)
        plots = generate_plots(df, options)

        return {"plots": plots, "columns": list(df.columns)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)