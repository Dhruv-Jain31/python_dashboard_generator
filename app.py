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
import re
import warnings
from typing import Optional, Tuple
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

def is_date_column(series: pd.Series) -> bool:
    # Check if column contains date-like strings
    date_pattern = r'^\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4}'
    try:
        # First check regex pattern
        if series.str.match(date_pattern, na=False).any():
            return True
        # Then try parsing as datetime
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(series, errors='coerce')
            return parsed.notna().mean() > 0.5
    except:
        return False

def select_best_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    # Get column types
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col]) or (df[col].dtype == "object" and is_date_column(df[col]))]

    x_col, y_col, chart_type = None, None, None

    # Check if columns exist and are non-empty
    if len(numeric_cols) > 0:
        # Priority 1: Time-series (datetime x, numeric y)
        if len(datetime_cols) > 0:
            x_col = datetime_cols[0]
            y_col = numeric_cols[0]
            chart_type = 'line'
        # Priority 2: Bar (categorical x, numeric y)
        elif len(categorical_cols) > 0 and len(df[categorical_cols[0]].dropna().unique()) < 20:
            x_col = categorical_cols[0]
            y_col = numeric_cols[0]
            chart_type = 'bar'
        # Priority 3: Scatter (numeric x, numeric y)
        elif len(numeric_cols) >= 2:
            x_col = numeric_cols[0]
            y_col = numeric_cols[1]
            chart_type = 'scatter'
        # Fallback: Histogram (single numeric)
        else:
            x_col = numeric_cols[0]
            y_col = None
            chart_type = 'histogram'
    elif len(categorical_cols) > 0:
        # Fallback: Pie chart for categorical
        x_col = categorical_cols[0]
        y_col = None
        chart_type = 'pie'

    return x_col, y_col, chart_type

def generate_plots(df: pd.DataFrame, options: VisualizationOptions = None) -> list:
    plots = []
    
    # Handle missing values
    df = df.fillna(df.select_dtypes(include=['float64', 'int64']).mean())
    df = df.fillna(df.select_dtypes(include=['object']).mode().iloc[0])
    
    # Parse datetime columns
    for col in df.columns:
        if df[col].dtype == "object" and is_date_column(df[col]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass
    
    # Get column types
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    datetime_cols = df.select_dtypes(include=['datetime64']).columns
    
    # Validate DataFrame
    if df.empty or len(df.columns) == 0:
        raise ValueError("Uploaded CSV is empty or contains no valid columns")
    
    # Auto-select x, y, and chart type if not specified
    auto_x, auto_y, auto_chart = select_best_columns(df)
    
    # User-specified plot
    if options and options.chart_type and options.x_column and options.y_column:
        if options.x_column not in df.columns or options.y_column not in df.columns:
            raise ValueError(f"Selected columns {options.x_column} or {options.y_column} not found in CSV")
        if options.chart_type == "scatter":
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
    # 1. Auto-selected plot
    if auto_x and auto_chart == 'line' and auto_y:
        fig = go.Figure()
        for col in numeric_cols[:3]:
            fig.add_trace(go.Scatter(
                x=df[auto_x],
                y=df[col],
                mode='lines',
                name=col
            ))
        fig.update_layout(
            title=f'Trends over {auto_x}',
            xaxis_title=auto_x,
            yaxis_title='Value'
        )
        plots.append(fig.to_json())
    elif auto_x and auto_y and auto_chart == 'bar':
        fig = px.bar(
            df,
            x=auto_x,
            y=auto_y,
            title=f"{auto_y} by {auto_x}"
        )
        plots.append(fig.to_json())
    elif auto_x and auto_y and auto_chart == 'scatter':
        fig = px.scatter(
            df,
            x=auto_x,
            y=auto_y,
            title=f"{auto_y} vs {auto_x}",
            hover_data=[df.index]
        )
        plots.append(fig.to_json())
    elif auto_x and auto_chart == 'histogram':
        fig = px.histogram(
            df,
            x=auto_x,
            title=f'Histogram of {auto_x}',
            nbins=30
        )
        plots.append(fig.to_json())
    elif auto_x and auto_chart == 'pie':
        value_counts = df[auto_x].value_counts()
        fig = px.pie(
            names=value_counts.index,
            values=value_counts.values,
            title=f'Pie Chart of {auto_x}'
        )
        plots.append(fig.to_json())
    
    # 2. Bar chart for categorical columns
    for cat_col in categorical_cols:
        if len(df[cat_col].dropna().unique()) < 20:
            value_counts = df[cat_col].value_counts()
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f'Distribution of {cat_col}',
                labels={'x': cat_col, 'y': 'Count'}
            )
            plots.append(fig.to_json())
    
    # 3. Histogram for numeric columns
    for num_col in numeric_cols:
        fig = px.histogram(
            df,
            x=num_col,
            title=f'Histogram of {num_col}',
            nbins=30
        )
        plots.append(fig.to_json())
    
    # 4. Box plot for categorical and numeric
    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
        fig = px.box(
            df,
            x=categorical_cols[0],
            y=numeric_cols[0],
            title=f'Box Plot of {numeric_cols[0]} by {categorical_cols[0]}'
        )
        plots.append(fig.to_json())
    
    # 5. Pie chart for categorical
    for cat_col in categorical_cols:
        if len(df[cat_col].dropna().unique()) < 10:
            value_counts = df[cat_col].value_counts()
            fig = px.pie(
                names=value_counts.index,
                values=value_counts.values,
                title=f'Pie Chart of {cat_col}'
            )
        plots.append(fig.to_json())
    
    # 6. Heatmap for numeric correlations
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='Viridis'
        ))
        fig.update_layout(title='Correlation Heatmap')
        plots.append(fig.to_json())
    
    if not plots:
        raise ValueError("No valid plots could be generated from the CSV")
    
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
        
        # Validate CSV
        if df.empty or len(df.columns) == 0:
            raise ValueError("Uploaded CSV is empty or contains no valid columns")
        
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