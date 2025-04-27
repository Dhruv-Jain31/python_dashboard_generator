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
    date_pattern = r'^\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4}'
    try:
        if series.str.match(date_pattern, na=False).any():
            return True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(series, errors='coerce')
            return parsed.notna().mean() > 0.5
    except:
        return False

def clean_numeric_column(series: pd.Series) -> pd.Series:
    # Only apply to columns that look like numeric strings
    if series.dtype == "object":
        # Use str.match to avoid regex warning
        numeric_pattern = r'^\d{1,3}(?:,\d{3})*(?:\.\d+)?$'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            is_numeric = series.str.match(numeric_pattern, na=False).any()
        if is_numeric:
            try:
                cleaned = series.str.replace(',', '').astype(float)
                return pd.to_numeric(cleaned, errors='coerce')
            except:
                return series
    return series

def select_best_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    # Clean numeric-like columns
    df_cleaned = df.copy()
    for col in df_cleaned.columns:
        df_cleaned[col] = clean_numeric_column(df_cleaned[col])
    
    numeric_cols = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns
    datetime_cols = [col for col in df_cleaned.columns if pd.api.types.is_datetime64_any_dtype(df_cleaned[col]) or (df_cleaned[col].dtype == "object" and is_date_column(df_cleaned[col]))]

    x_col, y_col, chart_type = None, None, None

    if len(numeric_cols) > 0:
        valid_numeric_cols = [col for col in numeric_cols if df_cleaned[col].notna().any()]
        if len(valid_numeric_cols) > 0:
            if len(datetime_cols) > 0:
                x_col = datetime_cols[0]
                y_col = valid_numeric_cols[0]
                chart_type = 'line'
            elif len(categorical_cols) > 0 and len(df_cleaned[categorical_cols[0]].dropna().unique()) < 20:
                x_col = categorical_cols[0]
                y_col = valid_numeric_cols[0]
                chart_type = 'bar'
            elif len(valid_numeric_cols) >= 2:
                x_col = valid_numeric_cols[0]
                y_col = valid_numeric_cols[1]
                chart_type = 'scatter'
            else:
                x_col = valid_numeric_cols[0]
                y_col = None
                chart_type = 'histogram'
    if len(categorical_cols) > 0 and not x_col:
        x_col = categorical_cols[0]
        y_col = None
        chart_type = 'pie'

    return x_col, y_col, chart_type

def generate_plots(df: pd.DataFrame, options: VisualizationOptions = None) -> list:
    plots = []
    
    # Make a copy to avoid modifying the original
    df_cleaned = df.copy()
    
    # Handle missing values
    numeric_means = df_cleaned.select_dtypes(include=['float64', 'int64']).mean()
    object_modes = df_cleaned.select_dtypes(include=['object']).mode().iloc[0]
    df_cleaned = df_cleaned.fillna(numeric_means).fillna(object_modes)
    
    # Parse datetime columns
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == "object" and is_date_column(df_cleaned[col]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')
                except:
                    pass
    
    # Clean numeric-like columns
    for col in df_cleaned.columns:
        df_cleaned[col] = clean_numeric_column(df_cleaned[col])
    
    # Get column types
    numeric_cols = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns
    datetime_cols = df_cleaned.select_dtypes(include=['datetime64']).columns
    
    # Validate DataFrame
    if df_cleaned.empty or len(df_cleaned.columns) == 0:
        raise ValueError("Uploaded CSV is empty or contains no valid columns")
    
    # Validate data presence
    if df_cleaned.isna().all().all():
        raise ValueError("All data in CSV is NaN after processing")
    
    print(f"Numeric columns: {numeric_cols.tolist()}")
    print(f"Categorical columns: {categorical_cols.tolist()}")
    print(f"Datetime columns: {datetime_cols}")
    
    # Auto-select x, y, and chart type if not specified
    auto_x, auto_y, auto_chart = select_best_columns(df_cleaned)
    print(f"Selected columns: x={auto_x}, y={auto_y}, chart_type={auto_chart}")
    
    # User-specified plot
    if options and options.chart_type and options.x_column and options.y_column:
        if options.x_column not in df_cleaned.columns or options.y_column not in df_cleaned.columns:
            raise ValueError(f"Selected columns {options.x_column} or {options.y_column} not found in CSV")
        if options.chart_type == "scatter":
            if df_cleaned[options.x_column].notna().any() and df_cleaned[options.y_column].notna().any():
                fig = px.scatter(
                    df_cleaned,
                    x=options.x_column,
                    y=options.y_column,
                    title=f"{options.y_column} vs {options.x_column}",
                    hover_data=[df_cleaned.index]
                )
                plots.append(fig.to_json())
        elif options.chart_type == "bar" and options.x_column in categorical_cols and options.y_column in numeric_cols:
            if df_cleaned[options.x_column].notna().any() and df_cleaned[options.y_column].notna().any():
                fig = px.bar(
                    df_cleaned,
                    x=options.x_column,
                    y=options.y_column,
                    title=f"{options.y_column} by {options.x_column}"
                )
                plots.append(fig.to_json())
    
    # Default plots
    # 1. Auto-selected plot
    if auto_x and auto_chart == 'line' and auto_y:
        if df_cleaned[auto_x].notna().any() and df_cleaned[auto_y].notna().any():
            fig = go.Figure()
            for col in numeric_cols[:3]:
                if df_cleaned[col].notna().any():
                    fig.add_trace(go.Scatter(
                        x=df_cleaned[auto_x],
                        y=df_cleaned[col],
                        mode='lines+markers',
                        name=col
                    ))
            fig.update_layout(
                title=f'Trends over {auto_x}',
                xaxis_title=auto_x,
                yaxis_title='Value'
            )
            if fig.data:
                plots.append(fig.to_json())
    elif auto_x and auto_y and auto_chart == 'bar':
        if df_cleaned[auto_x].notna().any() and df_cleaned[auto_y].notna().any():
            fig = px.bar(
                df_cleaned,
                x=auto_x,
                y=auto_y,
                title=f"{auto_y} by {auto_x}"
            )
            plots.append(fig.to_json())
    elif auto_x and auto_y and auto_chart == 'scatter':
        if df_cleaned[auto_x].notna().any() and df_cleaned[auto_y].notna().any():
            fig = px.scatter(
                df_cleaned,
                x=auto_x,
                y=auto_y,
                title=f"{auto_y} vs {auto_x}",
                hover_data=[df_cleaned.index]
            )
            plots.append(fig.to_json())
    elif auto_x and auto_chart == 'histogram':
        if df_cleaned[auto_x].notna().any():
            fig = px.histogram(
                df_cleaned,
                x=auto_x,
                title=f'Histogram of {auto_x}',
                nbins=30
            )
            plots.append(fig.to_json())
    elif auto_x and auto_chart == 'pie':
        if df_cleaned[auto_x].notna().any():
            value_counts = df_cleaned[auto_x].value_counts()
            fig = px.pie(
                names=value_counts.index,
                values=value_counts.values,
                title=f'Pie Chart of {auto_x}'
            )
            plots.append(fig.to_json())
    
    # 2. Bar chart for categorical columns
    for cat_col in categorical_cols:
        if len(df_cleaned[cat_col].dropna().unique()) < 20 and df_cleaned[cat_col].notna().any():
            value_counts = df_cleaned[cat_col].value_counts()
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f'Distribution of {cat_col}',
                labels={'x': cat_col, 'y': 'Count'}
            )
            plots.append(fig.to_json())
    
    # 3. Histogram for all numeric columns
    for num_col in numeric_cols:
        if df_cleaned[num_col].notna().any():
            data = df_cleaned[num_col].dropna()
            if len(data) > 0:
                print(f"Generating histogram for {num_col} with data:\n{data.head()}")
                # Dynamically set nbins based on data range
                data_range = data.max() - data.min()
                nbins = min(30, max(10, int(len(data) / 5))) if data_range != 0 else 10
                fig = px.histogram(
                    df_cleaned,  # Use full DataFrame
                    x=num_col,   # Explicitly specify x
                    title=f'Histogram of {num_col}',
                    nbins=nbins
                )
                fig.update_layout(xaxis_title=num_col, yaxis_title='Count')
                plots.append(fig.to_json())
    
    # 4. Box plots for categorical and numeric
    for cat_col in categorical_cols:
        for num_col in numeric_cols[:3]:
            if len(df_cleaned[cat_col].dropna().unique()) < 20 and df_cleaned[cat_col].notna().any() and df_cleaned[num_col].notna().any():
                fig = px.box(
                    df_cleaned,
                    x=cat_col,
                    y=num_col,
                    title=f'Box Plot of {num_col} by {cat_col}'
                )
                plots.append(fig.to_json())
    
    # 5. Pie chart for categorical
    for cat_col in categorical_cols:
        if len(df_cleaned[cat_col].dropna().unique()) < 10 and df_cleaned[cat_col].notna().any():
            value_counts = df_cleaned[cat_col].value_counts()
            fig = px.pie(
                names=value_counts.index,
                values=value_counts.values,
                title=f'Pie Chart of {cat_col}'
            )
            plots.append(fig.to_json())
    
    # 6. Heatmap for numeric correlations
    if len(numeric_cols) >= 2:
        corr_matrix = df_cleaned[numeric_cols].corr()
        if not corr_matrix.empty and not corr_matrix.isna().all().all():
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='Viridis',
                zmin=-1,
                zmax=1
            ))
            fig.update_layout(title='Correlation Heatmap')
            plots.append(fig.to_json())
    
    if not plots:
        raise ValueError("No valid plots could be generated from the CSV")
    
    print(f"Generated plots JSON: {len(plots)} plots")  # Debug JSON count
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
        
        # Log DataFrame for debugging
        print(f"Uploaded CSV columns: {df.columns.tolist()}")
        print(f"First few rows:\n{df.head()}")
        
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