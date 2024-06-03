import numpy as np
import pandas as pd
import scipy.stats as st
from typing import Union

from bokeh.plotting import figure
from bokeh.models import Slope, ColumnDataSource, HoverTool
from bokeh.io import export_png

from . import Dataset
from .trainers import Trainer

def plot_rt_scatter(df, output_filename: str = None):
    if not output_filename:
        # view the plot in a Jupyter notebook
        from bokeh.io import output_notebook, show
        output_notebook()

    # calculate linear fit
    slope, intercept = np.polyfit(df.y, df.y_pred, 1)

    p = figure()
    p.scatter("y", "y_pred", size=3, color="#5E676F", source=df)

    p.add_layout(Slope(gradient=1, y_intercept=0, line_width=1, line_color='#384049', line_alpha=0.5))
    p.add_layout(Slope(gradient=slope, y_intercept=intercept, line_color='#D02937'))

    # Axis Label
    p.xaxis.axis_label = 'Library RT'
    p.yaxis.axis_label = 'Predicted RT'
    p.axis.axis_label_text_color = "#384049"
    p.axis.axis_label_text_font_style = 'bold'
    # Axis Line
    p.axis.axis_line_color = "#384049"
    # Axis Ticks
    p.axis.minor_tick_in = 0
    p.axis.minor_tick_out = 3
    p.axis.minor_tick_line_color = "#384049"
    p.axis.major_tick_in = 0
    p.axis.major_tick_out = 6
    p.axis.major_tick_line_color = "#384049"

    # Grid
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    # Hover data
    hover = HoverTool()
    if "Name" in df.columns:
        hover.tooltips = """
            <div>
                <span style="font-weight: bold; color: #384049;">@Name<br></span>
                <span style="color: #384049;">@y, @y_pred<br></span>
            </div>
        """
    else:
        hover.tooltips = """
            <div>
                <span style="color: #384049;">@y, @y_pred<br></span>
            </div>
        """
    p.add_tools(hover)

    if output_filename:
        if not output_filename.lower().endswith('.png'):
            output_filename += '.png'
        export_png(p, filename=output_filename)
    else:
        show(p)


def outlier_identification(trainer: Trainer, dataset: Dataset, prediction_column: str, confidence_interval: float = 95, output_filename: str = None):
    """
    """

    if not output_filename:
        # view the plot in a Jupyter notebook
        from bokeh.io import output_notebook, show
        output_notebook()
    
    # predict RT
    data = dataset.get_training_data(include_metadata=True)
    X = data.drop(['Name', 'InChIKey', 'SMILES', dataset.target_column], axis=1)
    y = data[dataset.target_column].values

    y_pred = trainer.predict(X)
    rt_error = y - y_pred

    # calculate CI and linear fit
    ci = st.norm.ppf(confidence_interval / 100, loc=np.mean(rt_error), scale=np.std(rt_error))
    slope, intercept = np.polyfit(y, y_pred, 1)

    # find outliers
    a_t, b_t = np.array([0, intercept + ci]), np.array([1, slope + intercept + ci])
    a_b, b_b = np.array([0, intercept - ci]), np.array([1, slope + intercept - ci])

    def is_in_ci(row):
        is_above = lambda p, a, b: np.cross(p - a, b - a) < 0
        p = np.array([row.y, row.y_pred])
        
        return is_above(p, a_b, b_b) and not is_above(p, a_t, b_t)

    y_df = pd.DataFrame({'y': y, 'y_pred': y_pred}, index=X.index)
    y_df['in_ci'] = y_df.apply(is_in_ci, axis=1)
    y_df = pd.merge(data["Name"], y_df, left_index=True, right_index=True)
    source_ann = y_df[y_df.in_ci]
    source_out = y_df[~y_df.in_ci]

    # plot
    p = figure()
    p.xaxis.axis_label = f'Experimental {dataset.target_column}'
    p.yaxis.axis_label = f'Predicted {dataset.target_column}'
    p.scatter('y', 'y_pred', size=3, color="#5E676F", source=source_ann)
    p.scatter('y', 'y_pred', size=3, color='#D02937', legend_label='Outliers', source=source_out)

    p.add_layout(Slope(gradient=1, y_intercept=0, line_width=1, line_color='#384049', line_alpha=0.5))

    p.add_layout(Slope(gradient=slope, y_intercept=intercept, line_color='#092DF7'))
    p.add_layout(Slope(gradient=slope, y_intercept=intercept + ci, line_color='#092DF7', line_dash='dashed', line_alpha=0.75))
    p.add_layout(Slope(gradient=slope, y_intercept=intercept - ci, line_color='#092DF7', line_dash='dashed', line_alpha=0.75))

    p.legend.location = "top_left"

    # Axis Label
    p.axis.axis_label_text_color = "#384049"
    p.axis.axis_label_text_font_style = 'bold'
    # Axis Line
    p.axis.axis_line_color = "#384049"
    # Axis Ticks
    p.axis.minor_tick_in = 0
    p.axis.minor_tick_out = 3
    p.axis.minor_tick_line_color = "#384049"
    p.axis.major_tick_in = 0
    p.axis.major_tick_out = 6
    p.axis.major_tick_line_color = "#384049"

    # Grid
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    # Hover data
    hover = HoverTool()
    hover.tooltips = """
        <div>
            <span style="font-weight: bold; color: #384049;">@Name<br></span>
            <span style="color: #384049;">@y, @y_pred<br></span>
        </div>
    """
    p.add_tools(hover)

    if output_filename:
        if not output_filename.lower().endswith('.png'):
            output_filename += '.png'
        export_png(p, filename=output_filename)
    else:
        show(p)

    # return annotated dataframe and outliers
    annotated = data
    annotated = annotated[[c for c in annotated.columns if c in ['Name', 'InChIKey', 'SMILES', dataset.target_column]]].copy()
    annotated[prediction_column] = y_pred

    df = annotated[['Name', dataset.target_column]]
    df_rtp = y_df[~y_df.in_ci][['y_pred']]
    df_rtp.columns = [prediction_column]
    outliers = df.join(df_rtp, how='inner')
    
    return annotated, outliers

def plot_feature_importance(trainer: Trainer, dataset: Union[Dataset, pd.DataFrame] = None, output_filename: str = None):
    if not output_filename:
        # view the plot in a Jupyter notebook
        from bokeh.io import output_notebook, show
        output_notebook()
    
    if dataset is not None:
        df = trainer.feature_importance(dataset)
    else:
        df = trainer.feature_importance()
    df = df.iloc[0:20, :]
    df.sort_values(by="importance", inplace=True)
    source = ColumnDataSource(data=df)

    p = figure(y_range=df["feature"], title="Feature importance (top 20) \n")
    p.hbar(y="feature", right="importance", source=source, height=0.8,
           fill_color="#D02937", line_color = '#D02937')
    
    # Axis Label
    p.xaxis.axis_label = 'Importance'
    p.axis.axis_label_text_color = "#384049"
    p.axis.axis_label_text_font_style = 'bold'
    # Axis Line
    p.axis.axis_line_color = "#384049"
    # Axis Ticks
    p.axis.minor_tick_in = 0
    p.axis.minor_tick_out = 3
    p.axis.minor_tick_line_color = "#384049"
    p.axis.major_tick_in = 0
    p.axis.major_tick_out = 6
    p.axis.major_tick_line_color = "#384049"
    # Axis X range
    p.x_range.start = 0

    # Grid
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    # Title
    p.title.align = "center"
    p.title.text_color = "#384049"
    p.title.text_font_size = "15px"

    # Hover data
    hover = HoverTool()
    hover.tooltips = """
        <div>
            <span style="font-weight: bold; color: #384049;">@feature:</span>
            <span style="color: #384049;">@importance</span>
        </div>
    """
    p.add_tools(hover)

    if output_filename:
        if not output_filename.lower().endswith('.png'):
            output_filename += '.png'
        export_png(p, filename=output_filename)
    else:
        show(p)
