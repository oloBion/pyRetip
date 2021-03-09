import numpy as np
import pandas as pd
import scipy.stats as st

from retip import Dataset, Trainer


def outlier_identification(trainer: Trainer, dataset: Dataset, output_filename: str = None):
    """
    """

    from bokeh.models import Slope
    from bokeh.plotting import figure

    if not output_filename:
        # view the plot in a Jupyter notebook
        from bokeh.io import output_notebook, show
        output_notebook()
    
    # predict RT
    data = dataset.get_data()
    X = data.drop('RT', axis=1)
    y = data.RT.values

    y_pred = trainer.predict(X)
    rt_error = y - y_pred

    # calculate CI and linear fit
    ci = st.norm.ppf(0.95, loc=np.mean(rt_error), scale=np.std(rt_error))
    slope, intercept = par = np.polyfit(y, y_pred, 1)

    # find outliers
    a_t, b_t = np.array([0, intercept + ci]), np.array([1, slope + intercept + ci])
    a_b, b_b = np.array([0, intercept - ci]), np.array([1, slope + intercept - ci])

    def is_in_ci(row):
        is_above = lambda p, a, b: np.cross(p - a, b - a) < 0
        p = np.array([row.y, row.y_pred])
        
        return is_above(p, a_b, b_b) and not is_above(p, a_t, b_t)

    y_df = pd.DataFrame({'y': y, 'y_pred': y_pred}, index=X.index)
    y_df['in_ci'] = y_df.apply(is_in_ci, axis=1)

    # plot
    p = figure()
    p.xaxis.axis_label = 'Library RT'
    p.yaxis.axis_label = 'Predicted RT'
    p.scatter(y_df[y_df.in_ci].y, y_df[y_df.in_ci].y_pred, size=3)
    p.scatter(y_df[~y_df.in_ci].y, y_df[~y_df.in_ci].y_pred, size=3, color='red', legend_label='Outliers')

    p.add_layout(Slope(gradient=1, y_intercept=0, line_width=1, line_color='black', line_alpha=0.5))

    p.add_layout(Slope(gradient=slope, y_intercept=intercept, line_color='blue'))
    p.add_layout(Slope(gradient=slope, y_intercept=intercept + ci, line_color='blue', line_dash='dashed', line_alpha=0.75))
    p.add_layout(Slope(gradient=slope, y_intercept=intercept - ci, line_color='blue', line_dash='dashed', line_alpha=0.75))

    p.legend.location = "top_left"

    if not output_filename:
        show(p)

    # return outliers
    df = dataset.df[['Name', 'RT']]
    df_rtp = y_df[~y_df.in_ci][['y_pred']]
    df_rtp.columns = ['RTP']
    
    return df.join(df_rtp, how='inner')
