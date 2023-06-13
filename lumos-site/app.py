import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from streamlit_option_menu import option_menu
banner_image = Image.open('JOB TRACE VISULAZATION.png')

st.image(banner_image)

nav_bar_horizontal = option_menu(None, ["Job Run Time", "Model 2", "Model 3"], default_index=0, orientation="horizontal")
system_models_jrt = ["Mira", "Blue Waters", "Philly", "Helios"]

with st.form("select_chart_model_jrt"):
    #cdf
    st.write("# Select the chart you want to view")
    chart_select_radio_jrt = st.radio("Chart Selection", [None, "CDF Run Time Chart", "Detailed Run Time Distribution Chart"])
    submit = st.form_submit_button("Select")
    if submit:
        st.write(f"You have selected to view: {chart_select_radio_jrt}")
    if chart_select_radio_jrt == "CDF Run Time Chart":

        min_value_exp_run_time_slider = 0
        max_value_exp_run_time_slider = 8 

        with st.sidebar.form("CDF_chart_form_jrt"):
            st.write("## Adjust the following settings to change the CDF chart:")
            selected_system_models_jrt = []
            with st.expander("Select System Model(s)"):
                for item in system_models_jrt:
                    model_checkbox_jrt = st.checkbox(item)
                    if model_checkbox_jrt:
                        selected_system_models_jrt.append(item)

            cdf_frequency_slider_jrt = st.slider("Choose frequency range", min_value=0, max_value=100, step=20)
            cdf_run_time_slider_jrt = st.slider("Choose run time range (in powers of 10)", min_value_exp_run_time_slider, max_value_exp_run_time_slider, step=1)
            cdf_run_time_value_slider_jrt = int(10**cdf_run_time_slider_jrt)
                        
            submit_cdf_sidebar_button = st.form_submit_button("Apply")

        #Alex code here for displaying the cdf chart

    elif chart_select_radio_jrt == "Detailed Run Time Distribution Chart":
        #drt = detailed run time
        drt_time_ranges = ["0sec to 30sec", "30sec to 10min", "10min to 1h", "1h to 12h", "more than 12h"]
        with st.sidebar.form("detailed_run_time_form_jrt"):
            st.write("## Adjust the following settings to change the detailed run time chart:")
            drt_selected_system_models_jrt = []
            drt_selected_time_range_jrt = []
            with st.expander("Select System Model(s)"):
                for item in system_models_jrt:
                    drt_model_checkbox_jrt = st.checkbox(item)
                    if drt_model_checkbox_jrt:
                        drt_selected_system_models_jrt.append(item)
            drt_frequency_slider_jrt = st.slider("Choose frequency range", min_value=0.0, max_value=0.6, step=0.1)
            with st.expander("Select Run Time Range"):
                for item in drt_time_ranges:
                    drt_time_range_checkbox_jrt = st.checkbox(item)
                    if drt_time_range_checkbox_jrt:
                        drt_selected_time_range_jrt.append(item)

            submit_drt_sidebar_button = st.form_submit_button("Apply")
        #Alex code here for displaying the detailed run time chart

    