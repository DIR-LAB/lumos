import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import warnings
from streamlit_option_menu import option_menu
from matplotlib import pyplot as plt
import seaborn
from datetime import datetime
from collections import Counter, defaultdict
import json
import time 
import matplotlib
banner_image = Image.open('JOB TRACE VISULAZATION.png')

st.image(banner_image)

nav_bar_horizontal = option_menu(None, ["Job Run Time", "Job Wait Time", "Model 3"], default_index=0, orientation="horizontal")

system_models_jrt = ["Mira", "Blue Waters", "Philly", "Helios"]

bw_df = pd.read_csv("../data_blue_waters.csv")
mira_df_2 = pd.read_csv("../data_mira.csv")
hl_df = pd.read_csv("../data_helios.csv")
philly_df = pd.read_csv("../data_philly.csv")

columns=["job", "user", "project", "state", "gpu_num", "cpu_num", "node_num", "submit_time", "wait_time", "run_time", "wall_time", "node_hour"]

chart_select_radio_jrt = None;

if nav_bar_horizontal == "Job Run Time":
    with st.form("select_chart_model_jrt"):
        #cdf
        st.write("### Select the chart you want to view")
        chart_select_radio_jrt = st.radio("Chart Selection", [None, "CDF Run Time Chart", "Detailed Run Time Distribution Chart"], horizontal=True)
        submit = st.form_submit_button("Select")
        if submit:
            st.write(f"You have selected: {chart_select_radio_jrt}")
            if chart_select_radio_jrt == "CDF Run Time Chart":
                st.markdown('<script>scrollToSection("cdf_chart_section")</script>', unsafe_allow_html=True)
            elif chart_select_radio_jrt == "Detailed Run Time Distribution Chart":
                st.markdown('<script>scrollToSection("drt_chart_section")</script>', unsafe_allow_html=True)

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
            if submit_cdf_sidebar_button:
                if len(selected_system_models_jrt) > 1:
                    with st.spinner("Loading...."):
                        time.sleep(7)
                    st.success("Done!")
                else:
                    st.write("Please select the system models to see the graph")
            
        
        #Alex code here for displaying the cdf chart
        # Plots Figure 1(a) from page 3, 3.1.1
        st.markdown("<a name='cdf_chart_section'></a>", unsafe_allow_html=True)

        def plot_cdf(x, bins, xlabel, ylabel="Frequency (%)", color="", linestyle="--"):
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)

            x = np.sort(x)
            cdf = 100 * np.arange(len(x)) / float(len(x))

            if color:
                plt.plot(x, cdf, linestyle=linestyle, linewidth=5, color=color)
            else:
                plt.plot(x, cdf, linestyle=linestyle, linewidth=5)

            plt.xlabel(xlabel, fontsize=20)
            plt.ylabel(ylabel, fontsize=20)
            plt.margins(0)
            plt.ylim(0, cdf_frequency_slider_jrt) 
            plt.xlim(10**0, cdf_run_time_value_slider_jrt) 
            

            plt.grid(True)

        plt.style.use("default")
        
        if len(selected_system_models_jrt) >= 1:
            for item in system_models_jrt:
                if "Blue Waters" in selected_system_models_jrt:
                    plot_cdf(bw_df["run_time"], 1000,"Time (s)", linestyle=":", color="blue")
                if "Mira" in selected_system_models_jrt:
                    plot_cdf(mira_df_2["run_time"], 1000,"Time (s)", linestyle="--", color="red")
                if "Philly" in selected_system_models_jrt:
                    plot_cdf(philly_df["run_time"], 1000,"Time (s)", linestyle="-.", color="green")
                if "Helios" in selected_system_models_jrt:
                    plot_cdf(hl_df["run_time"], 10009999,"Job Run Time (s)", linestyle="--", color="violet")
            
            plt.rc('legend',fontsize=12)
            plt.legend(selected_system_models_jrt, loc="lower right")
        else:
            st.write("## Please select one or more system models in the sidebar to plot the graph.")

        #avoiding the user warning for now
        warnings.filterwarnings("ignore", message="Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.")
        
        plt.xscale("log")
        plt.show()
        st.pyplot(plt.gcf())

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
        # Plots Figure 1(b) from page 3, 3.1.1
        st.markdown("<a name='drt_chart_section'></a>", unsafe_allow_html=True)
        def lt_xs(data, t1, t2):
            lt10min_jobs_num = len(data[data < t2][data >= t1])
            all_jobs_num = len(data)
            return lt10min_jobs_num / all_jobs_num

        def lt_xs_all(t1, t2):
            res = []
            res.append(lt_xs(bw_df["run_time"], t1, t2))
            res.append(lt_xs(mira_df_2["run_time"], t1, t2))
            res.append(lt_xs(philly_df["run_time"], t1, t2))
            res.append(lt_xs(hl_df["run_time"], t1, t2))
            return res

        x = [0, 30, 600, 3600, 12 * 3600, 100000]
        x_value = np.array([1, 2, 3, 4, 5])
        labels = ['0~30s', '30s~10min', '10min~1h', '1h~12h', "more than 12h"]
        bw = []
        mr = []
        ply = []
        hl = []
        width = 0.2

        for i in range(1, len(x)):
            res = lt_xs_all(x[i-1], x[i])
            bw.append(res[0])
            mr.append(res[1])
            ply.append(res[2])
            hl.append(res[3])

        for model in system_models_jrt:
                if "Blue Waters" in drt_selected_system_models_jrt:
                    plt.bar(x_value - 3 * width / 2, bw, width, edgecolor='black', hatch="x", color="blue")
                if "Mira" in drt_selected_system_models_jrt:
                    plt.bar(x_value - width / 2, mr, width, edgecolor='black', hatch="\\", color="red")
                if "Philly" in drt_selected_system_models_jrt:
                    plt.bar(x_value + width / 2, ply, width, edgecolor='black', hatch=".", color="green")
                if "Helios" in drt_selected_system_models_jrt:
                   plt.bar(x_value + 3 * width / 2, hl, width, edgecolor='black', hatch="-", color="violet")
      

        plt.xticks(x_value, labels)
        plt.legend(drt_selected_system_models_jrt, prop={'size': 14}, loc="upper right")
        plt.xlabel("Job Run Time (s)", fontsize=20)
        plt.show()
        st.pyplot()

elif nav_bar_horizontal == "Job Wait Time":
    st.write("Hello, Welcome to Job Wait Time")
else:
    st.write("Hello, Welcome to Model 3")
