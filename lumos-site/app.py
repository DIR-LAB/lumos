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
import os 

curr_dir = os.path.dirname(__file__)

banner_image_path = os.path.join(curr_dir, 'images/JOB TRACE VISULAZATION.png')

data_blue_waters_path = os.path.join(curr_dir, 'data/data_blue_waters.csv')
data_mira_path = os.path.join(curr_dir, 'data/data_mira.csv')
data_helios_path = os.path.join(curr_dir, 'data/data_helios.csv')
data_philly_path = os.path.join(curr_dir, 'data/data_philly.csv')

banner_image = Image.open(banner_image_path)
st.image(banner_image)

bw_df = pd.read_csv(data_blue_waters_path)
mira_df_2 = pd.read_csv(data_mira_path)
hl_df = pd.read_csv(data_helios_path)
philly_df = pd.read_csv(data_philly_path)

nav_bar_horizontal = option_menu(None, ["Job Run Time", "Job Arrival Pattern", "Sys Util & Res Occu", "Job Waiting Time"], default_index=0, orientation="horizontal")

system_models_jrt = ["Mira", "Blue Waters", "Philly", "Helios"]

message = st.empty()
columns=["job", "user", "project", "state", "gpu_num", "cpu_num", "node_num", "submit_time", "wait_time", "run_time", "wall_time", "node_hour"]

if nav_bar_horizontal == "Job Run Time":
    system_models_jrt = ["Mira", "Blue Waters", "Philly", "Helios"]
    with st.form("select_chart_model_jrt"):
        #cdf
        st.write("### Select a chart you want to view")
        chart_select_radio_jrt = st.radio("Chart Selection", [None, "CDF Run Time Chart", "Detailed Run Time Distribution Chart"], horizontal=True)
        submit = st.form_submit_button("Select")
        if submit:
            if not chart_select_radio_jrt is None:
                st.write(f"**You have selected**: {chart_select_radio_jrt}")
            else:
                text_color = "red"
                st.markdown(f'<span style="color:{text_color}">You have selected "None", please select an other option to view chart.</span>', unsafe_allow_html=True)
            if chart_select_radio_jrt == "CDF Run Time Chart":
                st.markdown('<script>scrollToSection("cdf_chart_section")</script>', unsafe_allow_html=True)
            elif chart_select_radio_jrt == "Detailed Run Time Distribution Chart":
                st.markdown('<script>scrollToSection("drt_chart_section")</script>', unsafe_allow_html=True)

    if chart_select_radio_jrt == "CDF Run Time Chart":
        min_value_exp_run_time_slider = 0
        max_value_exp_run_time_slider = 8 
        selected_system_models_jrt = system_models_jrt.copy() 
        st.markdown("<h2 style='text-align: center; color: black;'>CDF of Run Time Chart</h2>", unsafe_allow_html=True)
        st.sidebar.markdown("<h1 style='text-align: center; color: Black;'>Chart Customization Panel</h1>", unsafe_allow_html=True)

        with st.spinner("In progress...., Please do not change any settings now"):  
            with st.sidebar.form("CDF_chart_form_jrt"):
                st.write("## Alter the following settings to customize the CDF chart:")
                selected_system_models_jrt = system_models_jrt.copy() 
                with st.expander("**Select System Model(s)**", expanded=True):
                    for item in system_models_jrt:
                        model_checkbox_jrt = st.checkbox(item, True)
                        if not model_checkbox_jrt:
                            selected_system_models_jrt.remove(item)
                cdf_frequency_slider_jrt = st.slider("**Adjust frequency range (y-axis):**", min_value=0, max_value=100, step=20, value=100)
                cdf_run_time_slider_jrt = st.slider("**Adjust run time range (in powers of 10) (x-axis):**", min_value_exp_run_time_slider, max_value_exp_run_time_slider, step=1, value=8)
                cdf_run_time_slider_value_jrt = int(10**cdf_run_time_slider_jrt)
                            
                submit_cdf_sidebar_button = st.form_submit_button("Apply")
                if submit_cdf_sidebar_button:
                    if len(selected_system_models_jrt) < 1:
                         text_color = "red"
                         st.markdown(f'<span style="color:{text_color}">Please select one or more system model(s) in the sidebar to plot the CDF chart.</span>', unsafe_allow_html=True)
                    else:
                        pass;
                
            # Plots Figure 1(a) from page 3, 3.1.1
            st.markdown("<a name='cdf_chart_section'></a>", unsafe_allow_html=True)
            def plot_detailed_run_time_distribution(data, bins, xlabel, ylabel="Frequency (%)", color="", linestyle="--"):
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                counts, bin_edges = np.histogram(data, bins=bins)
                counts = counts / float(sum(counts))
                bin_width = bin_edges[1] - bin_edges[0]
                bin_centers = bin_edges[:-1] + bin_width / 2
                if color:
                    plt.bar(bin_centers, counts * 100, width=bin_width, color=color)
                else:
                    plt.bar(bin_centers, counts * 100, width=bin_width)
            def plot_cdf(x, bins, xlabel, ylabel="Frequency (%)", color="", linestyle="--"):
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                x = np.sort(x)
                cdf = 100 * np.arange(len(x)) / float(len(x))
                if color:
                    plt.plot(x, cdf, linestyle=linestyle, linewidth=5, color=color)
                else:
                    plt.plot(x, cdf, linestyle=linestyle, linewidth=5)
                plt.xlabel(xlabel, fontsize=14)
                plt.ylabel(ylabel, fontsize=14)
                plt.margins(0)
                plt.ylim(0, cdf_frequency_slider_jrt) 
                plt.xlim(10**0, cdf_run_time_slider_value_jrt) 
                plt.grid(True)
            plt.style.use("default")

            if len(selected_system_models_jrt) >= 1:
                for item in system_models_jrt:
                    if "Blue Waters" in selected_system_models_jrt:
                        plot_cdf(bw_df["run_time"], 1000, "Time (s)", linestyle=":", color="blue")
                    if "Mira" in selected_system_models_jrt:
                        plot_cdf(mira_df_2["run_time"], 1000, "Time (s)", linestyle="--", color="red")
                    if "Philly" in selected_system_models_jrt:
                        plot_cdf(philly_df["run_time"], 1000, "Time (s)", linestyle="-.", color="green")
                    if "Helios" in selected_system_models_jrt:
                        plot_cdf(hl_df["run_time"], 10009999, "Job Run Time (s)", linestyle="--", color="violet")
                
                plt.rc('legend', fontsize=12)
                plt.legend(selected_system_models_jrt, loc="lower right")
                # Avoiding this user warning for now
                # warnings.filterwarnings("ignore", message="Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.")
                plt.xscale("log")
                st.pyplot()
                
                with st.expander("**CDF Run Time Chart Description:**", expanded=True):
                         st.write("Displays a Cumulative Distribution Functions (CDFs) of the runtime comparisons of the four job traces (Blue Waters, Mira, Philly, and Helios).")
            else:
                st.write("## Please select one or more system model(s) in the sidebar to plot the chart.")

    elif chart_select_radio_jrt == "Detailed Run Time Distribution Chart":
            st.markdown("<h2 style='text-align: center; color: black;'>Detailed Run Time Distribution Chart</h2>", unsafe_allow_html=True)
            st.sidebar.markdown("<h1 style='text-align: center; color: Black;'>Chart Customization Panel</h1>", unsafe_allow_html=True)

            # drt = detailed run time
            drt_time_ranges = ['0~30s', '30s~10min', '10min~1h', '1h~12h', "more than 12h"]
            drt_selected_system_models_jrt = system_models_jrt.copy()
            drt_selected_time_range_jrt = drt_time_ranges.copy()

            with st.spinner("In progress...., Please do not change any settings now"): 
                with st.sidebar.form("detailed_run_time_form_jrt"):
                    st.write("### Alter the following settings to customize the detailed run time chart:")
                    with st.expander("**Select System Model(s)**", expanded=True):
                        for item in system_models_jrt:
                            drt_model_checkbox_jrt = st.checkbox(item, True)
                            if not drt_model_checkbox_jrt:
                                drt_selected_system_models_jrt.remove(item)
                    drt_frequency_slider_jrt = st.slider("**Adjust frequency range (y-axis):**", min_value=0.0, max_value=0.6, step=0.1, value=0.6)
                    with st.expander("**Select Run Time Range (x-axis):**", expanded=True):
                        for item in drt_time_ranges:
                            drt_time_range_checkbox_jrt = st.checkbox(item, True)
                            if not drt_time_range_checkbox_jrt:
                                drt_selected_time_range_jrt.remove(item)
                    submit_drt_sidebar_button = st.form_submit_button("Apply")
                    if submit_drt_sidebar_button:
                        if len(drt_selected_system_models_jrt) < 1:
                            text_color = "red"
                            st.markdown(f'<span style="color:{text_color}">Please select one or more system model(s) in the sidebar to plot the CDF chart.</span>', unsafe_allow_html=True)
                        else:
                            pass;
                # Plots Figure 1(b) from page 4, 3.1.2
                st.markdown("<a name='drt_chart_section'></a>", unsafe_allow_html=True)
                
                def plot_detailed_run_time_distribution(data, bins, xlabel, ylabel="Frequency (%)", color="", linestyle="--"):
                    plt.xticks(fontsize=16)
                    plt.yticks(fontsize=16)
                    counts, bin_edges = np.histogram(data, bins=bins)
                    counts = counts / float(sum(counts))
                    bin_width = bin_edges[1] - bin_edges[0]
                    bin_centers = bin_edges[:-1] + bin_width / 2
                    if color:
                        plt.bar(bin_centers, counts * 100, width=bin_width, color=color)
                    else:
                        plt.bar(bin_centers, counts * 100, width=bin_width)
                    plt.xlabel(xlabel, fontsize=14)
                    plt.ylabel(ylabel, fontsize=14)
                    plt.margins(0)
                    plt.ylim(0, drt_frequency_slider_jrt * 100)
                    plt.xlim(0, drt_selected_system_models_jrt)
                    plt.grid(True)
                plt.style.use("default")
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
                    if labels[i-1] in drt_selected_time_range_jrt:
                        res = lt_xs_all(x[i-1], x[i])
                        bw.append(res[0])
                        mr.append(res[1])
                        ply.append(res[2])
                        hl.append(res[3])
                x_value_selected = np.arange(1, len(drt_selected_time_range_jrt) + 1)

                if len(drt_selected_system_models_jrt) >= 1 and len(drt_selected_time_range_jrt) >= 1:
                    for model in system_models_jrt:
                            if "Blue Waters" in drt_selected_system_models_jrt:
                                plt.bar(x_value_selected - 3 * width / 2, bw, width, edgecolor='black', hatch="x", color="blue")
                            if "Mira" in drt_selected_system_models_jrt:
                                plt.bar(x_value_selected - width / 2, mr, width, edgecolor='black', hatch="\\", color="red")
                            if "Philly" in drt_selected_system_models_jrt:
                                plt.bar(x_value_selected + width / 2, ply, width, edgecolor='black', hatch=".", color="green")
                            if "Helios" in drt_selected_system_models_jrt:
                                plt.bar(x_value_selected + 3 * width / 2, hl, width, edgecolor='black', hatch="-", color="violet")
                    plt.ylim(0.00, drt_frequency_slider_jrt)
                    plt.xticks(x_value_selected, drt_selected_time_range_jrt)
                    plt.legend(drt_selected_system_models_jrt, prop={'size': 12}, loc="upper right")
                    plt.ylabel("Frequency (%)", fontsize=14)
                    plt.xlabel("Job Run Time (s)", fontsize=14)
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.pyplot()
                    with st.expander("**Detailed Run Time Distribution Chart Description:**", expanded=True):
                        st.write("Displays a bar chart of the four job traces categorized by run times (30 sec, 1 min, 10 mins, 1h, and 12+hrs) alongside the frequency in which they occur.")

            
                elif len(drt_selected_system_models_jrt) >= 1 and len(drt_selected_time_range_jrt) < 1:
                    st.write("## Please select one or more time ranges in the sidebar to plot the chart.")
                elif len(drt_selected_system_models_jrt) < 1 and len(drt_selected_time_range_jrt) >= 1:
                    st.write("## Please select one or more system models in the sidebar to plot the chart.")
                else:
                    st.write("## Please select one or more system models and time ranges in the sidebar to plot the chart.")

# Job Arrival pattern page code
elif nav_bar_horizontal == "Job Arrival Pattern":
    system_models_jap = ["Blue Waters", "Mira", "Philly", "Helios"]
    chart_select_radio_jap = None;

    def get_time_of_day(time, timestamp=True):
            if timestamp:
                time = datetime.fromtimestamp(time)
            else:
                time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
            return (time.hour + (time.minute>30))%24, datetime.strftime(time, '%Y-%m-%d')

    def get_day_of_week(time):
            time = datetime.fromtimestamp(time)
            return time.isocalendar()[2], time.isocalendar()[1]
            
    def plot_time_submit(submit_time, xlabel, ylabel="Number of Submitted Jobs", week=False, marker="o", color=""):
            if week == True:
                time, days = list(zip(*[get_time_of_day(i) for i in submit_time]))
                dd = Counter()
                for i in time:
                    dd[i] += 1
                keys = sorted(dd.keys())
                n = len(set(days))
            else:
                days, weeks = list(zip(*[get_day_of_week(i) for i in submit_time]))
                dd = Counter()
                for i in days:
                    dd[i] += 1
                keys = sorted(dd.keys())
                n = len(set(weeks))
            plt.plot(keys, [np.array(dd[j])/n for j in keys], marker=marker, linewidth=3, markersize=12, color=color)


    with st.form("select_chart_model_jap"):
        st.write("#### Select a chart you want to view")
        chart_select_radio_jap = st.radio("Chart Selection", [None, "Daily Submit Pattern", "Weekly Submit Pattern", "Job Arrival Interval"], horizontal=True)
        submit_chart_radio_button_jap = st.form_submit_button("Select")
        if submit_chart_radio_button_jap:
            if chart_select_radio_jap is not None:
                    st.write(f"**You have selected:** {chart_select_radio_jap}")
            else:
                text_color = "red"
                st.markdown(f'<span style="color:{text_color}">You have selected "None", please select an other option to view chart.</span>', unsafe_allow_html=True)

            if chart_select_radio_jap == "Daily Submit Pattern":
                st.markdown('<script>scrollToSection("dsp_chart_section")</script>', unsafe_allow_html=True)
            elif chart_select_radio_jap == "Weekly Submit Pattern":
                st.markdown('<script>scrollToSection("wsp_chart_section")</script>', unsafe_allow_html=True)
            elif chart_select_radio_jap == "Job Arrival Interval":
                st.markdown('<script>scrollToSection("jap_chart_section")</script>', unsafe_allow_html=True)
        
            
    #  Code for individual charts             
    if chart_select_radio_jap == "Daily Submit Pattern":
        st.markdown("<h2 style='text-align: center; color: black;'>Daily Submit Pattern Chart</h2>", unsafe_allow_html=True)
        st.sidebar.markdown("<h1 style='text-align: center; color: Black;'>Chart Customization Panel</h1>", unsafe_allow_html=True)
        
        dsp_selected_system_models_jap = system_models_jap.copy()
        with st.spinner("In progress...., Please do not change any settings now"):   
            with st.sidebar.form("dsp_personal_parameters_update_form"):
                st.write("### Alter the following settings to customize the Daily Submit Pattern chart:")
                with st.expander("**Select System Model(s)**", expanded=True):
                    for item in system_models_jap:
                        dsp_model_checkbox_jap = st.checkbox(item, True)
                        if not dsp_model_checkbox_jap:
                            dsp_selected_system_models_jap.remove(item)
                dsp_job_count_slider_jap = st.slider("**Adjust Job Submit Count Range (y-axis):**", min_value=0, max_value=180, step=20, value=180)
                dsp_hour_of_the_day_slider_jap = st.slider("**Adjust Hour of the Day Range (x-axis):**", min_value=-1, max_value=24, step=1, value=24)
                dap_submit_parameters_button_jap = st.form_submit_button("Apply Changes")
                if dap_submit_parameters_button_jap:
                    if len(dsp_selected_system_models_jap) < 1:
                        text_color = "red"
                        st.markdown(f'<span style = "color: {text_color}">Please set system model(s) above first and then adjust the parameters here.</span>', unsafe_allow_html=True)
                    else:
                        pass;
                        

            plt.figure(figsize=(12,7))
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)

            if len(dsp_selected_system_models_jap) >=1 :

                for item in dsp_selected_system_models_jap:
                    if "Blue Waters" in dsp_selected_system_models_jap:
                        plot_time_submit(bw_df["submit_time"], xlabel="Hour of the Day", week=True,marker="^", color="blue")
                    if "Mira" in dsp_selected_system_models_jap:
                        plot_time_submit(mira_df_2["submit_time"], xlabel="Hour of the Day", week=True,marker="o", color="red")
                    if "Philly" in dsp_selected_system_models_jap:
                        plot_time_submit(philly_df["submit_time"], xlabel="Hour of the Day", week=True,marker="s", color="green")
                    if "Helios" in dsp_selected_system_models_jap:
                        plot_time_submit(hl_df["submit_time"], xlabel="Hour of the Day", week=True,marker="d", color="violet") 
        
                plt.xlabel("Hour of the Day", fontsize=18)
                plt.ylabel("Job Submit Count", fontsize=18)
                st.set_option('deprecation.showPyplotGlobalUse', False)
                plt.ylim(0, dsp_job_count_slider_jap)
                plt.xlim(-1, dsp_hour_of_the_day_slider_jap)
                plt.tight_layout()
                plt.grid(True)
                plt.legend(dsp_selected_system_models_jap,  prop={'size': 14}, loc="upper right")
                plt.rc('legend',fontsize=20)
                st.pyplot()

                with st.expander("**Daily Submit Pattern Chart Description:**", expanded=True):
                    st.write("Displays a chart presenting the job arrival counts of each job trace for each hour of the day")
            else:
                st.write("## Please select one or more system model(s) from sidebar to plot the chart")

    elif chart_select_radio_jap == "Weekly Submit Pattern":
        st.markdown("<h2 style='text-align: center; color: black;'>Weekly Submit Pattern Chart</h2>", unsafe_allow_html=True)
        st.sidebar.markdown("<h1 style='text-align: center; color: Black;'>Chart Customization Panel</h1>", unsafe_allow_html=True)
       
        wsp_selected_system_models_jap = system_models_jap.copy()
        with st.spinner("In progress...., Please do not change any settings now"):   
            with st.sidebar.form("wsp_personal_parameters_update_form"):
                st.write("### Alter the following settings to customize the Weekly Submit Pattern chart:")
                with st.expander("**Select System Model(s)**", expanded=True):
                    for item in system_models_jap:
                        wsp_model_checkbox_jap = st.checkbox(item, True)
                        if not wsp_model_checkbox_jap:
                            wsp_selected_system_models_jap.remove(item)
                wsp_job_count_slider_jap = st.slider("**Adjust Job Submit Count Range (y-axis):**", min_value=0, max_value=3000, step=500, value=3000)
                wsp_hour_of_the_day_slider_jap = st.slider("**Adjust Day of the Week Range (x-axis):**", min_value=0, max_value=8, step=1, value=8)
                wsp_submit_parameters_button_jap = st.form_submit_button("Apply Changes")
                if wsp_submit_parameters_button_jap:
                    if len(wsp_selected_system_models_jap) < 1:
                        text_color = "red"
                        st.markdown(f'<span style = "color: {text_color}">Please select one or more system model(s) and click "Apply Changes".</span>', unsafe_allow_html=True)
                    else:
                        pass;

            plt.figure(figsize=(12,7))
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16) 

            if len(wsp_selected_system_models_jap) >= 1:
                for item in wsp_selected_system_models_jap:
                    if "Blue Waters" in wsp_selected_system_models_jap:
                        plot_time_submit(bw_df["submit_time"], xlabel="Day of the Week", week=False,marker="^", color="blue")
                    if "Mira" in wsp_selected_system_models_jap:
                        plot_time_submit(mira_df_2["submit_time"], xlabel="Day of the Week", week=False,marker="o", color="red")
                    if "Philly" in wsp_selected_system_models_jap:
                        plot_time_submit(philly_df["submit_time"], xlabel="Day of the Week", week=False,marker="s", color="green")
                    if "Helios" in wsp_selected_system_models_jap:
                        plot_time_submit(hl_df["submit_time"], xlabel="Day of the Week", week=False,marker="d", color="violet")
                plt.xlabel("Day of the Week", fontsize=20)
                plt.ylabel("Job Submit Count", fontsize=20)
                plt.ylim(0, wsp_job_count_slider_jap)
                plt.tight_layout()
                plt.xlim(0, wsp_hour_of_the_day_slider_jap)
                plt.grid(True)
                plt.legend(wsp_selected_system_models_jap,  prop={'size': 14}, loc="upper right")
                st.set_option('deprecation.showPyplotGlobalUse', False)
                plt.rc('legend',fontsize=20)
                st.pyplot()

                with st.expander("**Weekly Submit Pattern Chart Description:**", expanded=True):
                                st.write("Displays a chart presenting the job arrival counts of each job trace for each day of the week")
            else:
                st.write("## Please select one or more system model(s) from sidebar to plot the chart")

    elif chart_select_radio_jap == "Job Arrival Interval":
        st.markdown("<h2 style='text-align: center; color: black;'>Job Arrival Interval Chart</h2>", unsafe_allow_html=True)
        st.sidebar.markdown("<h1 style='text-align: center; color: Black;'>Chart Customization Panel</h1>", unsafe_allow_html=True)  
       
        jap_min_value_exp_arrival_interval_slider = 0
        jap_max_value_exp_arrival_interval_slider = 8 
        jai_selected_system_models_jap = system_models_jap.copy()

        with st.spinner("In progress...., Please do not change any settings now"): 
            with st.sidebar.form("jai_personal_parameters_update_form"):
                st.write("### Alter the following settings to customize the Job Arrival Interval chart:")
                with st.expander("**Select System Model(s)**", expanded=True):
                    for item in system_models_jap:
                        jai_model_checkbox_jap = st.checkbox(item, True)
                        if not jai_model_checkbox_jap:
                            jai_selected_system_models_jap.remove(item)
                jai_job_count_slider_jap = st.slider("**Adjust Frequency Range (y-axis):**", min_value=0, max_value=100, step=20, value=100)
                jai_hour_of_the_day_slider_jap = st.slider("**Adjust Job Arrival Interval Range (in powers of 10) (x-axis):**", jap_min_value_exp_arrival_interval_slider, jap_max_value_exp_arrival_interval_slider, step=1, value=8)
                jai_hour_of_the_day_slider_value_jap = int(10**jai_hour_of_the_day_slider_jap)
                jai_submit_parameters_button_jap = st.form_submit_button("Apply Changes")
                if jai_submit_parameters_button_jap:
                    if len(jai_selected_system_models_jap) < 1:
                        text_color = "red"
                        st.markdown(f'<span style = "color: {text_color}">Please select one or more system model(s) and click "Apply Changes".</span>', unsafe_allow_html=True)
                    else:
                        pass;
                    
            # Alex your code here
            # define function for plotting CDF 
            def plot_cdf(x,bins ,xlabel, ylabel="Frequency (%)",color="", linestyle="--"):
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16) 
                x = np.sort(x)
                cdf = 100*np.arange(len(x)) / float(len(x))
                if color:
                    plt.plot(x, cdf, linestyle=linestyle, linewidth=5, color=color)
                else:
                    plt.plot(x, cdf, linestyle=linestyle, linewidth=5)
                plt.xlabel(xlabel, fontsize=20)
                plt.ylabel(ylabel, fontsize=20)
                plt.margins(0)
                plt.ylim(0, 100)
                plt.grid(True)


            # Job Arrival Interval (s) Fig 2c
            def get_interval(a, peak=False):
                def get_time_of_day2(time):
                    time = datetime.fromtimestamp(time)
                    return (time.hour + (time.minute>30))%24
                if peak:
                    z = a.apply(get_time_of_day2)
                    b = a-a.shift(1)
                    c = b[(z>=8) & (z<=17)]
                    return c
                return a-a.shift(1)
            
            # define function for plotting CDF
            def plot_cdf(x,bins ,xlabel, ylabel="Frequency (%)",color="", linestyle="--"):
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16) 
                
                x = np.sort(x)
                cdf = 100*np.arange(len(x)) / float(len(x))

                if color:
                    plt.plot(x, cdf, linestyle=linestyle, linewidth=5, color=color)
                else:
                    plt.plot(x, cdf, linestyle=linestyle, linewidth=5)

                plt.xlabel(xlabel, fontsize=20)
                plt.ylabel(ylabel, fontsize=20)
                plt.margins(0)
                plt.ylim(0, jai_job_count_slider_jap)
                plt.xlim(int(10 ** jap_min_value_exp_arrival_interval_slider), jai_hour_of_the_day_slider_value_jap)

                plt.grid(True)
            
            plt.style.use("default")

            plt.figure(figsize=[6,5])

            if len(jai_selected_system_models_jap) >= 1:
                for item in jai_selected_system_models_jap:
                    if "Blue Waters" in jai_selected_system_models_jap:
                        plot_cdf(get_interval(bw_df["submit_time"]), 1000,"Time (s)", linestyle=":")
                    if "Mira" in jai_selected_system_models_jap:
                        plot_cdf(get_interval(mira_df_2["submit_time"]), 1000,"Time (s)", linestyle="--")
                    if "Philly" in jai_selected_system_models_jap:
                        plot_cdf(get_interval(philly_df["submit_time"]), 1000,"Time (s)", linestyle="-.")
                    if "Helios" in jai_selected_system_models_jap:
                        plot_cdf(get_interval(hl_df["submit_time"]), 10009999,"Job Arrival Interval (s)", linestyle="--")
                    
                plt.rc('legend',fontsize=22)
                plt.legend(jai_selected_system_models_jap, loc = "upper right", prop={'size': 14})
                st.set_option('deprecation.showPyplotGlobalUse', False)
                plt.xscale("log")
                st.pyplot()

                with st.expander("**Job Arrival Interval:**", expanded=True):
                            st.write("Displays a Cumulative Distribution Functions (CDF) of job arrival interval(s) comparison of the four job traces (Blue Waters, Mira, Philly, and Helios).")
            else:
                st.write("## Please select one or more system model(s) from sidebar to plot the chart")
# System Utilization and Resource Occupation page

# System Utilization and Resource Occupation page
elif nav_bar_horizontal == "Sys Util & Res Occu":
    select_cpu_gpu_radio_suaro = None
    select_cpu_radio_suaro = None
    select_gpu_radio_suaro = None
    chart_options_suaro = ["Blue Waters CPU", "Mira CPU", "Blue Waters GPU", 
                "Philly GPU", "Helios GPU", "Philly GPU-SchedGym"]
    cpu_chart_options_suaro = ["Blue Waters CPU", "Mira CPU"]
    gpu_chart_options_suaro = ["Blue Waters GPU", 
                "Philly GPU", "Helios GPU", "Philly GPU-SchedGym"]
    selected_charts_list_suaro = []

    with st.form("select_charts_checkbox_main_form_suaro"): 
        st.write("### Please select one or more option(s) below to view there charts")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<h4 style="text-align: center;">CPU Charts</h4>', unsafe_allow_html=True)
            for item in cpu_chart_options_suaro:
                chart_selected_suaro = st.checkbox(item)
                if chart_selected_suaro:
                    selected_charts_list_suaro.append(item)
        with col2:
            st.markdown('<h4 style="text-align: center;">GPU Charts</h4>', unsafe_allow_html=True)
            for item in gpu_chart_options_suaro:
                chart_selected_suaro = st.checkbox(item)
                if chart_selected_suaro:
                    selected_charts_list_suaro.append(item)
        select_charts_checkbox_main_form_button_suaro = st.form_submit_button("Load Charts")
        if select_charts_checkbox_main_form_button_suaro:
            if len(selected_charts_list_suaro) >= 1:
                st.write(f'**You have selected:** {selected_charts_list_suaro}')
            else:
                st.markdown("<h5 style='color: red;'>You have not selected any chart options above, please select one or more chart option(s) to load the charts</h5>", unsafe_allow_html=True)
        else: 
            pass

    st.sidebar.markdown("<h1 style='text-align: center; color: Black;'>Chart Customization Panel</h1>", unsafe_allow_html=True)
    with st.sidebar.form("sidebar_form_suaro"):
        st.write("### Alter the following settings to customize the selected chart(s):")
        sys_utilization_slider_suaro = st.slider("**Adjust System Utilization Range (Y-axis):**", min_value = 0, max_value=100, value=100, step=20)
        time_slider_suaro = st.slider("**Adjust Time Range (X-axis):**", min_value=0, max_value=120, value=120, step=20)
        submit_button_sidebar_suaro = st.form_submit_button("Apply Changes")
    
    def plot_util(data, total_nodes, key="node_num", color='b'):
        data = data.copy()
        start_time = list(data["submit_time"])[0]
        end_time = list(data["submit_time"])[-1]
        duration = end_time - start_time
        days = int(duration/(24*3600))
        days_usage = [0]*days
        data["start_time"] = data.apply(lambda row: row["submit_time"] + row["wait_time"]-start_time, axis=1)
        data["end_time"] = data.apply(lambda row: row["start_time"] + row["run_time"], axis=1)
        data["start_day"] = data.apply(lambda row: int(row["start_time"]/(24*3600)), axis=1)
        data["end_day"] = data.apply(lambda row: int(row["end_time"]/(24*3600))+1, axis=1)
    #     print(data[data["start_day"]==60])
    #     print(data[data["start_day"]==90])
        for index, row in data.iterrows():
            for i in range(int(row["start_day"]), int(row["end_day"])):
                if i <len(days_usage):
                    days_usage[i] += row[key]*(min(row["end_time"], (i+1)*24*3600)-max(row["start_time"], i*24*3600))
    #     for i in range(len(days_usage)):
    #         days_usage[i] = min((total_nodes*24*3600), days_usage[i])
    #     days_usage = days_usage[10:130]
        # print(np.mean(np.array(days_usage)/(total_nodes*24*3600)))
        plt.bar(range(len(days_usage)), 100*np.array(days_usage)/(total_nodes*24*3600), color=color)
        plt.plot([-10, 150], [80]*2, color="black", linestyle="--")
        plt.ylim(0, sys_utilization_slider_suaro)
        plt.xlim(0, time_slider_suaro)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        plt.xlabel("Time (Days)", fontsize=26)
        plt.ylabel("System Utilization(%)", fontsize=26)
        st.pyplot()
        
    with st.spinner("In Progess... Please do not change any settings now"):

            col1, col2 = st.columns(2)
            for idx, item in enumerate(selected_charts_list_suaro):
                col_logic_cal_suaro = col1 if idx % 2 == 0 else col2
                if item == "Blue Waters CPU":
                    with col_logic_cal_suaro:
                        st.markdown("<h4 style='text-align: center;'>Blue Waters CPU Chart</h4>", unsafe_allow_html=True)
                        plot_util(bw_df[1000:], 22636*32, "cpu_num", color="#1f77b4")  
                elif item == "Mira CPU":
                    with col_logic_cal_suaro:
                        st.markdown("<h4 style='text-align: center;'>Mira CPU Chart</h4>", unsafe_allow_html=True)
                        plot_util(mira_df_2, 49152, color='#ff7f0e')
                elif item == "Blue Waters GPU":
                    with col_logic_cal_suaro:
                        st.markdown("<h4 style='text-align: center;'>Blue Waters GPU Chart</h4>", unsafe_allow_html=True)
                        plot_util(bw_df[1000:], 4228, "gpu_num", color="#1f77b4")
                elif item == "Philly GPU":
                    with col_logic_cal_suaro:
                        st.markdown("<h4 style='text-align: center;'>Philly GPU Chart</h4>", unsafe_allow_html=True)
                        plot_util(philly_df, 2490, "gpu_num", color='#2ca02c')
                elif item == "Helios GPU":
                    with col_logic_cal_suaro:
                        st.markdown("<h4 style='text-align: center;'>Helios GPU Chart</h4>", unsafe_allow_html=True)
                        plot_util(hl_df, 1080, "gpu_num")
                elif item == "Philly GPU-SchedGym":
                    with col_logic_cal_suaro:
                        st.markdown("<h4 style='text-align: center;'>Philly GPU-SchedGym Chart</h4>", unsafe_allow_html=True)
                        data_philly_df_schedule_path = os.path.join(curr_dir, 'data/philly_df_schedule.csv')
                        ppppp = pd.read_csv(data_philly_df_schedule_path)
                        plot_util(ppppp, 2490, "gpu_num", color='#9467bd')

# Job Waiting Time Page
elif nav_bar_horizontal == "Job Waiting Time":
    chart_select_radio_jwt = None
    system_models_jwt = ["Blue Waters", "Mira", "Philly", "Helios"]

    def plot_cdf(x,bins ,xlabel, ylabel="Frequency (%)",color="", linestyle="--"):
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16) 
        x = np.sort(x)
        cdf = 100*np.arange(len(x)) / float(len(x))
        if color:
            plt.plot(x, cdf, linestyle=linestyle, linewidth=5, color=color)
        else:
            plt.plot(x, cdf, linestyle=linestyle, linewidth=5)
        plt.xlabel(xlabel, fontsize=20)
        plt.ylabel(ylabel, fontsize=20)
        plt.margins(0)
        plt.ylim(0, 100)
        plt.grid(True)

    #Function to calculate Average Wait Time charts
    def plot_percentage_corehour(selected_job_sizes, frequency_value, selected_models, run_time=False):
            plt.style.use("default")
            traces = selected_models
            # Chart - Average Waiting Time w.r.t Job Run Time
            short_job_size_dic = {'Blue Waters': 1.74, 'Mira': 4.70, 'Philly': 1.17, 'Helios': 1.97}
            middle_job_size_dic = {'Blue Waters': 62.07, 'Mira': 81.24, 'Philly': 14.32, 'Helios': 22.28}
            long_job_size_dic = {'Blue Waters': 36.18, 'Mira': 14.05, 'Philly': 84.51, 'Helios': 75.75}

            small_job_size_dic = {'Blue Waters': 86.21, 'Mira': 34.12, 'Philly': 18.48, 'Helios': 4.57}
            middle2_job_size_dic = {'Blue Waters': 4.48, 'Mira': 46.63, 'Philly': 68.87, 'Helios': 37.93}
            large_job_size_dic = {'Blue Waters': 9.31, 'Mira': 19.25, 'Philly': 12.65, 'Helios': 57.50}

            if run_time:
                status = {}
                if "Short" in selected_job_sizes:
                    status['Short'] = [short_job_size_dic[system_model] for system_model in short_job_size_dic if system_model in selected_models]
                else:
                    pass
                if "Middle" in selected_job_sizes:
                    status['Middle'] = [middle_job_size_dic[system_model] for system_model in middle_job_size_dic if system_model in selected_models]
                else:
                    pass
                if "Long" in selected_job_sizes:
                    status['Long'] = [long_job_size_dic[system_model] for system_model in long_job_size_dic if system_model in selected_models]
                else:
                    pass
            else:
                status = {}
                if "Small" in selected_job_sizes:
                    status['Small'] = [small_job_size_dic[system_model] for system_model in small_job_size_dic if system_model in selected_models]
                else:
                    pass
                if "Middle" in selected_job_sizes:
                    status['Middle'] = [middle2_job_size_dic[system_model] for system_model in middle2_job_size_dic if system_model in selected_models]
                else:
                    pass
                if "Large" in selected_job_sizes:
                    status['Large'] = [large_job_size_dic[system_model] for system_model in large_job_size_dic if system_model in selected_models]
                else:
                    pass

            x = np.arange(len(traces))  # the label locations
            width = 0.25  # the width of the bars
            multiplier = 0

            fig, ax = plt.subplots()
            hatches= ["-", ".", "x", "-"]
            for i, (attribute, measurement) in enumerate(status.items()):
                offset = width * multiplier
                rects = ax.bar(x + offset, measurement, width, label=attribute, hatch=hatches[i], edgecolor='black')
                ax.bar_label(rects, padding=3)
                multiplier += 1

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel('Percentage (%)', fontsize=18)
            ax.set_xlabel('Traces', fontsize=18)
            ax.set_xticks(x + width, traces, fontsize=15)
            ax.legend(fontsize=14, loc="upper right")
            ax.set_ylim(0, frequency_value)
            plt.grid(axis="y")
            st.pyplot(fig)

    with st.form("select_chart_model_jwt"):
        st.write("### Select a chart you want to view")
        chart_select_radio_jwt = st.radio("Chart Selection", [None, "CDF of Wait Time", "CDF of Turnaround Time", "Avg waiting Time w.r.t Job Size", "Avg Waiting Time w.r.t Job Run Time"])
        chart_selection_submit_button = st.form_submit_button("Select")
        if chart_selection_submit_button:
            if not chart_select_radio_jwt is None:
                st.write(f"**You have selected:** {chart_select_radio_jwt}")
            else:
                st.markdown('<h6 style="color: red;">You have selected "None", please select an other option to view chart.</h6>', unsafe_allow_html=True)
        else:
            pass
    
    if chart_select_radio_jwt == "CDF of Wait Time":
        #cdfowt - CDF of Wait Time
        cdfowt_selected_system_models_jwt = system_models_jwt.copy()
        cdfowt_min_value_exp_arrival_interval_slider = 0
        cdfowt_max_value_exp_arrival_interval_slider = 8 

        st.sidebar.markdown("<h1 style='text-align: center; color: Black;'>Chart Customization Panel</h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: black;'>CDF of Wait Time</h2>", unsafe_allow_html=True)
        

        with st.spinner("In progress...., Please do not change any settings now"):
            with st.sidebar.form("cdfowt_personal_parameters_update_form"):
                st.write("### Alter the following settings to customize the CDF of Wait Time Chart:")

                with st.expander("**Select System Model(s)**", expanded=True):
                    for item in system_models_jwt:
                        cdfowt_model_checkbox_jwt = st.checkbox(item, True)
                        if not cdfowt_model_checkbox_jwt:
                            cdfowt_selected_system_models_jwt.remove(item)
                        else:
                            pass
                
                cdfowt_frequency_slider_jwt = st.slider("**Adjust Frequency(%) Range (Y-axis):**", min_value=0, max_value=100, value=100, step=20)
                cdfowt_job_wait_time_slider_jwt = st.slider("**Adjust Job Wait Time Range (in powers of 10) (X-axis):**", cdfowt_min_value_exp_arrival_interval_slider, cdfowt_max_value_exp_arrival_interval_slider, value=cdfowt_max_value_exp_arrival_interval_slider, step=1)
                cdfowt_job_wait_time_slider_value_jwt = int(10 ** cdfowt_job_wait_time_slider_jwt)         
                cdfowt_submit_parameters_button_jwt = st.form_submit_button("Apply Changes")

            #Graph Code
            if len(cdfowt_selected_system_models_jwt) >= 1:
                for items in system_models_jwt:
                    if "Blue Waters" in cdfowt_selected_system_models_jwt:
                        plot_cdf(bw_df["wait_time"], 100000, "Job Wait Time (s)")
                    if "Mira" in cdfowt_selected_system_models_jwt:
                        plot_cdf(mira_df_2["wait_time"], 100000, "Job Wait Time (s)")
                    if "Philly" in cdfowt_selected_system_models_jwt:
                        plot_cdf(philly_df[10000:130000]["wait_time"], 100000, "Job Wait Time (s)")
                    if "Helios" in cdfowt_selected_system_models_jwt:
                        plot_cdf(hl_df["wait_time"], 100000, "Job Wait Time (s)")

                plt.ylim(0, cdfowt_frequency_slider_jwt)
                plt.xlim(int(10**cdfowt_min_value_exp_arrival_interval_slider), cdfowt_job_wait_time_slider_value_jwt)
                plt.rc('legend', fontsize=12)
                plt.legend(cdfowt_selected_system_models_jwt, loc="lower right")
                st.set_option('deprecation.showPyplotGlobalUse', False)
                plt.xscale("log")
                st.pyplot()

            with st.expander("**CDF of wait Time Chart Description:**", expanded=True):
                            st.write("Compares the CDF of the waiting time of each job across the four traces")

    
    elif chart_select_radio_jwt == "CDF of Turnaround Time":
        cdfott_selected_system_models_jwt = system_models_jwt.copy()
        cdfott_min_value_exp_arrival_interval_slider = 0
        cdfott_max_value_exp_arrival_interval_slider = 8 

        st.sidebar.markdown("<h1 style='text-align: center; color: Black;'>Chart Customization Panel</h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: black;'>CDF of Turnaround Time</h2>", unsafe_allow_html=True)

        
        with st.spinner("In progress...., Please do not change any settings now"):
            with st.sidebar.form("cdfott_personal_parameters_update_form"):
                st.write("### Alter the following settings to customize the CDF of Turnaround Time Chart:")

                with st.expander("**Select System Model(s)**", expanded=True):
                    for item in system_models_jwt:
                        cdfott_model_checkbox_jwt = st.checkbox(item, True)
                        if not cdfott_model_checkbox_jwt:
                            cdfott_selected_system_models_jwt.remove(item)
                        else:
                            pass

                cdfott_frequency_slider_jwt = st.slider("**Adjust Frequency(%) Range (Y-axis):**", min_value=0, max_value=100, value=100, step=20)
                cdfott_turnaround_time_slider_jwt = st.slider("**Adjust Turnaround Time Range (in powers of 10) (X-axis):**", cdfott_min_value_exp_arrival_interval_slider, cdfott_max_value_exp_arrival_interval_slider, value=8, step=1)
                cdfott_turnaround_time_slider_value_jwt = int(10 ** cdfott_turnaround_time_slider_jwt)         
                cdfott_submit_parameters_button_jwt = st.form_submit_button("Apply Changes")

            #Graph Code


            with st.expander("**CDF of Turnaround Time Chart Description:**", expanded=True):
                            st.write("Description Goes Here")

    elif chart_select_radio_jwt == "Avg waiting Time w.r.t Job Size":
        # AWTJS: Avg waiting Time w.r.t Job Size
        awtjs_selected_system_models_jwt = system_models_jwt.copy()
        awtjs_job_sizes_list_jwt = ["Small", "Middle", "Large"]
        awtjs_job_sizes_selected_list_jwt = awtjs_job_sizes_list_jwt.copy()

        st.sidebar.markdown("<h1 style='text-align: center; color: Black;'>Chart Customization Panel</h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: black;'>Avg Waiting Time w.r.t Job Size Chart</h2>", unsafe_allow_html=True)

        with st.spinner("In progress...., Please do not change any settings now"):
            with st.sidebar.form("awtjs_personal_parameters_update_form"):
                st.write("### Alter the following settings to customize the Avg Waiting Time w.r.t Job Size chart:")
                with st.expander("**Select Job Size(s)**", expanded=True):
                    for item in awtjs_job_sizes_list_jwt:
                        awtjs_job_size_checkbox_jwt = st.checkbox(item, True)
                        if not awtjs_job_size_checkbox_jwt:
                            awtjs_job_sizes_selected_list_jwt.remove(item)
                        else:
                            pass
                
                awtjs_avg_wait_time_slider_jwt = st.slider("**Adjust Average Wait Time (hours) Range (Y-axis):**", min_value=0, max_value=100, value=100, step=10)
            
                with st.expander("**Select System Model(s) (x-axis)**", expanded=True):
                    for item in system_models_jwt:
                        awtjs_model_checkbox_jwt = st.checkbox(item, True)
                        if not awtjs_model_checkbox_jwt:
                            awtjs_selected_system_models_jwt.remove(item)
                        else:
                            pass
                awtjs_submit_parameters_button_jwt = st.form_submit_button("Apply Changes")
            
            if len(awtjs_job_sizes_selected_list_jwt) >= 1 and len(awtjs_selected_system_models_jwt) >= 1:
                plot_percentage_corehour(awtjs_job_sizes_selected_list_jwt, awtjs_avg_wait_time_slider_jwt, awtjs_selected_system_models_jwt)
                with st.expander("**Avg Waiting Time w.r.t Job Size Chart Description:**", expanded=True):
                    st.write("Description Goes Here")
            elif len(awtjs_job_sizes_selected_list_jwt) < 1 and len(awtjs_selected_system_models_jwt) >= 1:
                st.write("## Please select one or more job size(s) from sidebar to plot the chart")
            elif len(awtjs_job_sizes_selected_list_jwt) >= 1 and len(awtjs_selected_system_models_jwt) < 1:
                st.write("## Please select one or more system model(s) from sidebar to plot the chart")
            else:
                st.write("## Please select one or more system model(s) and job size(s) from sidebar to plot the chart")

    
    elif chart_select_radio_jwt == "Avg Waiting Time w.r.t Job Run Time":
        # AWTJRT: Average Waiting Time w.r.t Job Run Time
        awtjrt_selected_system_models_jwt = system_models_jwt.copy()
        awtjrt_job_run_time_list_jwt = ["Short", "Middle", "Long"]
        awtjrt_job_run_time_selected_list_jwt = awtjrt_job_run_time_list_jwt.copy()

        st.sidebar.markdown("<h1 style='text-align: center; color: Black;'>Chart Customization Panel</h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: black;'>Avg Waiting Time w.r.t Job Run Time Chart</h2>", unsafe_allow_html=True)
       
        with st.spinner("In progress...., Please do not change any settings now"):
            with st.sidebar.form("awtjrt_personal_parameters_update_form"):
                st.write("### Alter the following settings to customize the Avg Waiting Time w.r.t Job Run Time chart:")
                with st.expander("**Select Job Run Time(s)**", expanded=True):
                    for item in awtjrt_job_run_time_list_jwt:
                        awtjrt_job_run_time_checkbox_jwt = st.checkbox(item, True)
                        if not awtjrt_job_run_time_checkbox_jwt:
                            awtjrt_job_run_time_selected_list_jwt.remove(item)
                        else:
                            pass

                awtjrt_avg_wait_time_slider_jwt = st.slider("**Adjust Average Wait Time (hours) Range (Y-axis):**", min_value=0, max_value=100, value=100, step=10)

                with st.expander("**Select System Model(s) (x-axis)**", expanded=True):
                    for item in system_models_jwt:
                        awtjrt_model_checkbox_jwt = st.checkbox(item, True)
                        if not awtjrt_model_checkbox_jwt:
                            awtjrt_selected_system_models_jwt.remove(item)
                        else:
                            pass
                        
                awtjrt_submit_parameters_button_jwt = st.form_submit_button("Apply Changes")
        
            if len(awtjrt_job_run_time_selected_list_jwt) >= 1 and len(awtjrt_selected_system_models_jwt) >= 1:
                plot_percentage_corehour(awtjrt_job_run_time_selected_list_jwt, awtjrt_avg_wait_time_slider_jwt, awtjrt_selected_system_models_jwt, True)
                with st.expander("**Avg Waiting Time w.r.t Job Run Time Chart Description:**", expanded=True):
                    st.write("Description Goes Here")
            elif len(awtjrt_job_run_time_selected_list_jwt) < 1 and len(awtjrt_selected_system_models_jwt) >= 1:
                st.write("## Please select one or more job run time(s) from sidebar to plot the chart")
            elif len(awtjrt_job_run_time_selected_list_jwt) >= 1 and len(awtjrt_selected_system_models_jwt) < 1:
                st.write("## Please select one or more system model(s) from sidebar to plot the chart")
            else:
                st.write("## Please select one or more system model(s) and job run time(s) from sidebar to plot the chart")
    else:
        pass

else:
    st.write("Please select a section from the navigation bar.")

