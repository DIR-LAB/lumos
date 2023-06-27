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

nav_bar_horizontal = option_menu(None, ["Job Run Time", "Job Arrival Pattern", "Model 3"], default_index=0, orientation="horizontal")

system_models_jrt = ["Mira", "Blue Waters", "Philly", "Helios"]

bw_df = pd.read_csv("../data_blue_waters.csv")
mira_df_2 = pd.read_csv("../data_mira.csv")
hl_df = pd.read_csv("../data_helios.csv")
philly_df = pd.read_csv("../data_philly.csv")
message = st.empty()
columns=["job", "user", "project", "state", "gpu_num", "cpu_num", "node_num", "submit_time", "wait_time", "run_time", "wall_time", "node_hour"]

if nav_bar_horizontal == "Job Run Time":
    system_models_jrt = ["Mira", "Blue Waters", "Philly", "Helios"]
    with st.form("select_chart_model_jrt"):
        #cdf
        st.write("### Select the chart you want to view")
        chart_select_radio_jrt = st.radio("Chart Selection", [None, "CDF Run Time Chart", "Detailed Run Time Distribution Chart"], horizontal=True)
        submit = st.form_submit_button("Select")
        if submit:
            st.write(f"**You have selected**: {chart_select_radio_jrt}")
            if chart_select_radio_jrt == "CDF Run Time Chart":
                st.markdown('<script>scrollToSection("cdf_chart_section")</script>', unsafe_allow_html=True)
            elif chart_select_radio_jrt == "Detailed Run Time Distribution Chart":
                st.markdown('<script>scrollToSection("drt_chart_section")</script>', unsafe_allow_html=True)

    if chart_select_radio_jrt == "CDF Run Time Chart":
        min_value_exp_run_time_slider = 0
        max_value_exp_run_time_slider = 8 
        selected_system_models_jrt = system_models_jrt.copy() 
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
                cdf_frequency_slider_jrt = st.slider("**Adjust frequency range (y-axis)**", min_value=0, max_value=100, step=20, value=100)
                cdf_run_time_slider_jrt = st.slider("**Adjust run time range (in powers of 10) (x-axis)**", min_value_exp_run_time_slider, max_value_exp_run_time_slider, step=1, value=8)
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
                st.markdown("<h2 style='text-align: center; color: black;'>CDF of Run Time Chart</h2>", unsafe_allow_html=True)

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
                
                with st.expander("**CDF Run Time Chart Description:**", expanded=True):
                         st.write("Displays a Cumulative Distribution Functions (CDFs) of the runtime comparisons of the four job traces (Blue Waters, Mira, Philly, and Helios).")
            else:
                st.write("## Please select one or more system model(s) in the sidebar to plot the chart.")

    elif chart_select_radio_jrt == "Detailed Run Time Distribution Chart":
            st.markdown("<h2 style='text-align: center; color: black;'>Detailed Run Time Distribution Chart</h2>", unsafe_allow_html=True)
            # drt = detailed run time
            drt_time_ranges = ['0~30s', '30s~10min', '10min~1h', '1h~12h', "more than 12h"]
            drt_selected_system_models_jrt = system_models_jrt.copy()
            drt_selected_time_range_jrt = drt_time_ranges.copy()
            with st.spinner("In progress...., Please do not change any settings now"): 
                with st.sidebar.form("detailed_run_time_form_jrt"):
                    st.write("## Alter the following settings to customize the detailed run time chart:")
                    with st.expander("**Select System Model(s)**", expanded=True):
                        for item in system_models_jrt:
                            drt_model_checkbox_jrt = st.checkbox(item, True)
                            if not drt_model_checkbox_jrt:
                                drt_selected_system_models_jrt.remove(item)
                    drt_frequency_slider_jrt = st.slider("**Adjust frequency range (y-axis)**", min_value=0.0, max_value=0.6, step=0.1, value=0.6)
                    with st.expander("**Select Run Time Range (x-axis)**", expanded=True):
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
        dsp_selected_system_models_jap = system_models_jap.copy()
        with st.spinner("In progress...., Please do not change any settings now"): 
            st.sidebar.markdown("<h1 style='text-align: center; color: Black;'>Chart Customization Panel</h1>", unsafe_allow_html=True)  
            with st.sidebar.form("dsp_personal_parameters_update_form"):
                st.write("## Alter the following settings to customize the Daily Submit Pattern chart:")
                with st.expander("**Select System Model(s)**", expanded=True):
                    for item in system_models_jap:
                        dsp_model_checkbox_jap = st.checkbox(item, True)
                        if not dsp_model_checkbox_jap:
                            dsp_selected_system_models_jap.remove(item)
                dsp_job_count_slider_jap = st.slider("**Adjust Job Submit Count Range (y-axis)**", min_value=0, max_value=180, step=20, value=180)
                dsp_hour_of_the_day_slider_jap = st.slider("**Adjust Hour of the Day Range (x-axis)**", min_value=-1, max_value=24, step=1, value=24)
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

    elif chart_select_radio_jap == "Weekly Submit Pattern":
        st.markdown("<h2 style='text-align: center; color: black;'>Weekly Submit Pattern Chart</h2>", unsafe_allow_html=True)
        wsp_selected_system_models_jap = system_models_jap.copy()
        with st.spinner("In progress...., Please do not change any settings now"): 
            st.sidebar.markdown("<h1 style='text-align: center; color: Black;'>Chart Customization Panel</h1>", unsafe_allow_html=True)  
            with st.sidebar.form("wsp_personal_parameters_update_form"):
                st.write("## Alter the following settings to customize the Weekly Submit Pattern chart:")
                with st.expander("**Select System Model(s)**", expanded=True):
                    for item in system_models_jap:
                        wsp_model_checkbox_jap = st.checkbox(item, True)
                        if not wsp_model_checkbox_jap:
                            wsp_selected_system_models_jap.remove(item)
                wsp_job_count_slider_jap = st.slider("Adjust Job Submit Count Range (y-axis):", min_value=0, max_value=3000, step=500, value=3000)
                wsp_hour_of_the_day_slider_jap = st.slider("Adjust Day of the Week Range (x-axis):", min_value=0, max_value=8, step=1, value=8)
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



    elif chart_select_radio_jap == "Job Arrival Interval":
        jap_min_value_exp_arrival_interval_slider = 0
        jap_max_value_exp_arrival_interval_slider = 8 
        jai_selected_system_models_jap = system_models_jap.copy()
        st.markdown("<h2 style='text-align: center; color: black;'>Job Arrival Interval Chart</h2>", unsafe_allow_html=True)
        with st.spinner("In progress...., Please do not change any settings now"): 
            st.sidebar.markdown("<h1 style='text-align: center; color: Black;'>Chart Customization Panel</h1>", unsafe_allow_html=True)  
            with st.sidebar.form("jai_personal_parameters_update_form"):
                st.write("## Alter the following settings to customize the Job Arrival Interval chart:")
                with st.expander("**Select System Model(s)**", expanded=True):
                    for item in system_models_jap:
                        jai_model_checkbox_jap = st.checkbox(item, True)
                        if not jai_model_checkbox_jap:
                            jai_selected_system_models_jap.remove(item)
                jai_job_count_slider_jap = st.slider("Adjust Frequency Range (y-axis):", min_value=0, max_value=100, step=20, value=100)
                jai_hour_of_the_day_slider_jap = st.slider("Adjust Job Arrival Interval Range (in powers of 10) (x-axis):", jap_min_value_exp_arrival_interval_slider, jap_max_value_exp_arrival_interval_slider, step=1, value=8)
                jai_hour_of_the_day_slider_value_jap = int(10**jai_hour_of_the_day_slider_jap)
                jai_submit_parameters_button_jap = st.form_submit_button("Apply Changes")
                if jai_submit_parameters_button_jap:
                    if len(jai_selected_system_models_jap) < 1:
                        text_color = "red"
                        st.markdown(f'<span style = "color: {text_color}">Please select one or more system model(s) and click "Apply Changes".</span>', unsafe_allow_html=True)
                    else:
                        pass;
                    
            # Alex your code here
<<<<<<< HEAD
=======
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

>>>>>>> 1c86165dba4451c60db1dbf2d55acfbf09cb868d

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
                plt.xlim(1, jai_hour_of_the_day_slider_value_jap)

                plt.grid(True)
            
            plt.style.use("default")

            plt.figure(figsize=[6,5])

            if len in (jai_selected_system_models_jap) >= 1:
                for item in system_models_jap:
                    if "Blue Waters" in jai_selected_system_models_jap:
                        plot_cdf(get_interval(bw_df["submit_time"]), 1000,"Time (s)", linestyle=":")
                    if "Mira" in jai_selected_system_models_jap:
                        plot_cdf(get_interval(mira_df_2["submit_time"]), 1000,"Time (s)", linestyle="--")
                    if "Philly" in jai_selected_system_models_jap:
                        plot_cdf(get_interval(philly_df["submit_time"]), 1000,"Time (s)", linestyle="-.")
                    if "Helios" in jai_selected_system_models_jap:
                        plot_cdf(get_interval(hl_df["submit_time"]), 10009999,"Job Arrival Interval (s)", linestyle="--")
                
            plt.rc('legend',fontsize=22)
            plt.legend(["bw", "mira", "philly","helios"], loc = "upper right")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.xscale("log")
            st.pyplot(plt.gcf())


            with st.expander("**Job Arrival Interval:**", expanded=True):
                         st.write("Displays a Cumulative Distribution Functions (CDF) of job arrival interval(s) comparison of the four job traces (Blue Waters, Mira, Philly, and Helios).")

elif nav_bar_horizontal == "Model 3":
    st.write("Model 3")


    
    
   

    # Time - 3.08 - Mira CPU
    
    # Time - 32.5 S - blue waters CPU
   
    # Time - 24.71 - blue waters GPU

    # Time - 11.39 - Philly GPU
    
    # Time - 11.93 - Helios GPU 
   
    # Time - 11.34 - Philly GPU - SchedGym 


elif nav_bar_horizontal == "Job Waiting Time":
    # Code for "Model 3" section goes here
    st.write("This is the 'Model 4' section.")

else:
    st.write("Please select a section from the navigation bar.")

