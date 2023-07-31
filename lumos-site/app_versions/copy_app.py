import streamlit as st
from PIL import Image
from heapq import *
from collections import defaultdict, Counter
import bisect
import numpy as np
import pandas as pd
import warnings
from streamlit_option_menu import option_menu
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from datetime import datetime
from collections import Counter, defaultdict
import json
import time 
import matplotlib
import os 

st.set_page_config(page_title="Job Trace Visualization Application", page_icon="📊")
curr_dir = os.path.dirname(__file__)

banner_image_path = os.path.join(curr_dir, 'images/App Banner Image.png')

data_blue_waters_path = os.path.join(curr_dir, 'data/data_blue_waters.csv')
data_mira_path = os.path.join(curr_dir, 'data/data_mira.csv')
data_helios_path = os.path.join(curr_dir, 'data/data_helios.csv')
data_philly_path = os.path.join(curr_dir, 'data/data_philly.csv')
data_philly_gpu_schedule_path = os.path.join(curr_dir, 'data/philly_df_schedule.csv')

banner_image = Image.open(banner_image_path)
st.image(banner_image)

@st.cache_data
def load_data():
    bw = pd.read_csv(data_blue_waters_path)
    mira = pd.read_csv(data_mira_path)
    hl = pd.read_csv(data_helios_path)
    philly = pd.read_csv(data_philly_path)
    philly_gpu = pd.read_csv(data_philly_gpu_schedule_path)
    return bw, mira, hl, philly, philly_gpu


bw_df, mira_df_2, hl_df, philly_df, philly_gpu_schedule_df = load_data()

styles = {
    "nav-link-selected": {
         "background-color": "black",
    }
}

#Common title, button, and loading text variables
chart_selection_form_title = "Charts Selection Form"
chart_selection_form_load_charts_text = "Select/Deselect charts below and then click 'Load Charts' to apply your changes."
chart_side_by_side_checkbox_highlight_text = "Select one or more charts in 'Chart Selection Form' above to view charts side by side"
chart_description_expander_title = "Charts Description"
chart_view_settings_title = "Charts View Settings"


spinner_text = "In progress...., Please do not change any settings now"

main_nav = option_menu(options=["Job Geometric Characteristics", "Job Failure Characteristics", "User Behavior Characteristics"],
                                 menu_title="Pick a characteristic to view available model options",
                                 icons=["bi-1-circle", "bi-2-circle", "bi-3-circle"],
                                 styles=styles, orientation="horizontal", menu_icon="bi-segmented-nav")

columns=["job", "user", "project", "state", "gpu_num", "cpu_num", "node_num", "submit_time", "wait_time", "run_time", "wall_time", "node_hour"]


if main_nav == "Job Geometric Characteristics":
    nav_bar_horizontal = option_menu("Job Geometric: Pick a model to load related charts",
     ["Job Run Time", "Job Arrival Pattern", "Sys Util & Res Occu", "Job Waiting Time"],
     default_index=0, orientation="vertical", menu_icon="bi-list")

    if nav_bar_horizontal == "Job Run Time":
        jrt_system_models_jgc = ["Mira", "Blue Waters", "Philly", "Helios"]
        jrt_selected_system_models_jgc = jrt_system_models_jgc.copy() 
        
        jrt_chart_selection_left_col_options_jgc = ["CDF Run Time"]
        jrt_chart_selection_right_col_options_jgc = ["Detailed Run Time Distribution"]
        jrt_chart_selection_options_jgc = jrt_chart_selection_left_col_options_jgc + jrt_chart_selection_right_col_options_jgc
        jrt_chart_selected_list_jgc = jrt_chart_selection_options_jgc.copy()
        x_value_selected = None
        
        
        jrt_drt_time_ranges_jgc = ['0~30s', '30s~10min', '10min~1h', '1h~12h', "more than 12h"]
        jrt_drt_selected_time_range_jgc = jrt_drt_time_ranges_jgc.copy()
        
        with st.form("jrt_chart_selection_form_jgc"):
            st.write(f"### **{chart_selection_form_title}**")
            st.write(f'**{chart_selection_form_load_charts_text}**')
            col1, col2 = st.columns(2)
            with col1 :
                for item in jrt_chart_selection_left_col_options_jgc:
                    jrt_chart_selection_check_box_left_option_jgc = st.checkbox(item, True)
                    if not jrt_chart_selection_check_box_left_option_jgc:
                        jrt_chart_selected_list_jgc.remove(item)
            with col2:
                for item2 in jrt_chart_selection_right_col_options_jgc:
                    jrt_chart_selection_check_box_right_option_jgc = st.checkbox(item2, True)
                    if not jrt_chart_selection_check_box_right_option_jgc:
                        jrt_chart_selected_list_jgc.remove(item2)

            jrt_chart_selection_check_box_submission_button_jgc = st.form_submit_button("Load Charts")
            
            if jrt_chart_selection_check_box_submission_button_jgc:
                if len(jrt_chart_selected_list_jgc) >= 1:
                    st.write(f"**You Have Selected:** {jrt_chart_selected_list_jgc}")
                else:
                    st.markdown("<h5 style='color: red'>Please select one or more charts options above and then click 'Load Charts'</h5>", unsafe_allow_html=True)
            else:
                pass
            
        
        if len(jrt_chart_selected_list_jgc) >= 1:
            st.sidebar.markdown("<h1 style='text-align: center; color: Black;'>Chart Customization Panel</h1>", unsafe_allow_html=True)
            
            min_value_exp_run_time_slider = 0
            max_value_exp_run_time_slider = 8 

            with st.sidebar.form("jrt_sidebar_form_jgc"):
                st.write("### Alter the following settings to customize the selected chart(s):")
                
                with st.expander("**Select System Model(s)**", expanded=True):
                        for item in jrt_system_models_jgc:
                            jrt_model_checkbox_jpc = st.checkbox(item, True)
                            if not jrt_model_checkbox_jpc:
                                jrt_selected_system_models_jgc.remove(item)
                                
                if "CDF Run Time" in jrt_chart_selected_list_jgc:                
                    with st.expander("**CDF Run Time Chart (X and Y - axis)**", expanded=True):       
                        jrt_cdf_frequency_slider_jgc = st.slider("**Adjust frequency range (Y-axis):**", min_value=0, max_value=100, step=20, value=100)
                        jrt_cdf_run_time_slider_jgc = st.slider("**Adjust run time range (in powers of 10) (X-axis):**", min_value_exp_run_time_slider, max_value_exp_run_time_slider, step=1, value=8)
                        jrt_cdf_run_time_slider_value_jgc = int(10**jrt_cdf_run_time_slider_jgc)
                        
                if "Detailed Run Time Distribution" in jrt_chart_selected_list_jgc:
                    with st.expander("**Detailed Run Time Distribution Chart (X and Y - axis)**", expanded=True):
                        jrt_drt_frequency_slider_jgc = st.slider("**Adjust frequency range (Y-axis):**", min_value=0.0, max_value=0.6, step=0.1, value=0.6)
                        st.write("##### **Select Run Time Range (X-axis):**")
                        for item in jrt_drt_time_ranges_jgc:
                            jrt_drt_time_range_checkbox_jgc = st.checkbox(item, True)
                            if not jrt_drt_time_range_checkbox_jgc:
                                    jrt_drt_selected_time_range_jgc.remove(item)  
                                    
                jrt_submit_parameters_button_jgc = st.form_submit_button("Apply Changes")
                
                
            with st.expander(f"**{chart_view_settings_title}**", expanded=True):
                jrt_check_box_view_side_by_side_jgc = st.checkbox("Select to view charts side by side")
                
            # Plots Figure 1(a) from page 3, 3.1.
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
                plt.ylim(0, jrt_cdf_frequency_slider_jgc) 
                plt.xlim(10**0, jrt_cdf_run_time_slider_value_jgc) 
                plt.grid(True)
                plt.style.use("default")
                
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
                plt.xlim(0, jrt_drt_selected_system_models_jgc)
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
                        
            def polt_cdf_job_run_time(side_by_side, chart_title):
                if side_by_side:
                    st.markdown(f"<h4 style='text-align: center;'>{chart_title}</h4>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h2 style='text-align: center;'>{chart_title}</h2>", unsafe_allow_html=True)                   
                for item in jrt_system_models_jgc:
                    if "Blue Waters" in jrt_selected_system_models_jgc:
                        plot_cdf(bw_df["run_time"], 1000, "Time (s)", linestyle=":", color="blue")
                    if "Mira" in jrt_selected_system_models_jgc:
                        plot_cdf(mira_df_2["run_time"], 1000, "Time (s)", linestyle="--", color="red")
                    if "Philly" in jrt_selected_system_models_jgc:
                        plot_cdf(philly_df["run_time"], 1000, "Time (s)", linestyle="-.", color="green")
                    if "Helios" in jrt_selected_system_models_jgc:
                        plot_cdf(hl_df["run_time"], 10009999, "Job Run Time (s)", linestyle="--", color="violet")
                        
                plt.rc('legend', fontsize=12)
                plt.legend(jrt_selected_system_models_jgc, loc="lower right")
                plt.xscale("log")
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()
                
            def plot_detailed_run_time(side_by_side, chart_title): 
                x = [0, 30, 600, 3600, 12 * 3600, 100000]
                x_value = np.array([1, 2, 3, 4, 5])
                labels = ['0~30s', '30s~10min', '10min~1h', '1h~12h', "more than 12h"]
                bw = []
                mr = []
                ply = []
                hl = []
                width = 0.2
                
                for i in range(1, len(x)):
                    if labels[i-1] in jrt_drt_selected_time_range_jgc:
                        res = lt_xs_all(x[i-1], x[i])
                        bw.append(res[0])
                        mr.append(res[1])
                        ply.append(res[2])
                        hl.append(res[3])
                                
                x_value_selected = np.arange(1, len(jrt_drt_selected_time_range_jgc) + 1)
                
                if side_by_side:
                    st.markdown(f"<h4 style='text-align: center;'>{chart_title}</h4>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h2 style='text-align: center;'>{chart_title}</h2>", unsafe_allow_html=True)
                    
                for model in jrt_system_models_jgc:
                        if "Blue Waters" in jrt_selected_system_models_jgc:
                            plt.bar(x_value_selected - 3 * width / 2, bw, width, edgecolor='black', hatch="x", color="blue")
                        if "Mira" in jrt_selected_system_models_jgc:
                            plt.bar(x_value_selected - width / 2, mr, width, edgecolor='black', hatch="\\", color="red")
                        if "Philly" in jrt_selected_system_models_jgc:
                            plt.bar(x_value_selected + width / 2, ply, width, edgecolor='black', hatch=".", color="green")
                        if "Helios" in jrt_selected_system_models_jgc:
                            plt.bar(x_value_selected + 3 * width / 2, hl, width, edgecolor='black', hatch="-", color="violet")
                plt.ylim(0.00, jrt_drt_frequency_slider_jgc)
                plt.xticks(x_value_selected, jrt_drt_selected_time_range_jgc)
                plt.legend(jrt_selected_system_models_jgc, prop={'size': 12}, loc="upper right")
                plt.ylabel("Frequency (%)", fontsize=14)
                plt.xlabel("Job Run Time (s)", fontsize=14)
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()

            with st.spinner(spinner_text):   
                st.markdown("<h1 style='text-align: center; color: black;'>Comparisons Of Run Time Among Four DataSets Charts</h1>", unsafe_allow_html=True) 
                if len(jrt_selected_system_models_jgc) >= 1:
                    if jrt_check_box_view_side_by_side_jgc:
                        col1, col2 = st.columns(2)
                        for idx, item in enumerate(jrt_chart_selected_list_jgc):
                            jrt_col_logic_cal_jgc = col1 if idx % 2 == 0 else col2
                            if item == "CDF Run Time":
                                with jrt_col_logic_cal_jgc:
                                    polt_cdf_job_run_time(True, "CDF Run Time")
                            elif item == "Detailed Run Time Distribution":
                                with jrt_col_logic_cal_jgc:
                                    if len(jrt_drt_selected_time_range_jgc) >= 1:
                                        plot_detailed_run_time(True, "Detailed Run Time Distribution")
                                    else:
                                        st.markdown("<h4 style='color: red'>Detailed Run Time Distribution: Please select one or more 'Run Time Range' options (X-axis) from sidebar to plot this chart</h4>", unsafe_allow_html=True)         
                            else:
                                pass
                    else:
                        if "CDF Run Time" in jrt_chart_selected_list_jgc:
                            polt_cdf_job_run_time(False, "CDF Run Time")
                        else:
                            pass
                        if "Detailed Run Time Distribution" in jrt_chart_selected_list_jgc:
                            if len(jrt_drt_selected_time_range_jgc) >= 1:
                                    plot_detailed_run_time(False, "Detailed Run Time Distribution")
                            else:
                                st.markdown("<h2 style='color: red'>Detailed Run Time Distribution: Please select one or more 'Run Time Range' options (X-axis) from sidebar to plot this chart</h2>", unsafe_allow_html=True)         
       
                        else:
                            pass  
                        
                    with st.expander(f"**{chart_description_expander_title}**", expanded=True):
                        st.write("**CDF Of Run Time:** Displays a Cumulative Distribution Functions (CDFs) of the runtime comparisons of the four job traces (Blue Waters, Mira, Philly, and Helios).")
                        st.write("**Detailed Run Time Distribution:** Displays a bar chart of the four job traces categorized by run times (30 sec, 1 min, 10 mins, 1h, and 12+hrs) alongside the frequency in which they occur.")
                else:
                    st.markdown("<h2 style='color: red'>Please select one or more system model(s) from sidebar to plot the chart</h2>", unsafe_allow_html=True)         
                    

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
            st.markdown("<h2 style='text-align: center;'>Daily Submit Pattern Chart</h2>", unsafe_allow_html=True)
            st.sidebar.markdown("<h1 style='text-align: center;'>Chart Customization Panel</h1>", unsafe_allow_html=True)
            
            dsp_selected_system_models_jap = system_models_jap.copy()
            with st.spinner("In progress...., Please do not change any settings now"):   
                with st.sidebar.form("dsp_personal_parameters_update_form"):
                    st.write("### Alter the following settings to customize the Daily Submit Pattern chart:")
                    with st.expander("**Select System Model(s)**", expanded=True):
                        for item in system_models_jap:
                            dsp_model_checkbox_jap = st.checkbox(item, True)
                            if not dsp_model_checkbox_jap:
                                dsp_selected_system_models_jap.remove(item)
                    dsp_job_count_slider_jap = st.slider("**Adjust Job Submit Count Range (Y-axis):**", min_value=0, max_value=180, step=20, value=180)
                    dsp_hour_of_the_day_slider_jap = st.slider("**Adjust Hour of the Day Range (X-axis):**", min_value=-1, max_value=24, step=1, value=24)
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
            st.markdown("<h2 style='text-align: center;'>Weekly Submit Pattern Chart</h2>", unsafe_allow_html=True)
            st.sidebar.markdown("<h1 style='text-align: center;'>Chart Customization Panel</h1>", unsafe_allow_html=True)
        
            wsp_selected_system_models_jap = system_models_jap.copy()
            with st.spinner("In progress...., Please do not change any settings now"):   
                with st.sidebar.form("wsp_personal_parameters_update_form"):
                    st.write("### Alter the following settings to customize the Weekly Submit Pattern chart:")
                    with st.expander("**Select System Model(s)**", expanded=True):
                        for item in system_models_jap:
                            wsp_model_checkbox_jap = st.checkbox(item, True)
                            if not wsp_model_checkbox_jap:
                                wsp_selected_system_models_jap.remove(item)
                    wsp_job_count_slider_jap = st.slider("**Adjust Job Submit Count Range (Y-axis):**", min_value=0, max_value=3000, step=500, value=3000)
                    wsp_hour_of_the_day_slider_jap = st.slider("**Adjust Day of the Week Range (X-axis):**", min_value=0, max_value=8, step=1, value=8)
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
            st.markdown("<h2 style='text-align: center;'>Job Arrival Interval Chart</h2>", unsafe_allow_html=True)
            st.sidebar.markdown("<h1 style='text-align: center;'>Chart Customization Panel</h1>", unsafe_allow_html=True)  
        
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
                    jai_hour_of_the_day_slider_jap = st.slider("**Adjust Job Arrival Interval Range (in powers of 10) (X-axis):**", jap_min_value_exp_arrival_interval_slider, jap_max_value_exp_arrival_interval_slider, step=1, value=8)
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
    elif nav_bar_horizontal == "Sys Util & Res Occu":
        suaro_cpu_chart_options_jgc = ["Blue Waters CPU", "Mira CPU"]
        suaro_gpu_chart_options_jgc = ["Blue Waters GPU", 
                    "Philly GPU", "Helios GPU", "Philly GPU-SchedGym"]
        suaro_chart_options_jgc = suaro_cpu_chart_options_jgc + suaro_gpu_chart_options_jgc
        suaro_selected_charts_list_jgc = suaro_chart_options_jgc.copy()

        def plot_util_jgc(data, total_nodes, key="node_num", color='b', side_by_side=False, chart_title=None):
            data = data.copy()
            start_time = data["submit_time"].min()
            end_time = data["submit_time"].max()
            duration = end_time - start_time
            days = int(duration/(86400))
            days_usage = np.zeros(days)
            
            if side_by_side:
                st.markdown(f"<h4 style='text-align: center;'>{chart_title}</h4>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h2 style='text-align: center;'>{chart_title}</h2>", unsafe_allow_html=True)

            data["start_time"] = data["submit_time"] + data["wait_time"] - start_time
            data["end_time"] = data["start_time"] + data["run_time"]
            data["start_day"] = (data["start_time"]/(86400)).astype(int)
            data["end_day"] = (data["end_time"]/(86400)).astype(int) + 1

            clipped_start_days = np.clip(data["start_day"], 0, days-1)
            clipped_end_days = np.clip(data["end_day"], 0, days)

            for day in range(days):
                mask = (clipped_start_days <= day) & (day < clipped_end_days)
                days_usage[day] += np.sum(data.loc[mask, key] * (np.minimum(data.loc[mask, "end_time"], (day + 1) * 86400) - np.maximum(data.loc[mask, "start_time"], day * 86400)))

            plt.bar(range(len(days_usage)), 100 * days_usage / (total_nodes * 86400), color=color)
            plt.plot([-10, 150], [80] * 2, color="black", linestyle="--")
            plt.ylim(0, suaro_sys_utilization_slider_jgc)
            plt.xlim(0, suaro_time_slider_jgc)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.xlabel("Time (Days)", fontsize=26)
            plt.ylabel("System Utilization(%)", fontsize=26)
            st.pyplot()

        with st.form("suaro_select_charts_checkbox_main_form_jgc"): 
            st.write(f"### **{chart_selection_form_title}**")
            st.write(f'**{chart_selection_form_load_charts_text}**')
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<h4 style="text-align: center;">CPU Charts</h4>', unsafe_allow_html=True)
                for item in suaro_cpu_chart_options_jgc:
                    suaro_chart_selected_jgc = st.checkbox(item, True)
                    if not suaro_chart_selected_jgc:
                        suaro_selected_charts_list_jgc.remove(item)
            with col2:
                st.markdown('<h4 style="text-align: center;">GPU Charts</h4>', unsafe_allow_html=True)
                for item in suaro_gpu_chart_options_jgc:
                    suaro_chart_selected_jgc = st.checkbox(item, True)
                    if not suaro_chart_selected_jgc:
                        suaro_selected_charts_list_jgc.remove(item)
                        
            suaro_select_charts_checkbox_main_form_button_jgc = st.form_submit_button("Load Charts")
            
            if suaro_select_charts_checkbox_main_form_button_jgc:
                if len(suaro_selected_charts_list_jgc) >= 1:
                    st.write(f'**You have selected:** {suaro_selected_charts_list_jgc}')
                else:
                    st.markdown("<h5 style='color: red;'>You have not selected any chart options above, please select one or more chart option(s) to load the charts.</h5>", unsafe_allow_html=True)
            else: 
                pass

        if len(suaro_selected_charts_list_jgc) >= 1:
            st.sidebar.markdown("<h1 style='text-align: center;'>Chart Customization Panel</h1>", unsafe_allow_html=True)
            
            with st.sidebar.form("suaro_sidebar_form_jgc"):
                st.write("### Alter the following settings to customize the selected chart(s):")
                suaro_sys_utilization_slider_jgc = st.slider("**Adjust System Utilization Range (Y-axis):**", min_value = 0, max_value=100, value=100, step=10)
                suaro_time_slider_jgc = st.slider("**Adjust Time Range (X-axis):**", min_value=0, max_value=120, value=120, step=10)
                suaro_submit_button_sidebar_jgc = st.form_submit_button("Apply Changes")
                if suaro_submit_button_sidebar_jgc:
                    if len(suaro_selected_charts_list_jgc) < 1:
                        st.markdown("<h5 style='color: red;'>Please select one or more chart option(s) from the menu in the main screen to load the charts.</h5>", unsafe_allow_html=True)
                    else:
                        pass
                    
            with st.expander(f"**{chart_view_settings_title}**", expanded=True):
                    suaro_check_box_view_side_by_side_jgc = st.checkbox("Select to view charts side by side")

            with st.spinner("In Progess... Please do not change any settings now"):
                st.markdown("<h1 style='text-align: center; color: black;'>The System Utilization Across Multiple Systems Charts</h1>", unsafe_allow_html=True)
                
                if suaro_check_box_view_side_by_side_jgc:          
                    col1, col2 = st.columns(2)
                    for idx, item in enumerate(suaro_selected_charts_list_jgc):
                        suaro_col_logic_cal_jgc = col1 if idx % 2 == 0 else col2
                        if item == "Blue Waters CPU":
                            with suaro_col_logic_cal_jgc:
                                plot_util_jgc(bw_df[1000:], 22636*32, "cpu_num", color="#1f77b4", side_by_side=True, chart_title="Blue Waters CPU Chart")  
                        elif item == "Mira CPU":
                            with suaro_col_logic_cal_jgc:
                                plot_util_jgc(mira_df_2, 49152, color='#ff7f0e', side_by_side=True, chart_title="Mira CPU Chart")
                        elif item == "Blue Waters GPU":
                            with suaro_col_logic_cal_jgc:
                                plot_util_jgc(bw_df[1000:], 4228, "gpu_num", color="#1f77b4", side_by_side=True, chart_title="Blue Waters GPU Chart")
                        elif item == "Philly GPU":
                            with suaro_col_logic_cal_jgc:
                                plot_util_jgc(philly_df, 2490, "gpu_num", color='#2ca02c', side_by_side=True, chart_title="Philly GPU Chart")
                        elif item == "Helios GPU":
                            with suaro_col_logic_cal_jgc:
                                plot_util_jgc(hl_df, 1080, "gpu_num", side_by_side=True, chart_title="Helios GPU Chart")
                        elif item == "Philly GPU-SchedGym":
                            with suaro_col_logic_cal_jgc:
                                plot_util_jgc(philly_gpu_schedule_df, 2490, "gpu_num", color='#9467bd', side_by_side=True, chart_title="Philly GPU-SchedGym Chart")
                        else:
                            pass
                else:
                    for item in suaro_selected_charts_list_jgc:
                        if item == "Blue Waters CPU":
                            plot_util_jgc(bw_df[1000:], 22636*32, "cpu_num", color="#1f77b4", side_by_side=False, chart_title="Blue Waters CPU Chart")  
                        elif item == "Mira CPU":
                            plot_util_jgc(mira_df_2, 49152, color='#ff7f0e', side_by_side=False, chart_title="Mira CPU Chart")
                        elif item == "Blue Waters GPU":
                            plot_util_jgc(bw_df[1000:], 4228, "gpu_num", color="#1f77b4", side_by_side=False, chart_title="Blue Waters GPU Chart")
                        elif item == "Philly GPU":
                            plot_util_jgc(philly_df, 2490, "gpu_num", color='#2ca02c', side_by_side=False, chart_title="Philly GPU Chart")
                        elif item == "Helios GPU":
                            plot_util_jgc(hl_df, 1080, "gpu_num", side_by_side=False, chart_title="Helios GPU Chart")
                        elif item == "Philly GPU-SchedGym":
                            plot_util_jgc(philly_gpu_schedule_df, 2490, "gpu_num", color='#9467bd', side_by_side=False, chart_title="Philly GPU-SchedGym Chart")
                        else:
                            pass
                        
                with st.expander(f"**{chart_description_expander_title}**", expanded=True):
                                st.write("**The System Utilization Across Multiple Systems Charts:** ADD")

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
            chart_selection_submit_button_jwt = st.form_submit_button("Select")
            if chart_selection_submit_button_jwt:
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

            st.sidebar.markdown("<h1 style='text-align: center;'>Chart Customization Panel</h1>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center;'>CDF of Wait Time</h2>", unsafe_allow_html=True)
            

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

                    plt.ylabel('Frequency (%)', fontsize=18)
                    plt.xlabel('Time Range', fontsize=18)
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

            st.sidebar.markdown("<h1 style='text-align: center;'>Chart Customization Panel</h1>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center;'>CDF of Turnaround Time</h2>", unsafe_allow_html=True)

            
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
                if len(cdfott_selected_system_models_jwt) >= 1:
                    for items in system_models_jwt:
                        if "Blue Waters" in cdfott_selected_system_models_jwt:
                            plot_cdf(bw_df["wait_time"]+bw_df["run_time"], 100000, "Job Wait Time (s)", linestyle=":")
                        if "Mira" in cdfott_selected_system_models_jwt:
                            plot_cdf(mira_df_2["wait_time"]+mira_df_2["run_time"], 100000, "Job Wait Time (s)", linestyle="--")
                        if "Philly" in cdfott_selected_system_models_jwt:
                            plot_cdf(philly_df["wait_time"]+philly_df["run_time"], 100000, "Job Wait Time (s)", linestyle="-.")
                        if "Helios" in cdfott_selected_system_models_jwt:
                            plot_cdf(hl_df["wait_time"]+hl_df["run_time"], 100000, "Job Turnaround Time (s)", linestyle="--")
                plt.xscale("log")
                plt.ylabel('Frequency (%)', fontsize=18)
                plt.xlabel('Time Range', fontsize=18)
                st.set_option('deprecation.showPyplotGlobalUse', False)
                plt.xlim(int(10 ** cdfott_min_value_exp_arrival_interval_slider), cdfott_turnaround_time_slider_value_jwt)
                plt.ylim(0, cdfott_frequency_slider_jwt)
                plt.rc('legend', fontsize=12)
                plt.legend(cdfott_selected_system_models_jwt, loc="upper left")
                st.pyplot()

                with st.expander("**CDF of Turnaround Time Chart Description:**", expanded=True):
                                st.write("Description Goes Here")

        elif chart_select_radio_jwt == "Avg waiting Time w.r.t Job Size":
            # AWTJS: Avg waiting Time w.r.t Job Size
            awtjs_selected_system_models_jwt = system_models_jwt.copy()
            awtjs_job_sizes_list_jwt = ["Small", "Middle", "Large"]
            awtjs_job_sizes_selected_list_jwt = awtjs_job_sizes_list_jwt.copy()

            st.sidebar.markdown("<h1 style='text-align: center;'>Chart Customization Panel</h1>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center;'>Avg Waiting Time w.r.t Job Size Chart</h2>", unsafe_allow_html=True)

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
                
                    with st.expander("**Select System Model(s) (X-axis)**", expanded=True):
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

            st.sidebar.markdown("<h1 style='text-align: center;'>Chart Customization Panel</h1>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center;'>Avg Waiting Time w.r.t Job Run Time Chart</h2>", unsafe_allow_html=True)
        
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

                    with st.expander("**Select System Model(s) (X-axis)**", expanded=True):
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

elif main_nav == "Job Failure Characteristics":
    nav_bar_jfc = option_menu("Job Failure: Pick a model to load related charts", ["Job Failures Distribution", "Correlation between Job Failure and Job Geometries"], 
    default_index=0, orientation="vertical", menu_icon="bi-list")
    system_models_jfc = ["Blue Waters", "Mira", "Philly", "Helios"]

    # Job Failures Distribution charts plotting function.
    def plot_percentage_status(selected_status, frequency_value, selected_models, job_counts=True):
        plt.style.use("default")
        traces = selected_models
        pass_dict = {'Blue Waters': 64.99, 'Mira': 70.05, 'Philly': 59.58, 'Helios': 64.72}
        failed_dict = {'Blue Waters': 7.26, 'Mira': 9.01, 'Philly': 30.90, 'Helios': 14.06}
        killed_dict = {'Blue Waters': 27.74, 'Mira': 20.94, 'Philly': 9.52, 'Helios': 21.15}

        pass_dict_2 = {'Blue Waters': 53.64, 'Mira': 56.94, 'Philly': 33.78, 'Helios': 52.42}
        failed_dict_2 = {'Blue Waters': 4.91, 'Mira': 5.78, 'Philly': 33.40, 'Helios': 6.64}
        killed_dict_2 = {'Blue Waters': 41.45, 'Mira': 37.28, 'Philly': 32.82, 'Helios': 40.94}

        if job_counts:
            status = {}
            if "Pass" in selected_status:
                status['Pass'] = [pass_dict[system_model] for system_model in pass_dict if system_model in selected_models]
            if "Failed" in selected_status:
                status['Failed'] = [failed_dict[system_model] for system_model in failed_dict if system_model in selected_models]
            if "Killed" in selected_status:
                status['Killed'] = [killed_dict[system_model] for system_model in killed_dict if system_model in selected_models]
        else:
            status = {}
            if "Pass" in selected_status:
                status['Pass'] = [pass_dict_2[system_model] for system_model in pass_dict_2 if system_model in selected_models]
            if "Failed" in selected_status:
                status['Failed'] = [failed_dict_2[system_model] for system_model in failed_dict_2 if system_model in selected_models]
            if "Killed" in selected_status:
                status['Killed'] = [killed_dict_2[system_model] for system_model in killed_dict_2 if system_model in selected_models]


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
        ax.set_xticks(x + width, traces, fontsize=12)
        ax.legend(fontsize=15)
        ax.set_ylim(0, frequency_value)
        plt.grid(axis="y")
        st.pyplot(fig)
    
    # Alex - Correlation between Job Failure and Job Geometries charts function here (only function)
    # Creates bar plot to compare percentage distribution of run time across Blue Water, Mira, Philly, and Helios
    def plot_status_over(run_time=False):

        plt.style.use("default")
        traces = ("bw", "mira", "philly", "helios")
        if run_time:
            bw = [[77.38959920341603, 10.274344986835594, 12.336055809748379],
                [53.084873994352385, 5.706217699397944, 41.20890830624967],
                [59.617663499161544, 2.222470653996646, 38.15986584684181]]
            mira = [[76.86672302056917, 15.995115995115993, 7.13816098431483],
                    [64.34694050751082, 2.5799881184757703, 33.07307137401341],
                    [2.564102564102564, 0, 97.43589743589743]]
            philly = [[60.49915507604315, 32.72672126175311, 6.774123662203735],
                    [60.98551323079041, 22.593854306458827, 16.420632462750763],
                    [39.69247516668935, 35.07960266702953, 25.227922166281125]]
            hl = [[65.05428807036834, 15.161489829576691, 19.784222100054976],
                [64.9508786495088, 4.849868548498685, 30.199252801992525],
                [49.60118168389956, 8.951255539143279, 41.44756277695716]]
            z = ["Short", "Middle", "Long"]
        else:
            bw = [[67.00871394770195, 7.114774934564608, 25.876511117733443],
                [47.096774193548384, 1.2903225806451613, 51.61290322580645],
                [67.36842105263158, 4.2105263157894735, 28.421052631578945]]
            mira = [[70.73658165207462, 8.288922725542443, 20.974495622382946],
                    [58.49765258215962, 19.342723004694836, 22.15962441314554],
                    [65.33957845433255, 13.817330210772832, 20.843091334894616]]
            philly = [[67.6981199964019, 24.18899801287136, 8.112881990726734],
                    [26.483950799689726, 58.283160344254426, 15.232888856055848],
                    [20.27687296416938, 63.27361563517915, 16.449511400651463]]
            hl = [[57.48994568597371, 21.9826692648949, 20.527385049131393],
                [45.79295637720701, 22.9936964907329, 31.213347132060086],
                [36.688236653570605, 13.173099144904091, 50.13866420152531]]
            z = ["Small", "Middle", "Large"]
            status = {
                'Small': (15.61, 12.22, 5.68, 0.3),
                'Middle': (143.62, 50.96, 15.84, 0.4),
                'Large': (53.33, 42.83, 13.26, 0.53),
            }

        x = np.arange(len(traces))  # the label locations
        width = 0.25  # the width of the bars
        multiplier = 0

        fig, ax = plt.subplots()
        hatches = ["-", ".", "x", "-"]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for i, measurement in enumerate(zip(bw, mira, philly, hl)):
            offset = width * multiplier
            prev = np.array([0.0] * 4)
            # print(*zip(*measurement))
            for k, j in enumerate(zip(*measurement)):
                rects = ax.bar(x + offset, j, width, hatch=hatches[k], color=colors[k], edgecolor='black', bottom=prev)
                prev += np.array(j)
            # ax.bar_label(rects, padding=3)
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Percentage (%)', fontsize=20)
        # ax.set_xlabel('Traces', fontsize=20)
        # ax.set_title('Penguin attributes by species')
        ax.set_xticks(np.delete(np.arange(16) * 0.25, [3, 7, 11, 15]), 4 * z, fontsize=10, rotation=45)
        # ax.set_xticks(x + width, traces, fontsize=15)
        ax.set_ylim(0, 100)
        plt.grid(axis="y")
        st.pyplot(fig)

    

    if nav_bar_jfc == "Job Failures Distribution":
        jfd_chart_selection_options_jfc = ["Job Count w.r.t Job Status", "Core Hours w.r.t Job Status"] 
        jfd_chart_selected_list_jfc = jfd_chart_selection_options_jfc.copy()

        with st.form("jfd_chart_selection_form_jfc"):
            st.write(f"### **{chart_selection_form_title}**")
            st.write(f'**{chart_selection_form_load_charts_text}**')
            col1, col2 = st.columns(2)
            with col1 :
                jfd_chart_selection_check_box_left_option_jfc = st.checkbox(jfd_chart_selection_options_jfc[0], True)
                if not jfd_chart_selection_check_box_left_option_jfc:
                     jfd_chart_selected_list_jfc.remove(jfd_chart_selection_options_jfc[0])
            with col2:
                jfd_chart_selection_check_box_right_option_jfc = st.checkbox(jfd_chart_selection_options_jfc[1], True)
                if not jfd_chart_selection_check_box_right_option_jfc:
                     jfd_chart_selected_list_jfc.remove(jfd_chart_selection_options_jfc[1])

            jfd_chart_selection_check_box_submission_button_jfc = st.form_submit_button("Load Charts")
            if jfd_chart_selection_check_box_submission_button_jfc:
                if len(jfd_chart_selected_list_jfc) >= 1:
                    st.write(f"**You Have Selected:** {jfd_chart_selected_list_jfc}")
                else:
                    st.markdown("<h5 style='color: red'>Please select one or more charts options above and then click 'Load Charts'</h5>", unsafe_allow_html=True)
            else:
                pass
        
        if len(jfd_chart_selected_list_jfc) >= 1:
            st.sidebar.markdown("<h1 style='text-align: center;'>Chart Customization Panel</h1>", unsafe_allow_html=True)

            with st.sidebar.form("jfd_sidebar_form_jfc"):
                # jfd: Job Failures Distribution
                jfd_selected_system_models_jfc = system_models_jfc.copy()
                jfd_job_status_list_jfc = ["Pass", "Failed", "Killed"]
                jfd_job_status_selected_list_jfc = jfd_job_status_list_jfc.copy()

                st.write("### Alter the following settings to customize the selected chart(s):")
                with st.expander("**Select Job Status(es)**", expanded=True):
                    for item in jfd_job_status_list_jfc:
                        jfd_job_status_checkbox_jfc = st.checkbox(item, True)
                        if not jfd_job_status_checkbox_jfc:
                            jfd_job_status_selected_list_jfc.remove(item)
                        else:
                            pass

                jfd_percentage_slider_jfc = st.slider("**Adjust Percentage Range (Y-axis):**", min_value=0, max_value=100, value=100, step=20)
                with st.expander("**Select System Model(s) (X-axis)**", expanded=True):
                    for item in system_models_jfc:
                        jfd_model_checkbox_jfc = st.checkbox(item, True)
                        if not jfd_model_checkbox_jfc:
                            jfd_selected_system_models_jfc.remove(item)
                        else:
                            pass

                jfd_submit_parameters_button_jfc = st.form_submit_button("Apply Changes")
           
            if len(jfd_job_status_selected_list_jfc) >= 1 and len(jfd_selected_system_models_jfc) >= 1:        
                with st.expander(f"**{chart_view_settings_title}**", expanded=True):
                    jfd_check_box_view_side_by_side_jfc = st.checkbox("Select to view charts side by side")
                    
                with st.spinner(spinner_text):
                    st.markdown("<h1 style='text-align: center; color: black;'>The Distribution Of Different Job Statuses Charts</h1>", unsafe_allow_html=True)

                    if jfd_check_box_view_side_by_side_jfc:
                        if len(jfd_chart_selected_list_jfc) >= 1:
                            col1, col2 = st.columns(2)
                            for idx, item in enumerate(jfd_chart_selected_list_jfc):
                                jfd_col_logic_cal_jfc = col1 if idx % 2 == 0 else col2
                                if item == "Job Count w.r.t Job Status":
                                    with jfd_col_logic_cal_jfc:
                                        st.markdown("<h4 style='text-align: center;'>Job Count w.r.t Job Status</h4>", unsafe_allow_html=True)
                                        plot_percentage_status(jfd_job_status_selected_list_jfc, jfd_percentage_slider_jfc, jfd_selected_system_models_jfc, True)
                                elif item == "Core Hours w.r.t Job Status":
                                    with jfd_col_logic_cal_jfc:
                                        st.markdown("<h4 style='text-align: center;'>Core Hours w.r.t Job Status</h4>", unsafe_allow_html=True)
                                        plot_percentage_status(jfd_job_status_selected_list_jfc, jfd_percentage_slider_jfc, jfd_selected_system_models_jfc, False)
                                else:
                                    pass
                        else:
                            pass
                    else:
                        if "Job Count w.r.t Job Status" in jfd_chart_selected_list_jfc:
                            st.markdown("<h2 style='text-align: center;'>Job Count w.r.t Job Status</h2>", unsafe_allow_html=True)
                            plot_percentage_status(jfd_job_status_selected_list_jfc, jfd_percentage_slider_jfc, jfd_selected_system_models_jfc, True)
                        else:
                            pass
                        if "Core Hours w.r.t Job Status" in jfd_chart_selected_list_jfc:
                            st.markdown("<h2 style='text-align: center;'>Core Hours w.r.t Job Status</h2>", unsafe_allow_html=True)
                            plot_percentage_status(jfd_job_status_selected_list_jfc, jfd_percentage_slider_jfc, jfd_selected_system_models_jfc, False)
                        else:
                            pass
            
                with st.expander(f"**{chart_description_expander_title}**", expanded=True):
                    st.write("**Job Count w.r.t Job Status:** This depicts the total number of jobs classified according to their completion status - Pass, Failed, or Killed. It helps in analyzing job execution trends.")
                    st.write("**Core Hours w.r.t Job Status:** This quantifies the total computing resources consumed by jobs, segmented by their final status. It assists in understanding resource utilization in different scenarios.") 
           
            elif len(jfd_job_status_selected_list_jfc) < 1 and len(jfd_selected_system_models_jfc) >= 1:
                st.markdown("<h2 style='color: red'>Please select one or more job status(es) from the sidebar to plot the chart</h2>", unsafe_allow_html=True)         

            elif len(jfd_job_status_selected_list_jfc) >= 1 and len(jfd_selected_system_models_jfc) < 1:
                st.markdown("<h2 style='color: red'>Please select one or more system model(s) from the sidebar to plot the chart</h2>", unsafe_allow_html=True)
                

            else: # len(jfd_job_status_selected_list_jfc) < 1 and len(jfd_selected_system_models_jfc) < 1
                st.markdown("<h2 style='color: red'>Please select one or more job status(es) and system model(s) from the sidebar to plot the chart</h2>", unsafe_allow_html=True)

        else:
            pass
        
    elif nav_bar_jfc == "Correlation between Job Failure and Job Geometries":
        cbjfajg_chart_title_jfc = "Chart Selection Form"
        cbjfajg_chart_checkbox_title_jfc = "Select one or more charts"
        cbjfajg_chart_selection_options_jfc = ["Job Status w.r.t Job Size", "Job Status w.r.t Job Run Time"]
        cbjfajg_chart_selected_list_jfc = cbjfajg_chart_selection_options_jfc.copy()

        with st.form("cbjfajg_chart_selection_form_jfc"):
            st.write(f"### **{chart_selection_form_title}**")
            st.write(f'**{chart_selection_form_load_charts_text}**')
            col1, col2 = st.columns(2)
            with col1 :
                cbjfajg_chart_selection_check_box_left_option_jfc = st.checkbox(cbjfajg_chart_selection_options_jfc[0], True)
                if not cbjfajg_chart_selection_check_box_left_option_jfc:
                    cbjfajg_chart_selected_list_jfc.remove(cbjfajg_chart_selection_options_jfc[0])
            with col2:
                cbjfajg_chart_selection_check_box_right_option_jfc = st.checkbox(cbjfajg_chart_selection_options_jfc[1], True)
                if not cbjfajg_chart_selection_check_box_right_option_jfc:
                    cbjfajg_chart_selected_list_jfc.remove(cbjfajg_chart_selection_options_jfc[1])

            cbjfajg_chart_selection_check_box_submission_button_jfc = st.form_submit_button("Load Charts")
            if cbjfajg_chart_selection_check_box_submission_button_jfc:
                if len(cbjfajg_chart_selected_list_jfc) >= 1:
                    st.write(f"**You Have Selected:** {cbjfajg_chart_selected_list_jfc}")
                else:
                    st.markdown("<h5 style='color: red'>Please select one or more charts options above and then click 'Load Charts'</h5>", unsafe_allow_html=True)
            else:
                pass

        if len(cbjfajg_chart_selected_list_jfc) >= 1:
            st.sidebar.markdown("<h1 style='text-align: center;'>Chart Customization Panel</h1>", unsafe_allow_html=True)

            with st.sidebar.form("cbjfajg_sidebar_form_jfc"):
                # jfd: Job Failures Distribution
                cbjfajg_selected_system_models_jfc = system_models_jfc.copy()
                cbjfajg_job_size_list_jfc = ["Small", "Middle", "Large"]
                cbjfajg_job_size_selected_list_jfc = cbjfajg_job_size_list_jfc.copy()

                st.write("### Alter the following settings to customize the selected chart(s):")
                with st.expander("**Select Job Status(es)**", expanded=True):
                    for item in cbjfajg_job_size_list_jfc:
                        cbjfajg_job_status_checkbox_jfc = st.checkbox(item, True)
                        if not cbjfajg_job_status_checkbox_jfc:
                            cbjfajg_job_size_selected_list_jfc.remove(item)
                        else:
                            pass

                cbjfajg_percentage_slider_jfc = st.slider("**Adjust Percentage Range (Y-axis):**", min_value=0, max_value=100, value=100, step=20)
                
                with st.expander("**Select System Model(s) (X-axis)**", expanded=True):
                    for item in system_models_jfc:
                        cbjfajg_model_checkbox_jfc = st.checkbox(item, True)
                        if not cbjfajg_model_checkbox_jfc:
                            cbjfajg_selected_system_models_jfc.remove(item)
                        else:
                            pass

                cbjfajg_submit_parameters_button_jfc = st.form_submit_button("Apply Changes")
            
            if len(cbjfajg_job_size_selected_list_jfc) >= 1 and len(cbjfajg_selected_system_models_jfc) >= 1:        
                with st.expander(f"**{chart_view_settings_title}**", expanded=True):
                    cbjfajg_check_box_view_side_by_side_jfc = st.checkbox("Select to view charts side by side")

                with st.spinner(spinner_text):
                    st.markdown("<h1 style='text-align: center; color: black;'>Job Failure v.s Job Runtime And Job Requested Resources Charts</h1>", unsafe_allow_html=True)
                    if cbjfajg_check_box_view_side_by_side_jfc:
                        if len(cbjfajg_chart_selected_list_jfc) >= 1:
                            col1, col2 = st.columns(2)
                            for idx, item in enumerate(cbjfajg_chart_selected_list_jfc):
                                cbjfajg_col_logic_cal_jfc = col1 if idx % 2 == 0 else col2
                                if item == "Job Status w.r.t Job Size":
                                    with cbjfajg_col_logic_cal_jfc:
                                        st.markdown("<h4 style='text-align: center;'>Job Status w.r.t Job Size</h4>", unsafe_allow_html=True)
                                        # Alex - call function with the parameters to plot Job Status w.r.t Job Size
                                        plot_status_over()
                                elif item == "Job Status w.r.t Job Run Time":
                                    with cbjfajg_col_logic_cal_jfc:
                                        st.markdown("<h4 style='text-align: center;'>Job Status w.r.t Job Run Time</h4>", unsafe_allow_html=True)
                                        # Alex - call function with the parameters to plot Job Status w.r.t Job Run Time
                                        plot_status_over(True)
                        else:
                            pass
                    else:
                        if "Job Status w.r.t Job Size" in cbjfajg_chart_selected_list_jfc:
                            st.markdown("<h2 style='text-align: center;'>Job Status w.r.t Job Size</h2>", unsafe_allow_html=True)
                            #Alex - call function with the parameters to plot Job Status w.r.t Job Size
                            plot_status_over()
                        else:
                            pass
                        if "Job Status w.r.t Job Run Time" in cbjfajg_chart_selected_list_jfc:
                            st.markdown("<h2 style='text-align: center;'>Job Status w.r.t Job Run Time</h2>", unsafe_allow_html=True)
                            #Alex - call function with the parameters to plot Job Status w.r.t Job Run Time
                            plot_status_over(True)
                        else:
                            pass

                with st.expander(f"**{chart_description_expander_title}**", expanded=True):
                    st.write("**Job Status w.r.t Job Size:** This chart illustrates the status of jobs (Pass, Failed, Killed) with respect to their sizes. It provides insight into how job size may impact completion status, thereby helping to predict potential job execution outcomes.")
                    st.write("**Job Status w.r.t Job Run Time:** This visualization represents the correlation between job status and job run time. By analyzing job completion (Pass, Failed, Killed) in relation to run time, it aids in understanding the efficiency of jobs and can assist in identifying potential bottlenecks or issues in job execution.")
           
            elif len(cbjfajg_job_size_selected_list_jfc) < 1 and len(cbjfajg_selected_system_models_jfc) >= 1:
                st.markdown("<h2 style='color: red'>Please select one or more job status(es) from the sidebar to plot the chart</h2>", unsafe_allow_html=True)         

            elif len(cbjfajg_job_size_selected_list_jfc) >= 1 and len(cbjfajg_selected_system_models_jfc) < 1:
                st.markdown("<h2 style='color: red'>Please select one or more system model(s) from the sidebar to plot the chart</h2>", unsafe_allow_html=True)

            else: 
                st.markdown("<h2 style='color: red'>Please select one or more job status(es) and system model(s) from the sidebar to plot the chart</h2>", unsafe_allow_html=True)


elif main_nav == "User Behavior Characteristics":
    ubc_nav_bar = option_menu("User Behavior: Pick a model to load related charts", ["Users’ Repeated Behaviors", "Users’ Submission Behaviors", "Correlation between Job Run Time and Job Statuses"], 
    default_index=0, orientation="vertical", menu_icon="bi-list")

    if ubc_nav_bar == "Users’ Repeated Behaviors":
        urb_chart_title_ubc = "Chart Selection Form"
        urb_chart_checkbox_title_ubc = "Select one or more charts"
        urb_chart_selection_left_col_options_ubc = ["Blue Waters", "Mira"]
        urb_chart_selection_right_col_options_ubc = ["Philly", "Helios"]
        urb_chart_selection_options_ubc = urb_chart_selection_left_col_options_ubc + urb_chart_selection_right_col_options_ubc
        urb_chart_selected_list_ubc = urb_chart_selection_options_ubc.copy()

        with st.form("urb_chart_selection_form_ubc"):
            st.write(f"### **{chart_selection_form_title}**")
            st.write(f'**{chart_selection_form_load_charts_text}**')
            col1, col2 = st.columns(2)
            with col1 :
                for item in urb_chart_selection_left_col_options_ubc:
                    urb_chart_selection_check_box_left_option_ubc = st.checkbox(item, True)
                    if not urb_chart_selection_check_box_left_option_ubc:
                        urb_chart_selected_list_ubc.remove(item)
            with col2:
                for item2 in urb_chart_selection_right_col_options_ubc:
                    urb_chart_selection_check_box_right_option_ubc = st.checkbox(item2, True)
                    if not urb_chart_selection_check_box_right_option_ubc:
                        urb_chart_selected_list_ubc.remove(item2)
            urb_chart_selection_check_box_submission_button_ubc = st.form_submit_button("Load Charts")

            if urb_chart_selection_check_box_submission_button_ubc:
                if len(urb_chart_selected_list_ubc) >= 1:
                    st.write(f"**You Have Selected:** {urb_chart_selected_list_ubc}")
                else:
                    st.markdown("<h5 style='color: red'>Please select one or more charts options above and then click 'Load Charts'</h5>", unsafe_allow_html=True)
            else:
                pass

        if len(urb_chart_selected_list_ubc) >= 1:
            st.sidebar.markdown("<h1 style='text-align: center; color: Black;'>Chart Customization Panel</h1>", unsafe_allow_html=True)

            with st.sidebar.form("urb_sidebar_form_ubc"):
                st.write("### Alter the following settings to customize the selected chart(s):")
                urb_percentage_slider_ubc = st.slider("**Adjust Percentage Range (Y-axis):**", min_value=0, max_value=100, value=100, step=20)
                urb_no_of_top_groups_per_user_slider_ubc = st.slider("**Adjust No Of Top Groups Per User (X-axis):**", min_value=0, max_value=10, value=10, step=1)
                urb_submit_parameters_button_ubc = st.form_submit_button("Apply Changes")

            with st.expander(f"**{chart_view_settings_title}**", expanded=True):
                urb_check_box_view_side_by_side_ubc = st.checkbox("Select to view charts side by side")

            with st.spinner(spinner_text):
                st.markdown("<h2 style='text-align: center; color: black;'>The Resource-Configuration Group Per User Charts</h2>", unsafe_allow_html=True)

                def plot_123(a, color, chart_title, x_axis_value, y_axis_value):
                    fig, ax = plt.subplots()
                    x_axis_value_calculate = x_axis_value

                    x_axis_value_calculate = x_axis_value + 1
                    x_values = list(range(1, x_axis_value_calculate))
                    y_values = np.array(a)*100
                    ax.bar(x_values, y_values, color=color)
    
                    ax.set_ylabel('Percentage (%)', fontsize=20)
                    ax.set_xlabel('Number of Top Groups Per User', fontsize=20)
                    ax.set_xticks(list(range(1, x_axis_value_calculate)))
                    ax.set_xticklabels(list(range(1, x_axis_value_calculate)), fontsize=15)
                    ax.set_ylim(0, y_axis_value)         
                    ax.grid(axis="y")
                    st.markdown(f"<h2 style='text-align: center;'>{chart_title}</h2>", unsafe_allow_html=True)
                    st.pyplot(fig)

                #a,b,c,d - Blue waters, Mira, Philly, and Helios     
                a = [0.6194840399786984, 0.7729370934866642, 0.8425218118648454, 0.8906973579175446, 0.9238788917664792, 0.9479635533005115, 0.9659413769736639, 0.9788211703158228, 0.9842781315514204, 0.987734831866639]
                b = [0.6918350088912488, 0.8533482445948762, 0.921081711512026, 0.9533918131448507, 0.9710197995695022, 0.9810033596267114, 0.9872495542508333, 0.9916599140171835, 0.9944420135092896, 0.9964546220465884]
                c = [0.28569096620357964, 0.4384045247520146, 0.545916628344075, 0.6263372405355048, 0.6897181499719287, 0.7429051624867624, 0.7877784887121456, 0.8257544812862695, 0.8583802658301265, 0.8858856158005057]
                d = [0.3412589175944932, 0.5253771632298813, 0.6401852895114848, 0.7268169396811582, 0.7918618794877094, 0.8394237557838181, 0.8733033543091736, 0.9005927265133411, 0.9214560290971314, 0.9370205635505027]
                colors = []
                x = []
                urb_chart_titles_ubc = []
                urb_x_axis_slice_end_value_ubc = urb_no_of_top_groups_per_user_slider_ubc

                for item in urb_chart_selected_list_ubc:
                    if "Blue Waters" == item:
                        x.append(a[:urb_x_axis_slice_end_value_ubc])
                        urb_chart_titles_ubc.append("Blue Waters")
                        colors.append('#1f77b4')
                    elif "Mira" == item:
                        x.append(b[:urb_x_axis_slice_end_value_ubc])
                        urb_chart_titles_ubc.append("Mira")
                        colors.append('#ff7f0e')
                    elif "Philly" == item:
                        x.append(c[:urb_x_axis_slice_end_value_ubc])
                        urb_chart_titles_ubc.append("Philly")
                        colors.append('#2ca02c')
                    elif "Helios" == item:
                        x.append(d[:urb_x_axis_slice_end_value_ubc]) 
                        urb_chart_titles_ubc.append("Helios")
                        colors.append('#d62728')
                    else:
                        pass

                if urb_check_box_view_side_by_side_ubc:
                        col1, col2 = st.columns(2)
                        for idx, item in enumerate(urb_chart_selected_list_ubc):
                            urb_col_logic_cal_ubc = col1 if idx % 2 == 0 else col2
                            if item == "Blue Waters":
                                with urb_col_logic_cal_ubc:
                                    plot_123(a, colors[0], "Blue Waters", urb_no_of_top_groups_per_user_slider_ubc, urb_percentage_slider_ubc)
                            elif item == "Mira":
                                with urb_col_logic_cal_ubc:
                                    plot_123(b, colors[1], "Mira", urb_no_of_top_groups_per_user_slider_ubc, urb_percentage_slider_ubc)
                            elif item == "Philly":
                                with urb_col_logic_cal_ubc:
                                    plot_123(c, colors[2], "Philly", urb_no_of_top_groups_per_user_slider_ubc, urb_percentage_slider_ubc)
                            elif item == "Helios":
                                with urb_col_logic_cal_ubc:
                                    plot_123(d, colors[3], "Helios", urb_no_of_top_groups_per_user_slider_ubc, urb_percentage_slider_ubc)
                            else:
                                pass          
                else:
                    for i, j, z in zip(x, colors, urb_chart_titles_ubc):
                        plot_123(i, j, z, urb_no_of_top_groups_per_user_slider_ubc, urb_percentage_slider_ubc)
                    

                with st.expander(f"**{chart_description_expander_title}**", expanded=True):
                    st.write("**The Resource-Configuration Groups Per User:** This chart visualizes the repeated job submission patterns based on resource configurations (number of nodes and run time). It shows that nearly 90% of all jobs fall within the top 10 largest groups of similar job configurations, indicating high repetition in user job submissions. Additionally, it compares repetition across different systems (Philly, Helios, Blue Waters, Mira), revealing less repeated patterns in deep learning workloads on Philly and Helios.")
        else:
            pass

    elif ubc_nav_bar == "Users’ Submission Behaviors":
        usb_chart_title_ubc = "Chart Selection Form"
        usb_chart_checkbox_title_ubc = "Select one or more charts"
        usb_chart_selection_left_col_options_ubc = ["Blue Waters", "Mira"]
        usb_chart_selection_right_col_options_ubc = ["Philly", "Helios"]
        usb_chart_selection_options_ubc = usb_chart_selection_left_col_options_ubc + usb_chart_selection_right_col_options_ubc
        usb_chart_selected_list_ubc = usb_chart_selection_options_ubc.copy()
        usb_job_size_list_ubc = ["Short Queue", "Middle Queue", "Long Queue"]
        usb_job_size_selected_list_ubc = usb_job_size_list_ubc.copy()

        with st.form("usb_chart_selection_form_ubc"):
            st.write(f"### **{chart_selection_form_title}**")
            st.write(f'**{chart_selection_form_load_charts_text}**')
            col1, col2 = st.columns(2)
            with col1 :
                for item in usb_chart_selection_left_col_options_ubc:
                    usb_chart_selection_check_box_left_option_ubc = st.checkbox(item, True)
                    if not usb_chart_selection_check_box_left_option_ubc:
                        usb_chart_selected_list_ubc.remove(item)
            with col2:
                for item2 in usb_chart_selection_right_col_options_ubc:
                    usb_chart_selection_check_box_right_option_ubc = st.checkbox(item2, True)
                    if not usb_chart_selection_check_box_right_option_ubc:
                        usb_chart_selected_list_ubc.remove(item2)
            usb_chart_selection_check_box_submission_button_ubc = st.form_submit_button("Load Charts")

            if usb_chart_selection_check_box_submission_button_ubc:
                if len(usb_chart_selected_list_ubc) >= 1:
                    st.write(f"**You Have Selected:** {usb_chart_selected_list_ubc}")
                else:
                    st.markdown("<h5 style='color: red'>Please select one or more charts options above and then click 'Load Charts'</h5>", unsafe_allow_html=True)
            else:
                pass

        if len(usb_chart_selected_list_ubc) >= 1:
            st.sidebar.markdown("<h1 style='text-align: center; color: Black;'>Chart Customization Panel</h1>", unsafe_allow_html=True)

            with st.sidebar.form("usb_sidebar_form_ubc"):
                st.write("### Alter the following settings to customize the selected chart(s):")
                usb_percentage_slider_ubc = st.slider("**Adjust Percentage (%) (Y-axis):**", min_value=0, max_value=100, value=100, step=20)
                with st.expander("**Select Job Size(s) (X-axis)**", expanded=True):
                    for item in usb_job_size_list_ubc:
                        usb_job_size_checkbox_ubc = st.checkbox(item, True)
                        if not usb_job_size_checkbox_ubc:
                            usb_job_size_selected_list_ubc.remove(item)
                        else:
                            pass
                usb_submit_parameters_button_ubc = st.form_submit_button("Apply Changes")

            with st.expander(f"**{chart_view_settings_title}**", expanded=True):
                    usb_check_box_view_side_by_side_ubc = st.checkbox("Select to view charts side by side") 
                    
            #Alex Graph Code Here
            def analyze_util_and_user_behavior(data, total_nodes, gpu=False):
                queue = [] # waiting:0, running:1
                util_time_user = defaultdict(list) # util, time, user
                util_node_user = defaultdict(list)
                util_time = []
                cur_util = 0
                
                for index, row in data.iterrows():
                    row["start_time"] = row["submit_time"] + row["wait_time"]
                    row["end_time"] = row["start_time"] + row["run_time"]
                    row["index"] = index
                    # only for helios
                    # total_nodes = gpu_num_total[bisect.bisect(gpu_date, row["submit_time"])-1]
                    
                    while queue and queue[0][0]<=row["submit_time"]:
                        temp = heappop(queue)
                        cur_time = temp[0]
                        job_type = temp[1]
                        job = temp[3]
                        
                        if job_type == "waiting":
                            heappush(queue, (job["end_time"], "running", job["index"], job))
                            cur_util += job["gpu_num"] if gpu else job["node_num"]
                        elif job_type == "running":
                            cur_util -= job["gpu_num"] if gpu else job["node_num"]
                        else:
                            raise NotImplementedError
                            
                        if util_time and util_time[-1][0] == cur_time:
                            util_time[-1][1] = cur_util/total_nodes
                        else:
                            util_time.append([cur_time, cur_util/total_nodes])
                            
                    heappush(queue, (row["start_time"], "waiting", row["index"] , row))
                    util_time_user[row["user"]].append([row["submit_time"], cur_util/total_nodes])
                    util_node_user[row["user"]].append([row["gpu_num"] if gpu else row["node_num"], cur_util/total_nodes])
                    
                    if index % 10000 == 0:
                        print(index)
                        
                return util_time, util_time_user, util_node_user


            def analyze_queue_and_user_behavior(data, gpu=False):
                data = data.copy()
                data["index"] = data.index
                queue = [] # waiting:0, running:1
                util_time_user = defaultdict(list) # util, time, user
                util_node_user = defaultdict(list)
                util_time = []
                cur_wait = 0
                
                for index, row in data.iterrows():
                    start_time = row["submit_time"] + row["wait_time"]
                    end_time = start_time + row["run_time"]
                    # only for helios
                    # total_nodes = gpu_num_total[bisect.bisect(gpu_date, row["submit_time"])-1]
                    
                    while queue and queue[0][0]<=row["submit_time"]:
                        temp = heappop(queue)
                        cur_time = temp[0]
                        job_type = temp[1]
                        job = temp[3]
                        
                        if job_type == "waiting":
                            heappush(queue, (job["submit_time"] + job["wait_time"]+job["run_time"], "running", job["index"], job))
                            cur_wait -= 1
                        elif job_type == "running":
                            pass
                                # cur_util -= job["gpu_num"] if gpu else job["node_num"]
                        else:
                            raise NotImplementedError
                            
                        util_time.append([cur_time, cur_wait])
                        
                    heappush(queue, (start_time, "waiting", index, row))
                    util_time_user[row["user"]].append([row["run_time"], cur_wait])
                    util_node_user[row["user"]].append([row["gpu_num"] if gpu else row["node_num"], cur_wait])
                    
                    cur_wait += 1
                    
                    if index % 10000 == 0:
                        print(index)
                        
                return util_time, util_time_user, util_node_user


            def plot_util_node(un, data, bars, user="per", xlabel="mira"):
                hatches= ["-", ".", "x", "-"]
                
                if user == "per":
                    users = list(data.groupby("user").count().sort_values(by="job", ascending=False).index[:6])
                    stride = 0.2
                    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

                    for ui, user in enumerate(users):
                        print(ui, user)

                        x = ["{:.0%}".format((1+index)*stride)
                                                for index in range(int(1/stride))]
                        buckets = [Counter() for _ in range(int(1/stride))]
                        
                        for node, util in un[user]:
                            b = int(min(1/stride-1, util/stride))
                            buckets[b][node] += 1
                            
                        for i in buckets:
                            s = sum(i.values())
                            for j in i:
                                i[j] /= s
                                
                        prevy = np.array([0]*len(x))
                        prev_bar = -1
                        
                        for bar in bars:
                            y = np.array([sum(i[j] for j in i.keys() if prev_bar<j <=bar) for i in buckets])
                            axes[ui//3, ui%3].bar(x, y, bottom=prevy)
                            prevy = y+prevy
                            prev_bar = bar
                            
                        axes[ui//3, ui%3].legend(bars)
                else:
                    all_un = []
                    for i in un:
                        all_un.extend(un[i])
                        
                    max_wait = max(all_un, key=lambda x: x[1])[1]
                    stride = 1/3
                    fig, axes = plt.subplots(1, 1, figsize=(3, 5))

                    x = ["{}".format((1+index)*stride*max_wait)
                                            for index in range(int(1/stride))]
                    buckets = [Counter() for _ in range(int(1/stride))]
                    
                    for node, util in all_un:
                        b = int(min(1/stride-1, (util/max_wait)/stride))
                        buckets[b][node] += 1
                        
                    for i in buckets:
                        i[0] = 0
                        s = sum(list(i.values()))
                        for j in i:
                            i[j] /= s*0.01
                            
                    prevy = np.array([0]*len(x))
                    prev_bar = -1
                    
                    for index, bar in enumerate(bars):
                        y = np.array([sum(i[j] for j in i.keys() if prev_bar<j <=bar) for i in buckets])
                        axes.bar(x, y, bottom=prevy, hatch=hatches[index], edgecolor="black")
                        prevy = y+prevy
                        prev_bar = bar
                        
                    axes.set_xticks(x, ["Short Queue", "Middle Queue", "Long Queue"], rotation=45)
                    axes.set_xlabel(xlabel, fontsize=20)
                    axes.set_ylabel("Percentage (%)", fontsize=20)
                    st.pyplot(fig)
                    # axes.legend([ "Minimal", "Short", "Middle", "Long"], fontsize=30, bbox_to_anchor=(0.5, 2.05),
                    # ncol=4, fancybox=True, shadow=True)
                    
            bw_queue_time, bw_queue_time_user,bw_queue_node_user = analyze_queue_and_user_behavior(bw_df, gpu=False)
            bars = [1, 22636//10, 3*22636//10, 1000000]
            plot_util_node(bw_queue_node_user, bw_df, bars, "all", "bw")
                    
            mira_queue_time, mira_queue_time_user, mira_queue_node_user = analyze_queue_and_user_behavior(mira_df_2, gpu=False)
            bars = [ 512, 49152//10, 3*49152//10, 49152]
            plot_util_node(mira_queue_node_user, mira_df_2, bars, "all", "mira")

            phi_queue_time, phi_queue_time_user,phi_queue_node_user = analyze_queue_and_user_behavior(philly_df, gpu=True)
            bars = [1, 1, 8, 256]
            plot_util_node(phi_queue_node_user, philly_df, bars, "all", "philly")

            queue_time, queue_time_user,queue_node_user = analyze_queue_and_user_behavior(hl_df, gpu=True)
            bars = [1, 1, 8, 256]
            plot_util_node(queue_node_user, hl_df, bars, "all", "helios")
                        
                        
            with st.expander(f"**{chart_description_expander_title}**", expanded=True):
                    st.write("**The Median Runtime Of Different Types Of Jobs Charts:** ")
                    st.write("**The Resource-configuration Groups per User:** This chart visualizes the repeated job submission patterns based on resource configurations (number of nodes and run time). It shows that nearly 90% of all jobs fall within the top 10 largest groups of similar job configurations, indicating high repetition in user job submissions. Additionally, it compares repetition across different systems (Philly, Helios, Blue Waters, Mira), revealing less repeated patterns in deep learning workloads on Philly and Helios.")    

    elif ubc_nav_bar == "Correlation between Job Run Time and Job Statuses":
        cbjrtajs_chart_title_ubc = "Chart Selection Form"
        cbjrtajs_chart_checkbox_title_ubc = "Select one or more charts"
        cbjrtajs_chart_selection_left_col_options_ubc = ["Blue Waters", "Mira"]
        cbjrtajs_chart_selection_right_col_options_ubc = ["Philly", "Helios"]
        cbjrtajs_chart_selection_options_ubc = cbjrtajs_chart_selection_left_col_options_ubc + cbjrtajs_chart_selection_right_col_options_ubc
        cbjrtajs_chart_selected_list_ubc = cbjrtajs_chart_selection_options_ubc.copy()
        cbjrtajs_job_status_list_ubc = ["Pass", "Failed", "Killed"]
        cbjrtajs_job_status_selected_list_ubc = cbjrtajs_job_status_list_ubc.copy()
        
        #Y-axis min and max 
        cbjrtajs_min_value_exp_run_time_slider_ubc = 0
        cbjrtajs_max_value_exp_run_time_slider_ubc = 6 

        with st.form("cbjrtajs_chart_selection_form_ubc"):
            st.write(f"### **{chart_selection_form_title}**")
            st.write(f'**{chart_selection_form_load_charts_text}**')
            col1, col2 = st.columns(2)
            with col1 :
                for item in cbjrtajs_chart_selection_left_col_options_ubc:
                    cbjrtajs_chart_selection_check_box_left_option_ubc = st.checkbox(item, True)
                    if not cbjrtajs_chart_selection_check_box_left_option_ubc:
                        cbjrtajs_chart_selected_list_ubc.remove(item)
            with col2:
                for item2 in cbjrtajs_chart_selection_right_col_options_ubc:
                    cbjrtajs_chart_selection_check_box_right_option_ubc = st.checkbox(item2, True)
                    if not cbjrtajs_chart_selection_check_box_right_option_ubc:
                        cbjrtajs_chart_selected_list_ubc.remove(item2)
            cbjrtajs_chart_selection_check_box_submission_button_ubc = st.form_submit_button("Load Charts")

            if cbjrtajs_chart_selection_check_box_submission_button_ubc:
                if len(cbjrtajs_chart_selected_list_ubc) >= 1:
                    st.write(f"**You Have Selected:** {cbjrtajs_chart_selected_list_ubc}")
                else:
                    st.markdown("<h5 style='color: red'>Please select one or more charts options above and then click 'Load Charts'</h5>", unsafe_allow_html=True)
            else:
                pass

        if len(cbjrtajs_chart_selected_list_ubc) >= 1:
            st.sidebar.markdown("<h1 style='text-align: center; color: Black;'>Chart Customization Panel</h1>", unsafe_allow_html=True)

            with st.sidebar.form("cbjrtajs_sidebar_form_ubc"):
                st.write("### Alter the following settings to customize the selected chart(s):")
                
                cbjrtajs_percentage_slider_ubc = st.slider("**Adjust Job Run Time (in powers of 10) (Y-axis):**", min_value=cbjrtajs_min_value_exp_run_time_slider_ubc, max_value=cbjrtajs_max_value_exp_run_time_slider_ubc, value=6, step=1)
                cbjrtajs_percentage_slider_value_ubc = int(10**cbjrtajs_percentage_slider_ubc)
                
                with st.expander("**Select Job Status(es) (X-axis)**", expanded=True):
                    for item in cbjrtajs_job_status_list_ubc:
                        cbjrtajs_job_status_checkbox_ubc = st.checkbox(item, True)
                        if not cbjrtajs_job_status_checkbox_ubc:
                            cbjrtajs_job_status_selected_list_ubc.remove(item)
                        else:
                            pass
                cbjrtajs_submit_parameters_button_ubc = st.form_submit_button("Apply Changes")

            with st.expander(f"**{chart_view_settings_title}**", expanded=True):
                cbjrtajs_check_box_view_side_by_side_ubc = st.checkbox("Select to view charts side by side")
                
            #Function to plot the charts
            def plot_attribute_per_ml(u, data, state="state", status=None ,all_user=False, side_by_side = False, chart_title=None):
                plt.style.use("default")
                rows = list(data.groupby(u).sum().sort_values(by="node_hour", ascending=False).index[:3])

                if all_user:
                    mean_run_time = [data["run_time"]]
                    selected_run_times = []


                    if side_by_side:
                        st.markdown(f"<h4 style='text-align: center;'>{chart_title}</h4>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<h2 style='text-align: center;'>{chart_title}</h2>", unsafe_allow_html=True)


                    for idx, item in enumerate(status):
                        if item == "Pass":
                            st0_run_time = [data.groupby([state])["run_time"].apply(list).get(status[idx],0)]
                            selected_run_times.append(st0_run_time)
                        elif item == "Failed":
                            st1_run_time = [data.groupby([state])["run_time"].apply(list).get(status[idx],0)]
                            selected_run_times.append(st1_run_time)   
                        elif item == "Killed":
                            st2_run_time = [data.groupby([state])["run_time"].apply(list).get(status[idx],0)]
                            selected_run_times.append(st2_run_time)
                        else:
                            pass


                    fig, axes = plt.subplots(1, 1, figsize=(4, 3))

                    for index, i in enumerate(zip(*selected_run_times)):
                        k = [np.log10(np.array(j)+1) for j in i]
                        seaborn.violinplot(data=k,ax=axes, scale="width")

                    ax = axes
                    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))

                    ymin, ymax = ax.get_ylim()
                    tick_range = np.arange(np.floor(ymin), ymax)
                    ax.yaxis.set_ticks(tick_range)
                    ax.yaxis.set_ticks([np.log10(x) for p in tick_range for x in np.linspace(10 ** p, 10 ** (p + 1), 10)], minor=True)
                    ax.yaxis.grid(True)

                    ax.set_xticks([y for y in range(len(status))])
                    ax.set_xticklabels(status, fontsize=24)
                    ax.set_ylabel('Job Run time (s)', fontsize=20)
                    st.pyplot(fig)

                else:
                    mean_run_time = [data.groupby(u)["run_time"].apply(list).loc[i] for i in rows]
                    selected_run_times = []

                    if side_by_side:
                        st.markdown(f"<h4 style='text-align: center;'>{chart_title}</h4>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<h2 style='text-align: center;'>{chart_title}</h2>", unsafe_allow_html=True)

                    for idx, item in enumerate(status):
                        if item == "Pass":
                            st0_run_time = [data.groupby([u, state])["run_time"].apply(list).loc[i].get(status[0],0) for i in rows]
                            selected_run_times.append(st0_run_time)
                        elif item == "Failed":
                            st1_run_time = [data.groupby([u, state])["run_time"].apply(list).loc[i].get(status[1],0) for i in rows]
                            selected_run_times.append(st1_run_time)
                        elif item == "Killed":
                            st2_run_time = [data.groupby([u, state])["run_time"].apply(list).loc[i].get(status[2],0) for i in rows]
                            selected_run_times.append(st2_run_time)
                        else:
                            pass

                    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

                    for index, i in enumerate(zip(*selected_run_times)):
                        k = [np.log10(np.array(j)+1) for j in i]
                        seaborn.violinplot(data=k,ax=axes[index%3], scale="width")

                    for index, ax in enumerate(axes.flatten()):
                        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
                        ymin, ymax = ax.get_ylim()
                        tick_range = np.arange(np.floor(ymin), ymax)
                        ax.yaxis.set_ticks(tick_range, fontsize=20)
                        ax.yaxis.set_ticks([np.log10(x) for p in tick_range for x in np.linspace(10 ** p, 10 ** (p + 1), 10)], minor=True,)
                        ax.yaxis.grid(True)
                        ax.set_xticks([y for y in range(status)])
                        ax.set_xticklabels(status, fontsize=15)
                        ax.set_xlabel('User '+str(index+1), fontsize=20)

                        if index == 0:
                            ax.set_ylabel('Job Run time (s)', fontsize=20)

            with st.spinner(spinner_text):

                if len(cbjrtajs_job_status_selected_list_ubc) >= 1:
                    st.markdown("<h1 style='text-align: center; color: black;'>The Median Runtime Of Different Types Of Jobs Charts</h1>", unsafe_allow_html=True)
                    if cbjrtajs_check_box_view_side_by_side_ubc:
                        col1, col2 = st.columns(2)
                        for idx, item in enumerate(cbjrtajs_chart_selected_list_ubc):
                            cbjrtajs_col_logic_cal_ubc = col1 if idx % 2 == 0 else col2
                            if item == "Blue Waters":
                                with cbjrtajs_col_logic_cal_ubc:
                                    plot_attribute_per_ml("user", data=bw_df, state="new_status", status=cbjrtajs_job_status_selected_list_ubc, all_user=True, side_by_side = True, chart_title = "Blue Waters")
                            elif item == "Mira":
                                with cbjrtajs_col_logic_cal_ubc:
                                    plot_attribute_per_ml("user", data=mira_df_2, state="new_status", status=cbjrtajs_job_status_selected_list_ubc, all_user=True, side_by_side = True, chart_title = "Mira")
                            elif item == "Philly":
                                with cbjrtajs_col_logic_cal_ubc:
                                    plot_attribute_per_ml("user", data=philly_df, state="state", status=cbjrtajs_job_status_selected_list_ubc, all_user=True, side_by_side = True, chart_title = "Philly")
                            elif item == "Helios":
                                with cbjrtajs_col_logic_cal_ubc:
                                    plot_attribute_per_ml("user", data=hl_df, state="state", status=cbjrtajs_job_status_selected_list_ubc, all_user=True, side_by_side = True, chart_title = "Helios")
                            else:
                                pass                           
                    else:
                        for item in cbjrtajs_chart_selected_list_ubc:
                            if item == "Blue Waters":
                                plot_attribute_per_ml("user", data=bw_df, state="new_status", status=cbjrtajs_job_status_selected_list_ubc, all_user=True, side_by_side = False, chart_title = "Blue Waters")                          
                            elif item == "Mira":
                                plot_attribute_per_ml("user", data=mira_df_2, state="new_status", status=cbjrtajs_job_status_selected_list_ubc, all_user=True, side_by_side = False, chart_title = "Mira")                        
                            elif item == "Philly":
                                plot_attribute_per_ml("user", data=philly_df, state="state", status=cbjrtajs_job_status_selected_list_ubc, all_user=True, side_by_side = False, chart_title = "Philly")                      
                            elif item == "Helios":
                                plot_attribute_per_ml("user", data=hl_df, state="state", status=cbjrtajs_job_status_selected_list_ubc, all_user=True, side_by_side = False, chart_title = "Helios")
                            else:
                                pass
                else:
                    st.markdown("<h2 style='color: red'>Please select one or more job status(es) from the sidebar to plot the chart(s)</h2>", unsafe_allow_html=True)

                with st.expander(f"**{chart_description_expander_title}**", expanded=True):
                    st.write("**The Median Runtime Of Different Types Of Jobs Charts:** ")
    else:
        pass
else:
    pass
