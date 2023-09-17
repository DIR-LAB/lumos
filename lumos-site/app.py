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
from collections import deque
from matplotlib.lines import Line2D

st.set_page_config(page_title="Job Trace Visualization Application", page_icon="📊")
curr_dir = os.path.dirname(__file__)

banner_image_path = os.path.join(curr_dir, 'images/App Banner Image.png')

data_blue_waters_path = os.path.join(curr_dir, 'data/data_blue_waters.csv')
data_mira_path = os.path.join(curr_dir, 'data/data_mira.csv')
data_helios_path = os.path.join(curr_dir, 'data/data_helios.csv')
data_philly_path = os.path.join(curr_dir, 'data/data_philly.csv')
data_philly_gpu_schedule_path = os.path.join(curr_dir, 'data/philly_df_schedule.csv')
data_supercloud_path = os.path.join(curr_dir, 'data/data_supercloud.csv')
data_theta_path = os.path.join(curr_dir, 'data/data_theta.csv')
data_thetagpu_path = os.path.join(curr_dir, 'data/data_thetagpu.csv')

banner_image = Image.open(banner_image_path)
st.image(banner_image)

@st.cache_data
def load_data():
    bw = pd.read_csv(data_blue_waters_path)
    mira = pd.read_csv(data_mira_path)
    hl = pd.read_csv(data_helios_path)
    philly = pd.read_csv(data_philly_path)
    philly_gpu = pd.read_csv(data_philly_gpu_schedule_path)
    supercloud = pd.read_csv(data_supercloud_path)
    theta = pd.read_csv(data_theta_path)
    thetagpu = pd.read_csv(data_thetagpu_path)
    return bw, mira, hl, philly, philly_gpu, supercloud, theta, thetagpu

bw_df, mira_df_2, hl_df, philly_df, philly_gpu_schedule_df, sc_df, th_df, th_gpu_df = load_data()

columns=["job", "user", "project", "state", "gpu_num", "cpu_num", "node_num", "submit_time", "wait_time", "run_time", "wall_time", "node_hour"]

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

if 'file_counter' not in st.session_state:
    st.session_state.file_counter = 2

user_entered_cluster_names = {}
user_uploaded_cluster_data_files = {}
with st.expander("**Upload Own Files Section**", expanded=False):
    with st.form("Upload_Files_form"):
        st.markdown("<h2 style = 'text-align: center; background-color: red'>This upload feature is still under construction, please do not upload anything.</h2>", unsafe_allow_html = True)
        st.markdown("<h3 style = 'text-align: center'>Provide Your CSV Data for Visual Analysis</h3>", unsafe_allow_html = True)
        st.markdown("<h6>Ensure the files your are uploading contains the essential columns and data for visual plotting. See the sample file for reference - <a href='https://gist.github.com/MonishSoundarRaj/56f2e24982b89761761b02dac481077c'>Click Here To View</a></h6>", unsafe_allow_html = True)
        st.markdown("<p style='color: teal; background-color: yellow; padding: 8px; border-radius: 5px;'>Note: We won't store any of your information, all your uploaded files will be automatically deleted after your session ends.</p>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
                add_more_files_button = st.form_submit_button("Add More Files") 
        with col2:
                remove_files_button = st.form_submit_button("Remove File")
                
        if add_more_files_button:
            st.session_state.file_counter += 1
        elif remove_files_button:
            st.session_state.file_counter -= 1
        else:
            pass
            
        for i in range(1, st.session_state.file_counter):
            cluster_name = st.text_input(f"{i}] Enter The Name of the Cluster:", help="Enter the name of the clusters for which you will be uploading the file below.")
            user_entered_cluster_names[i] = cluster_name
            
            uploaded_file = st.file_uploader(f"Upload File {i}", type=["csv"])
            if uploaded_file:
                user_uploaded_cluster_data_files[i] = uploaded_file
                
        submit_all_forms = st.form_submit_button("Submit All Files")

    if submit_all_forms: 
        for i, name in user_entered_cluster_names.items():
            get_uploaded_file = user_uploaded_cluster_data_files.get(i)
            
            if get_uploaded_file:
                uploaded_file_columns = pd.read_csv(get_uploaded_file).columns
                all_columns_present_check = set(columns).issubset(uploaded_file_columns)        

                if not all_columns_present_check:
                    user_uploaded_cluster_data_files[i] = None
                    st.markdown(f"<h6 style='color: red'>The file '{name}' is missing essential columns. Please re-upload with the required columns and click 'Submit All Files'. Required columns: {columns}</h6>", unsafe_allow_html = True)
                
main_nav = option_menu(options=["Job Geometric Characteristics", "Job Failure Characteristics", "User Behavior Characteristics"],
                                 menu_title="Pick a characteristic to view available model options",
                                 icons=["bi-1-circle", "bi-2-circle", "bi-3-circle"],
                                 styles=styles, orientation="horizontal", menu_icon="bi-segmented-nav")

if main_nav == "Job Geometric Characteristics":
    nav_bar_horizontal = option_menu("Job Geometric: Pick a model to load related charts",
     ["Job Run Time", "Job Arrival Pattern", "System Utilization & Resource Occupation", "Job Waiting Time"],
     default_index=0, orientation="vertical", menu_icon="bi-list")

    if nav_bar_horizontal == "Job Run Time":
        jrt_system_models_jgc = ["Blue Waters", "Mira", "Philly", "Helios", "Super Cloud", "Theta", "Theta GPU"] 
        jrt_selected_system_models_jgc = jrt_system_models_jgc.copy() 
        
        jrt_chart_selection_left_col_options_jgc = ["CDF Run Time"]
        jrt_chart_selection_right_col_options_jgc = ["Detailed Run Time Distribution"]
        jrt_chart_selection_options_jgc = jrt_chart_selection_left_col_options_jgc + jrt_chart_selection_right_col_options_jgc
        jrt_charts_selected_list_jgc = jrt_chart_selection_options_jgc.copy()
        x_value_selected = None
        
        jrt_drt_time_ranges_jgc = ['0~30s', '30s~10m', '10m~1h', '1h~12h', "more than 12h"]
        jrt_drt_selected_time_range_jgc = jrt_drt_time_ranges_jgc.copy()
        
        with st.form("jrt_chart_selection_form_jgc"):
            st.write(f"### **{chart_selection_form_title}**")
            st.write(f'**{chart_selection_form_load_charts_text}**')
            col1, col2 = st.columns(2)
            with col1 :
                for item in jrt_chart_selection_left_col_options_jgc:
                    jrt_chart_selection_check_box_left_option_jgc = st.checkbox(item, True)
                    if not jrt_chart_selection_check_box_left_option_jgc:
                        jrt_charts_selected_list_jgc.remove(item)
            with col2:
                for item2 in jrt_chart_selection_right_col_options_jgc:
                    jrt_chart_selection_check_box_right_option_jgc = st.checkbox(item2, True)
                    if not jrt_chart_selection_check_box_right_option_jgc:
                        jrt_charts_selected_list_jgc.remove(item2)

            jrt_chart_selection_check_box_submission_button_jgc = st.form_submit_button("Load Charts")
            
            if jrt_chart_selection_check_box_submission_button_jgc:
                if len(jrt_charts_selected_list_jgc) >= 1:
                    st.write(f"**You Have Selected:** {jrt_charts_selected_list_jgc}")
                else:
                    st.markdown("<h5 style='color: red'>Please select one or more charts options above and then click 'Load Charts'</h5>", unsafe_allow_html=True)
            else:
                pass
            
        if len(jrt_charts_selected_list_jgc) >= 1:
            st.sidebar.markdown("<h1 style='text-align: center;'>Chart Customization Panel</h1>", unsafe_allow_html=True)
            
            min_value_exp_run_time_slider = 0
            max_value_exp_run_time_slider = 8 

            with st.sidebar.form("jrt_sidebar_form_jgc"):
                st.write("### Alter the following settings to customize the selected chart(s):")
                
                with st.expander("**Select System Model(s)**", expanded=True):
                        for item in jrt_system_models_jgc:
                            jrt_model_checkbox_jpc = st.checkbox(item, True)
                            if not jrt_model_checkbox_jpc:
                                jrt_selected_system_models_jgc.remove(item)
                                
                if "CDF Run Time" in jrt_charts_selected_list_jgc:                
                    with st.expander("**CDF Run Time Chart (Y and X - axis)**", expanded=True):       
                        jrt_cdf_frequency_slider_jgc = st.slider("**Adjust frequency range (Y-axis):**", min_value=0, max_value=100, step=20, value=100)
                        jrt_cdf_run_time_slider_jgc = st.slider("**Adjust run time range (in powers of 10) (X-axis):**", min_value_exp_run_time_slider, max_value_exp_run_time_slider, step=1, value=8)
                        jrt_cdf_run_time_slider_value_jgc = int(10**jrt_cdf_run_time_slider_jgc)
                        
                if "Detailed Run Time Distribution" in jrt_charts_selected_list_jgc:
                    with st.expander("**Detailed Run Time Distribution Chart (Y and X - axis)**", expanded=True):
                        jrt_drt_frequency_slider_jgc = st.slider("**Adjust frequency range (Y-axis):**", min_value=0.0, max_value=0.6, step=0.1, value=0.6)
                        st.write("##### **Select Run Time Range (X-axis):**")
                        for item in jrt_drt_time_ranges_jgc:
                            jrt_drt_time_range_checkbox_jgc = st.checkbox(item, True)
                            if not jrt_drt_time_range_checkbox_jgc:
                                    jrt_drt_selected_time_range_jgc.remove(item)  
                                    
                jrt_submit_parameters_button_jgc = st.form_submit_button("Apply Changes")
                  
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
                res.append(lt_xs(sc_df["run_time"], t1, t2))
                res.append(lt_xs(th_df["run_time"], t1, t2))
                res.append(lt_xs(th_gpu_df["run_time"], t1, t2))    
                return res
                        
            def polt_cdf_job_run_time(side_by_side, chart_title):
                if side_by_side:
                    st.markdown(f"<h4 style='text-align: center;'>{chart_title}</h4>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h2 style='text-align: center;'>{chart_title}</h2>", unsafe_allow_html=True)   
                
                system_data = {
                    'Blue Waters': (bw_df["run_time"], ":", "blue"),
                    'Mira': (mira_df_2["run_time"], "--", "orange"),
                    'Philly': (philly_df["run_time"], "-.", "green"),
                    'Helios': (hl_df["run_time"], "--", "red"),
                    'Super Cloud': (sc_df["run_time"], (1, (6,1)), "lightblue"),
                    'Theta': (th_df["run_time"], "solid", "grey"),
                    'Theta GPU': (th_gpu_df["run_time"], (0, (5,1)), "violet")
                }             
                
                for system, (data, linestyle, color) in system_data.items():
                    if system in jrt_selected_system_models_jgc:
                        plot_cdf(data, 10000, "Job Run Time (s)", linestyle=linestyle, color=color)
                        
                plt.rc('legend', fontsize=12)
                plt.legend(jrt_selected_system_models_jgc, loc="lower right")
                plt.xscale("log")
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()
                
            def plot_detailed_run_time(side_by_side, chart_title): 
                x = [0, 30, 600, 3600, 12 * 3600, 100000]
                x_value = np.array([1, 2, 3, 4, 5])
                labels = ['0~30s', '30s~10m', '10m~1h', '1h~12h', "more than 12h"]
                bw = []
                mr = []
                ply = []
                hl = []
                sc = []
                th = []
                th_gpu = []
                width = 0.12
                
                for i in range(1, len(x)):
                    if labels[i-1] in jrt_drt_selected_time_range_jgc:
                        res = lt_xs_all(x[i-1], x[i])
                        bw.append(res[0])
                        mr.append(res[1])
                        ply.append(res[2])
                        hl.append(res[3])
                        sc.append(res[4])
                        th.append(res[5])
                        th_gpu.append(res[6])
                                
                x_value_selected = np.arange(1, len(jrt_drt_selected_time_range_jgc) + 1)
                
                if side_by_side:
                    st.markdown(f"<h4 style='text-align: center;'>{chart_title}</h4>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h2 style='text-align: center;'>{chart_title}</h2>", unsafe_allow_html=True)
                
                system_data = {
                    "Blue Waters": ((x_value_selected - 7 * width / 2), bw, "x", "blue"),
                    "Mira": ((x_value_selected - 5 * width / 2), mr, "\\", "orange"),
                    "Philly": ((x_value_selected - 3 * width / 2), ply, ".", "green"),
                    "Helios": ((x_value_selected - width / 2), hl, "-", "red"),
                    "Super Cloud": ((x_value_selected + width / 2), sc, "*", "lightblue"),
                    "Theta": ((x_value_selected + 3 * width/2), th, "/", "grey"),
                    "Theta GPU": ((x_value_selected + 5 * width/2), th_gpu, "|", "violet")
                }

                for system, (value, data, hatch, color) in system_data.items():
                    if system in jrt_selected_system_models_jgc:
                        plt.bar(value, data, width, edgecolor="black", hatch=hatch, color=color)
                            
                plt.ylim(0.00, jrt_drt_frequency_slider_jgc)
                plt.xticks(x_value_selected, jrt_drt_selected_time_range_jgc)
                plt.legend(jrt_selected_system_models_jgc, prop={'size': 10}, loc="upper right")
                plt.ylabel("Frequency (%)", fontsize=14)
                plt.xlabel("Job Run Time (s)", fontsize=14)
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()
            
            with st.expander(f"**{chart_view_settings_title}**", expanded=True):
                jrt_check_box_view_side_by_side_jgc = st.checkbox("Select to view charts side by side")

            with st.spinner(spinner_text):   
                st.markdown("<h1 style='text-align: center;'>Comparisons Of Run Time Among Four DataSets Charts</h1>", unsafe_allow_html=True) 
                if len(jrt_selected_system_models_jgc) >= 1:
                    if jrt_check_box_view_side_by_side_jgc:
                        col1, col2 = st.columns(2)
                        for idx, item in enumerate(jrt_charts_selected_list_jgc):
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
                        if "CDF Run Time" in jrt_charts_selected_list_jgc:
                            polt_cdf_job_run_time(False, "CDF Run Time")
                        else:
                            pass
                        if "Detailed Run Time Distribution" in jrt_charts_selected_list_jgc:
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
        jap_system_models_jgc = ["Blue Waters", "Mira", "Philly", "Helios", "Super Cloud", "Theta", "Theta GPU"]
        jap_selected_system_models_jgc = jap_system_models_jgc.copy()
        jap_chart_selection_left_col_options_jgc = ["Daily Submit Pattern", "Weekly Submit Pattern"]
        jap_chart_selection_right_col_options_jgc = ["Job Arrival Interval"]
        jap_chart_selection_options_jgc = jap_chart_selection_left_col_options_jgc + jap_chart_selection_right_col_options_jgc
        jap_charts_selected_list_jgc = jap_chart_selection_options_jgc.copy()
        
        with st.form("jap_select_chart_model_jgc"):
            st.write(f"### **{chart_selection_form_title}**")
            st.write(f'**{chart_selection_form_load_charts_text}**')
            col1, col2 = st.columns(2)
            with col1 :
                for item in jap_chart_selection_left_col_options_jgc:
                    jap_chart_selection_check_box_left_option_jgc = st.checkbox(item, True)
                    if not jap_chart_selection_check_box_left_option_jgc:
                        jap_charts_selected_list_jgc.remove(item)
            with col2:
                for item2 in jap_chart_selection_right_col_options_jgc:
                    jap_chart_selection_check_box_right_option_jgc = st.checkbox(item2, True)
                    if not jap_chart_selection_check_box_right_option_jgc:
                        jap_charts_selected_list_jgc.remove(item2)

            jap_chart_selection_check_box_submission_button_jgc = st.form_submit_button("Load Charts")
            
            if jap_chart_selection_check_box_submission_button_jgc:
                if len(jap_charts_selected_list_jgc) >= 1:
                        st.write(f"**You have selected:** {jap_charts_selected_list_jgc}")
                else:
                    st.markdown("<h5 style='color: red'>Please select one or more charts options above and then click 'Load Charts'</h5>", unsafe_allow_html=True)           
            else:
                pass
        
        if len(jap_charts_selected_list_jgc) >= 1:
            st.sidebar.markdown("<h1 style='text-align: center;'>Chart Customization Panel</h1>", unsafe_allow_html=True)

            with st.sidebar.form("jap_sidebar_form_jgc"):
                st.write("### Alter the following settings to customize the selected chart(s):")
                
                with st.expander("**Select System Model(s)**", expanded=True):
                        for item in jap_system_models_jgc:
                            jap_model_checkbox_jgc = st.checkbox(item, True)
                            if not jap_model_checkbox_jgc:
                                jap_selected_system_models_jgc.remove(item)
                                
                if "Daily Submit Pattern" in jap_charts_selected_list_jgc:                
                    with st.expander("**Daily Submit Pattern Chart (Y and X - axis)**", expanded=True):  
                        jap_dsp_job_count_slider_jgc = st.slider("**Adjust Job Submit Count Range (Y-axis):**", min_value=0, max_value=180, step=20, value=180)
                        jap_dsp_hour_of_the_day_slider_jgc = st.slider("**Adjust Hour of the Day Range (X-axis):**", min_value=-1, max_value=24, step=1, value=24)  
                       
                        
                if "Weekly Submit Pattern" in jap_charts_selected_list_jgc:
                    with st.expander("**Weekly Submit Pattern Chart (Y and X - axis)**", expanded=True):
                        jap_wsp_job_count_slider_jgc = st.slider("**Adjust Job Submit Count Range (Y-axis):**", min_value=0, max_value=3000, step=500, value=3000)
                        jap_wsp_hour_of_the_day_slider_jgc = st.slider("**Adjust Day of the Week Range (X-axis):**", min_value=0, max_value=8, step=1, value=8)
                        
                                    
                if "Job Arrival Interval" in jap_charts_selected_list_jgc:
                    with st.expander("**Job Arrival Interval Chart (Y and X - axis)**", expanded=True):
                        
                        jap_jai_min_value_exp_arrival_interval_slider_jgc = 0
                        jap_jai_max_value_exp_arrival_interval_slider_jgc = 8 
                        
                        jap_jai_job_count_slider_jgc = st.slider("**Adjust Frequency Range (y-axis):**", min_value=0, max_value=100, step=20, value=100)
                        jap_jai_hour_of_the_day_slider_jgc = st.slider("**Adjust Job Arrival Interval Range (in powers of 10) (X-axis):**", jap_jai_min_value_exp_arrival_interval_slider_jgc, jap_jai_max_value_exp_arrival_interval_slider_jgc, step=1, value=8)
                        jap_jai_hour_of_the_day_slider_value_jgc = int(10**jap_jai_hour_of_the_day_slider_jgc)
                                      
                jap_submit_parameters_button_jgc = st.form_submit_button("Apply Changes")
                
            def get_time_of_day(time, timestamp=True):
                if timestamp:
                    time = datetime.fromtimestamp(time)
                else:
                    time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
                
                return (time.hour + (time.minute > 30)) % 24, time.strftime('%Y-%m-%d')

            def get_day_of_week(time):
                time = datetime.fromtimestamp(time)
                iso_calendar = time.isocalendar()
                return iso_calendar[2], iso_calendar[1]

            def plot_time_submit(submit_time, xlabel, ylabel="Number of Submitted Jobs", week=False, marker="o", color=""):
                if week:
                    times_and_days = [get_time_of_day(i) for i in submit_time]
                    times = [td[0] for td in times_and_days]
                    dd = Counter(times)
                    days = {td[1] for td in times_and_days}
                    n = len(days)
                else:
                    days_and_weeks = [get_day_of_week(i) for i in submit_time]
                    days = [dw[0] for dw in days_and_weeks]
                    dd = Counter(days)
                    weeks = {dw[1] for dw in days_and_weeks}
                    n = len(weeks)
                keys = sorted(dd.keys())
                avg_values = [np.array(dd[key])/n for key in keys]
                plt.plot(keys, avg_values, marker=marker, linewidth=3, markersize=12, color=color)
            
            def plot_daily_submit_pattern(side_by_side, chart_title):
                if side_by_side:
                    st.markdown(f"<h4 style='text-align: center;'>{chart_title}</h4>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h2 style='text-align: center;'>{chart_title}</h2>", unsafe_allow_html=True)

                plt.figure(figsize=(12,7))
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)

                system_data = {
                    "Blue Waters": (bw_df["submit_time"], "^", "blue"),
                    "Mira": (mira_df_2["submit_time"], "o", "orange"),
                    "Philly": (philly_df["submit_time"], "s", "green"),
                    "Helios": (hl_df["submit_time"], "d", "red"),
                    "Super Cloud": (sc_df["submit_time"], "<", "lightblue"),
                    "Theta": (th_df["submit_time"], "x", "grey"),
                    "Theta GPU": (th_gpu_df["submit_time"], ">", "violet")
                }

                for system, (data, marker, color) in system_data.items():
                    if system in jap_selected_system_models_jgc:
                        plot_time_submit(data, xlabel="Hour of the Day", week=True, marker=marker, color=color)

                plt.xlabel("Hour of the Day", fontsize=22)
                plt.ylabel("Job Submit Count", fontsize=22)
                st.set_option('deprecation.showPyplotGlobalUse', False)
                plt.ylim(0, jap_dsp_job_count_slider_jgc)
                plt.xlim(-1, jap_dsp_hour_of_the_day_slider_jgc)
                plt.tight_layout()
                plt.grid(True)
                plt.legend(jap_selected_system_models_jgc, prop={'size': 14}, loc="upper right")
                plt.rc('legend', fontsize=20)
                st.pyplot()
            
            def plot_weekly_submit_pattern(side_by_side, chart_title):
                
                if side_by_side:
                    st.markdown(f"<h4 style='text-align: center;'>{chart_title}</h4>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h2 style='text-align: center;'>{chart_title}</h2>", unsafe_allow_html=True)

                plt.figure(figsize=(12,7))
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16) 

                system_data = {
                    "Blue Waters": (bw_df["submit_time"], "^", "blue"),
                    "Mira": (mira_df_2["submit_time"], "o", "orange"),
                    "Philly": (philly_df["submit_time"], "s", "green"),
                    "Helios": (hl_df["submit_time"], "d", "red"),
                    "Super Cloud": (sc_df["submit_time"], "<", "lightblue"),
                    "Theta": (th_df["submit_time"], "x", "grey"),
                    "Theta GPU": (th_gpu_df["submit_time"], ">", "violet")
                }

                for system, (data, marker, color) in system_data.items():
                    if system in jap_selected_system_models_jgc:
                        plot_time_submit(data, xlabel="Day of the Week", week=False, marker=marker, color=color)

                plt.xlabel("Day of the Week", fontsize=22)
                plt.ylabel("Job Submit Count", fontsize=22)
                plt.ylim(0, jap_wsp_job_count_slider_jgc)
                plt.tight_layout()
                plt.xlim(0, jap_wsp_hour_of_the_day_slider_jgc)
                plt.grid(True)
                plt.legend(jap_selected_system_models_jgc,  prop={'size': 12}, loc="upper right")
                st.set_option('deprecation.showPyplotGlobalUse', False)
                plt.rc('legend',fontsize=20)
                st.pyplot()

            def plot_job_arrival_interval(side_by_side ,chart_title): 
                if side_by_side:
                    st.markdown(f"<h4 style='text-align: center;'>{chart_title}</h4>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h2 style='text-align: center;'>{chart_title}</h2>", unsafe_allow_html=True)
                
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
                def plot_cdf(x, bins ,xlabel, ylabel="Frequency (%)",color="", linestyle="--"):
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
                    plt.ylim(0, jap_jai_job_count_slider_jgc)
                    plt.xlim(int(10 ** jap_jai_min_value_exp_arrival_interval_slider_jgc), jap_jai_hour_of_the_day_slider_value_jgc)

                    plt.grid(True)
                
                plt.style.use("default")

                plt.figure(figsize=[6,5])
                
                system_data = {
                    "Blue Waters": (get_interval(bw_df["submit_time"]), 1000, "Job Arrival Interval (s)", ":",  "blue"),
                    "Mira": (get_interval(mira_df_2["submit_time"]), 1000, "Job Arrival Interval (s)", "--",  "orange"),
                    "Philly": (get_interval(philly_df["submit_time"]), 1000, "Job Arrival Interval (s)","-.", "green"),
                    "Helios": (get_interval(hl_df["submit_time"]), 10009999, "Job Arrival Interval (s)", "--", "red"),
                    "Super Cloud": (get_interval(sc_df["submit_time"]), 1000, "Job Arrival Interval (s)", (1, (6,1)),  "lightblue"),
                    "Theta": (get_interval(th_df["submit_time"]), 1000, "Job Arrival Interval (s)",  "solid", "grey"),
                    "Theta GPU": (get_interval(th_gpu_df["submit_time"]), 1000, "Job Arrival Interval (s)",(0, (5, 1)), "violet")
                }

                for system, (data, value, xlabel, linestyle, color) in system_data.items():
                    if system in jap_selected_system_models_jgc:
                        plot_cdf(data, value, xlabel, linestyle=linestyle)
                    
                plt.rc('legend',fontsize=22)
                plt.legend(jap_selected_system_models_jgc, loc = "upper right", prop={'size': 12})
                st.set_option('deprecation.showPyplotGlobalUse', False)
                plt.xscale("log")
                st.pyplot()   
                       
            with st.expander(f"**{chart_view_settings_title}**", expanded=True):
                jap_check_box_view_side_by_side_jgc = st.checkbox("Select to view charts side by side")   
                
            with st.spinner(spinner_text):
                st.markdown("<h1 style='text-align: center;'>Comparisons Of Job Arrival Patterns</h1>", unsafe_allow_html=True)
                if len(jap_selected_system_models_jgc) >= 1:
                    if jap_check_box_view_side_by_side_jgc:          
                        col1, col2 = st.columns(2)
                        for idx, item in enumerate(jap_charts_selected_list_jgc):
                            jap_col_logic_cal_jgc = col1 if idx % 2 == 0 else col2
                            if item == "Daily Submit Pattern":
                                with jap_col_logic_cal_jgc:
                                    plot_daily_submit_pattern(True, "Daily Submit Pattern")  
                            elif item == "Weekly Submit Pattern":
                                with jap_col_logic_cal_jgc:
                                    plot_weekly_submit_pattern(True, "Weekly Submit Pattern")    
                            elif item == "Job Arrival Interval":
                                with jap_col_logic_cal_jgc:
                                    plot_job_arrival_interval(True, "Job Arrival Interval")
                            else:
                                pass
                    else:
                        for item in jap_charts_selected_list_jgc:
                            if item == "Daily Submit Pattern":
                                plot_daily_submit_pattern(False, "Daily Submit Pattern")  
                            elif item == "Weekly Submit Pattern":
                                plot_weekly_submit_pattern(False, "Weekly Submit Pattern")    
                            elif item == "Job Arrival Interval":
                                plot_job_arrival_interval(False, "Job Arrival Interval")
                            else:
                                pass
                
                    with st.expander(f"**{chart_description_expander_title}**", expanded=True):
                        st.write("**Daily Submit Pattern Chart Description:** Displays a chart presenting the job arrival counts of each job trace for each hour of the day")
                        st.write("**Weekly Submit Pattern Chart Description:** Displays a chart presenting the job arrival counts of each job trace for each day of the week")
                        st.write("**Job Arrival Interval:** Displays a Cumulative Distribution Functions (CDF) of job arrival interval(s) comparison of the four job traces (Blue Waters, Mira, Philly, and Helios).")          
                else:
                    st.markdown("<h2 style='color: red'>Please select one or more system model(s) from sidebar to plot the chart</h2>", unsafe_allow_html=True)  
             
    elif nav_bar_horizontal == "System Utilization & Resource Occupation":
        suaro_cpu_charts_options_jgc = ["Blue Waters CPU", "Mira CPU", "Super Cloud CPU", "Theta CPU"]
        suaro_gpu_charts_options_jgc = ["Blue Waters GPU", 
                    "Philly GPU", "Helios GPU", "Super Cloud GPU", "Philly GPU-SchedGym", "Theta GPU"]
        suaro_charts_options_jgc = suaro_cpu_charts_options_jgc + suaro_gpu_charts_options_jgc
        suaro_charts_selected_list_jgc = suaro_charts_options_jgc.copy()
        
        #Additional charts
        suaro_system_models_jgc = ["Blue Waters", "Mira", "Philly", "Helios", "Super Cloud", "Theta", "Theta GPU"]
        suaro_selected_system_models_jgc = suaro_system_models_jgc.copy()
        suaro_other_charts_left_options_jgc = ["Core Hours w.r.t Job Size", "Core Hours w.r.t Job Run Time"]
        suaro_other_charts_right_options_jgc = ["CDF of Requested Cores", "CDF of Core Hour"]
        suaro_other_charts_options_jgc = suaro_other_charts_left_options_jgc + suaro_other_charts_right_options_jgc
        suaro_other_charts_selected_list_jgc = []

        with st.form("suaro_select_charts_checkbox_main_form_jgc"): 
            st.write(f"### **{chart_selection_form_title}**")
            st.write(f'**{chart_selection_form_load_charts_text}**')
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<h4 style="text-align: center;">CPU Charts</h4>', unsafe_allow_html=True)
                for item in suaro_cpu_charts_options_jgc:
                    suaro_chart_selected_jgc = st.checkbox(item, True)
                    if not suaro_chart_selected_jgc:
                        suaro_charts_selected_list_jgc.remove(item)
            with col2:
                st.markdown('<h4 style="text-align: center;">GPU Charts</h4>', unsafe_allow_html=True)
                for item in suaro_gpu_charts_options_jgc:
                    suaro_chart_selected_jgc = st.checkbox(item, True)
                    if not suaro_chart_selected_jgc:
                        suaro_charts_selected_list_jgc.remove(item)
                   
            with col1:
                st.markdown("<br><br><br>", unsafe_allow_html=True)
                st.markdown('<h4 style="text-align: center;">Core Hour Charts</h4>', unsafe_allow_html=True)
                for item in suaro_other_charts_left_options_jgc:
                    suaro_other_chart_selected_jgc = st.checkbox(item)
                    if suaro_other_chart_selected_jgc:
                        suaro_other_charts_selected_list_jgc.append(item)
            with col2:
                st.markdown('<h4 style="text-align: center;">CDF Charts</h4>', unsafe_allow_html=True)
                for item in suaro_other_charts_right_options_jgc:
                    suaro_other_chart_selected_jgc = st.checkbox(item)
                    if suaro_other_chart_selected_jgc:
                        suaro_other_charts_selected_list_jgc.append(item)
    
                        
            suaro_select_charts_checkbox_main_form_button_jgc = st.form_submit_button("Load Charts")
            
            if suaro_select_charts_checkbox_main_form_button_jgc:
                # fix this to include the other charts
                if len(suaro_charts_selected_list_jgc) >= 1 or len(suaro_other_charts_selected_list_jgc) >= 1:
                    st.write(f'**You have selected:** {suaro_charts_selected_list_jgc} {suaro_other_charts_selected_list_jgc}')
                else:
                    st.markdown("<h5 style='color: red;'>You have not selected any chart options above, please select one or more chart option(s) to load the charts.</h5>", unsafe_allow_html=True)
            else: 
                pass

        if len(suaro_charts_selected_list_jgc) >= 1 or len(suaro_other_charts_selected_list_jgc) >= 1:
            st.sidebar.markdown("<h1 style='text-align: center;'>Chart Customization Panel</h1>", unsafe_allow_html=True)
             
            with st.sidebar.form("suaro_sidebar_form_jgc"):
                st.write("### Alter the following settings to customize the selected chart(s):")
                if len(suaro_charts_selected_list_jgc) >= 1:
                    with st.expander("**System Utilization Charts control (Y and X - axis)**", expanded=True):
                        suaro_sys_utilization_slider_jgc = st.slider("**Adjust System Utilization Range (Y-axis):**", min_value = 0, max_value=100, value=100, step=10)
                        suaro_time_slider_jgc = st.slider("**Adjust Time Range (X-axis):**", min_value=0, max_value=120, value=120, step=10)
                        
                if len(suaro_other_charts_selected_list_jgc) >= 1:
                    with st.expander("**Select System Model(s)**", expanded=True):
                        st.write("Pick the models below to change the X-axis of core hours charts and legend of CDF charts")
                        for item in suaro_system_models_jgc:
                            suaro_model_checkbox_jgc = st.checkbox(item, True)
                            if not suaro_model_checkbox_jgc:
                                suaro_selected_system_models_jgc.remove(item)

                    if "Core Hours w.r.t Job Size" in suaro_other_charts_selected_list_jgc:
                        suaro_chwjs_job_sizes_list_jgc = ["Small", "Middle", "Large"]
                        suaro_chwjs_job_sizes_selected_list_jgc = suaro_chwjs_job_sizes_list_jgc.copy()
                        with st.expander("**Core Hours w.r.t Job Size Chart Controls (Legend and Y-axis)**", expanded=True):
                            st.write("##### **Select Job Size(s) (Legend):**")             
                            for item in suaro_chwjs_job_sizes_list_jgc:
                                suaro_chwjs_job_size_checkbox_jgc = st.checkbox(item, True)
                                if not suaro_chwjs_job_size_checkbox_jgc:
                                    suaro_chwjs_job_sizes_selected_list_jgc.remove(item)
                                else:
                                    pass
                                
                            suaro_chwjs_percentage_slider_jgc = st.slider("**Adjust Percentage (%) Range (Y-axis):**", min_value=0, max_value=100, value=100, step=10) 
                         
                    if "Core Hours w.r.t Job Run Time" in suaro_other_charts_selected_list_jgc:
                        suaro_chwjrt_job_runtime_list_jgc = ["Short", "Medium", "Long"]
                        suaro_chwjrt_job_runtime_selected_list_jgc = suaro_chwjrt_job_runtime_list_jgc.copy()
                        with st.expander("**Core Hours w.r.t Job Run Time Chart Controls (Legend and Y-axis)**", expanded=True):
                            st.write("##### **Select Job Run Time(s) (Legend):**")      
                            for item in suaro_chwjrt_job_runtime_list_jgc:
                                suaro_chwjrt_job_runtime_checkbox_jgc = st.checkbox(item, True)
                                if not suaro_chwjrt_job_runtime_checkbox_jgc:
                                   suaro_chwjrt_job_runtime_selected_list_jgc.remove(item)
                                else:
                                    pass
                            suaro_chwjrt_percentage_slider_jgc = st.slider("**Adjust Percentage(%) Range (Y -axis):**", min_value=0, max_value=100, value=100, step=10) 
                       
                    if "CDF of Requested Cores" in suaro_other_charts_selected_list_jgc:
                        with st.expander("**CDF of Requested Cores Chart Controls (Y and X-axis)**", expanded=True):
                            suaro_cdforc_min_value_exp_arrival_interval_slider_jgc = 0
                            suaro_cdforc_max_value_exp_arrival_interval_slider_jgc = 6
                            suaro_cdforc_frequency_slider_jgc = st.slider("**Adjust Frequency(%) Range (Y-axis) :**", min_value=0, max_value=100, value=100, step=20)
                            suaro_cdforc_core_slider_jgc = st.slider("**Adjust Core Range (in powers of 10) (X-axis) :**", suaro_cdforc_min_value_exp_arrival_interval_slider_jgc, suaro_cdforc_max_value_exp_arrival_interval_slider_jgc, value=8, step=1)
                            suaro_cdforc_core_slider_value_jgc = int(10 ** suaro_cdforc_core_slider_jgc) 
                            
                    if "CDF of Core Hour" in suaro_other_charts_selected_list_jgc:
                        with st.expander("**CDF of Core Hour Chart Controls (Y and X-axis)**", expanded=True): 
                            suaro_cdfoch_min_value_exp_arrival_interval_slider_jgc = -3
                            suaro_cdfoch_max_value_exp_arrival_interval_slider_jgc = 8
                            suaro_cdfoch_frequency_slider_jgc = st.slider("**Adjust Frequency(%) Range (Y -axis) :**", min_value=0, max_value=100, value=100, step=20)
                            suaro_cdfoch_core_hour_slider_jgc = st.slider("**Adjust Core Hour Range (in powers of 10) (X-axis) :**", suaro_cdfoch_min_value_exp_arrival_interval_slider_jgc, suaro_cdfoch_max_value_exp_arrival_interval_slider_jgc, value=8, step=1)
                            suaro_cdfoch_core_hour_slider_value_jgc = int(10 ** suaro_cdfoch_core_hour_slider_jgc)                       
                            
                suaro_submit_button_sidebar_jgc = st.form_submit_button("Apply Changes")
                    
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
                    
            with st.expander(f"**{chart_view_settings_title}**", expanded=True):
                    suaro_check_box_view_side_by_side_jgc = st.checkbox("Select to view charts side by side")
            
            with st.spinner("In Progess... Please do not change any settings now"):
                if len(suaro_charts_selected_list_jgc) >= 1:
                    st.markdown("<h1 style='text-align: center;'>The System Utilization Across Multiple Systems Charts</h1>", unsafe_allow_html=True)
                    
                if suaro_check_box_view_side_by_side_jgc:          
                    col1, col2 = st.columns(2)
                    for idx, item in enumerate(suaro_charts_selected_list_jgc):
                        suaro_col_logic_cal_jgc = col1 if idx % 2 == 0 else col2
                        if item == "Blue Waters CPU":
                            with suaro_col_logic_cal_jgc:
                                plot_util_jgc(bw_df[1000:], 22636*32, "cpu_num", color="blue", side_by_side=True, chart_title="Blue Waters CPU Chart")  
                        elif item == "Mira CPU":
                            with suaro_col_logic_cal_jgc:
                                plot_util_jgc(mira_df_2, 49152, color='orange', side_by_side=True, chart_title="Mira CPU Chart")
                        elif item == "Super Cloud CPU":
                            with suaro_col_logic_cal_jgc:
                                plot_util_jgc(sc_df, 704, color='lightblue', side_by_side=True, chart_title="Super Cloud CPU Chart")
                        elif item == "Theta CPU":
                            with suaro_col_logic_cal_jgc:
                                plot_util_jgc(th_df, 4392, color='violet', side_by_side=True, chart_title="Theta CPU Chart")
                        elif item == "Blue Waters GPU":
                            with suaro_col_logic_cal_jgc:
                                plot_util_jgc(bw_df[1000:], 4228, "gpu_num", color="#1f77b4", side_by_side=True, chart_title="Blue Waters GPU Chart")
                        elif item == "Philly GPU":
                            with suaro_col_logic_cal_jgc:
                                plot_util_jgc(philly_df, 2490, "gpu_num", color='#2ca02c', side_by_side=True, chart_title="Philly GPU Chart")
                        elif item == "Helios GPU":
                            with suaro_col_logic_cal_jgc:
                                plot_util_jgc(hl_df, 1080, "gpu_num", side_by_side=True, chart_title="Helios GPU Chart")
                        elif item == "Super Cloud GPU":
                            with suaro_col_logic_cal_jgc:
                                plot_util_jgc(sc_df, 448, "gpu_num", color='#9467bd', side_by_side=True, chart_title="Super Cloud GPU Chart")
                        elif item == "Philly GPU-SchedGym":
                            with suaro_col_logic_cal_jgc:
                                plot_util_jgc(philly_gpu_schedule_df, 2490, "gpu_num", color='#9467bd', side_by_side=True, chart_title="Philly GPU-SchedGym Chart")
                        elif item == "Theta GPU":
                            with suaro_col_logic_cal_jgc:
                                plot_util_jgc(th_gpu_df, 24, "gpu_num", color='#9467bd', side_by_side=True, chart_title="Theta GPU Chart")
                        else:
                            pass
                else:
                    for item in suaro_charts_selected_list_jgc:
                        if item == "Blue Waters CPU":
                            plot_util_jgc(bw_df[1000:], 22636*32, "cpu_num", color="blue", side_by_side=False, chart_title="Blue Waters CPU Chart")  
                        elif item == "Mira CPU":
                            plot_util_jgc(mira_df_2, 49152, color='orange', side_by_side=False, chart_title="Mira CPU Chart")
                        elif item == "Super Cloud CPU":
                            plot_util_jgc(sc_df, 704, color='lightblue', side_by_side=False, chart_title="Super Cloud CPU Chart")
                        elif item == "Theta CPU":
                            plot_util_jgc(th_df, 4392, color='violet', side_by_side=False, chart_title="Theta CPU Chart")
                        elif item == "Blue Waters GPU":
                            plot_util_jgc(bw_df[1000:], 4228, "gpu_num", color="#1f77b4", side_by_side=False, chart_title="Blue Waters GPU Chart")
                        elif item == "Philly GPU":
                            plot_util_jgc(philly_df, 2490, "gpu_num", color='#2ca02c', side_by_side=False, chart_title="Philly GPU Chart")
                        elif item == "Helios GPU":
                            plot_util_jgc(hl_df, 1080, "gpu_num", side_by_side=False, chart_title="Helios GPU Chart")
                        elif item == "Super Cloud GPU":
                            plot_util_jgc(sc_df, 448, "gpu_num", color='#9467bd', side_by_side=False, chart_title="Super Cloud GPU Chart")
                        elif item == "Philly GPU-SchedGym":
                            plot_util_jgc(philly_gpu_schedule_df, 2490, "gpu_num", color='#9467bd', side_by_side=False, chart_title="Philly GPU-SchedGym Chart")
                        elif item == "Theta GPU":
                            plot_util_jgc(th_gpu_df, 24, "gpu_num", color='#9467bd', side_by_side=False, chart_title="Theta GPU Chart")
                        else:
                            pass
                        
                        
                def plot_percentage_corehour(selected_job_sizes, frequency_value, selected_models, run_time=False, side_by_side=False, chart_title="none"):
                    plt.style.use("default")
                    traces = selected_models
                    
                    if side_by_side:
                        st.markdown(f"<h4 style='text-align: center;'>{chart_title}</h4>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<h2 style='text-align: center;'>{chart_title}</h2>", unsafe_allow_html=True)
                        
                    short_job_size_dic = {'Blue Waters': 1.74, 'Mira': 4.70, 'Philly': 1.17, 'Helios': 1.97, 'Super Cloud': 2.78, 'Theta': 2.23, 'Theta GPU': 5.60}
                    medium_job_size_dic = {'Blue Waters': 62.07, 'Mira': 81.24, 'Philly': 14.32, 'Helios': 22.28, 'Super Cloud': 32.19, 'Theta': 78.47, 'Theta GPU': 94.34}
                    long_job_size_dic = {'Blue Waters': 36.18, 'Mira': 14.05, 'Philly': 84.51, 'Helios': 75.75, 'Super Cloud': 65.03, 'Theta': 19.30, 'Theta GPU': 0.07}

                    small_job_size_dic = {'Blue Waters': 86.21, 'Mira': 34.12, 'Philly': 18.48, 'Helios': 4.57, 'Super Cloud': 99.65, 'Theta': 17.43, 'Theta GPU': 53.24}
                    middle_job_size_dic = {'Blue Waters': 4.48, 'Mira': 46.63, 'Philly': 68.87, 'Helios': 37.93, 'Super Cloud': 0.35, 'Theta': 48.13, 'Theta GPU': 38.77}
                    large_job_size_dic = {'Blue Waters': 9.31, 'Mira': 19.25, 'Philly': 12.65, 'Helios': 57.50, 'Super Cloud': 0.0, 'Theta': 34.44, 'Theta GPU': 7.98}

                    if run_time:
                        status = {}
                        if "Short" in selected_job_sizes:
                            status['Short'] = [short_job_size_dic[system_model] for system_model in short_job_size_dic if system_model in selected_models]
                        if "Medium" in selected_job_sizes:
                            status['Medium'] = [medium_job_size_dic[system_model] for system_model in medium_job_size_dic if system_model in selected_models]
                        if "Long" in selected_job_sizes:
                            status['Long'] = [long_job_size_dic[system_model] for system_model in long_job_size_dic if system_model in selected_models]
                    else:
                        status = {}
                        if "Small" in selected_job_sizes:
                            status['Small'] = [small_job_size_dic[system_model] for system_model in small_job_size_dic if system_model in selected_models]
                        if "Middle" in selected_job_sizes:
                            status['Middle'] = [middle_job_size_dic[system_model] for system_model in middle_job_size_dic if system_model in selected_models]
                        if "Large" in selected_job_sizes:
                            status['Large'] = [large_job_size_dic[system_model] for system_model in large_job_size_dic if system_model in selected_models]

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
                    ax.set_ylabel('Percentage (%)', fontsize=20)
                    traces_ticks = []
                    for item in traces:
                        if item == "Blue Waters":
                            traces_ticks.append("bw")
                        elif item == "Mira":
                            traces_ticks.append("mr")
                        elif item == "Philly":
                            traces_ticks.append("phi")
                        elif item == "Helios":
                            traces_ticks.append("hl")
                        elif item == "Super Cloud":
                            traces_ticks.append("sc")
                        elif item == "Theta":
                            traces_ticks.append("th")
                        else:
                            traces_ticks.append("th_gpu")
                            
                    ax.set_xticks(x + width, traces_ticks, fontsize=15)
                    ax.legend(fontsize=15)
                    ax.set_ylim(0, frequency_value)
                    plt.grid(axis="y")
                    st.pyplot()  
                
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
                    plt.ylim(0, 100)
                    plt.xlim(10**0, 10**8)
                    plt.grid(True)
                    plt.style.use("default")
                
                bw_values = np.where(bw_df["cpu_num"] != 0, bw_df["cpu_num"], bw_df["gpu_num"] * 16)
                sc_values = np.where(sc_df["cpu_num"] != 0, sc_df["cpu_num"], sc_df["gpu_num"] * 16)

                system_data_cdforc = {
                    'Blue Waters': (bw_values, ":", "blue"),
                    'Mira': (mira_df_2["node_num"].values, "--", "orange"),
                    'Philly': (philly_df["gpu_num"].values, "-.", "green"),
                    'Helios': (hl_df["gpu_num"].values, "--", "red"),
                    'Super Cloud': (sc_values, (1, (6,1)), "lightblue"),
                    'Theta': (th_df["cpu_num"].values, "solid", "grey"),
                    'Theta GPU': (th_gpu_df["gpu_num"].values, (0, (5,1)), "violet")
                }

                bw_cdfoch_values = bw_values * bw_df["run_time"] / 3600
                sc_cdfoch_values = sc_values * sc_df["run_time"] / 3600

                system_data_cdfoch = {
                    'Blue Waters': (bw_cdfoch_values, ":", "blue"),
                    'Mira': (mira_df_2["node_num"] * mira_df_2["run_time"] / 3600, "--", "orange"),
                    'Philly': (philly_df["gpu_num"] * philly_df["run_time"] / 3600, "-.", "green"),
                    'Helios': (hl_df["gpu_num"] * hl_df["run_time"] / 3600, "--", "red"),
                    'Super Cloud': (sc_cdfoch_values, (1, (6,1)), "lightblue"),
                    'Theta': (th_df["node_num"] * th_df["run_time"] / 3600, "solid", "grey"),
                    'Theta GPU': (th_gpu_df["node_num"] * th_gpu_df["run_time"] / 3600, (0, (5,1)), "violet")
                }
                
                def suaro_cdforc_plot_jgc(y_axis_value, x_axis_value, selected_models, side_by_side = False, chart_title= "none"):
                    
                    if side_by_side:
                        st.markdown(f"<h4 style='text-align: center;'>{chart_title}</h4>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<h2 style='text-align: center;'>{chart_title}</h2>", unsafe_allow_html=True)
                        
                    for system, (data, linestyle, color) in system_data_cdforc.items():
                        if system in selected_models:
                            plot_cdf(data, 100000, "Core", linestyle=linestyle, color=color)
                    plt.legend(selected_models)
                    plt.xscale("log")
                    plt.rc('legend', fontsize=23)  
                    plt.ylim(0, y_axis_value)
                    plt.xlim(10**0, 10**x_axis_value)
                    st.pyplot()
            
                def suaro_cdfoch_plot_jgc(y_axis_value, x_axis_value, selected_models, side_by_side = False, chart_title= "none"):
                    
                    if side_by_side:
                        st.markdown(f"<h4 style='text-align: center;'>{chart_title}</h4>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<h2 style='text-align: center;'>{chart_title}</h2>", unsafe_allow_html=True)
                    
                    for system, (data, linestyle, color) in system_data_cdfoch.items():
                        if system in selected_models:
                            plot_cdf(data, 100000, "Core Hour", linestyle=linestyle, color=color)
                            
                    plt.legend(selected_models)
                    plt.xscale("log")
                    plt.rc('legend', fontsize=23)
                    plt.ylim(0, y_axis_value)
                    plt.xlim(10**-3, 10**x_axis_value)
                    st.pyplot()
                
                if "Core Hours w.r.t Job Size" in suaro_other_charts_selected_list_jgc or "Core Hours w.r.t Job Run Time" in suaro_other_charts_selected_list_jgc:
                    st.markdown("<h1 style='text-align: center;'>Core Hour Domination of Different Types of Jobs Charts</h1>", unsafe_allow_html=True)   
                if suaro_check_box_view_side_by_side_jgc:          
                    col1, col2 = st.columns(2)
                    for idx, item in enumerate(suaro_other_charts_selected_list_jgc):
                        suaro_col_logic_cal_jgc = col1 if idx % 2 == 0 else col2
                        if item == "Core Hours w.r.t Job Size":
                            with suaro_col_logic_cal_jgc:
                               plot_percentage_corehour(suaro_chwjs_job_sizes_selected_list_jgc, suaro_chwjs_percentage_slider_jgc, suaro_selected_system_models_jgc, False, True, "Core Hours w.r.t Job Size")
                        elif item == "Core Hours w.r.t Job Run Time":
                            with suaro_col_logic_cal_jgc:
                               plot_percentage_corehour(suaro_chwjrt_job_runtime_selected_list_jgc, suaro_chwjrt_percentage_slider_jgc, suaro_selected_system_models_jgc, True, True, "Core Hours w.r.t Job Run Time")  
                        else:
                            pass
                else: 
                    for idx, item in enumerate(suaro_other_charts_selected_list_jgc):
                        if item == "Core Hours w.r.t Job Size":
                               plot_percentage_corehour(suaro_chwjs_job_sizes_selected_list_jgc, suaro_chwjs_percentage_slider_jgc, suaro_selected_system_models_jgc, False, False, "Core Hours w.r.t Job Size")
                        elif item == "Core Hours w.r.t Job Run Time":
                               plot_percentage_corehour(suaro_chwjrt_job_runtime_selected_list_jgc, suaro_chwjrt_percentage_slider_jgc, suaro_selected_system_models_jgc, True, False, "Core Hours w.r.t Job Run Time")  
                        else:
                            pass 
                
                if "CDF of Requested Cores" in suaro_other_charts_selected_list_jgc or "CDF of Core Hour" in suaro_other_charts_selected_list_jgc:      
                    st.markdown("<h1 style='text-align: center;'>Comparisons of Core and Core Hour Among Four Datasets Charts</h1>", unsafe_allow_html=True)
                if suaro_check_box_view_side_by_side_jgc:          
                    col1, col2 = st.columns(2)
                    for idx, item in enumerate(suaro_other_charts_selected_list_jgc):
                        suaro_col_logic_cal_jgc = col1 if idx % 2 == 0 else col2
                        if item == "CDF of Requested Cores":
                            with suaro_col_logic_cal_jgc:
                                suaro_cdforc_plot_jgc(suaro_cdforc_frequency_slider_jgc, suaro_cdforc_core_slider_jgc, suaro_selected_system_models_jgc, True, "CDF of Requested Cores")
                        elif item ==  "CDF of Core Hour":
                            with suaro_col_logic_cal_jgc:
                                suaro_cdfoch_plot_jgc(suaro_cdfoch_frequency_slider_jgc, suaro_cdfoch_core_hour_slider_jgc, suaro_selected_system_models_jgc, True, "CDF of Core Hour")     
                        else:
                            pass
                else:  
                    for idx, item in enumerate(suaro_other_charts_selected_list_jgc):
                        if item == "CDF of Requested Cores":
                            suaro_cdforc_plot_jgc(suaro_cdforc_frequency_slider_jgc, suaro_cdforc_core_slider_jgc, suaro_selected_system_models_jgc, False, "CDF of Requested Cores")
                        elif item ==  "CDF of Core Hour":
                            suaro_cdfoch_plot_jgc(suaro_cdfoch_frequency_slider_jgc, suaro_cdfoch_core_hour_slider_jgc, suaro_selected_system_models_jgc, False, "CDF of Core Hour")     
                        else:
                            pass
                    
                with st.expander(f"**{chart_description_expander_title}**", expanded=True):
                    st.write("""
                    **System Utilization Overview:** Analyzing four systems, Blue Waters showcases both CPU and GPU usage due to its combined design, while Philly and Helios focus on GPU utilization, as their CPUs are auxiliary. Philly and Helios have notably lower utilization, often under 80% even with waiting jobs. Two issues with Philly are an initial low utilization phase and a potentially ineffective Fair scheduler. Testing with SchedGym using the FCFS+Backfilling approach showed utilization could hit 100%, up from the original 80% peak.
                    """)

    elif nav_bar_horizontal == "Job Waiting Time":
        jwt_system_models_jgc = ["Blue Waters", "Mira", "Philly", "Helios", "Super Cloud", "Theta", "Theta GPU"]
        jwt_selected_system_models_jgc = jwt_system_models_jgc.copy()
        jwt_cdf_chart_options_jgc = ["CDF of Wait Time", "CDF of Turnaround Time"]
        jwt_avg_wait_time_chart_options_jgc = ["Avg waiting Time w.r.t Job Size",  "Avg Waiting Time w.r.t Job Run Time"]
        jwt_chart_options_jgc = jwt_cdf_chart_options_jgc + jwt_avg_wait_time_chart_options_jgc
        jwt_charts_selected_list_jgc = jwt_chart_options_jgc.copy()
                
        with st.form("jwt_select_charts_checkbox_main_form_jgc"): 
            st.write(f"### **{chart_selection_form_title}**")
            st.write(f'**{chart_selection_form_load_charts_text}**')
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<h4 style="text-align: center;">CDF Charts</h4>', unsafe_allow_html=True)
                for item in jwt_cdf_chart_options_jgc:
                    jwt_chart_selected_jgc = st.checkbox(item, True)
                    if not jwt_chart_selected_jgc:
                        jwt_charts_selected_list_jgc.remove(item)
            with col2:
                st.markdown('<h4 style="text-align: center;">Avg Wait Time Charts</h4>', unsafe_allow_html=True)
                for item in jwt_avg_wait_time_chart_options_jgc:
                    jwt_chart_selected_jgc = st.checkbox(item, True)
                    if not jwt_chart_selected_jgc:
                        jwt_charts_selected_list_jgc.remove(item)
                        
            jwt_select_charts_checkbox_main_form_button_jgc = st.form_submit_button("Load Charts")
            
            if jwt_select_charts_checkbox_main_form_button_jgc:
                if len(jwt_charts_selected_list_jgc) >= 1:
                    st.write(f'**You have selected:** {jwt_charts_selected_list_jgc}')
                else:
                    st.markdown("<h5 style='color: red;'>You have not selected any chart options above, please select one or more chart option(s) to load the charts.</h5>", unsafe_allow_html=True)
            else: 
                pass
            
        if len(jwt_charts_selected_list_jgc) >= 1:
            st.sidebar.markdown("<h1 style='text-align: center;'>Chart Customization Panel</h1>", unsafe_allow_html=True)

            with st.sidebar.form("jwt_sidebar_form_jgc"):
                    st.write("### Alter the following settings to customize the selected chart(s):")
                    
                    with st.expander("**Select System Model(s)**", expanded=True):
                        st.write("##### Pick the models below to change the legend of CDF's charts and X-axis of the Avg Wait Time charts")
                        for item in jwt_system_models_jgc:
                            jwt_model_checkbox_jpc = st.checkbox(item, True)
                            if not jwt_model_checkbox_jpc:
                                jwt_selected_system_models_jgc.remove(item)
                                    
                    if "CDF of Wait Time" in jwt_charts_selected_list_jgc:                
                        with st.expander("**CDF of Wait Time Controls (Y and X - axis)**", expanded=True): 
                            jwt_cdfowt_min_value_exp_arrival_interval_slider_jgc = 0
                            jwt_cdfowt_max_value_exp_arrival_interval_slider_jgc = 8 
                            jwt_cdfowt_frequency_slider_jgc = st.slider("**Adjust Frequency(%) Range (Y-axis):**", min_value=0, max_value=100, value=100, step=20)
                            jwt_cdfowt_job_wait_time_slider_jgc = st.slider("**Adjust Job Wait Time Range (in powers of 10) (X-axis):**", jwt_cdfowt_min_value_exp_arrival_interval_slider_jgc, jwt_cdfowt_max_value_exp_arrival_interval_slider_jgc, value=jwt_cdfowt_max_value_exp_arrival_interval_slider_jgc, step=1)
                            jwt_cdfowt_job_wait_time_slider_value_jgc = int(10 ** jwt_cdfowt_job_wait_time_slider_jgc)       
                    
                    if "CDF of Turnaround Time" in jwt_charts_selected_list_jgc:
                        with st.expander("**CDF of Turnaround Time Controls (Y and X - axis)**", expanded=True): 
                            jwt_cdfott_min_value_exp_arrival_interval_slider_jgc = 0
                            jwt_cdfott_max_value_exp_arrival_interval_slider_jgc = 8
                            jwt_cdfott_frequency_slider_jgc = st.slider("**Adjust Frequency(%) Range (Y-axis) :**", min_value=0, max_value=100, value=100, step=20)
                            jwt_cdfott_turnaround_time_slider_jgc = st.slider("**Adjust Turnaround Time Range (in powers of 10) (X-axis) :**", jwt_cdfott_min_value_exp_arrival_interval_slider_jgc, jwt_cdfott_max_value_exp_arrival_interval_slider_jgc, value=8, step=1)
                            jwt_cdfott_turnaround_time_slider_value_jgc = int(10 ** jwt_cdfott_turnaround_time_slider_jgc) 
                        
                    if "Avg waiting Time w.r.t Job Size" in jwt_charts_selected_list_jgc:
                        jwt_awtjs_job_sizes_list_jgc = ["Small", "Middle", "Large"]
                        jwt_awtjs_job_sizes_selected_list_jgc = jwt_awtjs_job_sizes_list_jgc.copy()
                        
                        with st.expander("**Avg waiting Time w.r.t Job Size Chart Controls (legend and Y-axis)**", expanded=True): 
                            st.write("##### **Select Job Size(s) (Legend):**")             
                            for item in jwt_awtjs_job_sizes_list_jgc:
                                jwt_awtjs_job_size_checkbox_jgc = st.checkbox(item, True)
                                if not jwt_awtjs_job_size_checkbox_jgc:
                                    jwt_awtjs_job_sizes_selected_list_jgc.remove(item)
                                else:
                                    pass
                            jwt_awtjs_avg_wait_time_slider_jgc = st.slider("**Adjust Average Wait Time (hours) Range (Y-axis):**", min_value=0, max_value=100, value=100, step=10)
                    
                    if "Avg Waiting Time w.r.t Job Run Time" in jwt_charts_selected_list_jgc:
                            jwt_awtjrt_job_run_time_list_jgc = ["Short", "Medium", "Long"]
                            jwt_awtjrt_job_run_time_selected_list_jgc = jwt_awtjrt_job_run_time_list_jgc.copy()
                            
                            with st.expander("**Avg waiting Time w.r.t Job Run Time Chart Controls (legend and Y-axis)**", expanded=True):  
                                st.write("##### **Select Job Run Time(s) (Legend):**") 
                                for item in jwt_awtjrt_job_run_time_list_jgc:
                                    jwt_awtjrt_job_run_time_checkbox_jgc = st.checkbox(item, True)
                                    if not jwt_awtjrt_job_run_time_checkbox_jgc:
                                        jwt_awtjrt_job_run_time_selected_list_jgc.remove(item)
                                    else:
                                        pass
                                jwt_awtjrt_avg_wait_time_slider_jgc = st.slider("**Adjust Average Wait Time (hours) Range (Y-axis) :**", min_value=0, max_value=100, value=100, step=10)
                                        
                    jwt_submit_parameters_button_jgc = st.form_submit_button("Apply Changes")
                         
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
                
                short_job_size_dic = {'Blue Waters': 1.74, 'Mira': 4.70, 'Philly': 1.17, 'Helios': 1.97, 'Super Cloud': 2.78, 'Theta': 2.23, 'Theta GPU': 5.60}
                medium_job_size_dic = {'Blue Waters': 62.07, 'Mira': 81.24, 'Philly': 14.32, 'Helios': 22.28, 'Super Cloud': 32.19, 'Theta': 78.47, 'Theta GPU': 94.34}
                long_job_size_dic = {'Blue Waters': 36.18, 'Mira': 14.05, 'Philly': 84.51, 'Helios': 75.75, 'Super Cloud': 65.03, 'Theta': 19.30, 'Theta GPU': 0.07}

                small_job_size_dic = {'Blue Waters': 86.21, 'Mira': 34.12, 'Philly': 18.48, 'Helios': 4.57, 'Super Cloud': 99.65, 'Theta': 17.43, 'Theta GPU': 53.24}
                middle_job_size_dic = {'Blue Waters': 4.48, 'Mira': 46.63, 'Philly': 68.87, 'Helios': 37.93, 'Super Cloud': 0.35, 'Theta': 48.13, 'Theta GPU': 38.77}
                large_job_size_dic = {'Blue Waters': 9.31, 'Mira': 19.25, 'Philly': 12.65, 'Helios': 57.50, 'Super Cloud': 0.0, 'Theta': 34.44, 'Theta GPU': 7.98}
            
                if run_time:
                    status = {}
                    if "Short" in selected_job_sizes:
                        status['Short'] = [short_job_size_dic[system_model] for system_model in short_job_size_dic if system_model in selected_models]
                    else:
                        pass
                    if "Medium" in selected_job_sizes:
                        status['Medium'] = [medium_job_size_dic[system_model] for system_model in medium_job_size_dic if system_model in selected_models]
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
                        status['Middle'] = [middle_job_size_dic[system_model] for system_model in middle_job_size_dic if system_model in selected_models]
                    else:
                        pass
                    if "Large" in selected_job_sizes:
                        status['Large'] = [large_job_size_dic[system_model] for system_model in large_job_size_dic if system_model in selected_models]
                    else:
                        pass

                x = np.arange(len(traces))  
                width = 0.20 
                multiplier = 0

                fig, ax = plt.subplots()
                hatches= ["-", ".", "x", "-"]
                
                for i, (attribute, measurement) in enumerate(status.items()):
                    offset = width * multiplier
                    rects = ax.bar(x + offset, measurement, width, label=attribute, hatch=hatches[i], edgecolor='black')
                    ax.bar_label(rects, padding=3, fontsize = 8)
                    multiplier += 1
                
                traces_ticks = []
                for item in traces:
                    if item == "Blue Waters":
                        traces_ticks.append("bw")
                    elif item == "Mira":
                        traces_ticks.append("mr")
                    elif item == "Philly":
                        traces_ticks.append("phi")
                    elif item == "Helios":
                        traces_ticks.append("hl")
                    elif item == "Super Cloud":
                        traces_ticks.append("sc")
                    elif item == "Theta":
                        traces_ticks.append("th")
                    else:
                        traces_ticks.append("th_gpu")
                    
                ax.set_ylabel('Percentage (%)', fontsize=18)
                ax.set_xlabel('Traces', fontsize=18)
                ax.set_xticks(x + width, traces_ticks, fontsize=12)
                ax.legend(fontsize=14, loc="upper right")
                ax.set_ylim(0, frequency_value)
                plt.grid(axis="y")
                st.pyplot(fig)
                    
            def plot_cdf_wait_time(side_by_side, chart_title):
                if side_by_side:
                    st.markdown(f"<h5 style='text-align: center;'>{chart_title}</h5>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h2 style='text-align: center;'>{chart_title}</h2>", unsafe_allow_html=True)
                
                system_data= {
                    "Blue Waters": {"data": bw_df["wait_time"], "color": "blue", "linestyle": ":"},
                    "Mira": {"data": mira_df_2["wait_time"], "color": "orange", "linestyle": "--"},
                    "Philly": {"data": philly_df[10000:130000]["wait_time"], "color": "green", "linestyle": "-."},
                    "Helios": {"data": hl_df["wait_time"], "color": "red", "linestyle": "--"},
                    "Super Cloud": {"data": sc_df["wait_time"], "color": "lightblue", "linestyle": (1, (6,1))},
                    "Theta": {"data": th_df["wait_time"], "color": "grey", "linestyle": "solid"},
                    "Theta GPU": {"data": th_gpu_df["wait_time"], "color": "violet", "linestyle": (0, (5,1))}
                }

                for cluster in jwt_selected_system_models_jgc:
                    if cluster in system_data:
                        cluster_data = system_data[cluster]
                        plot_cdf(cluster_data["data"], 100000, "Job Wait Time (s)", color=cluster_data["color"], linestyle=cluster_data["linestyle"])

                plt.ylabel('Frequency (%)', fontsize=18)
                plt.xlabel('Time Range', fontsize=18)
                plt.ylim(0, jwt_cdfowt_frequency_slider_jgc)
                plt.xlim(int(10**jwt_cdfowt_min_value_exp_arrival_interval_slider_jgc), jwt_cdfowt_job_wait_time_slider_value_jgc)
                plt.rc('legend', fontsize=12)
                plt.legend(jwt_selected_system_models_jgc, loc="lower right")
                st.set_option('deprecation.showPyplotGlobalUse', False)
                plt.xscale("log")
                st.pyplot()
            
            def plot_cdf_turnaround_time(side_by_side, chart_title):
                if side_by_side:
                    st.markdown(f"<h5 style='text-align: center;'>{chart_title}</h5>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h2 style='text-align: center;'>{chart_title}</h2>", unsafe_allow_html=True)
                
                        
                system_data = {
                    "Blue Waters": {"data": bw_df["wait_time"] + bw_df["run_time"], "color": "blue", "linestyle": ":"},
                    "Mira": {"data": mira_df_2["wait_time"] + mira_df_2["run_time"], "color": "orange", "linestyle": "--"},
                    "Philly": {"data": philly_df["wait_time"] + philly_df["run_time"], "color": "green", "linestyle": "-."},
                    "Helios": {"data": hl_df["wait_time"] + hl_df["run_time"], "color": "red", "linestyle": "--"},
                    "Super Cloud": {"data": sc_df["wait_time"] + sc_df["run_time"], "color": "lightblue", "linestyle": (1, (6,1))},
                    "Theta": {"data": th_df["wait_time"] + th_df["run_time"], "color": "grey", "linestyle": "solid"},
                    "Theta GPU": {"data": th_gpu_df["wait_time"] + th_gpu_df["run_time"], "color": "violet", "linestyle": (0, (5,1))}
                }

                for cluster in jwt_selected_system_models_jgc:
                    if cluster in system_data:
                        cluster_data = system_data[cluster]
                        plot_cdf(cluster_data["data"], 100000, "Job Turnaround Time", color=cluster_data["color"], linestyle=cluster_data["linestyle"])
                
                plt.xscale("log")
                plt.ylabel('Frequency (%)', fontsize=18)
                plt.xlabel('Time Range', fontsize=18)
                st.set_option('deprecation.showPyplotGlobalUse', False)
                plt.xlim(int(10 ** jwt_cdfott_min_value_exp_arrival_interval_slider_jgc), jwt_cdfott_turnaround_time_slider_value_jgc)
                plt.ylim(0, jwt_cdfott_frequency_slider_jgc)
                plt.rc('legend', fontsize=12)
                plt.legend(jwt_selected_system_models_jgc, loc="upper left")
                st.pyplot()
                    
            def plot_avg_wait_wrt_time_job_size(side_by_side, chart_title):
                if side_by_side:
                    st.markdown(f"<h5 style='text-align: center;'>{chart_title}</h5>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h2 style='text-align: center;'>{chart_title}</h2>", unsafe_allow_html=True)
                    
                plot_percentage_corehour(jwt_awtjs_job_sizes_selected_list_jgc, jwt_awtjs_avg_wait_time_slider_jgc, jwt_selected_system_models_jgc)
            
            def plot_avg_wait_time_wrt_job_run_time(side_by_side, chart_title):
                if side_by_side:
                    st.markdown(f"<h5 style='text-align: center;'>{chart_title}</h5>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h2 style='text-align: center;'>{chart_title}</h2>", unsafe_allow_html=True)
                    
                plot_percentage_corehour(jwt_awtjrt_job_run_time_selected_list_jgc, jwt_awtjrt_avg_wait_time_slider_jgc, jwt_selected_system_models_jgc, True)
                
            with st.expander(f"**{chart_view_settings_title}**", expanded=True):
                    jwt_check_box_view_side_by_side_jgc = st.checkbox("Select to view charts side by side")

            with st.spinner(spinner_text):    
                st.markdown("<h1 style='text-align: center;'>The Job Wait Time Charts</h1>", unsafe_allow_html=True)
                if len(jwt_selected_system_models_jgc) >= 1:
                    if jwt_check_box_view_side_by_side_jgc:          
                        col1, col2 = st.columns(2)
                        for idx, item in enumerate(jwt_charts_selected_list_jgc):
                            jwt_col_logic_cal_jgc = col1 if idx % 2 == 0 else col2
                            if item == "CDF of Wait Time":
                                with jwt_col_logic_cal_jgc:
                                    plot_cdf_wait_time(True, "CDF of Wait Time")  
                            elif item == "CDF of Turnaround Time":
                                with jwt_col_logic_cal_jgc:
                                    plot_cdf_turnaround_time(True, "CDF of Turnaround Time")
                            elif item == "Avg waiting Time w.r.t Job Size":
                                with jwt_col_logic_cal_jgc:
                                    if len(jwt_awtjs_job_sizes_selected_list_jgc) >= 1:
                                        plot_avg_wait_wrt_time_job_size(True, "Avg waiting Time w.r.t Job Size")
                                    else:    
                                        st.markdown("<h6 style='color: red'>Please select one or more 'Job size(s)' and click 'Apply Changes' in sidebar to plot this chart</h6>", unsafe_allow_html=True)  
                            elif item == "Avg Waiting Time w.r.t Job Run Time":
                                with jwt_col_logic_cal_jgc:
                                    if len(jwt_awtjrt_job_run_time_selected_list_jgc) >= 1:
                                        plot_avg_wait_time_wrt_job_run_time(True, "Avg Waiting Time w.r.t Job Run Time")
                                    else:
                                        st.markdown("<h6 style='color: red'>Please select one or more 'Job Run Time(s)' and click 'Apply Changes' in sidebar to plot this chart</h6>", unsafe_allow_html=True)
                            else:
                                pass
                    else:
                        for item in jwt_charts_selected_list_jgc:
                            if item == "CDF of Wait Time":
                                plot_cdf_wait_time(False, "CDF of Wait Time")  
                            elif item == "CDF of Turnaround Time":
                                plot_cdf_turnaround_time(False, "CDF of Turnaround Time")
                            elif item == "Avg waiting Time w.r.t Job Size":
                                if len(jwt_awtjs_job_sizes_selected_list_jgc) >= 1:
                                    plot_avg_wait_wrt_time_job_size(False, "Avg waiting Time w.r.t Job Size")
                                else:    
                                    st.markdown("<h3 style='color: red'>Please select one or more 'Job size(s)' and click 'Apply Changes' in sidebar to plot this chart</h3>", unsafe_allow_html=True)  
                            elif item == "Avg Waiting Time w.r.t Job Run Time":
                                if len(jwt_awtjrt_job_run_time_selected_list_jgc) >= 1:
                                    plot_avg_wait_time_wrt_job_run_time(False, "Avg Waiting Time w.r.t Job Run Time")
                                else:
                                    st.markdown("<h3 style='color: red'>Please select one or more 'Job Run Time(s)' and click 'Apply Changes' in sidebar to plot this chart</h3>", unsafe_allow_html=True)
                            else:
                                pass
                              
                    with st.expander(f"**{chart_description_expander_title}**", expanded=True):
                        st.write("**CDF of Wait Time:** This chart offers a visual interpretation of the cumulative distribution of job wait times. By observing the curve, users can easily discern the cumulative percentage of jobs that experience a given wait time or less. It's a valuable tool for understanding system efficiency and detecting bottlenecks in job processing.")
                        st.write("**CDF of Turnaround Time:** Representing the cumulative distribution of the total time from job submission to its completion, this chart sheds light on the overall job processing speed. A glance at the curve provides insights into the efficiency of the system and how promptly it can handle and execute submitted jobs.")
                        st.write("**Avg Waiting Time w.r.t Job Size:** By plotting the average waiting times against job sizes, this chart unveils patterns and correlations between the size of a job and the time it spends waiting in a queue. Larger jobs might require more resources and could potentially wait longer. This visualization is pivotal in understanding and optimizing resource allocation strategies.")
                        st.write("**Avg Waiting Time w.r.t Job Run Time:** This chart delves into the nuanced relationship between the projected runtimes of jobs and their actual waiting times. It can highlight if jobs with longer or shorter projected runtimes tend to wait more or less before getting executed. Such insights can guide better scheduling and resource management decisions.")
                else:
                    st.markdown("<h2 style='color: red'>Please select one or more system model(s) from the sidebar to plot the chart</h2>", unsafe_allow_html=True)
        else:
            pass
    else:
        pass

elif main_nav == "Job Failure Characteristics":
    nav_bar_jfc = option_menu("Job Failure: Pick a model to load related charts", ["Job Failures Distribution", "Correlation between Job Failure and Job Geometries"], 
    default_index=0, orientation="vertical", menu_icon="bi-list")

    # Job Failures Distribution charts plotting function.
    def plot_percentage_status(selected_status, frequency_value, selected_models, side_by_side, chart_title, job_counts=True):
        
        if side_by_side:
            st.markdown(f"<h4 style='text-align: center;'>{chart_title}</h4>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='text-align: center;'>{chart_title}</h2>", unsafe_allow_html=True)
           
        plt.style.use("default") 
        traces = selected_models
        
        pass_dict = {'Blue Waters': 64.99, 'Mira': 70.05, 'Philly': 59.58, 'Helios': 64.72, 'Super Cloud': 93.50, 'Theta': 63.91, 'Theta GPU': 24.44}
        failed_dict = {'Blue Waters': 7.26, 'Mira': 9.01, 'Philly': 30.90, 'Helios': 14.06, 'Super Cloud': 1.15, 'Theta': 7.92, 'Theta GPU': 55.43}
        killed_dict = {'Blue Waters': 27.74, 'Mira': 20.94, 'Philly': 9.52, 'Helios': 21.15, 'Super Cloud': 5.36, 'Theta': 28.17, 'Theta GPU': 20.13}

        pass_dict_2 = {'Blue Waters': 53.64, 'Mira': 56.94, 'Philly': 33.78, 'Helios': 52.42, 'Super Cloud': 91.70, 'Theta': 52.96, 'Theta GPU': 28.21}
        failed_dict_2 = {'Blue Waters': 4.91, 'Mira': 5.78, 'Philly': 33.40, 'Helios': 6.64, 'Super Cloud': 2.16, 'Theta': 3.68, 'Theta GPU': 66.37}
        killed_dict_2 = {'Blue Waters': 41.45, 'Mira': 37.28, 'Philly': 32.82, 'Helios': 40.94, 'Super Cloud': 6.15, 'Theta': 43.36, 'Theta GPU': 5.42}

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

        x = np.arange(len(traces))
        width = 0.20 
        multiplier = 0

        fig, ax = plt.subplots()
        hatches= ["-", ".", "x", "-"]
        for i, (attribute, measurement) in enumerate(status.items()):
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute, hatch=hatches[i], edgecolor='black')
            ax.bar_label(rects, padding=3)
            multiplier += 1
        
        traces_ticks = []
        for item in traces:
            if item == "Blue Waters":
                traces_ticks.append("bw")
            elif item == "Mira":
                traces_ticks.append("mr")
            elif item == "Philly":
                traces_ticks.append("phi")
            elif item == "Helios":
                traces_ticks.append("hl")
            elif item == "Super Cloud":
                traces_ticks.append("sc")
            elif item == "Theta":
                traces_ticks.append("th")
            else:
                traces_ticks.append("th_gpu")

        ax.set_ylabel('Percentage (%)', fontsize=18)
        ax.set_xticks(x + width, traces_ticks, fontsize=12)
        ax.legend(fontsize=15)
        ax.set_ylim(0, frequency_value)
        plt.grid(axis="y")
        st.pyplot(fig)
    
    def plot_status_over(selected_statuses, frequency_slider_value, selected_systems, side_by_side, chart_title, run_time=False):
        
        if side_by_side:
            st.markdown(f"<h4 style='text-align: center;'>{chart_title}</h4>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='text-align: center;'>{chart_title}</h2>", unsafe_allow_html=True)
        
        plt.style.use("default")
        traces = selected_systems
        
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
            sc = [[93.44135211267606, 0.8396619718309859, 5.718985915492958],
    [93.9306831500992, 1.402575972865197, 4.666740877035601],
    [90.88649155722325, 2.5891181988742966, 6.524390243902439]]
            th = [[72.29591836734693, 11.020408163265307, 16.683673469387756],
    [47.777189732733525, 1.587721619476052, 50.63508864779042],
    [0, 2.684563758389262, 97.31543624161074]]
            th_gpu = [[25.809303590859628, 46.78318824809576, 27.407508161044614],
    [21.88330591064422, 71.52259207695228, 6.594102012403494],
    [0, 100.0, 0]]

            z = ["Short","Middle","Long"]

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
            sc = [[93.42672413793103, 0.831953019123626, 5.74132284294534],
    [93.94928116761103, 1.0545788239406872, 4.996140008448283],
    [93.3411184411193, 1.7995799527664729, 4.859301606114214]]
            th = [[64.4215076529487, 8.045520709276168, 27.532971637775134],
    [48.06763285024155, 5.555555555555555, 46.3768115942029],
    [47.35099337748344, 1.6556291390728477, 50.993377483443716]]
            th_gpu = [[22.512866769247093, 53.446171804531225, 24.04096142622168],
    [34.30376270255424, 65.14693765449053, 0.549299642955232],
    [27.11864406779661, 72.88135593220339, 0]]

            z = ["Small","Middle","Large"]
            status = {
                'Small': (15.61, 12.22, 5.68, 0.3, 4.73, 4.28, 1.28),
                'Middle': (143.62, 50.96, 15.84, 0.4, 1.10, 30.98, 1.28),
                'Large': (53.33, 42.83, 13.26, 0.53, 0.0, 159.76, 5.23),
            }
            
        categories = ["Pass", "Failed", "Killed"]
        selected_indices = [categories.index(status) for status in selected_statuses]
        
        system_mapping = {
            "Blue Waters": bw,
            "Mira": mira,
            "Philly": philly,
            "Helios": hl,
            "Super Cloud": sc,
            "Theta": th,
            "Theta GPU": th_gpu
        }

        # Filter out the data of selected systems
        system_data = [system_mapping[item] for item in selected_systems if item in system_mapping]

        x = np.arange(len(traces))
        width = 0.25  
        multiplier = 0

        fig, ax = plt.subplots()
        hatches = ["-", ".", "x"]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        for i, measurement in enumerate(zip(*system_data)):
            offset = width * multiplier
            prev = np.array([0.0] * len(traces))
            for k, j in enumerate(zip(*measurement)):
                if k in selected_indices:
                    rects = ax.bar(x + offset, j, width, hatch=hatches[k], color=colors[k], edgecolor='black', bottom=prev)
                    prev += np.array(j)
            multiplier += 1

        ax.set_ylabel('Percentage (%)', fontsize=20)
        
        if len(traces) == 1:
            ax.set_xticks(np.arange(3) * 0.25, len(traces) * z, fontsize=10, rotation=45) 
        elif len(traces) == 2:
             ax.set_xticks(np.delete(np.arange(7) * 0.25, [3]), len(traces) * z, fontsize=10, rotation=45) 
        elif len(traces) == 3:
            ax.set_xticks(np.delete(np.arange(11) * 0.25, [3, 7]), len(traces) * z, fontsize=10, rotation=45) 
        elif len(traces) == 4:
            ax.set_xticks(np.delete(np.arange(15) * 0.25, [3, 7, 11]), len(traces) * z, fontsize=10, rotation=45) 
        elif len(traces) == 5:
            ax.set_xticks(np.delete(np.arange(19) * 0.25, [3, 7, 11, 15]), len(traces) * z, fontsize=10, rotation=45) 
        elif len(traces) == 6:
            ax.set_xticks(np.delete(np.arange(23) * 0.25, [3, 7, 11, 15, 19]), len(traces) * z, fontsize=10, rotation=45) 
        elif len(traces) == 7:
            ax.set_xticks(np.delete(np.arange(27) * 0.25, [3, 7, 11, 15, 19, 23]), len(traces) * z, fontsize=10, rotation=45) 
        else:
            pass
        
        traces_ticks = []
        for item in selected_systems:
            if item == "Blue Waters":
                traces_ticks.append("bw")
            elif item == "Mira":
                traces_ticks.append("mr")
            elif item == "Philly":
                traces_ticks.append("phi")
            elif item == "Helios":
                traces_ticks.append("hl")
            elif item == "Super Cloud":
                traces_ticks.append("sc")
            elif item == "Theta":
                traces_ticks.append("th")
            else:
                traces_ticks.append("th_gpu")
        
        ax.set_ylim(0, frequency_slider_value)
        legend1 = ax.legend([categories[i] for i in selected_indices], 
                    fontsize=15, loc='upper center', 
                    bbox_to_anchor=(0.5, 1.15), 
                    ncol=len(selected_indices))
        
        empty_handles = [Line2D([0], [0], color='none') for _ in range(len(traces_ticks))]
        legend2 = ax.legend(empty_handles, traces_ticks, 
                            fontsize=15, loc='lower center', 
                            bbox_to_anchor=(0.5, -0.3), 
                            ncol=len(traces_ticks), 
                            handlelength=0, borderpad=0.5)

        ax.add_artist(legend1)
        ax.add_artist(legend2)
        plt.tight_layout()
        plt.grid(axis="y")
        st.pyplot(fig)

    if nav_bar_jfc == "Job Failures Distribution":
        jfd_system_models_jfc = ["Blue Waters", "Mira", "Philly", "Helios", "Super Cloud", "Theta", "Theta GPU"]
        jfd_selected_system_models_jfc = jfd_system_models_jfc.copy()
        jfd_chart_selection_options_jfc = ["Job Count w.r.t Job Status", "Core Hours w.r.t Job Status"] 
        jfd_job_status_list_jfc = ["Pass", "Failed", "Killed"]
        jfd_job_status_selected_list_jfc = jfd_job_status_list_jfc.copy()
        jfd_charts_selected_list_jfc = jfd_chart_selection_options_jfc.copy()

        with st.form("jfd_chart_selection_form_jfc"):
            st.write(f"### **{chart_selection_form_title}**")
            st.write(f'**{chart_selection_form_load_charts_text}**')
            col1, col2 = st.columns(2)
            with col1 :
                jfd_chart_selection_check_box_left_option_jfc = st.checkbox(jfd_chart_selection_options_jfc[0], True)
                if not jfd_chart_selection_check_box_left_option_jfc:
                     jfd_charts_selected_list_jfc.remove(jfd_chart_selection_options_jfc[0])
            with col2:
                jfd_chart_selection_check_box_right_option_jfc = st.checkbox(jfd_chart_selection_options_jfc[1], True)
                if not jfd_chart_selection_check_box_right_option_jfc:
                     jfd_charts_selected_list_jfc.remove(jfd_chart_selection_options_jfc[1])

            jfd_chart_selection_check_box_submission_button_jfc = st.form_submit_button("Load Charts")
            if jfd_chart_selection_check_box_submission_button_jfc:
                if len(jfd_charts_selected_list_jfc) >= 1:
                    st.write(f"**You Have Selected:** {jfd_charts_selected_list_jfc}")
                else:
                    st.markdown("<h5 style='color: red'>Please select one or more charts options above and then click 'Load Charts'</h5>", unsafe_allow_html=True)
            else:
                pass
        
        if len(jfd_charts_selected_list_jfc) >= 1:
            st.sidebar.markdown("<h1 style='text-align: center;'>Chart Customization Panel</h1>", unsafe_allow_html=True)

            with st.sidebar.form("jfd_sidebar_form_jfc"):
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
                    for item in jfd_system_models_jfc:
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
                    st.markdown("<h1 style='text-align: center;'>The Distribution Of Different Job Statuses Charts</h1>", unsafe_allow_html=True)

                    if jfd_check_box_view_side_by_side_jfc:
                            col1, col2 = st.columns(2)
                            for idx, item in enumerate(jfd_charts_selected_list_jfc):
                                jfd_col_logic_cal_jfc = col1 if idx % 2 == 0 else col2
                                if item == "Job Count w.r.t Job Status":
                                    with jfd_col_logic_cal_jfc:
                                        plot_percentage_status(jfd_job_status_selected_list_jfc, jfd_percentage_slider_jfc, jfd_selected_system_models_jfc, True, "Job Count w.r.t Job Status", True)
                                elif item == "Core Hours w.r.t Job Status":
                                    with jfd_col_logic_cal_jfc:
                                        plot_percentage_status(jfd_job_status_selected_list_jfc, jfd_percentage_slider_jfc, jfd_selected_system_models_jfc, True, "Core Hours w.r.t Job Status", False)
                    else:
                        if "Job Count w.r.t Job Status" in jfd_charts_selected_list_jfc:
                            plot_percentage_status(jfd_job_status_selected_list_jfc, jfd_percentage_slider_jfc, jfd_selected_system_models_jfc, False, "Job Count w.r.t Job Status", True)
                        else:
                            pass
                        if "Core Hours w.r.t Job Status" in jfd_charts_selected_list_jfc:
                            plot_percentage_status(jfd_job_status_selected_list_jfc, jfd_percentage_slider_jfc, jfd_selected_system_models_jfc, False, "Core Hours w.r.t Job Status", False)
                        else:
                            pass
            
                with st.expander(f"**{chart_description_expander_title}**", expanded=True):
                    st.write("**Job Count w.r.t Job Status:** This depicts the total number of jobs classified according to their completion status - Pass, Failed, or Killed. It helps in analyzing job execution trends.")
                    st.write("**Core Hours w.r.t Job Status:** This quantifies the total computing resources consumed by jobs, segmented by their final status. It assists in understanding resource utilization in different scenarios.") 
           
            elif len(jfd_job_status_selected_list_jfc) < 1 and len(jfd_selected_system_models_jfc) >= 1:
                st.markdown("<h2 style='color: red'>Please select one or more job status(es) from the sidebar to plot the chart</h2>", unsafe_allow_html=True)         

            elif len(jfd_job_status_selected_list_jfc) >= 1 and len(jfd_selected_system_models_jfc) < 1:
                st.markdown("<h2 style='color: red'>Please select one or more system model(s) from the sidebar to plot the chart</h2>", unsafe_allow_html=True)
                
            else:
                st.markdown("<h2 style='color: red'>Please select one or more job status(es) and system model(s) from the sidebar to plot the chart</h2>", unsafe_allow_html=True)
        else:
            pass
        
    elif nav_bar_jfc == "Correlation between Job Failure and Job Geometries":
        cbjfajg_system_models_jfc = ["Blue Waters", "Mira", "Philly", "Helios", "Super Cloud", "Theta", "Theta GPU"]
        cbjfajg_selected_system_models_jfc = cbjfajg_system_models_jfc.copy()
        cbjfajg_chart_selection_options_jfc = ["Job Status w.r.t Job Size", "Job Status w.r.t Job Run Time"]
        cbjfajg_job_status_list_jfc = ["Pass", "Failed", "Killed"]
        cbjfajg_job_status_selected_list_jfc = cbjfajg_job_status_list_jfc.copy()
        cbjfajg_charts_selected_list_jfc = cbjfajg_chart_selection_options_jfc.copy()

        with st.form("cbjfajg_chart_selection_form_jfc"):
            st.write(f"### **{chart_selection_form_title}**")
            st.write(f'**{chart_selection_form_load_charts_text}**')
            col1, col2 = st.columns(2)
            with col1 :
                cbjfajg_chart_selection_check_box_left_option_jfc = st.checkbox(cbjfajg_chart_selection_options_jfc[0], True)
                if not cbjfajg_chart_selection_check_box_left_option_jfc:
                    cbjfajg_charts_selected_list_jfc.remove(cbjfajg_chart_selection_options_jfc[0])
            with col2:
                cbjfajg_chart_selection_check_box_right_option_jfc = st.checkbox(cbjfajg_chart_selection_options_jfc[1], True)
                if not cbjfajg_chart_selection_check_box_right_option_jfc:
                    cbjfajg_charts_selected_list_jfc.remove(cbjfajg_chart_selection_options_jfc[1])

            cbjfajg_chart_selection_check_box_submission_button_jfc = st.form_submit_button("Load Charts")
            
            if cbjfajg_chart_selection_check_box_submission_button_jfc:
                if len(cbjfajg_charts_selected_list_jfc) >= 1:
                    st.write(f"**You Have Selected:** {cbjfajg_charts_selected_list_jfc}")
                else:
                    st.markdown("<h5 style='color: red'>Please select one or more charts options above and then click 'Load Charts'</h5>", unsafe_allow_html=True)
            else:
                pass
            
        if len(cbjfajg_charts_selected_list_jfc) >= 1:
            st.sidebar.markdown("<h1 style='text-align: center;'>Chart Customization Panel</h1>", unsafe_allow_html=True)

            with st.sidebar.form("cbjfajg_sidebar_form_jfc"):
                st.write("### Alter the following settings to customize the selected chart(s):")
                with st.expander("**Select Job Status(es)**", expanded=True):
                    for item in cbjfajg_job_status_list_jfc:
                        cbjfajg_job_status_checkbox_jfc = st.checkbox(item, True)
                        if not cbjfajg_job_status_checkbox_jfc:
                            cbjfajg_job_status_selected_list_jfc.remove(item)
                        else:
                            pass

                cbjfajg_percentage_slider_jfc = st.slider("**Adjust Percentage Range (Y-axis):**", min_value=0, max_value=100, value=100, step=20)
                
                with st.expander("**Select System Model(s) (X-axis)**", expanded=True):
                    for item in cbjfajg_system_models_jfc:
                        cbjfajg_model_checkbox_jfc = st.checkbox(item, True)
                        if not cbjfajg_model_checkbox_jfc:
                            cbjfajg_selected_system_models_jfc.remove(item)
                        else:
                            pass

                cbjfajg_submit_parameters_button_jfc = st.form_submit_button("Apply Changes")
            
            if len(cbjfajg_job_status_selected_list_jfc) >= 1 and len(cbjfajg_selected_system_models_jfc) >= 1:        
                with st.expander(f"**{chart_view_settings_title}**", expanded=True):
                    cbjfajg_check_box_view_side_by_side_jfc = st.checkbox("Select to view charts side by side")

                with st.spinner(spinner_text):
                    st.markdown("<h1 style='text-align: center;'>Job Failure v.s Job Runtime And Job Requested Resources Charts</h1>", unsafe_allow_html=True)
                    if cbjfajg_check_box_view_side_by_side_jfc:
                        if len(cbjfajg_charts_selected_list_jfc) >= 1:
                            col1, col2 = st.columns(2)
                            for idx, item in enumerate(cbjfajg_charts_selected_list_jfc):
                                cbjfajg_col_logic_cal_jfc = col1 if idx % 2 == 0 else col2
                                if item == "Job Status w.r.t Job Size":
                                    with cbjfajg_col_logic_cal_jfc:
                                        plot_status_over(cbjfajg_job_status_selected_list_jfc, cbjfajg_percentage_slider_jfc, cbjfajg_selected_system_models_jfc, True, "Job Status w.r.t Job Size", False)
                                elif item == "Job Status w.r.t Job Run Time":
                                    with cbjfajg_col_logic_cal_jfc:
                                        plot_status_over(cbjfajg_job_status_selected_list_jfc, cbjfajg_percentage_slider_jfc, cbjfajg_selected_system_models_jfc, True, "Job Status w.r.t Job Run Time", True)
                        else:
                            pass
                    else:
                        if "Job Status w.r.t Job Size" in cbjfajg_charts_selected_list_jfc:
                            plot_status_over(cbjfajg_job_status_selected_list_jfc, cbjfajg_percentage_slider_jfc, cbjfajg_selected_system_models_jfc, False, "Job Status w.r.t Job Size", False)
                        else:
                            pass
                        if "Job Status w.r.t Job Run Time" in cbjfajg_charts_selected_list_jfc:
                            plot_status_over(cbjfajg_job_status_selected_list_jfc, cbjfajg_percentage_slider_jfc, cbjfajg_selected_system_models_jfc, False, "Job Status w.r.t Job Run Time", True)
                        else:
                            pass

                with st.expander(f"**{chart_description_expander_title}**", expanded=True):
                    st.write("**Job Status w.r.t Job Size:** This chart illustrates the status of jobs (Pass, Failed, Killed) with respect to their sizes. It provides insight into how job size may impact completion status, thereby helping to predict potential job execution outcomes.")
                    st.write("**Job Status w.r.t Job Run Time:** This visualization represents the correlation between job status and job run time. By analyzing job completion (Pass, Failed, Killed) in relation to run time, it aids in understanding the efficiency of jobs and can assist in identifying potential bottlenecks or issues in job execution.")
           
            elif len(cbjfajg_job_status_selected_list_jfc) < 1 and len(cbjfajg_selected_system_models_jfc) >= 1:
                st.markdown("<h2 style='color: red'>Please select one or more job status(es) from the sidebar to plot the chart</h2>", unsafe_allow_html=True)         

            elif len(cbjfajg_job_status_selected_list_jfc) >= 1 and len(cbjfajg_selected_system_models_jfc) < 1:
                st.markdown("<h2 style='color: red'>Please select one or more system model(s) from the sidebar to plot the chart</h2>", unsafe_allow_html=True)

            else: 
                st.markdown("<h2 style='color: red'>Please select one or more job status(es) and system model(s) from the sidebar to plot the chart</h2>", unsafe_allow_html=True)
    else:
        pass

elif main_nav == "User Behavior Characteristics":
    ubc_nav_bar = option_menu("User Behavior: Pick a model to load related charts", ["Users’ Repeated Behaviors", "Users’ Submission Behaviors", "Correlation between Job Run Time and Job Statuses"], 
    default_index=0, orientation="vertical", menu_icon="bi-list")

    if ubc_nav_bar == "Users’ Repeated Behaviors":
        urb_chart_selection_left_col_options_ubc = ["Blue Waters", "Mira", "Super Cloud", "Theta"]
        urb_chart_selection_right_col_options_ubc = ["Philly", "Helios", "Theta GPU"]
        urb_chart_selection_options_ubc = urb_chart_selection_left_col_options_ubc + urb_chart_selection_right_col_options_ubc
        urb_charts_selected_list_ubc = urb_chart_selection_options_ubc.copy()

        with st.form("urb_chart_selection_form_ubc"):
            st.write(f"### **{chart_selection_form_title}**")
            st.write(f'**{chart_selection_form_load_charts_text}**')
            col1, col2 = st.columns(2)
            with col1 :
                for item in urb_chart_selection_left_col_options_ubc:
                    urb_chart_selection_check_box_left_option_ubc = st.checkbox(item, True)
                    if not urb_chart_selection_check_box_left_option_ubc:
                        urb_charts_selected_list_ubc.remove(item)
            with col2:
                for item2 in urb_chart_selection_right_col_options_ubc:
                    urb_chart_selection_check_box_right_option_ubc = st.checkbox(item2, True)
                    if not urb_chart_selection_check_box_right_option_ubc:
                        urb_charts_selected_list_ubc.remove(item2)
            urb_chart_selection_check_box_submission_button_ubc = st.form_submit_button("Load Charts")

            if urb_chart_selection_check_box_submission_button_ubc:
                if len(urb_charts_selected_list_ubc) >= 1:
                    st.write(f"**You Have Selected:** {urb_charts_selected_list_ubc}")
                else:
                    st.markdown("<h5 style='color: red'>Please select one or more charts options above and then click 'Load Charts'</h5>", unsafe_allow_html=True)
            else:
                pass

        if len(urb_charts_selected_list_ubc) >= 1:
            st.sidebar.markdown("<h1 style='text-align: center;'>Chart Customization Panel</h1>", unsafe_allow_html=True)
            with st.sidebar.form("urb_sidebar_form_ubc"):
                st.write("### Alter the following settings to customize the selected chart(s):")
                urb_percentage_slider_ubc = st.slider("**Adjust Percentage Range (Y-axis):**", min_value=0, max_value=100, value=100, step=20)
                urb_no_of_top_groups_per_user_slider_ubc = st.slider("**Adjust No Of Top Groups Per User (X-axis):**", min_value=0, max_value=10, value=10, step=1)
                urb_submit_parameters_button_ubc = st.form_submit_button("Apply Changes")

            with st.expander(f"**{chart_view_settings_title}**", expanded=True):
                urb_check_box_view_side_by_side_ubc = st.checkbox("Select to view charts side by side")

            with st.spinner(spinner_text):
                st.markdown("<h2 style='text-align: center;'>The Resource-Configuration Group Per User Charts</h2>", unsafe_allow_html=True)

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
                a =[0.6194840399786984, 0.7729370934866642, 0.8425218118648454, 0.8906973579175446, 0.9238788917664792, 0.9479635533005115, 0.9659413769736639, 0.9788211703158228, 0.9842781315514204, 0.987734831866639]
                b =[0.6918350088912488, 0.8533482445948762, 0.921081711512026, 0.9533918131448507, 0.9710197995695022, 0.9810033596267114, 0.9872495542508333, 0.9916599140171835, 0.9944420135092896, 0.9964546220465884]
                c =[0.28569096620357964, 0.4384045247520146, 0.545916628344075, 0.6263372405355048, 0.6897181499719287, 0.7429051624867624, 0.7877784887121456, 0.8257544812862695, 0.8583802658301265, 0.8858856158005057]
                d = [0.3412589175944932, 0.5253771632298813, 0.6401852895114848, 0.7268169396811582, 0.7918618794877094, 0.8394237557838181, 0.8733033543091736, 0.9005927265133411, 0.9214560290971314, 0.9370205635505027]
                e = [0.34546400428022095, 0.5045269734214348, 0.6100105002787073, 0.6842132563596424, 0.7400938395959942, 0.7848193747836403, 0.820948816948782, 0.8503210056286024, 0.8747755404127091, 0.8953885311573875]
                f = [0.7518056020202507, 0.8558925637313898, 0.899646125474168, 0.9293387269242847, 0.9477965077320011, 0.9625355964106591, 0.9713036041859342, 0.9774447097600966, 0.9824553576493343, 0.9861430093312263]
                g = [0.5456415087066954, 0.6540775322952421, 0.736053563391097, 0.7961297625424151, 0.8294600959018047, 0.8593692636586076, 0.8821934734113498, 0.9031370845055565, 0.9197546224720765, 0.9343792133174184] 
                
                colors = []
                x = []
                urb_chart_titles_ubc = []
                urb_x_axis_slice_end_value_ubc = urb_no_of_top_groups_per_user_slider_ubc

                for item in urb_charts_selected_list_ubc:
                    if "Blue Waters" == item:
                        x.append(a[:urb_x_axis_slice_end_value_ubc])
                        urb_chart_titles_ubc.append("Blue Waters")
                        colors.append('blue')
                    elif "Mira" == item:
                        x.append(b[:urb_x_axis_slice_end_value_ubc])
                        urb_chart_titles_ubc.append("Mira")
                        colors.append('orange')
                    elif "Philly" == item:
                        x.append(c[:urb_x_axis_slice_end_value_ubc])
                        urb_chart_titles_ubc.append("Philly")
                        colors.append('green')
                    elif "Helios" == item:
                        x.append(d[:urb_x_axis_slice_end_value_ubc]) 
                        urb_chart_titles_ubc.append("Helios")
                        colors.append('red')
                    elif "Super Cloud" == item:
                        x.append(e[:urb_x_axis_slice_end_value_ubc])
                        urb_chart_titles_ubc.append("Super Cloud")
                        colors.append('lightblue')
                    elif "Theta" == item:
                        x.append(f[:urb_x_axis_slice_end_value_ubc])
                        urb_chart_titles_ubc.append("Theta")
                        colors.append('grey')
                    elif "Theta GPU" == item:
                        x.append(g[:urb_x_axis_slice_end_value_ubc])
                        urb_chart_titles_ubc.append("Theta GPU")
                        colors.append('violet')
                    else:
                        pass

                if urb_check_box_view_side_by_side_ubc:
                    col1, col2 = st.columns(2)
                    for idx, item in enumerate(urb_charts_selected_list_ubc):
                        urb_col_logic_cal_ubc = col1 if idx % 2 == 0 else col2
                        if item == "Blue Waters":
                            with urb_col_logic_cal_ubc:
                                plot_123(a[:urb_x_axis_slice_end_value_ubc], colors[idx], "Blue Waters", urb_no_of_top_groups_per_user_slider_ubc, urb_percentage_slider_ubc)
                        elif item == "Mira":
                            with urb_col_logic_cal_ubc:
                                plot_123(b[:urb_x_axis_slice_end_value_ubc], colors[idx], "Mira", urb_no_of_top_groups_per_user_slider_ubc, urb_percentage_slider_ubc)
                        elif item == "Philly":
                            with urb_col_logic_cal_ubc:
                                plot_123(c[:urb_x_axis_slice_end_value_ubc], colors[idx], "Philly", urb_no_of_top_groups_per_user_slider_ubc, urb_percentage_slider_ubc)
                        elif item == "Helios":
                            with urb_col_logic_cal_ubc:
                                plot_123(d[:urb_x_axis_slice_end_value_ubc], colors[idx], "Helios", urb_no_of_top_groups_per_user_slider_ubc, urb_percentage_slider_ubc)
                        elif item == "Super Cloud":
                            with urb_col_logic_cal_ubc:
                                plot_123(e[:urb_x_axis_slice_end_value_ubc], colors[idx], "Super Cloud", urb_no_of_top_groups_per_user_slider_ubc, urb_percentage_slider_ubc)
                        elif item == "Theta":
                            with urb_col_logic_cal_ubc:
                                plot_123(f[:urb_x_axis_slice_end_value_ubc], colors[idx], "Theta", urb_no_of_top_groups_per_user_slider_ubc, urb_percentage_slider_ubc)
                        elif item == "Theta GPU":
                            with urb_col_logic_cal_ubc:
                                plot_123(g[:urb_x_axis_slice_end_value_ubc], colors[idx], "Theta GPU", urb_no_of_top_groups_per_user_slider_ubc, urb_percentage_slider_ubc)
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
        usb_chart_selection_left_col_options_ubc = ["Blue Waters", "Mira", "Super Cloud", "Theta"]
        usb_chart_selection_right_col_options_ubc = ["Philly", "Helios", "Theta GPU"]
        usb_chart_selection_options_ubc = usb_chart_selection_left_col_options_ubc + usb_chart_selection_right_col_options_ubc
        usb_charts_selected_list_ubc = usb_chart_selection_options_ubc.copy()
        usb_job_sizes_list_ubc = ["Minimal", "Small", "Middle", "Large"]
        usb_job_sizes_selected_list_ubc = usb_job_sizes_list_ubc.copy()
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
                        usb_charts_selected_list_ubc.remove(item)
            with col2:
                for item2 in usb_chart_selection_right_col_options_ubc:
                    usb_chart_selection_check_box_right_option_ubc = st.checkbox(item2, True)
                    if not usb_chart_selection_check_box_right_option_ubc:
                        usb_charts_selected_list_ubc.remove(item2)
                        
            usb_chart_selection_check_box_submission_button_ubc = st.form_submit_button("Load Charts")

            if usb_chart_selection_check_box_submission_button_ubc:
                if len(usb_charts_selected_list_ubc) >= 1:
                    st.write(f"**You Have Selected:** {usb_charts_selected_list_ubc}")
                else:
                    st.markdown("<h5 style='color: red'>Please select one or more charts options above and then click 'Load Charts'</h5>", unsafe_allow_html=True)
            else:
                pass
            
        if len(usb_charts_selected_list_ubc) >= 1:
            st.sidebar.markdown("<h1 style='text-align: center;'>Chart Customization Panel</h1>", unsafe_allow_html=True)
            
            with st.sidebar.form("usb_sidebar_form_ubc"):
                st.write("### Alter the following settings to customize the selected chart(s):")
                
                # Hide it for now - Future code to make the job status dynamic 
                # with st.expander("**Select Job Status(es)**", expanded=True):
                    # for item in usb_job_sizes_list_ubc:
                    #     usb_job_sizes_checkbox_ubc = st.checkbox(item, True)
                    #     if not usb_job_sizes_checkbox_ubc:
                    #         usb_job_sizes_selected_list_ubc.remove(item)
                    #     else:
                    #         pass
                        
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
            
            with st.spinner(spinner_text):
                def analyze_queue_and_user_behavior(data, gpu=False):
                    data["start_time"] = data["submit_time"] + data["wait_time"]
                    data["end_time"] = data["start_time"] + data["run_time"]
                    data["index"] = data.index
                    queue = []
                    util_time_user = defaultdict(list)
                    util_node_user = defaultdict(list)
                    cur_wait = 0
                    for row in data.itertuples(index=False):
                        while queue and queue[0][0] <= row.submit_time:
                            _, job_type, _, job = heappop(queue)
                            if job_type == "waiting":
                                heappush(queue, (job.start_time + job.run_time, "running", job.index, job))
                                cur_wait -= 1
                            elif job_type == "running":
                                pass
                            else:
                                raise NotImplementedError
                        heappush(queue, (row.start_time, "waiting", row.index, row))
                        util_time_user[row.user].append([row.run_time, cur_wait])
                        util_node_user[row.user].append([row.gpu_num if gpu else row.node_num, cur_wait])
                        cur_wait += 1
                    return util_time_user,util_node_user

                def plot_util_node(un, data, bars, user="per", chart_title="mira", selected_job_statuses=["Minimal"], frequency_value_slider=100, selected_job_sizes = ["short queue"], side_by_side=False):
                    if side_by_side:
                        st.markdown(f"<h4 style='text-align: center;'>{chart_title}</h4>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<h1 style='text-align: center;'>{chart_title}</h1>", unsafe_allow_html=True)   
                    hatches= ["-", ".", "x", "-"]
                    fig, axes = plt.subplots(1, 1, figsize=(3, 5))
                    all_un = []
                    for i in un:
                        all_un.extend(un[i])
                    max_wait = max(all_un, key=lambda x: x[1])[1]
                    stride = 1/3
                    labels_map = {0: "Short Queue", 1: "Middle Queue", 2: "Long Queue"}
                    x = []
                    for index in range(int(1/stride)):
                        label = labels_map.get(index)
                        if label in selected_job_sizes:
                            x.append("{}".format((1+index)*stride*max_wait))
                    
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
                    
                    for bar_index, bar in enumerate(bars):
                        selected_indices = [i for i in range(int(1/stride)) if labels_map[i] in selected_job_sizes]
                        y = np.array([sum(i[j] for j in i.keys() if prev_bar<j <=bar) for bucket_index, i in enumerate(buckets) if bucket_index in selected_indices])
                        axes.bar(x, y, bottom=prevy, hatch=hatches[bar_index], edgecolor="black")
                        prevy = y+prevy
                        prev_bar = bar
                        
                    axes.set_xticks(x)
                    axes.set_xticklabels([labels_map[i] for i in range(int(1/stride)) if labels_map[i] in selected_job_sizes], rotation=45)
                    axes.set_ylabel("Percentage (%)", fontsize=20)
                    axes.set_ylim(0, frequency_value_slider)
                    
                    if side_by_side:
                        axes.legend(selected_job_statuses, fontsize=6, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4)  
                    else:
                        axes.legend(selected_job_statuses, fontsize=12, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4) 
                                
                    st.pyplot(fig)

                @st.cache_data
                def system_user_time_node():
                    bw_util_time_user, bw_queue_node_user = analyze_queue_and_user_behavior(bw_df, gpu=False)
                    mira_util_time_user, mira_queue_node_user = analyze_queue_and_user_behavior(mira_df_2, gpu=False)
                    phi_util_time_user, phi_queue_node_user = analyze_queue_and_user_behavior(philly_df, gpu=True)
                    hl_util_time_user, hl_queue_node_user = analyze_queue_and_user_behavior(hl_df, gpu=True)
                    sc_util_time_user, sc_queue_node_user = analyze_queue_and_user_behavior(sc_df, gpu=False)
                    th_util_time_user, th_queue_node_user = analyze_queue_and_user_behavior(th_df, gpu=False)
                    th_gpu_util_time_user, th_gpu_queue_node_user = analyze_queue_and_user_behavior(th_gpu_df, gpu=True)
                    
                    return (
                        bw_util_time_user, mira_util_time_user, phi_util_time_user, hl_util_time_user, sc_util_time_user, th_util_time_user, th_gpu_util_time_user, 
                        bw_queue_node_user, mira_queue_node_user, phi_queue_node_user, hl_queue_node_user, sc_queue_node_user, th_queue_node_user, th_gpu_queue_node_user
                    )

              
                (
                    bw_util_time_user, mira_util_time_user, phi_util_time_user, hl_util_time_user, sc_util_time_user, th_util_time_user, th_gpu_util_time_user,
                    bw_queue_node_user, mira_queue_node_user, phi_queue_node_user, hl_queue_node_user, sc_queue_node_user, th_queue_node_user, th_gpu_queue_node_user
                ) = system_user_time_node()

                
                values_dict = {
                    "Minimal": (1, 512, 1, 1, 1, 1, 1),
                    "Small": (22636//10, 49152//10, 1, 1, 704//10, 4392//10, 1),
                    "Middle": (3*22636//10, 3*49152//10, 8, 8, 3*704//10, 3*4392//10, 8),
                    "Large": (1000000, 49152, 256, 256, 704, 4392, 256)
                }
                
                bw_bars, mira_bars, phi_bars, hl_bars, sc_bars, th_bars, th_gpu_bars = [], [], [], [], [], [], []
                # bw_rt_bars, mira_rt_bars, phi_rt_bars, hl_rt_bars, sc_rt_bars, th_rt_bars, th_gpu_rt_bars = [], [], [], [], [], [], []
                
                for item in usb_job_sizes_selected_list_ubc:
                    if item in values_dict:
                        bw, mira, phi, hl, sc, th, th_gpu = values_dict[item]
                        bw_bars.append(bw)
                        mira_bars.append(mira)
                        phi_bars.append(phi)
                        hl_bars.append(hl)
                        sc_bars.append(sc)
                        th_bars.append(th)
                        th_gpu_bars.append(th_gpu)
                    
                system_data = {
                    'Blue Waters': (bw_queue_node_user, bw_util_time_user, bw_df, bw_bars),
                    'Mira': (mira_queue_node_user, mira_util_time_user, mira_df_2, mira_bars),
                    'Philly': (phi_queue_node_user, phi_util_time_user, philly_df, phi_bars),
                    'Helios': (hl_queue_node_user, hl_util_time_user, hl_df, hl_bars),
                    'Super Cloud': (sc_queue_node_user, sc_util_time_user, sc_df, sc_bars),
                    'Theta': (th_queue_node_user, th_util_time_user, th_df, th_bars),
                    'Theta GPU': (th_gpu_queue_node_user, th_gpu_util_time_user, th_gpu_df, th_gpu_bars)
                }

                if len(usb_job_size_selected_list_ubc) >= 1:
                    # Job sizes
                    st.markdown("<h1 style='text-align: center;'>Submitted Jobs' Sizes Impacted By Queue Length Charts</h1>", unsafe_allow_html=True)
                    if usb_check_box_view_side_by_side_ubc:
                        col1, col2 = st.columns(2)
                        for idx, (system, (node_user, time_user, data, bars)) in enumerate(system_data.items()):
                            usb_col_logic_cal_ubc = col1 if idx % 2 == 0 else col2
                            if system in usb_charts_selected_list_ubc:
                                with usb_col_logic_cal_ubc:
                                    plot_util_node(node_user, data, bars, "all", system, usb_job_sizes_selected_list_ubc, usb_percentage_slider_ubc, usb_job_size_selected_list_ubc, True)
                    else:
                        for idx, (system, (node_user, time_user, data, bars)) in enumerate(system_data.items()):
                            if system in usb_charts_selected_list_ubc:
                                plot_util_node(node_user, data, bars, "all", system, usb_job_sizes_selected_list_ubc, usb_percentage_slider_ubc, usb_job_size_selected_list_ubc, False)   
                    
                    # Job Run time              
                    st.markdown("<h1 style='text-align: center;'>Submitted Jobs' Runtime Impacted By Queue Length Charts</h1>", unsafe_allow_html=True)  
                    runtime_bar = [60, 3600, 3600*24, 1000000000]
                    usb_job_runtime_selected_list_ubc = ["Minimal", "Short", "Middle", "Long"]
                    if usb_check_box_view_side_by_side_ubc:
                        col1, col2 = st.columns(2)
                        for idx, (system, (node_user, time_user, data, bars)) in enumerate(system_data.items()):
                            usb_col_logic_cal_ubc = col1 if idx % 2 == 0 else col2
                            if system in usb_charts_selected_list_ubc:
                                with usb_col_logic_cal_ubc:
                                    plot_util_node(time_user, data, runtime_bar, "all", system, usb_job_runtime_selected_list_ubc, usb_percentage_slider_ubc, usb_job_size_selected_list_ubc, True)
                    else:
                        for idx, (system, (node_user, time_user, data, bars)) in enumerate(system_data.items()):
                            if system in usb_charts_selected_list_ubc:
                                plot_util_node(time_user, data, runtime_bar, "all", system, usb_job_runtime_selected_list_ubc, usb_percentage_slider_ubc, usb_job_size_selected_list_ubc, False)    
                    
                       
                    with st.expander(f"**{chart_description_expander_title}**", expanded=True):
                        st.write("""
                        **Submitted Jobs' Sizes Impacted By Queue Length:** This chart provides a visual analysis of how the sizes of submitted jobs correlate with the length of the job queue. It seeks to understand whether the queue length (how many jobs are waiting) has any bearing on the sizes of jobs that get submitted. For instance, when the queue is long, are smaller or larger jobs predominantly submitted? Understanding this relationship can have implications for resource allocation, scheduling strategies, and overall system efficiency. Such insights can also inform decision-makers about optimizing queue management techniques.
                        """)
                else:
                    st.markdown("<h2 style='color: red'>Please select one or more job size(s) from the sidebar to plot the chart(s)</h2>", unsafe_allow_html=True)
                    
    elif ubc_nav_bar == "Correlation between Job Run Time and Job Statuses":
        cbjrtajs_chart_selection_left_col_options_ubc = ["Blue Waters", "Mira", "Super Cloud", "Theta"]
        cbjrtajs_chart_selection_right_col_options_ubc = ["Philly", "Helios", "Theta GPU"]
        cbjrtajs_chart_selection_options_ubc = cbjrtajs_chart_selection_left_col_options_ubc + cbjrtajs_chart_selection_right_col_options_ubc
        cbjrtajs_charts_selected_list_ubc = cbjrtajs_chart_selection_options_ubc.copy()
        cbjrtajs_job_status_list_ubc = ["Pass", "Failed", "Killed"]
        cbjrtajs_job_status_selected_list_ubc = cbjrtajs_job_status_list_ubc.copy() 
        cbjrtajs_min_value_exp_run_time_slider_ubc = -1
        cbjrtajs_max_value_exp_run_time_slider_ubc = 6
    
        with st.form("cbjrtajs_chart_selection_form_ubc"):
            st.write(f"### **{chart_selection_form_title}**")
            st.write(f'**{chart_selection_form_load_charts_text}**')
            col1, col2 = st.columns(2)
            with col1 :
                for item in cbjrtajs_chart_selection_left_col_options_ubc:
                    cbjrtajs_chart_selection_check_box_left_option_ubc = st.checkbox(item, True)
                    if not cbjrtajs_chart_selection_check_box_left_option_ubc:
                        cbjrtajs_charts_selected_list_ubc.remove(item)
            with col2:
                for item2 in cbjrtajs_chart_selection_right_col_options_ubc:
                    cbjrtajs_chart_selection_check_box_right_option_ubc = st.checkbox(item2, True)
                    if not cbjrtajs_chart_selection_check_box_right_option_ubc:
                        cbjrtajs_charts_selected_list_ubc.remove(item2)
            cbjrtajs_chart_selection_check_box_submission_button_ubc = st.form_submit_button("Load Charts")

            if cbjrtajs_chart_selection_check_box_submission_button_ubc:
                if len(cbjrtajs_charts_selected_list_ubc) >= 1:
                    st.write(f"**You Have Selected:** {cbjrtajs_charts_selected_list_ubc}")
                else:
                    st.markdown("<h5 style='color: red'>Please select one or more charts options above and then click 'Load Charts'</h5>", unsafe_allow_html=True)
            else:
                pass
            
        if len(cbjrtajs_charts_selected_list_ubc) >= 1:
            st.sidebar.markdown("<h1 style='text-align: center;'>Chart Customization Panel</h1>", unsafe_allow_html=True)
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
                
            def plot_attribute_per_ml(u, data, state="state", status=None, frequency_slider_value=None, all_user=False, side_by_side = False, chart_title=None):
                plt.style.use("default")
                rows = list(data.groupby(u).count().sort_values(by="job", ascending=False).index[:3])
                # rows = list(data.groupby(u).sum().sort_values(by="node_hour", ascending=False).index[:3])
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
                    sns.violinplot(data=k,ax=axes, scale="width")
                ax = axes
                
                ymin = 1 if frequency_slider_value < 10 else 10 * (int(np.log10(frequency_slider_value)) - 1)
                ymax = frequency_slider_value
                ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
                tick_range = np.arange(np.floor(np.log10(ymin)), np.ceil(np.log10(ymax)) + 1)
                ax.yaxis.set_ticks(tick_range)
                ax.yaxis.set_ticks([np.log10(x) for p in tick_range for x in np.linspace(10 ** p, 10 ** (p + 1), 10)], minor=True)
                ax.yaxis.grid(True)     
                ax.set_xticks([y for y in range(len(status))])
                ax.set_xticklabels(status, fontsize=24)      
                ax.set_ylabel('Job Run time (s)', fontsize=20)
                st.pyplot(fig)
            
            with st.spinner(spinner_text):
                if len(cbjrtajs_job_status_selected_list_ubc) >= 1:
                    st.markdown("<h1 style='text-align: center;'>The Median Runtime Of Different Types Of Jobs Charts</h1>", unsafe_allow_html=True)
                    if cbjrtajs_check_box_view_side_by_side_ubc:
                        col1, col2 = st.columns(2)
                        for idx, item in enumerate(cbjrtajs_charts_selected_list_ubc):
                            cbjrtajs_col_logic_cal_ubc = col1 if idx % 2 == 0 else col2
                            if item == "Blue Waters":
                                with cbjrtajs_col_logic_cal_ubc:
                                    plot_attribute_per_ml("user", data=bw_df, state="new_status", status=cbjrtajs_job_status_selected_list_ubc, frequency_slider_value = cbjrtajs_percentage_slider_value_ubc, all_user=True, side_by_side = True, chart_title = "Blue Waters")
                            elif item == "Mira":
                                with cbjrtajs_col_logic_cal_ubc:
                                    plot_attribute_per_ml("user", data=mira_df_2, state="new_status", status=cbjrtajs_job_status_selected_list_ubc, frequency_slider_value=  cbjrtajs_percentage_slider_value_ubc, all_user=True, side_by_side = True, chart_title = "Mira")
                            elif item == "Philly":
                                with cbjrtajs_col_logic_cal_ubc:
                                    plot_attribute_per_ml("user", data=philly_df, state="state", status=cbjrtajs_job_status_selected_list_ubc, frequency_slider_value=cbjrtajs_percentage_slider_value_ubc, all_user=True, side_by_side = True, chart_title = "Philly")
                            elif item == "Helios":
                                with cbjrtajs_col_logic_cal_ubc:
                                    plot_attribute_per_ml("user", data=hl_df, state="state", status=cbjrtajs_job_status_selected_list_ubc, frequency_slider_value=cbjrtajs_percentage_slider_value_ubc, all_user=True, side_by_side = True, chart_title = "Helios")
                            elif item == "Super Cloud":
                                with cbjrtajs_col_logic_cal_ubc:
                                    plot_attribute_per_ml("user", data=sc_df, state="new_status", status=cbjrtajs_job_status_selected_list_ubc, frequency_slider_value=cbjrtajs_percentage_slider_value_ubc, all_user=True, side_by_side = True, chart_title = "Super Cloud")
                            elif item == "Theta":
                                with cbjrtajs_col_logic_cal_ubc:
                                    plot_attribute_per_ml("user", data=th_df, state="new_status", status=cbjrtajs_job_status_selected_list_ubc, frequency_slider_value=cbjrtajs_percentage_slider_value_ubc, all_user=True, side_by_side = True, chart_title = "Theta")
                            elif item == "Theta GPU":
                                with cbjrtajs_col_logic_cal_ubc:
                                    plot_attribute_per_ml("user", data=th_gpu_df, state="new_status", status=cbjrtajs_job_status_selected_list_ubc, frequency_slider_value=cbjrtajs_percentage_slider_value_ubc, all_user=True, side_by_side = True, chart_title = "Theta GPU")
                            else:
                                pass                           
                    else:
                        for item in cbjrtajs_charts_selected_list_ubc:
                            if item == "Blue Waters":
                                plot_attribute_per_ml("user", data=bw_df, state="new_status", status=cbjrtajs_job_status_selected_list_ubc, frequency_slider_value=cbjrtajs_percentage_slider_value_ubc, all_user=True, side_by_side = False, chart_title = "Blue Waters")                          
                            elif item == "Mira":
                                plot_attribute_per_ml("user", data=mira_df_2, state="new_status", status=cbjrtajs_job_status_selected_list_ubc, frequency_slider_value=cbjrtajs_percentage_slider_value_ubc, all_user=True, side_by_side = False, chart_title = "Mira")                        
                            elif item == "Philly":
                                plot_attribute_per_ml("user", data=philly_df, state="state", status=cbjrtajs_job_status_selected_list_ubc, frequency_slider_value=cbjrtajs_percentage_slider_value_ubc,  all_user=True, side_by_side = False, chart_title = "Philly")                      
                            elif item == "Helios":
                                plot_attribute_per_ml("user", data=hl_df, state="state", status=cbjrtajs_job_status_selected_list_ubc, frequency_slider_value=cbjrtajs_percentage_slider_value_ubc, all_user=True, side_by_side = False, chart_title = "Helios")
                            elif item == "Super Cloud":
                                plot_attribute_per_ml("user", data=sc_df, state="new_status", status=cbjrtajs_job_status_selected_list_ubc, frequency_slider_value=cbjrtajs_percentage_slider_value_ubc, all_user=True, side_by_side = False, chart_title = "Super Cloud")
                            elif item == "Theta":
                                plot_attribute_per_ml("user", data=th_df, state="new_status", status=cbjrtajs_job_status_selected_list_ubc, frequency_slider_value=cbjrtajs_percentage_slider_value_ubc, all_user=True, side_by_side = False, chart_title = "Theta")
                            elif item == "Theta GPU":
                                plot_attribute_per_ml("user", data=th_gpu_df, state="new_status", status=cbjrtajs_job_status_selected_list_ubc, frequency_slider_value=cbjrtajs_percentage_slider_value_ubc, all_user=True, side_by_side = False, chart_title = "Theta GPU")
                            else:
                                pass
                    with st.expander(f"**{chart_description_expander_title}**", expanded=True):
                        st.write("""
                        **The Median Runtime Of Different Types Of Jobs Charts:** This chart presents the median runtime for various categories of jobs. By focusing on the median, the chart offers a central tendency of runtime, minimizing the impact of outliers or extreme values. Different types of jobs, based on their complexity or resource requirements, might have varying runtimes. Visualizing the median runtime can provide insights into the average expected execution duration for each job type, helping in resource allocation, scheduling, and understanding system efficiency for diverse tasks.
                        """)
                else:
                    st.markdown("<h2 style='color: red'>Please select one or more job status(es) from the sidebar to plot the chart(s)</h2>", unsafe_allow_html=True)
    else:
        pass
else:
    pass