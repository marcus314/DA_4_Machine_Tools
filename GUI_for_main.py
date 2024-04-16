import main 
import os
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import ttk
#from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from PIL import Image, ImageTk
from datetime import datetime
import csv
import pandas as pd
from sklearn.metrics import mean_squared_error
import shutil

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DT = 0.02
CUTOFF = 4#17
current_directory = os.path.dirname(os.path.abspath(__file__))
DO_SAVE_LOG = False
ITERATIONS = 1
plot_frame = None  # Define your plot_frame where you want the plot to be displayed
canvas = None  # For storing the FigureCanvasTkAgg object
current_toolbar = None


#def browse_file():
#    filepath = filedialog.askopenfilename()
#    filepath_entry.delete(0, tk.END)
#    filepath_entry.insert(0, filepath)


class ToolTip(object):
    def __init__(self, widget, text='widget info'):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

    def showtip(self):
        "Display text in tooltip window"
        self.x, self.y, _, _ = self.widget.bbox("insert")
        self.x += self.widget.winfo_rootx() + 25
        self.y += self.widget.winfo_rooty() + 25
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry("+%d+%d" % (self.x, self.y))
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                      background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                      font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

def createToolTip(widget, text):
    toolTip = ToolTip(widget, text)
    def enter(event):
        toolTip.showtip()
    def leave(event):
        toolTip.hidetip()
    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)

def get_iterations():
    try:
        return int(iterations_entry.get())
    except ValueError:
        return ITERATIONS  # Return default value if input is not a valid integer

def train_network():
    iterations = get_iterations()
    additional_descriptor = "GUI_TEST_"
    match this_versuch.get():
        case "Versuch 1 CMX Prozess": data_params_test = main.data_Versuch_1_CMX_prozess()
        case "Versuch 2 CMX Prozess": data_params_test = main.data_Versuch_2_CMX_prozess()
        case "Versuch 3 CMX Prozess": data_params_test = main.data_Versuch_3_CMX_prozess()
        case "Versuch 4 CMX Prozess": data_params_test = main.data_Versuch_4_CMX_prozess()
        case "Versuch 1 CMX Aircut": data_params_test = main.data_Versuch_1_CMX_aircut()
        case "Versuch 2 CMX Aircut": data_params_test = main.data_Versuch_2_CMX_aircut()
        case "Versuch 3 CMX Aircut": data_params_test = main.data_Versuch_3_CMX_aircut()
        case "Versuch 4 CMX Aircut": data_params_test = main.data_Versuch_4_CMX_aircut()
        case "Versuch 1 I40 Prozess": data_params_test = main.data_Versuch_1_I40_prozess()
        case "Versuch 2 I40 Prozess": data_params_test = main.data_Versuch_2_I40_prozess()
        case "Versuch 3 I40 Prozess": data_params_test = main.data_Versuch_3_I40_prozess()
        case "Versuch 4 I40 Prozess": data_params_test = main.data_Versuch_4_I40_prozess()
        case "Versuch 1 I40 Aircut": data_params_test = main.data_Versuch_1_I40_aircut()
        case "Versuch 2 I40 Aircut": data_params_test = main.data_Versuch_2_I40_aircut()
        case "Versuch 3 I40 Aircut": data_params_test = main.data_Versuch_3_I40_aircut()
        case "Versuch 4 I40 Aircut": data_params_test = main.data_Versuch_4_I40_aircut()

    match this_current.get():
        case "Current x": data_params_test.target_channels = ["cur_x"]
        case "Current y": data_params_test.target_channels = ["cur_y"]
        case "Current z": data_params_test.target_channels = ["cur_z"]
        case "Current sp": data_params_test.target_channels = ["cur_sp"]
    
    match this_model.get():
        case "NN_Cheap": ML_params_test = main.NN_Cheap()
        case "NN_Normal": ML_params_test = main.NN_Normal()
        case "RF_Cheap": ML_params_test = main.RF_Cheap()
        case "RF_Normal": ML_params_test = main.RF_Normal()
    #self.aug1arg2 = 3 #Method of dataset compilation. 1 = winner takes all, 2 = linear for all > 0, 3 = quadratic for all > 0
    match this_augmentation_method.get():
        case "Augment by Score (linear)": 
            aug_params_test = main.All_augments_by_score()
            aug_params_test.aug1arg2 = 2
        case "Augment by Score (best)":
            aug_params_test = main.All_augments_by_score()
            aug_params_test.aug1arg2 = 1
        case "Augment by Score (quadratic)":
            aug_params_test = main.All_augments_by_score()
            aug_params_test.aug1arg2 = 3
        case "Window Warping": aug_params_test = main.WindowWarp()
        case "Time Warping": aug_params_test = main.TimeWarp()
        case "Magnitude Warping": aug_params_test = main.MagnitudeWarp()
        case "Random Delete": aug_params_test = main.RandomDelete()
        case "Noise": aug_params_test = main.Noise()
        case "No Augment": aug_params_test = main.NoAugment()

    if "Augment by Score" in this_augmentation_method.get():
        current_directory = os.path.dirname(os.path.abspath(__file__))
        
        if "CMX Prozess" in this_versuch.get():
            scorefolder = "scores_CMX_process"
        if "I40 Prozess" in this_versuch.get():
            scorefolder = "scores_I40_process"
        if "CMX Aircut" in this_versuch.get():
            scorefolder = "scores_CMX_aircut"
        if "I40 Aircut" in this_versuch.get():
            scorefolder = "scores_I40_aircut"

        scorefolder = scorefolder+"_DEV"

        scores_source_dir = os.path.join(current_directory, "Auswertung", "bewertung_der_methoden", scorefolder)
        scores_target_dir = os.path.join(current_directory, "Auswertung", "bewertung_der_methoden")

        for item in os.listdir(scores_target_dir):
            # Construct the full path of the item
            item_path = os.path.join(scores_target_dir, item)
            # Check if the item is a file and ends with .csv
            if os.path.isfile(item_path) and item_path.endswith('.csv'):
                try:
                    os.remove(item_path)
                except OSError as e:
                    print(f"Error removing {item_path}: {e.strerror}")

        for item in os.listdir(scores_source_dir):
            if item.endswith('.csv'):
                source_path = os.path.join(scores_source_dir, item)
                target_path = os.path.join(scores_target_dir, item)
                shutil.copy2(source_path, target_path)


    global result_original_DEV
    global result_augmented_DEV
    global result_original_MSE
    global result_augmented_MSE
    global original_predicitions
    global original_original_values
    global augmented_predictions

    result_original_DEV,result_original_MSE, original_predicitions, original_original_values = main.run_testfunctions(data_params_test, main.NoAugment(), ML_params_test, additional_descriptor, iterations = iterations, augment_before_va = True, do_save_log = DO_SAVE_LOG, do_plot = False, plot_average_predictions = False)
    result_augmented_DEV, result_augmented_MSE, augmented_predictions, augmented_original_values = main.run_testfunctions(data_params_test, aug_params_test, ML_params_test, additional_descriptor, iterations = iterations, augment_before_va = True, do_save_log = DO_SAVE_LOG, do_plot = False, plot_average_predictions = False) 
    print(f"Result original DEV: {result_original_DEV}")
    print(f"Result augmented DEV: {result_augmented_DEV}")
    print(f"Result original MSE: {result_original_MSE}")
    print(f"Result augmented MSE: {result_augmented_MSE}")
    plot_current_graph()
    update_file_list()


def plot_current_graph():
    global canvas, original_predicitions, original_original_values, augmented_predictions, current_toolbar

    # Clear previous canvas if exists
    if canvas:
        canvas.get_tk_widget().destroy()
    if current_toolbar:
        current_toolbar.destroy()

    # Create the figure with 3 subplots (2 for bar charts, 1 for line plot)
    fig = plt.figure(figsize=(13, 5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 3], width_ratios=[1, 1])

    # Define subplots
    ax1 = fig.add_subplot(gs[0, 0])  # First bar chart
    ax2 = fig.add_subplot(gs[0, 1])  # Second bar chart
    ax3 = fig.add_subplot(gs[1, :])  # Line plot (spanning both columns)

    # Data for bar charts
    x_pos = np.arange(2)
    labels = ['Original data only', 'With synthetic data']
    val1, err1 = result_original_DEV
    val2, err2 = result_augmented_DEV
    val1_MSE, err1_MSE = result_original_MSE
    val2_MSE, err2_MSE = result_augmented_MSE
    values_DEV = [val1, val2]
    errors_DEV = [err1, err2]
    values_MSE = [val1_MSE, val2_MSE]
    errors_MSE = [err1_MSE, err2_MSE]
    bar_colors = ['#10408B', '#8CB63C']

    val1_MSE = mean_squared_error(original_original_values, original_predicitions)
    val2_MSE = mean_squared_error(original_original_values, augmented_predictions)
    values_MSE = [val1_MSE, val2_MSE]


    result_Q_meas = sum(original_original_values)
    result_Q_est = sum(original_predicitions)
    result_Q_est_aug = sum(augmented_predictions)
    result_deviation_percentage = abs(round(((result_Q_est-result_Q_meas)/result_Q_meas)*100,4))
    result_deviation_percentage_aug = abs(round(((result_Q_est_aug-result_Q_meas)/result_Q_meas)*100,4))
    values_DEV = [result_deviation_percentage, result_deviation_percentage_aug]

    # First bar chart
    #ax1.bar(x_pos, values_DEV, yerr=errors_DEV, align='center', alpha=0.5, ecolor='black', capsize=10, color=bar_colors)
    ax1.bar(x_pos, values_DEV, align='center', alpha=0.5, ecolor='black', capsize=10, color=bar_colors)
    ax1.set_ylabel('% Deviation')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels)
    #ax1.set_title('First Bar Plot')

    # Second bar chart (you can customize this as needed)
    #ax2.bar(x_pos, values_MSE, yerr=errors_MSE, align='center', alpha=0.5, ecolor='black', capsize=10, color=bar_colors)
    ax2.bar(x_pos, values_MSE, align='center', alpha=0.5, ecolor='black', capsize=10, color=bar_colors)
    #ax2.set_ylabel('MSE')
    ax2.set_ylabel('MSE', labelpad=10)
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels)
    #ax2.set_title('Second Bar Plot')
    for ax, values in zip([ax1, ax2], [values_DEV, values_MSE]):
        # Determine the bottom margin space based on the figure size and dpi to place text appropriately.
        bottom_margin = ax.get_ylim()[0] * 0.1  # Adjust this to control the space above the x-axis labels.
        for i, value in enumerate(values):
            # Place text just above the x-axis labels.
            ax.text(i, bottom_margin, f'{value:.4f}', ha='center', va='bottom')

    # Line plot
    num_samples = len(original_predicitions)
    time_interval = 1 / 50  # 50 Hz sampling rate
    time_values = np.arange(0, num_samples * time_interval, time_interval)
    ax3.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax3.plot(time_values, original_original_values, color='#DF9B1B', label='Measured Values')
    ax3.plot(time_values, original_predicitions, color='#10408B', label='Prediction using only real data')
    ax3.plot(time_values, augmented_predictions, color='#8CB63C', label='Prediction using chosen augmentation method')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Current in A')
    ax3.legend()

    # Embed the plot in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()
    current_toolbar = NavigationToolbar2Tk(canvas, plot_frame)
    current_toolbar.update()



def plot_current_graph_OLD():
    global canvas, original_predicitions, original_original_values, augmented_predictions, current_toolbar

    # Clear previous canvas if exists
    if canvas:
        canvas.get_tk_widget().destroy()
    if current_toolbar:
        current_toolbar.destroy()

    # Create the figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, figsize=(13, 5),gridspec_kw={'height_ratios': [1,3]}) #16 statt 10

    #fig = plt.figure(figsize=(13, 5))
    #gs = fig.add_gridspec(2, 2, height_ratios=[1, 3], width_ratios=[1, 1])

    #ax1 = fig.add_subplot(gs[0, 0])  # First bar chart
    #ax2 = fig.add_subplot(gs[0, 1])  # Second bar chart
    #ax3 = fig.add_subplot(gs[1, :])  # Line plot (spanning both columns)

    # First subplot for the bar chart
    x_pos = np.arange(2)
    labels = ['Original data only', 'With synthetic data']
    #values = [np.mean(original_original_values), np.mean(augmented_predictions)]
    #errors = [np.std(original_original_values), np.std(augmented_predictions)]
    val1, err1 = result_original_DEV
    val2, err2 = result_augmented_DEV
    values = [val1, val2]
    errors = [err1, err2]

    bar_colors = ['#10408B', '#8CB63C']
    ax1.bar(x_pos, values, yerr=errors, align='center', alpha=0.5, ecolor='black', capsize=10,color=bar_colors)
    ax1.set_ylabel('Deviation in %')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels)
    ax1.set_title('Bar plot with error bars')

    # Second subplot for the line plot
    num_samples = len(original_predicitions)
    time_interval = 1 / 50  # 50 Hz sampling rate
    time_values = np.arange(0, num_samples * time_interval, time_interval)
    #x_values = np.arange(len(original_predicitions))
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.plot(time_values, original_original_values, color = '#DF9B1B', label='Measured Values')
    ax2.plot(time_values, original_predicitions, color = '#10408B', label='Prediction using only real data')
    ax2.plot(time_values, augmented_predictions, color = '#8CB63C', label='Prediction using synthetic data')
    #ax2.set_title('Line Plot')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Current in A')
    ax2.legend()

    # Embed the plot in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()

    ####tk_canvas.create_window(0, 0, window=canvas.get_tk_widget(), anchor=tk.NW)
    canvas.get_tk_widget().pack()
    current_toolbar = NavigationToolbar2Tk(canvas, plot_frame)
    current_toolbar.update()
    canvas.get_tk_widget().pack()

def update_file_list():
    """Update the list of CSV files in the saved_graphs directory."""
    file_list.delete(0, tk.END)  # Clear current list
    saved_graphs_path = os.path.join(current_directory, "saved_graphs")
    for filename in os.listdir(saved_graphs_path):
        if filename.endswith(".csv"):
            file_list.insert(tk.END, filename)

def load_selected_file():
    """Load the selected file into a NumPy array."""
    selected_file = file_list.get(tk.ANCHOR)
    if selected_file:
        file_path = os.path.join(current_directory, "saved_graphs", selected_file)
        global read_data
        read_data = pd.read_csv(file_path).values
        global original_predicitions, original_original_values, augmented_predictions
        original_predicitions = read_data[:, 0]
        original_original_values = read_data[:, 1]
        augmented_predictions = read_data[:, 2]
        plot_current_graph()
        print("Data loaded from:", selected_file)

def save_current_graph():
    global original_predicitions, original_original_values, augmented_predictions
    print(f"original_predicitions {original_predicitions.shape[0]}")
    if original_predicitions.shape[0] == 3:
        print(f"Empty, not saving")
        return
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(current_directory, "saved_graphs", f"{timestamp}_{this_versuch.get()}_{this_current.get()}_{this_model.get()}_{this_augmentation_method.get()}_{int(iterations_entry.get())}_iterations.csv")
    combined_data = np.column_stack((original_predicitions, original_original_values, augmented_predictions))
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Original Predictions', 'Original Values', 'Augmented Predictions']) # Header
        writer.writerows(combined_data)
    print(f"Graph data saved as {filename}")
    update_file_list()


def on_closing():
    plt.close('all')
    root.destroy()

root = tk.Tk()
root.protocol("WM_DELETE_WINDOW", on_closing)
root.tk.call('tk', 'scaling', 1.5)
root.geometry("1500x800")
root.title("Synthetic Data Generator")

# Initialize the DoubleVar after the Tkinter root window is created
focus_slider_value = tk.DoubleVar()
speed_slider_value = tk.DoubleVar()
this_versuch = tk.StringVar()
this_current = tk.StringVar()
this_model = tk.StringVar()
this_augmentation_method = tk.StringVar()
iterations_var = tk.IntVar(value=ITERATIONS)
result_original_DEV = [0,0]
result_augmented_DEV = [0,0]
result_original_MSE = [0,0]
result_augmented_MSE = [0,0]
original_predicitions = np.array([0,0,0])
original_original_values = np.array([0,0,0])
augmented_predictions = np.array([0,0,0])
list_versuche = ["Versuch 1 CMX Prozess", "Versuch 2 CMX Prozess", "Versuch 3 CMX Prozess", "Versuch 4 CMX Prozess",
                 "Versuch 1 I40 Prozess", "Versuch 2 I40 Prozess", "Versuch 3 I40 Prozess", "Versuch 4 I40 Prozess",
                 "Versuch 1 CMX Aircut", "Versuch 2 CMX Aircut", "Versuch 3 CMX Aircut", "Versuch 4 CMX Aircut",
                 "Versuch 1 I40 Aircut", "Versuch 2 I40 Aircut", "Versuch 3 I40 Aircut", "Versuch 4 I40 Aircut"]
list_currents = ["Current x", "Current y", "Current z", "Current sp"]
list_models = ["RF_Cheap", "RF_Normal", "NN_Cheap", "NN_Normal"]
list_augmentation_methods = ["No Augment", "Augment by Score (linear)","Augment by Score (quadratic)", "Augment by Score (best)", "Window Warping", "Time Warping","Magnitude Warping", "Random Delete", "Noise"]

main_frame = ttk.Frame(root, padding="1") #10 statt 1
main_frame.grid(row=0, column=0, padx=2, pady=20, sticky=(tk.W, tk.E, tk.N, tk.S)) #20 statt 2

current_row = 0
#ttk.Label(main_frame, text="Maschinen-Constraints aus folgender Configurationsdatei lesen: ()").grid(row=current_row, column=0, sticky=(tk.E, tk.W))
#current_row +=1

#filepath_entry = ttk.Entry(main_frame, width=50)
#filepath_entry.grid(row=current_row, column=0, sticky=tk.W)
#browse_button = ttk.Button(main_frame, text="Browse", command=browse_file)
#browse_button.grid(row=current_row, column=1, sticky=tk.W)
#current_row +=1




ttk.Label(main_frame, text="    ").grid(row=current_row, column=2, sticky=(tk.W)) #sets a gap betwenn the columns
ttk.Label(main_frame, text="    ").grid(row=current_row, column=4, sticky=(tk.W)) #sets a gap betwenn the columns
ttk.Label(main_frame, text="    ").grid(row=current_row, column=6, sticky=(tk.W)) #sets a gap betwenn the columns

drop_menu = ttk.OptionMenu(main_frame, this_versuch, list_versuche[0], *list_versuche)
drop_menu.grid(row=current_row, column=1, sticky=(tk.W, tk.E))
ttk.Label(main_frame, text="Daten wählen:").grid(row=current_row, column=0, sticky=(tk.W))

current_row += 1

drop_menu = ttk.OptionMenu(main_frame, this_current, list_currents[0], *list_currents)
drop_menu.grid(row=current_row, column=1, sticky=(tk.W, tk.E))
ttk.Label(main_frame, text="Strom wählen:").grid(row=current_row, column=0, sticky=(tk.W))
current_row += 1

drop_menu = ttk.OptionMenu(main_frame, this_model, list_models[0], *list_models)
drop_menu.grid(row=current_row, column=1, sticky=(tk.W, tk.E))
ttk.Label(main_frame, text="ML-Modell wählen:").grid(row=current_row, column=0, sticky=(tk.W))
mlmodel_tooltip = ttk.Label(main_frame, text="ML-Modell wählen:")
mlmodel_tooltip.grid(row=current_row, column=0, sticky=(tk.W))
createToolTip(mlmodel_tooltip, "Achtung: NN/RF_Normal dauern auf Laptop sehr lange zu trainieren")
current_row += 1

drop_menu = ttk.OptionMenu(main_frame, this_augmentation_method, list_augmentation_methods[0], *list_augmentation_methods)
drop_menu.grid(row=current_row, column=1, sticky=(tk.W, tk.E))
ttk.Label(main_frame, text="Augmentierungs Methode wählen:").grid(row=current_row, column=0, sticky=(tk.W))
augmentation_label = ttk.Label(main_frame, text="Augmentierungs Methode wählen:")
augmentation_label.grid(row=current_row, column=0, sticky=(tk.W))
createToolTip(augmentation_label, "This is a tooltip")

current_row += 1

ttk.Label(main_frame, text="Set iterations:").grid(row=current_row, column=0, sticky=(tk.W))
iterations_tooltip = ttk.Label(main_frame, text="Set iterations:")
iterations_tooltip.grid(row=current_row, column=0, sticky=(tk.W))
createToolTip(iterations_tooltip, "Setzt, wie viele Modelle trainiert werden.\nVon den schätzungen dieser wird der Mittelwert gebildet")
iterations_entry = ttk.Entry(main_frame, textvariable=iterations_var)
iterations_entry.grid(row=current_row, column=1, sticky=(tk.W, tk.E))
current_row += 1

# Create a new frame specifically for the slider and labels

#focus_slider_frame = ttk.Frame(main_frame, padding="10", relief="solid", borderwidth=1)
#focus_slider_frame.grid(row=current_row, columnspan=2, sticky=(tk.W, tk.E))
#current_row += 1

current_row = 0
action_button = ttk.Button(main_frame, text="Train Model", command=train_network)
action_button.grid(row=current_row, column=3, sticky=(tk.W, tk.E))
current_row += 1


plot_current_button = ttk.Button(main_frame, text="Save Graph", command=save_current_graph)
plot_current_button.grid(row=current_row, column=3, sticky=(tk.W, tk.E))
current_row += 1



current_directory = os.path.dirname(os.path.abspath(__file__))
#root.iconbitmap(os.path.join(current_directory, "needed_for_GUI", "logo_kit_small.ico"))
#root.update()
icon = Image.open(os.path.join(current_directory, "needed_for_GUI", "logo_kit.ico"))
icon = ImageTk.PhotoImage(icon)
 
# Set the taskbar icon
root.iconphoto(True, icon)


wbk_logo = Image.open(os.path.join(current_directory, "needed_for_GUI", "logo_wbk.png"))
wbk_logoTk = ImageTk.PhotoImage(wbk_logo)
wbk_logo_label = ttk.Label(root, image = wbk_logoTk)
wbk_logo_label.image = wbk_logoTk
wbk_logo_label.grid(row = 0, column = 10,rowspan=5, sticky=(tk.W))

#plot_frame = ttk.Frame(root, width=1000, height=500) #, padding="10"
plot_frame = ttk.Frame(root, height=500) #, padding="10"
plot_frame.grid(row=7, column=0, columnspan= 11,rowspan = 6, sticky=(tk.W, tk.E))#statt, tk.N, tk.S))

def add_scrollbar_to_file_list():
    scrollbar = tk.Scrollbar(main_frame, orient="vertical")
    scrollbar.grid(row=0, column=6, rowspan=4, sticky=(tk.N, tk.S))
    file_list.configure(yscrollcommand=scrollbar.set)
    scrollbar.configure(command=file_list.yview)

file_list = tk.Listbox(main_frame, height=5, width=85)
file_list.grid(row=0, column=5, rowspan=5, sticky=(tk.N)) #sticky=(tk.N, tk.S, tk.E, tk.W)
load_button = ttk.Button(main_frame, text="Load Selected File", command=load_selected_file)
load_button.grid(row=4, column=5)
update_file_list()
add_scrollbar_to_file_list()

root.mainloop()

