import os
import csv
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from processor import Processor

class EndmemberAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Endmember Analyzer")

        # Create variables to store image and pixel values
        self.image_path = ""
        self.start_x, self.start_y = 0, 0
        self.rect_id = None
        self.p = Processor()
        self.img = None
        self.existing_data = {}
        # Create widgets
        self.create_widgets()

    def create_widgets(self):
        # Create Image Canvas
        self.canvas = tk.Canvas(self.root)
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)

        # Create Plot Canvas
        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        # Load Image Button
        load_button = tk.Button(self.root, text="Load Image", command=self.load_image)
        load_button.pack(side=tk.TOP)

        # Save Endmember Button
        save_button = tk.Button(self.root, text="Save Endmember", command=self.save_endmember)
        save_button.pack(side=tk.TOP)

        # Display RGB Values
        self.rgb_label = tk.Label(self.plot_frame, text="Average RGB Values:")
        self.rgb_label.pack()

        # Create matplotlib figure and canvas
        self.fig, self.ax = plt.subplots()
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_plot_widget = self.canvas_plot.get_tk_widget()
        self.canvas_plot_widget.pack()

        # Bind Mouse Click, Drag, and Release Events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)

    def load_image(self):
        # Open file dialog to select an image
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp;*.tiff")])

        if self.image_path:
            # Load image and display on canvas
            if self.image_path.find('.tif') != -1:  
                self.p.prepare_data(self.image_path)
                self.img = self.p.genFalseRGB(convertPIL=True)
                print(self.p.hsi_data.shape)
            else:
                self.img = Image.open(self.image_path)
            self.img = self.img.resize((256, 256))
            self.photo = ImageTk.PhotoImage(self.img)
            self.canvas.config(width=self.img.width, height=self.img.height)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
    
    def save_endmember(self):
        # Create a dialogue box for text entry
        endmember_name = simpledialog.askstring("Save Endmember", "Enter the endmember name:")
        endmembers_csv_path = self.create_endmember_directory()
        # Print the entered text to the console
        if endmember_name:
            print("Endmember Name:", endmember_name)

            if endmember_name in self.existing_data:
                # Prompt a confirmation dialog to override or cancel
                confirmation = messagebox.askyesno(
                    "Endmember Exists",
                    f"The endmember '{endmember_name}' already exists. Do you want to average the new spectral data?"
                )
                
                if not confirmation:
                    return  # If canceled, do not override
                else:
                    self.existing_data[endmember_name] = [(a + b) / 2 for a, b in zip(self.existing_data[endmember_name], self.average_values)] 
            else:
                self.existing_data[endmember_name] = self.average_values

        self.write_dict_to_csv(self.existing_data, endmembers_csv_path)

    def draw_rectangle(self, x1, y1, x2, y2):
        # Draw a rectangle on the canvas
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=1, stipple="gray50")

    def calculate_average_rgb(self, x1, y1, x2, y2):
        # Crop the image to the selected region
        region = self.img.crop((x1, y1, x2, y2))

        # Calculate the average RGB values
        rgb_array = np.array(region)
        average_rgb = np.mean(rgb_array, axis=(0, 1))

        return average_rgb.astype(int)
    
    def calculate_average_spectral(self, x1, y1, x2, y2):
        
        print("x1 is," , x1)
        print("x2 is," , x2)
        print("y1 is," , y1)
        print("y2 is," , y2)

        cropped_region = self.p.hsi_data[x2:y2, x1:y1, :]
        print(cropped_region.shape)
        print(cropped_region)
        average_hyper = np.mean(cropped_region, axis=(0, 1))

        print("Average hyper Values:", average_hyper)
        return average_hyper.astype(int)

    def update_plot(self, average_values):
        # Update the RGB label
        self.rgb_label.config(text=f"Average Reflectance Values: {average_values}")

        bands = np.arange(1, len(average_values) + 1)
        # Plot average RGB values
        self.ax.clear()
        self.ax.plot(bands, average_values, marker='o', linestyle='-', color='b')
        self.ax.set_title("Average Reflectance Values")
        self.ax.set_xlabel("Channel")
        self.ax.set_ylabel("Value")
        self.ax.set_ylim(0, 255)
        self.canvas_plot.draw()

    def on_canvas_click(self, event):
        # Get the starting pixel coordinates when the mouse is clicked
        self.start_x, self.start_y = event.x, event.y

    def on_canvas_drag(self, event):
        # Get the current pixel coordinates during dragging
        current_x, current_y = event.x, event.y

        # Draw a rectangle to visualize the selected region
        self.draw_rectangle(self.start_x, self.start_y, current_x, current_y)

    def on_canvas_release(self, event):
        # Get the final pixel coordinates when the mouse is released
        end_x, end_y = event.x, event.y

        # Ensure start_x is smaller than end_x, and start_y is smaller than end_y
        x1, x2 = min(self.start_x, end_x), max(self.start_x, end_x)
        y1, y2 = min(self.start_y, end_y), max(self.start_y, end_y)

        # Get the average RGB values of the selected region
        # img = Image.open(self.image_path)
        if self.image_path.find('.tif') != -1:
            self.average_values = self.calculate_average_spectral(x1, x2, y1, y2)
        else:
            self.average_values = self.calculate_average_rgb(x1, y1, x2, y2)

        # Update the plot with the average RGB values
        self.update_plot(self.average_values)
    
    def create_endmember_directory(self):
        # Get the parent directory (one level up)
        parent_directory = os.path.dirname(os.path.dirname(self.image_path))

        # Define the name of the directory to be created
        new_directory_name = "endmembers"
        new_directory_path = os.path.join(parent_directory, new_directory_name)

        # Check if the directory already exists
        if not os.path.exists(new_directory_path):
            # Create the directory
            os.makedirs(new_directory_path)
            print(f"Directory '{new_directory_name}' created at: {new_directory_path}")
        else:
            print(f"Directory '{new_directory_name}' already exists at: {new_directory_path}")
        
        endmembers_csv_path = os.path.join(new_directory_path, 'endmembers.csv')

        if not os.path.exists(endmembers_csv_path):
            
            # Create an empty 'endmembers.csv' file
            with open(endmembers_csv_path, 'w', newline='') as csvfile:
                print(f"'endmembers.csv' created in '{new_directory_name}' directory.")
        else:
            # Read existing CSV into a dictionary
            with open(endmembers_csv_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                self.existing_data = {col: [] for col in reader.fieldnames}

                for row in reader:
                    for col in reader.fieldnames:
                        try:
                            value = int(row[col])
                        except ValueError:
                            # Handle the case where the value cannot be converted to an integer
                            # You might want to handle this differently based on your requirements
                            value = 0  # Default value if conversion fails

                        self.existing_data[col].append(value)
        
        return endmembers_csv_path


    def write_dict_to_csv(self, data, csv_file_name):
        with open(csv_file_name, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)

            # Writing the header row
            csv_writer.writerow(data.keys())

            # Writing the data rows
            for i in range(max(len(values) for values in data.values())):
                row = [data[key][i] if i < len(data[key]) else '' for key in data.keys()]
                csv_writer.writerow(row)

        print(f'Data has been written to {csv_file_name}.')

if __name__ == "__main__":
    root = tk.Tk()
    app = EndmemberAnalyzer(root)
    root.mainloop()
