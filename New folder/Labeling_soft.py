import pandas as pd
import os
from tkinter import Tk, Label, Button, StringVar, filedialog

# File paths
INPUT_FILE = "sample_for_labeling.csv"  # Replace with your dataset file
OUTPUT_FILE = "labeled_data_2.csv"        # File where labeled data will be saved

df_1 = pd.read_csv(r"D:\Data_TO_WORK_On(30-12-2024).csv").sample(n=1200,random_state=42)

#Cleaning Data From Arabic/Other langauges  Instances
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

def detect_language(text):
    try:
        return detect(text)
    except:
        return None 

df_1['language'] = df_1['PROBLEM_SUMMARY'].apply(detect_language)

df_1 = df_1[df_1['language'] == 'en']

df_1.to_csv("Label These Please (Omar_1-6-2025).csv")

# Load the dataset
if os.path.exists(OUTPUT_FILE):
    df = pd.read_csv(OUTPUT_FILE,encoding='latin-1')
    print(f"Resuming labeling. Loaded {OUTPUT_FILE} with {len(df)} rows.")
else:
    df = df_1
    df["Label"] = -1  # Initialize labels with -1 (unlabeled)
    print(f"Loaded {INPUT_FILE} with {len(df_1)} rows.")
    

# Tkinter GUI setup
class LabelingApp:
    def __init__(self, master, dataframe):
        self.master = master
        self.df = dataframe
        self.index = self.find_next_unlabeled()
        self.current_text = StringVar(self.master)  # Ensure it is associated with the root window
        self.current_text.set(self.get_text())
        self.create_widgets()
        self.save_interval = 20
        self.changes_made = False

    def create_widgets(self):
        # Display text
        self.text_label = Label(self.master, textvariable=self.current_text, wraplength=700, justify="left")
        self.text_label.grid(row=0, column=0, columnspan=3, padx=10, pady=10)

        # Buttons for labeling
        self.button_coherent = Button(self.master, text="Non-Noise (0)", command=lambda: self.label_text(0))
        self.button_coherent.grid(row=1, column=0, padx=10, pady=10)

        self.button_incoherent = Button(self.master, text="Noise (1)", command=lambda: self.label_text(1))
        self.button_incoherent.grid(row=1, column=1, padx=10, pady=10)

        self.button_skip = Button(self.master, text="Skip", command=self.skip_text)
        self.button_skip.grid(row=1, column=2, padx=10, pady=10)

        self.button_quit = Button(self.master, text="Quit and Save", command=self.quit_and_save)
        self.button_quit.grid(row=2, column=1, pady=10)

    def get_text(self):
        if self.index is not None:
            return self.df.loc[self.index, "PROBLEM_SUMMARY"]
        return "All samples have been labeled!"

    def find_next_unlabeled(self):
        for idx, row in self.df.iterrows():
            if row["Label"] == -1:
                return idx
        return None

    def label_text(self, label):
        if self.index is not None:
            self.df.at[self.index, "Label"] = label
            self.index = self.find_next_unlabeled()
            self.current_text.set(self.get_text())
            self.changes_made = True
            if self.index is not None and self.index % self.save_interval == 0:
                self.save_progress()

    def skip_text(self):
        self.index = self.find_next_unlabeled()
        self.current_text.set(self.get_text())

    def save_progress(self):
        self.df.to_csv(OUTPUT_FILE, index=False)
        print(f"Progress saved to {OUTPUT_FILE}.")

    def quit_and_save(self):
        if self.changes_made:
            self.save_progress()
        self.master.quit()

# Run the application
if __name__ == "__main__":
    root = Tk()
    root.title("Manual Text Labeling")
    root.geometry("800x400")
    app = LabelingApp(root, df)
    root.mainloop()
