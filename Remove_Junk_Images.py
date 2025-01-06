from ultralytics import YOLO
import os
import colorama
from colorama import Fore, Style
import shutil
import torch

colorama.init()
folder_path = r"C:\temp\photos"

script_dir = os.path.dirname(os.path.realpath(__file__))
log_file = os.path.join(script_dir, os.path.basename(folder_path) + '_processed_files.log')
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

model = YOLO("yolov8n.pt") #.to(device)

class_names = model.model.names

def load_processed_files(log_file):
    """ Load the list of processed files from the log file """
    if os.path.exists(log_file):
        with open(log_file, "r", encoding='utf-8') as file:
            return set(file.read().splitlines())
    return set()
    
def update_log_file(log_file, file_path):
    """ Update the log file with the newly processed file path """
    with open(log_file, "a", encoding='utf-8') as file:
        file.write(file_path + "\n")
        
def detect_person_in_image(image_path, model, class_names):
    # Use the predict method with the image path
    results = model.predict(source=image_path, save=False)

    # Check if 'person' is among the detected classes
    person_detected = any(class_names[int(detection.cpu().numpy()[-1])] == 'person' for detection in results[0])

    return person_detected

total_files = 0
for fpath, _, filenames in os.walk(folder_path):
    for filename in filenames:
        total_files += 1

current_file = 0
processed_files = load_processed_files(log_file)

for root, dirs, files in os.walk(folder_path):
    for filename in files:
        file_path = os.path.join(root, filename)
        current_file += 1
        percentage_complete = (current_file / total_files) * 100
        if file_path in processed_files:
            print(f"SKIPPING: {filename}")
            continue
        
        update_log_file(log_file,file_path)
        
        print("")
        print(f"{Fore.GREEN}{percentage_complete:.3f}%{Style.RESET_ALL} - {current_file} of {total_files}: {Fore.YELLOW}{file_path}{Style.RESET_ALL}")
        #print(f"{Fore.YELLOW}{file_path}{Style.RESET_ALL}")
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                person_detected = detect_person_in_image(file_path, model, class_names)
            except Exception as e:
                print(f"{Fore.RED}ERROR: {filename}{e}{Style.RESET_ALL}")
            if person_detected == False:
                print(f"{Fore.CYAN}JUNK: Renaming {filename}{Style.RESET_ALL}")
                new_path = os.path.join(root, f"_junk_{filename}")
                shutil.move(file_path, new_path)
                update_log_file(log_file,new_path)
        else:
            print(f"{Fore.RED}SKIPPING: {filename}{Style.RESET_ALL}")
