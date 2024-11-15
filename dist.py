import os
import random
import shutil
import argparse

def get_next_folder_name(base_folder, n):
    """Generate folder names like hdr_data(n), but using a basic approach."""
    folder_name = base_folder + "/hdr_data" + str(n)  # Simple string concatenation
    return folder_name

def select_random_files(source_folder, destination_folder, num_folders, num_files_per_folder):
    # Make sure the destination folder exists
    if not os.path.exists(destination_folder):
        os.mkdir(destination_folder)  # Simple check and create folder
    
    # List all files in the source folder
    files = os.listdir(source_folder)
    only_files = []
    for file in files:
        if os.path.isfile(source_folder + "/" + file):
            only_files.append(file)  # Add files to list manually
    
    # If there are not enough files, print and stop
    if len(only_files) < num_folders * num_files_per_folder:
        print("Not enough files. We need " + str(num_folders * num_files_per_folder) + " files but found " + str(len(only_files)) + " files.")
        return
    
    # Select random files
    selected_files = random.sample(only_files, num_folders * num_files_per_folder)

    # Start creating folders and copying files
    for i in range(num_folders):
        folder_name = get_next_folder_name(destination_folder, i + 1)  # Call the folder naming function
        
        if not os.path.exists(folder_name):  # Make sure folder doesn't already exist
            os.mkdir(folder_name)  # Create the folder
            
        # Now select files for this folder
        files_to_copy = selected_files[i * num_files_per_folder: (i + 1) * num_files_per_folder]
        
        for file in files_to_copy:
            src = source_folder + "/" + file
            dest = folder_name + "/" + file
            
            shutil.copy2(src, dest)  # Copy file manually
            
        print("Copied " + str(num_files_per_folder) + " files to " + folder_name)

# Main part to run script with args
if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Pick random files from a source folder and copy them to new folders.')
    
    parser.add_argument('source_folder', help='The folder to pick files from')
    parser.add_argument('base_destination_folder', help='Where to put new folders')
    parser.add_argument('num_folders', type=int, help='How many folders to create')
    parser.add_argument('num_files_per_folder', type=int, help='How many files per folder')
    
    # Parse the arguments
    args = parser.parse_args()

    # Run the function with the arguments given
    select_random_files(args.source_folder, args.base_destination_folder, args.num_folders, args.num_files_per_folder)
