import os
import shutil


def find_and_move_files(parent_directory, destination_directory, filename):
    """
    Finds and moves specified files from parent directory and its subdirectories to a destination directory.

    :param parent_directory: parent directory
    :param destination_directory: destination directory
    :param filename: name of files to be found and moved
    """
    print(filename)
    counter = 1
    r=0
    for dirpath, dirnames, filenames in os.walk(parent_directory):
        for name in filenames:
            r+=1
            if name == filename:
                # Construct source and destination file paths
                source_file = os.path.join(dirpath, name)
                new_name = f'eeg_{counter}.csv'  # rename file with unique number
                destination_file = os.path.join(destination_directory, new_name)
                
                # Move and rename the file
                shutil.move(source_file, destination_file)
                print(f"Moved file {source_file} to {destination_file}")

                counter += 1
    print(r)
# Use the function
parent_directory = '/home/ayknes/Downloads/EEG/Data Acquisition/'
destination_directory = '/home/ayknes/Downloads/EEG/Data Acquisition/Data'
filename = 'EEG.csv'

find_and_move_files(parent_directory, destination_directory, filename)

