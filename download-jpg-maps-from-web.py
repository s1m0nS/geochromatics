#!/usr/bin/env python
# coding: utf-8

# #### Simple Python code to dowload PDF-s from web

#-------------------------------------------------------------------#

# https://media.digitalarkivet.no/view/58619/1?indexing

# first map
# https://media.digitalarkivet.no/view/58619/97

# last map
# https://media.digitalarkivet.no/view/58619/4678

#-------------------------------------------------------------------#

# Time the execution of the script
import time
startTime = time.time()

import urllib
import requests
import os
from urllib.parse import urlparse
from urllib.parse import unquote
import os

# DEFINE UPPER/LOWER LIMIT
upper = 10071111010071
lower = 10071107120848
c = upper - lower
print("Number of images to try: ", c)

# ADD URL
url = "https://urn.digitalarkivet.no/"
params = {"URN:NBN:no-a1450-ka": lower}

# generated url
gurl = url + urllib.parse.urlencode(params)
# print('Not cleaned: ', gurl)

# FUNCTIONS

# Function to download a PDF from a given URL
def download_pdf(url, save_dir):
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the URL to extract the filename
            filename = os.path.basename(urlparse(url).path)
            
            # Check if the file is a PDF
            if filename.endswith('.pdf'):
                # Construct the full path to save the PDF
                save_path = os.path.join(save_dir, filename)
                
                # Save the PDF to the specified directory
                with open(save_path, 'wb') as pdf_file:
                    pdf_file.write(response.content)
                print(f"Downloaded: {filename}")
            else:
                print(f"Not a PDF: {filename}")
        else:
            print(f"Failed to download: {url}")

    except Exception as e:
        print(f"Error: {e}")


# Function to download a JPG from a given URL
def download_pdf(url, save_dir):
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the URL to extract the filename
            filename = os.path.basename(urlparse(url).path)
            
            # Check if the file is a JPG
            if filename.endswith('.jpg'):
                # Construct the full path to save the JPG
                save_path = os.path.join(save_dir, filename)
                
                # Save the PDF to the specified directory
                with open(save_path, 'wb') as pdf_file:
                    pdf_file.write(response.content)
                print(f"Downloaded: {filename}")
            else:
                print(f"Not a JPG: {filename}")
        else:
            print(f"Failed to download: {url}")

    except Exception as e:
        print(f"Error: {e}")



def generate_url(url, params):
    """
    Generate URLs and remove '%' and '=' symbols from the URL
    Remove '%' and '=' symbols from a URL.
    """

    # Generate URL
    gurl = url + urllib.parse.urlencode(params)

    # Decode URL encoding using unquote
    decoded_url = unquote(gurl)
    
    # Remove '%' and '=' symbols
    cleaned_url = decoded_url.replace('%', '').replace('=', '')

    #print('Cleaned URL: ', cleaned_url)
    
    return cleaned_url

def increment_numbers(a, b):
    """
    Generate a list of numbers from 'a' to 'b' (inclusive).
    Args:
        a (int): The starting number.
        b (int): The ending number.
    Returns:
        list: A list of numbers from 'a' to 'b'.
    """
    if a <= b:
        return list(range(a, b + 1))
    else:
        return []
    
def remove_files_by_size(directory, size_in_bytes):
    try:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            # Check if it's a file and its size matches the specified size
            if os.path.isfile(file_path) and os.path.getsize(file_path) == size_in_bytes:
                os.remove(file_path)
                print(f"Removed file: {file_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
def add_jpg(strings):
    """Add jpg extension to strings."""
    return [string + ".jpg" for string in strings]

generate_url(url, params)

# EXAMPLE
list_of_numbers = increment_numbers(lower, upper) # saved to list

print(" # FIRST MAP: ", list_of_numbers[0])
print(" #  LAST MAP: ", list_of_numbers[-1])

# Generate map URLs
map_urls = []
for i in list_of_numbers:
    map_urls.append(generate_url(url, params={"URN:NBN:no-a1450-ka": i}))

input_list = map_urls

# Add JPG extension
map_jpg_urls = add_jpg(input_list)

# Print first and last
print(" # FIRST MAP URL: ", map_jpg_urls[0])
print(" #  LAST MAP URL: ", map_jpg_urls[-1])

# DOWNLOAD MAPS TO MAPS FOLDER

# Directory to save the downloaded PDFs
save_directory = "/home/shymon/Documents/phd/GEOREFERENCING/DATA/MAPS/"

# Create the save directory if it doesn't exist
os.makedirs(save_directory, exist_ok=True)

# IMAGES THAT DOES NOT EXIST ARE 97683 B BIG, WE NEED TO REMOVE THEM
# Remove files from folder that are exactly 97683 B
directory_to_search = save_directory
size_to_remove = 97683   # Size in bytes (97.7 kB)

# Loop through the list of URLs and download PDFs
# at each iteration remove the files that are not maps
for url in map_jpg_urls:
    download_pdf(url, save_directory)
    remove_files_by_size(directory_to_search, size_to_remove)


# Done
print("Done, maps have been dowloaded...")

# Print execution time
executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
