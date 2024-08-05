import requests
import os

base_url = "http://mocap.cs.cmu.edu/subjects/"


categories = {
    "walk": [
        {2: [1, 2]}, {5: [1]}, {6: [1]}, {7: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]},
        {8: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}, {9: [12]}, {10: [4]}, {12: [1, 2, 3]}, {15: [1, 3, 9, 14]},
        {16: [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25 , 26, 27, 28, 29, 30, 31, 32, 33, 34, 47, 58]},
        {17: [1, 2, 3, 4, 5, 6, 7, 8, 9]}, {26: [1]}, {27: [1]}, {29: [1]}, {32: [1, 2]},
        {35: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 28, 29, 30, 31, 32, 33, 34]},
        {36: [2, 3, 9]}, {37: [1]}, {38: [1, 2, 4]}, {39: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]},
        {40: [2, 3, 4, 5]}, {41: [2, 3, 4, 5, 6]}, {43: [1]}, {45: [1]}, {46: [1]}, {47: [1]}, {49: [1]},
        {55: [4]}, {56: [1]}        
    ],
    "run": [
        {2: [3]},
        {9: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},
        {16: [8, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57]},
        {35: [17, 18, 19, 20, 21, 22, 23, 24, 25, 26]},
        {38: [3]}
    ],
    "jump": [
        {13: [11, 13, 19, 32, 39, 40, 41, 42]},
        {16: [1, 2, 3, 4, 5, 6, 7, 9, 10]}, {49: [2, 3]}
    ]
}

data_dir = "data"
os.makedirs(os.path.join(data_dir, "asf"),exist_ok=True)


for category, values in categories.items():
    
    # Make subject directory
    category_dir = os.path.join(data_dir, category)
    os.makedirs(category_dir, exist_ok=True)
    
    for value in values:        
        for subject, trials in value.items():    
            # Get response from subject url
            subject_url = f"{base_url}{subject:02d}/"
            response = requests.get(subject_url)
            
            if response.status_code == 200:
                
                # Download asf file
                file = f"{subject:02d}.asf"
                if os.path.exists(os.path.join(data_dir,"asf", file)):
                    continue
                file_url = subject_url + file
                file_response = requests.get(file_url)
                if file_response.status_code == 200:
                    with open(os.path.join(data_dir,"asf", file), 'wb') as f:
                        f.write(file_response.content)
                    print(f"Donwload {file}")
                else:
                    print(f"Failed to download {file}, pat`h is '{file_url}'")
                
                # download amc files
                for trial in trials:
                    file = f"{subject:02d}_{trial:02d}.amc"
                    
                    if os.path.exists(os.path.join(category_dir, file)):
                        continue
                    
                    file_url = subject_url + file
                    file_response = requests.get(file_url)
                    
                    if file_response.status_code == 200:
                        with open(os.path.join(category_dir, file), 'wb') as f:
                            f.write(file_response.content)
                        print(f"Downloaded {file}")
                    else:
                        print(f"Failed to download {file}, path is '{file_url}'")
            else:
                print(f"Failed to access subject {subject}")
        

print("Finish")
            