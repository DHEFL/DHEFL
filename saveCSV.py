import csv
import config_FL
import os 
def save(path, data1,title1, data2, title2):
    
    index = config_FL.num_client()
    # specify the file name and location
    filename = path+str(index)+'.csv'
    
    if not os.path.exists(filename):
        # If the file does not exist, create a new file and write the header row
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([title1, title2])
    # open the file for writing (create it if it doesn't exist)
    with open(filename, mode='a', newline='') as file:
        # create a CSV writer object
        writer = csv.writer(file)
        # write the data rows
        #for i in range(len(data1)):
        writer.writerow([data1, data2])

