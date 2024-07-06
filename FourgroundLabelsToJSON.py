import pandas as pd
import numpy as np
from collections import defaultdict
import shutil
import os

def getLabel(df,pointer):
    col_name = df.columns[pointer]
    return col_name.split('_')[0]

def countDict(df,index):
    '''
    return dictionary <label,list:[count,pointer]>
    pointer represents where the label starts
    '''
    count_dict = defaultdict(list)
    # start at 1 because 0 is filename
    pointer = 1
    curr_label = None
    while pointer < len(df.columns)-9:
        if not np.isnan(df.iloc[index,pointer]):
            label = getLabel(df,pointer)
            if curr_label != label:
                curr_label = label
                count_dict[label] = [0,pointer]
            count_dict[label][0] += 1
            pointer += 9
        else:
            pointer += 1
    return count_dict

def split_position(position):
    '''
    position: [xctr, yctr, zctr, xlen, ylen, zlen, xrot, yrot, zrot]
    '''
    arr = np.array(position)
    location = arr[0:3]
    dimensions = arr[3:6]
    orientation = arr[6:9]
    return (location,dimensions,orientation)


frame_directory = r"C:\Users\hussa\OneDrive - University of Tennessee\UTC\Research\Fourgrounds"
data_directory = 'Fourgrounds_csv_data'
destination_directory = 'Fourgrounds_Labels'

os.makedirs(destination_directory, exist_ok=True) # create directory if non-existent

for file in os.listdir(data_directory):
    print(file)
    df = pd.read_csv(os.path.join(data_directory,file))

    # create directory for the file with the same name
    folder_path = os.path.join(destination_directory, file.split('.')[0])
    os.makedirs(folder_path, exist_ok=True)

    # create pcd directory 
    pcd_path = os.path.join(folder_path,'pcd')
    os.makedirs(pcd_path, exist_ok=True)
    
    # new dataframe to export to json
    column_names =['Filename','Type','Location','Dimensions','Orientation']
    new_df = pd.DataFrame(columns=column_names)

    for index, row in df.iterrows():
        count_dict = countDict(df,index)
        print(count_dict)

        # copy pcd file to pcd directory
        filename = df.iloc[index,0]
        file_path = os.path.join(frame_directory,filename)
        shutil.copy(file_path, os.path.join(pcd_path,filename.split('/')[1]))

        for k,v in count_dict.items():
            count = v[0]
            pointer = v[1]
            for i in range(count):
                position = []
                for j in range(i,count*9 + 1,count):
                    position.append(df.iloc[index,j+pointer])
            
                location, dimensions, orientation = split_position(position)
                # filename = df.iloc[index,0]

                new_df.loc[len(new_df.index)] = [filename,k,location,dimensions,orientation]

    new_df.to_json(os.path.join(folder_path,'label_data.json'),orient='records')