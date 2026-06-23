import os
import glob
import pandas as pd

def combine_csv_files(folder_path, output_file_name="streaming_data.csv"):
    search_path = os.path.join(folder_path, "*.csv")
    csv_files = glob.glob(search_path)
    if not csv_files:
        print(f"No CSV files found in the folder: {folder_path}")
        return
    print(f"Found {len(csv_files)} CSV files. Combining them now...")
    
    df_list = []
    for file in csv_files:
        file_name = os.path.basename(file)
        stock_name = file_name.split('_')[0]
        
        try:
            df = pd.read_csv(file)
            df.insert(0, 'Stock', stock_name)
            df_list.append(df)
        except Exception as e:
            print(f"Warning: Could not read {file_name}. Error: {e}")
            continue
    
    if not df_list:
        print("No valid data to combine.")
        return


    combined_df = pd.concat(df_list, ignore_index=True)
    output_path = os.path.join(folder_path, output_file_name)
    combined_df.to_csv(output_path, index=False)
    print(f"Success! Combined file saved to: {output_path}")


if __name__ == "__main__":
    TARGET_FOLDER = "./data/streaming" 
    combine_csv_files(TARGET_FOLDER)