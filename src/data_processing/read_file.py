import pandas as pd
import os

def read_comma_delimited_file(filepath):
    """
    Reads a comma-delimited .csv or .txt file from the given file path
    and returns it as a pandas DataFrame.

    :param filepath: str, path to the .csv or .txt file
    :return: pandas DataFrame or None if file does not exist or is not a .csv or .txt
    """
    if not os.path.exists(filepath):
        print(f"The file {filepath} does not exist.")
        return None
    
    if not filepath.lower().endswith(('.csv', '.txt')):
        print("The file must be a .csv or .txt file.")
        return None
    
    try:
        df = pd.read_csv(filepath, delimiter=',')
        return df
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

# Example usage
if __name__ == "__main__":
    file_path = r'C:\path\to\your\file.csv'  # Update this path to your file location
    dataframe = read_comma_delimited_file(file_path)
    if dataframe is not None:
         print(dataframe)
