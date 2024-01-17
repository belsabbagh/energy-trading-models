import pandas as pd 
from sklearn.model_selection import train_test_split



def split_and_save_df(input_file, output_folder): 
    # read data from ../data/duration/dataset.csv
    df = pd.read_csv(input_file)
    # split into train and test data
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    # save train and test data to ../data/duration/train.csv and ../data/duration/test.csv
    train.to_csv(output_folder + "train.csv", index=False)
    test.to_csv(output_folder + "test.csv", index=False)


def main(): 
    split_and_save_df("../data/duration/dataset.csv", "../data/duration/")


if __name__ == "__main__":
    main()