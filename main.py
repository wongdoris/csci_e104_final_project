from load_data import load_data
from train import train_pipeline
import utils

cellfile = "independent_cell_features_954"
datafile = "new_labels_0_10"
testfile = "independent_input"


def main():
    (
        drug1_loader_train,
        drug2_data_train,
        drug1_data_val,
        drug2_data_val,
        drug1_data_test,
        drug2_data_test,
    ) = load_data(cellfile, datafile, testfile)


if __name__ == "__main__":
    main()
