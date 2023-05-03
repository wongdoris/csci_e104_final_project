import random
from codes.create_data import creat_data, TestbedDataset
from torch_geometric.loader import DataLoader


def load_data(cellfile, datafile, testfile, train_split=0.9, batch_size=128):

    creat_data(datafile, cellfile)
    creat_data(testfile, cellfile)

    TRAIN_BATCH_SIZE = batch_size
    TEST_BATCH_SIZE = batch_size

    drug1_data = TestbedDataset(root="data", dataset=datafile + "_drug1")
    drug2_data = TestbedDataset(root="data", dataset=datafile + "_drug2")
    data_size = len(drug1_data)

    # split train, validation and test data
    random_num = random.sample(range(0, data_size), data_size)
    idx_train = random_num[0 : int(data_size * train_split)]
    idx_val = list(set(random_num) - set(idx_train))

    # training data
    drug1_data_train = drug1_data[idx_train]
    drug2_data_train = drug2_data[idx_train]
    drug1_loader_train = DataLoader(
        drug1_data_train, batch_size=TRAIN_BATCH_SIZE, shuffle=None
    )
    drug2_loader_train = DataLoader(
        drug2_data_train, batch_size=TRAIN_BATCH_SIZE, shuffle=None
    )

    # validation data
    drug1_data_val = drug1_data[idx_val]
    drug2_data_val = drug2_data[idx_val]
    drug1_loader_val = DataLoader(
        drug1_data_val, batch_size=TRAIN_BATCH_SIZE, shuffle=None
    )
    drug2_loader_val = DataLoader(
        drug2_data_val, batch_size=TRAIN_BATCH_SIZE, shuffle=None
    )

    # holdout testing data
    drug1_data_test = TestbedDataset(root="data", dataset=testfile + "_drug1")
    drug2_data_test = TestbedDataset(root="data", dataset=testfile + "_drug2")
    drug1_loader_test = DataLoader(
        drug1_data_test, batch_size=TEST_BATCH_SIZE, shuffle=None
    )
    drug2_loader_test = DataLoader(
        drug2_data_test, batch_size=TEST_BATCH_SIZE, shuffle=None
    )

    print("\nData loaded successfully.")
    print("Training set size:", len(drug1_data_train))
    print("Validation set size:", len(drug1_data_val))
    print("Testing set size:", len(drug1_data_test))

    return (
        drug1_loader_train,
        drug2_loader_train,
        drug1_loader_val,
        drug2_loader_val,
        drug1_loader_test,
        drug2_loader_test,
    )
