import argparse
import pandas as pd
import torch
import sys
from torch_geometric.loader import DataLoader
from codes.create_data import TestbedDataset
from codes.train_pipeline import predicting
from models import gat, gcn, gin, sage

cellfile = "independent_cell_features_954"


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-model",
        "--model",
        type=str,
        default="gcn",
        help="Type of model to train, options: gcn, gat, gin, sage",
    )

    parser.add_argument(
        "-name", "--name", type=str, default="model1", help="Model Name"
    )

    parser.add_argument(
        "-data",
        "--data",
        type=str,
        default="independent_input",
        help="Input data for prediction",
    )

    args = parser.parse_args()

    run(model_name=args.name, model_type=args.model, testfile=args.data)


def run(testfile, model_name, model_type):
    # creat data

    drug1_data_test = TestbedDataset(root="data", dataset=testfile + "_drug1")
    drug2_data_test = TestbedDataset(root="data", dataset=testfile + "_drug2")

    drug1_loader_test = DataLoader(drug1_data_test, batch_size=128, shuffle=None)
    drug2_loader_test = DataLoader(drug2_data_test, batch_size=128, shuffle=None)

    df_test = pd.read_csv("data/" + testfile + ".csv")
    df_smile = pd.read_csv("data/smiles.csv")
    drug_to_smile = {row["name"]: row["smile"] for i, row in df_smile.iterrows()}
    smile_to_drug = {v: k for k, v in drug_to_smile.items()}

    # CPU or GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # load model
    print("\nLoading {} model: {}".format(model_type.upper(), model_name))
    if model_type == "gat":
        model = gat.GATNet().to(device)
    elif model_type == "gcn":
        model = gcn.GCNNet().to(device)
    elif model_type == "gin":
        model = gin.GINNet().to(device)
    elif model_type == "sage":
        model = sage.SAGENet().to(device)

    path = "trained_model/" + model_name
    try:
        model.load_state_dict(torch.load(path))
    except:
        print("Wrong model type!")
        sys.exit()

    y_true, prob, y_pred = predicting(
        model, device, drug1_loader_test, drug2_loader_test
    )

    print("\nModel predictions: ")
    for i, row in df_test.iterrows():
        print(
            "{} drug1: {}, drug2: {}, cell: {}, True label: {} | Prediction: {:.0f} (score={:.3f})".format(
                i + 1,
                row.cell,
                smile_to_drug[row.drug1],
                smile_to_drug[row.drug2],
                row.label,
                y_pred[i],
                prob[i],
            )
        )

    df_pred = df_test.copy()
    df_pred["prediction"] = y_pred
    df_pred["probability"] = prob
    j_pred = df_pred.to_json(orient="records")

    n_ones_true = len(df_pred[df_pred.label == 1])
    n_ones_pred = len(df_pred[df_pred.prediction == 1])
    ncorrect = df_pred[df_pred.prediction == df_pred.label].prediction.count()
    print("\nNumber of 1s: True={}, Predicted={}".format(n_ones_true, n_ones_pred))
    print(
        "Number of 0s: True={}, Predicted={}".format(
            len(df_pred) - n_ones_true, len(df_pred) - n_ones_pred
        )
    )
    print(
        "\Correct predictions: {}/{} = {:.2%}".format(
            ncorrect, len(df_pred), ncorrect / len(df_pred)
        )
    )

    # write predictions to disk
    with open("data/processed/predictions", "w") as f:
        f.write(j_pred)
    print("\nPredictions written to data/processed/predictions.json \n")


if __name__ == "__main__":
    main()
