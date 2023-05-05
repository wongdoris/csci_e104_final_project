import os
import argparse
import numpy as np
import torch
from codes.utils import plot_training_epoch, compute_preformence
from codes.load_data import load_data
from codes.train_pipeline import train_pipeline, predicting
from models import gat, gcn, gin, sage

cellfile = "independent_cell_features_954"
datafile = "new_labels_0_10"
testfile = "independent_input"


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
        "-epoch", "--epoch", type=int, default=5, help="Number of epoch to train"
    )
    parser.add_argument(
        "-name", "--name", type=str, default="model1", help="Model Name"
    )

    args = parser.parse_args()

    run(model_name=args.name, model_type=args.model, nepoch=args.epoch)


def run(model_name, model_type, nepoch):

    # load train, test data
    (
        drug1_loader_train,
        drug2_loader_train,
        drug1_loader_val,
        drug2_loader_val,
        drug1_loader_test,
        drug2_loader_test,
    ) = load_data(cellfile, datafile, testfile, train_split=0.9, batch_size=128)

    # CPU or GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("\nThe code uses GPU...")
    else:
        device = torch.device("cpu")
        print("\nThe code uses CPU!!!")

    # create model
    print("\nCreating {} model".format(model_type.upper()))
    if model_type == "gat":
        model = gat.GATNet().to(device)
    elif model_type == "gcn":
        model = gcn.GCNNet().to(device)
    elif model_type == "gin":
        model = gin.GINNet().to(device)
    elif model_type == "sage":
        model = sage.SAGENet().to(device)

    print(model)
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_params])
    print(f"\nNumber of trainable parameterss: {params}\n")

    print("Start training model...")
    info_train, info_val = train_pipeline(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=0.0005),
        loss_fn=torch.nn.CrossEntropyLoss(),
        device=device,
        train_data=[drug1_loader_train, drug2_loader_train],
        validation_data=[drug1_loader_val, drug2_loader_val],
        nepoch=nepoch,
    )
    print("Training complete.")

    # save trained model

    if not os.path.exists("trained_model"):
        os.makedirs("trained_model")

    path = "trained_model/" + model_name
    print("Saving trained model to {}".format(path))
    torch.save(model.state_dict(), path)

    # performance on testing data
    T, S, Y = predicting(model, device, drug1_loader_test, drug2_loader_test)
    perf_test = compute_preformence(T, S, Y)

    print("\nPerformance on Test data: ")
    for k, v in perf_test.items():
        print("{} = {:.4f}".format(k, v))

    # plot training epochs
    plot_training_epoch(info_train, info_val, model_name)


if __name__ == "__main__":
    main()
