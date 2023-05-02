import numpy as np
import torch
import utils, load_data, train
from models import gat, gcn, gin, sage

cellfile = "independent_cell_features_954"
datafile = "new_labels_0_10"
testfile = "independent_input"


def main(model_name="gin_model1", model_type="gin", nepoch=3):

    # load train, test data
    (
        drug1_loader_train,
        drug2_loader_train,
        drug1_loader_val,
        drug2_loader_val,
        drug1_loader_test,
        drug2_loader_test,
    ) = load_data.load_data(
        cellfile, datafile, testfile, train_split=0.9, batch_size=128
    )

    # CPU or GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("\nThe code uses GPU...")
    else:
        device = torch.device("cpu")
        print("\nThe code uses CPU!!!")

    # create model
    print(f"\nCreating {model_type.upper()} model:")
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
    info_train, info_val = train.train_pipeline(
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
    path = "trained_model/" + model_name
    print(f"Saving trained model to {path}")
    torch.save(model.state_dict(), path)

    # performance on testing data
    T, S, Y = train.predicting(model, device, drug1_loader_test, drug2_loader_test)
    perf_test = utils.compute_preformence(T, S, Y)

    print("\nPerformance on Test data: ")
    for k, v in perf_test.items():
        print(f"{k} = {v:.4f}")

    # plot training epochs
    utils.plot_training_epoch(info_train, info_val)


if __name__ == "__main__":
    main()
