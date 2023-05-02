import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score

# training function at each epoch
def train(model, device, drug1_loader_train, drug2_loader_train, optimizer, loss_fn):
    """Train a GNN model and retuen average batch loss"""
    model.train()
    batch_cnt, batch_loss = 0, 0
    for batch_idx, data in enumerate(zip(drug1_loader_train, drug2_loader_train)):
        data1 = data[0]
        data2 = data[1]
        data1 = data1.to(device)
        data2 = data2.to(device)
        y = data[0].y.view(-1, 1).long().to(device)
        y = y.squeeze(1)
        optimizer.zero_grad()
        output = model(data1, data2)
        loss = loss_fn(output, y)
        batch_loss += loss.item()
        batch_cnt += 1
        loss.backward()
        optimizer.step()

    return batch_loss / batch_cnt


def validate(model, device, drug1_loader_val, drug2_loader_val, loss_fn):
    """Calculate average batch loss on validation data"""
    batch_cnt, batch_loss = 0, 0
    with torch.no_grad():
        for data in zip(drug1_loader_val, drug2_loader_val):
            data1 = data[0]
            data2 = data[1]
            data1 = data1.to(device)
            data2 = data2.to(device)
            y = data[0].y.view(-1, 1).long().to(device)
            y = y.squeeze(1)
            output = model(data1, data2)
            loss = loss_fn(output, y)
            batch_loss += loss.item()
            batch_cnt += 1

    return batch_loss / batch_cnt


def predicting(model, device, drug1_loader_test, drug2_loader_test):
    """Generate prediction of a GNN model given input data"""
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()

    with torch.no_grad():
        for data in zip(drug1_loader_test, drug2_loader_test):
            data1 = data[0]
            data2 = data[1]
            data1 = data1.to(device)
            data2 = data2.to(device)
            output = model(data1, data2)
            ys = F.softmax(output, 1).to("cpu").data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
            total_prelabels = torch.cat(
                (total_prelabels, torch.Tensor(predicted_labels)), 0
            )
            total_labels = torch.cat((total_labels, data1.y.view(-1, 1).cpu()), 0)
    return (
        total_labels.numpy().flatten(),
        total_preds.numpy().flatten(),
        total_prelabels.numpy().flatten(),
    )


def train_pipeline(
    model, optimizer, loss_fn, device, train_data, validation_data, nepoch=100
):
    """Combine train, validation and preiction into a model training pipeline"""

    info = {"loss": [], "acc": [], "auc": []}
    val_info = {"loss": [], "acc": [], "auc": []}

    for epoch in range(nepoch):
        loss = train(model, device, train_data[0], train_data[1], optimizer, loss_fn)
        info["loss"].append(loss)
        val_loss = validate(
            model, device, validation_data[0], validation_data[1], loss_fn
        )
        val_info["loss"].append(val_loss)
        # compute preformence on training data
        # T is correct label
        # S is predict score
        # Y is predict label
        T, S, Y = predicting(model, device, train_data[0], train_data[1])
        AUC = roc_auc_score(T, S)
        ACC = accuracy_score(T, Y)
        info["acc"].append(ACC)
        info["auc"].append(AUC)

        # compute preformence on validation data
        val_T, val_S, val_Y = predicting(
            model, device, validation_data[0], validation_data[1]
        )
        val_AUC = roc_auc_score(val_T, val_S)
        val_ACC = accuracy_score(val_T, val_Y)
        val_info["acc"].append(val_ACC)
        val_info["auc"].append(val_AUC)

        print(
            "Epoch {}: Training Loss={:.4f} Accuracy={:.4f} AUC={:.4f} | Validation Loss={:.4f} Accuracy={:.4f} AUC={:.4f}".format(
                epoch + 1, loss, ACC, AUC, val_loss, val_ACC, val_AUC
            )
        )
    return info, val_info

