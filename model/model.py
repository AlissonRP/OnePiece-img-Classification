# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import models

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transformations = T.Compose([
    T.Resize((98, 98)),
    T.ToTensor(),
    T.Normalize(mean=[0.5979, 0.5621, 0.5287],
                std=[0.3124, 0.3003, 0.3111])

])

# %%


# %%
torch.manual_seed(42)
categories = ['Brook', 'Chopper', 'Franky', 'Jinbe', 'Luffy',
              'Nami', 'Robin', 'Sanji', 'Usopp', 'Zoro']

df = datasets.ImageFolder("data", transform=transformations)
train_size = int(0.8 * len(df))
test_size = len(df) - train_size
df_train, df_test = torch.utils.data.random_split(df, [train_size, test_size])

# %%
imgs = torch.stack([img_t for img_t, _ in df_train], 3)
imgs.view(3, -1).mean(dim=1)  # mean
imgs.view(3, -1).std(dim=1)  # std

# %%
batch_size = 32
train_loader = DataLoader(df_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(df_test, batch_size=batch_size, shuffle=True)

# %%

model = models.densenet161(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

classifier_input = model.classifier.in_features
num_labels = len(categories)
final = nn.Sequential(nn.Linear(classifier_input, 1024),
                      nn.ReLU(),
                      nn.Linear(1024, 512),
                      nn.ReLU(),
                      nn.Linear(512, num_labels),
                      nn.LogSoftmax(dim=1))

model.classifier = final
model.to(device)


# %%

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters())

epochs = 8
for epoch in range(epochs):
    train_loss = 0
    val_loss = 0
    accuracy = 0

    # Training the model
    model.train()
    count = 0
    for inputs, labels in train_loader:
        # to device
        inputs, labels = inputs.to(device), labels.to(device)
        # Clear optimizers
        optimizer.zero_grad()
        output = model.forward(inputs)
        loss = criterion(output, labels)
        # Calculate gradients (backpropogation)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*inputs.size(0)
        count += 1

    # Evaluating the model
    model.eval()
    count = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model.forward(inputs)
            valloss = criterion(output, labels)
            val_loss += valloss.item()*inputs.size(0)

            output = torch.exp(output)
            top_p, top_class = output.topk(1, dim=1)
            # See how many of the classes were correct?
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            count += 1
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = val_loss/len(test_loader.dataset)
    print('Accuracy: ', accuracy/len(test_loader))
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))


# %%
def make_confusion_matrix(model, loader, n_classes):
    confusion_matrix = torch.zeros(n_classes, n_classes, dtype=torch.int64)
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            for t, p in zip(torch.as_tensor(labels, dtype=torch.int64).view(-1),
                            torch.as_tensor(predicted, dtype=torch.int64).view(-1)):
                confusion_matrix[t, p] += 1
    return confusion_matrix


def evaluate_accuracy(model, dataloader, classes, verbose=True):
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    confusion_matrix = make_confusion_matrix(model, dataloader, len(classes))
    if verbose:
        total_correct = 0.0
        total_prediction = 0.0
        for i, classname in enumerate(classes):
            correct_count = confusion_matrix[i][i].item()
            class_pred = torch.sum(confusion_matrix[i]).item()

            total_correct += correct_count
            total_prediction += class_pred

            accuracy = 100 * float(correct_count) / class_pred
            print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                                 accuracy))
    print("Global acccuracy is {:.1f}".format(
        100 * total_correct/total_prediction))
    return confusion_matrix


def test(model, dataloader, classes):
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    total_correct = 0.0
    total_prediction = 0.0
    for classname, correct_count in correct_pred.items():
        total_correct += correct_count
        total_prediction += total_pred[classname]
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                             accuracy))
    print("Global acccuracy is {:.1f}".format(
        100 * total_correct/total_prediction))

# %%


confusion_matrix = evaluate_accuracy(model, test_loader, categories)

# %%
torch.save(model.state_dict(), 'modelo')
