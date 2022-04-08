
import streamlit as st
import torch
import torchvision.transforms as T
import torch.nn as nn
import os
import io
from PIL import Image
st.set_page_config(page_title='One Piece Predict', page_icon='imgs/icon.png',
                   layout="centered", initial_sidebar_state="auto", menu_items=None)
map_location = torch.device('cpu')


def load_image():
    uploaded_file = st.file_uploader(
        label='Upload an image of an Mugiwara member and I\'ll give you the probabilities!')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))

    else:
        return None


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
categories = ['Brook', 'Chopper', 'Franky', 'Jinbe', 'Luffy',
              'Nami', 'Robin', 'Sanji', 'Usopp', 'Zoro']


def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from torchvision import models
    model = models.densenet161(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    classifier_input = model.classifier.in_features

    num_labels = len(categories)
    classifier = nn.Sequential(nn.Linear(classifier_input, 1024),
                               nn.ReLU(),
                               nn.Linear(1024, 512),
                               nn.ReLU(),
                               nn.Linear(512, num_labels),
                               nn.LogSoftmax(dim=1))
    # Replace default classifier with new classifier
    model.classifier = classifier
    model.to(device)

    model.load_state_dict(torch.load('model/modelo', map_location='cpu'))
    model.eval()
    return model


def load_labels():
    labels_path = 'https://raw.githubusercontent.com/AlissonRP/things/master/cat.txt'
    labels_file = os.path.basename(labels_path)
    if not os.path.exists(labels_file):
        wget.download(labels_path)
    with open(labels_file, "r") as f:
        categories = [s.strip() for s in f.readlines()]
        return categories

# %%


def predict(model, categories, imagem):
    transformations = T.Compose([
        T.Resize((102, 102)),
        T.ToTensor(),
        T.Normalize(mean=[0.5979, 0.5621, 0.5287], std=[0.3124, 0.3003, 0.3111])])
    img_tensor = transformations(imagem)
    batch = img_tensor.unsqueeze(0).to(device)
    model.eval()
    output = model(batch)
    logits = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(logits, 3)
    for i in range(top5_prob.size(0)):
        st.write(categories[top5_catid[i]], round(top5_prob[i].item(), 2))


def main():
    st.title('Mugiwara Pirates Image Classification')
    model = load_model()
    categories = load_labels()
    image = load_image()
    result = st.button('Run on image')
    if result:
        st.write('The probabilities are ...')
        predict(model, categories, image)
    st.write(
        "This app was created by [AlissonRP](https://github.com/AlissonRP), repository on [GitHub](https://github.com/AlissonRP/OP-image-classification)")


if __name__ == '__main__':
    main()


# %%

OnePiece-img-Classification