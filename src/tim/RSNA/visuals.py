import cv2
import pydicom as dicom
import torch
import numpy as np

# Implementation adapted from https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
def show_gradcam(img_dir, model, patient, inputs, target, df_all, ax1, ax2, threshold=.5):
    model.eval()
    
    outputs = model(inputs)
    model.zero_grad()
    outputs.backward()
    
    proba = torch.sigmoid(outputs).item()
    pred = (proba >= threshold)
    
    with torch.no_grad():
        # pull the gradients out of the model
        gradients = model.get_activations_gradient()
        # pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        # get the activations of the last convolutional layer
        activations = model.get_activations(inputs)
        # weight the channels by corresponding gradients
        for i in range(pooled_gradients.shape[0]):
            activations[:, i, :, :] *= pooled_gradients[i]
        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()
    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = np.maximum(heatmap.cpu(), 0)
    # normalize the heatmap
    heatmap /= torch.max(heatmap)

    img_path = ''.join([img_dir, '/', patient.patient_id, '.dcm'])

    image = dicom.dcmread(img_path)
    
    heatmap_x = cv2.resize(heatmap.numpy(), (448, 448))
    heatmap_x = np.pad(heatmap_x, 32)
    heatmap_x = cv2.resize(heatmap_x, (image.pixel_array.shape[1], image.pixel_array.shape[0]))
    heatmap_x = np.uint8(255 * heatmap_x)
    heatmap_x = cv2.applyColorMap(heatmap_x, cv2.COLORMAP_HOT)
    
    image = cv2.cvtColor(image.pixel_array, cv2.COLOR_GRAY2RGB)
    image_x = cv2.addWeighted(image, 1.0, heatmap_x, .3, 1.0)
    # Draw bounding box
    if target == 1:
        for row in df_all[df_all.patient_id==patient.patient_id].iloc[:, 1:-1].values:
            x, y, w, h = np.int64(row)
            image_x = cv2.rectangle(image_x, (x, y), (x+w, y+h), (255, 0, 0), 7)

    ax1.imshow(image)
    ax2.imshow(image_x)
