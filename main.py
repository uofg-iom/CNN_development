import cnn_model

def main():
    dataset_path = "00 - Datasets split by class - Watermark Removed"
    model_obj = cnn_model.SkinTypeModel(dataset_path)
    model = model_obj.build_model(len(model_obj.class_names), cnn_model.ModelType.ResNet152)
    model.summary()