import cnn_model
import os


def main():
    dataset_path = "CNN_development/00 - Datasets split by class - Watermark Removed/"
    model_callback_path = "CNN_development/model/"
    trained_model_path = "D:/YEAR_5/DST5/Trained Models/trained_model_120_epoch/trained_model_120_epoch/"
    
    # Initialise the SkinTypeModel object
    model_obj = cnn_model.SkinTypeModel(dataset_path, model_callback_path)
    
    # If a pre-trained model exists, load it here
    model_obj.load_model(trained_model_path)
    # model_obj.show_train_dataset()
    
    # ====== Build Model ======
    # model = model_obj.build_model(len(model_obj.class_names), cnn_model.ModelType.ResNet152)
    # model.summary()
    
    # ====== Train Model ======
    # model_obj.train_model()
    
    # ====== Evaluate and Infere Model ======
    # model_obj.eval_model()
    test_ds_path = "D:/YEAR_5/DST5/DST Code/CNN_development/test_dataset/"
    for image in os.listdir(test_ds_path + "01 - Acne/"):
        model_obj.infere_model(image)
    for image in os.listdir(test_ds_path + "02 - Wrinkles/"):
        model_obj.infere_model(image)
    for image in os.listdir(test_ds_path + "03 - Dry skin/"):
        model_obj.infere_model(image)
    for image in os.listdir(test_ds_path + "05 - Oily skin/"):
        model_obj.infere_model(image)

if __name__ == "__main__":
    main()