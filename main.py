import cnn_model

def main():
    dataset_path = "CNN_development/00 - Datasets split by class - Watermark Removed/"
    model_callback_path = "CNN_development/model/"
    model_obj = cnn_model.SkinTypeModel(dataset_path, model_callback_path)
    
    # model_obj.show_train_dataset()
    
    # ====== Build Model ======
    model = model_obj.build_model(len(model_obj.class_names), cnn_model.ModelType.ResNet152)
    model.summary()
    
    # ====== Train Model ======
    model_obj.train_model()
    
    # ====== Evaluate and Infere Model ======
    model_obj.eval_model()
    model_obj.infere_model()
    

if __name__ == "__main__":
    main()