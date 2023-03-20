import reading
import CNN_multi_class
import CNN_multi_utilities
import argparse


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description="CNN classifierfor Alzheimer's disease detection on multimodal Diffusion Images."
    )



    parser.add_argument(
        "-dpd",
        "--datapathdiff",
        metavar="",
        help="Path of the diffusion parameters map",
        default="Diffusion_parameters_maps-20230215T134959Z-001"
    )
    

    

   
    parser.add_argument(
        "-ep",
        "--epochs",
        metavar="",
        help="Number of epochs for training",
        default="1000",
    )
    
    parser.add_argument(
        "-bs",
        "--batchsize",
        metavar="",
        
        help="Batch size for training",
        default="10",
    )

    args = parser.parse_args()
    
    
    
    images, labels =CNN_multi_utilities.import_dataset()

    augmented_images, augmented_labels = CNN_multi_utilities.data_augmentation(images, labels)
    
    x_train, y_train, x_val, y_val, x_test, y_test=CNN_multi_utilities.train_val_test_split(augmented_images, augmented_labels)
    
    
    shape=(110, 110, 3)
    
    model=CNN_multi_class.MyModel(shape)
    
    option = input("Load Previous weights for training? Y or N: ")
    
    if option.lower()[0] == "n":
        model.compile_and_fit(x_train,y_train,x_val,y_val,x_test,y_test,args.epochs,args.batchsize)
    elif option.lower()[0] == "y":
        model.load('model.h5',x_train,y_train,x_val,y_val,x_test,y_test,args.epochs,args.batchsize)
    
