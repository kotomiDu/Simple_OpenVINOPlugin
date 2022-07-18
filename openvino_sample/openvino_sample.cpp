// openvino_sample.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <cnn.hpp>
#include <inference_engine.hpp>
#include <vector>
#include <memory>
#include <string>
#include <list>

#define EXTERN_DLL_EXPORT extern "C" __declspec(dllexport)

EXTERN_DLL_EXPORT Cnn *createModel() {
    return new Cnn();
}


EXTERN_DLL_EXPORT bool initModel(Cnn *modelcnn, const char *input_model, const char *device_name ) {
    Core ie;
    modelcnn->Init(input_model, ie, device_name);
    return modelcnn->is_initialized();
}
EXTERN_DLL_EXPORT  float* InferModel(Cnn *modelcnn,   const char *input_image_path, int* resultVerLength) {
    cv::Mat image = cv::imread(input_image_path);
    auto blobs = modelcnn->Infer(image);
    
    auto output_shape = blobs.begin()->second->getTensorDesc().getDims();

    LockedMemory<const void> blobMapped = as<MemoryBlob>(blobs.begin()->second)->rmap();
    float* output_data_pointer = blobMapped.as<float*>();
    //std::vector<float> output_data(output_data_pointer, output_data_pointer + output_shape[0] * output_shape[1]);
    int length = output_shape[0] * output_shape[1];
    *resultVerLength = length;

    float* resultVerts = new float[length];
    for (auto i = 0; i < length; i++) {
        resultVerts[i] = *(output_data_pointer + i);
     
    }
    return resultVerts;
}

int main() {
    std::string input_model = "OV_FP16//Resnet34_3inputs_448x448_20200609.xml";
    std::string input_image_path = "InitImg.png";
    const std::string device_name = "CPU";
    float* resultVerts =new float();
    int* resultVerLength = new int();
 
    Cnn *context = createModel();
    bool success = initModel(context, input_model.c_str(), device_name.c_str());
    std::cout << success;
    return 1;
    //MnistCNN(input_model.c_str(), input_image_path.c_str(), device_name.c_str(),  resultVerLength);
    //std::cout << resultVerts[1];
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
