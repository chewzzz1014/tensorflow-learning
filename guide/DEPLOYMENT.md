# Deploy Tensorflow Object Detection API Using Tensorflow Serving (HTTP Rest)

## Pre
   - Make sure that the trained model has been exported in SavedModel format
   - The saved model has the following file structure:
     ```
      saved_model/
      ├─ 1/
      │  ├─ assets/
      │  ├─ variables/
      │  └─ saved_model.pb  
     ```

## Steps: 
1. Install Tensorflow and TensorFlow Serving  
   ```
   sudo apt-get update
   pip install tensorflow

   echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && \
   curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
   
   sudo apt-get install tensorflow-model-server
   ```

2. Start serving
   ```
   # note that model_base_path is the path to saved model without the version number dir(tf will pick the latest version itself)
   # in this case, path is /path-to-directory/saved_model/
   tensorflow_model_server --rest_api_port=9000 --model_base_path="path-to-directory/tensorflow_serving/my_model" --model_name=sample
   ```
   
3. HTTP Rest
   - Client can send `POST request` to the server's port 9000 (or any other value specified in step 2) 
   - Refer (test_deploy_http.py)[] for the implementation!
## References
1. https://www.analyticsvidhya.com/blog/2020/04/build-your-own-object-detection-model-using-tensorflow-api/
