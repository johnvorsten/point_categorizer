## Description of files
lableing.py, labeling_main.py, labeling_test.py - Label data with a ranked list of clustering metrics.
The most important class and method is probably Labeling.ExtractLabels.calc_labels | This method calcualtes the possible combinations of {clustering metric, dimensionality reduction method, etc} and outputs the results of that clustering setup into a dictionary (ex frozenset({'8',False,'MDS','Dindex','euclidean','ward.D'}):[0 0,0,0],)
From there, a custom loss function is used to calculate how well that clustering setup performed relative to the 'correct' number of clusters. The result is a sorted list of items ['hyperparameter_dict','predictions','correct_k','loss']

## TODO
1. (done) Understand size of ranking model in memory (Is it a good idea to containerize the model and serve it? Will it be too large in  memroy?)
2. (done) Document model serving and understnad workflow & Create small diagram of workflow for future self
3. (done) Create transformer file and class specific to tensorflow ranking (Remove relative imports)
4. (done) Add data locally to this project folder
4. (done) Add SQL configuration file, and remove individual references in all files
4. Document serialization of context and peritem features? - How useful will this be?
5. (done) Dockerize model for serving
6. (done) Document input features and data types
7. (done) Create custom transformer for inputting raw data to model - probably requires serializing data?
8. (done) Create FastAPI for serving the models predictions
9. Create template data for input into a web form
10. Create visualizations of clustering process for website understnading

# Docker networking
Create a default network for ranking_serving `docker network create ranking_net --driver bridge`

# Docker (tensorflow serving)
Pull docker image `docker pull tensorflow/serving:latest`
Start tensorflow serving docker container `docker run --detach -p 8501:8501 --network ranking_net -v "C:\Users\Jvorsten\PythonProjects\ML\point_categorizer\ranking\final_model\Run_20191024002109model4/:/models/model4" -e MODEL_NAME=model4 --name tf_serving --restart=on-failure:5 tensorflow/serving &`
Logs - 'Successfully loaded servable version {name: model4 version: 1572051525'
Connect this container to the bridge network `docker network connect ranking_net tf_serving`

## Docker (fastapi model4_serving.py)
build the docker image with docker `docker build . --tag ranking_serving`
Run the image for serving `docker run --name ranking_serving --publish 8004:8004 --network ranking_net --log-driver local --restart=on-failure:5 --detach ranking_serving`
Create request `curl -X 'POST' \
  'http://localhost:8004/clustering-ranking/model4predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"n_instance": 445.0, "n_features": 161.0, "len_var": 0..18, "uniq_ratio": 2.76, "n_len1": 0.01, "n_len2": 0.01, "n_len3": 0.78, "n_len4": 0.22, "n_len5": 0.0, "n_len6": 0.0, "n_len7": 0.0}'`
We must publish the containers port to a port on the host machine because we are not creating a networked group of containers. An Nginx proxy will pass requests to this host and port for predictions. It will no run as part of another service or container group
Connect this container to the bridge network `docker network connect ranking_net ranking_serving`
This container can now communicate with tf_serving through the bridge network