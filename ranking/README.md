## Description of files
lableing.py, labeling_main.py, labeling_test.py - Label data with a ranked list of clustering metrics.
The most important class and method is probably Labeling.ExtractLabels.calc_labels | This method calcualtes the possible combinations of {clustering metric, dimensionality reduction method, etc} and outputs the results of that clustering setup into a dictionary (ex frozenset({'8',False,'MDS','Dindex','euclidean','ward.D'}):[0 0,0,0],)
From there, a custom loss function is used to calculate how well that clustering setup performed relative to the 'correct' number of clusters. The result is a sorted list of items ['hyperparameter_dict','predictions','correct_k','loss']

## TODO
1. Understand size of ranking model in memory (Is it a good idea to containerize the model and serve it? Will it be too large in  memroy?)
2. Document model serving and understnad workflow
    Create small diagram of workflow for future self
3. Create transformer file and class specific to tensorflow ranking (Remove relative imports)
4. Add data locally to this project folder
4. Add SQL configuration file, and remove individual references in all files
4. Output all clustering resuts to a SQL database for saving (see Labeling.ExtractLabels.calc_labels | prevent code smell)
4. Document serialization of context and peritem features? - How useful will this be?
5. Dockerize model for serving
6. Document input features and data types
7. Create custom transformer for inputting raw data to model - probably requires serializing data?
8. Create FastAPI for serving the models predictions
9. Create template data for input into a web form
10. Create visualizations of clustering process for website understnading

# Tensorflow serving
Start tensorflow serving docker container `docker run -t --rm -p 8502:8502 -v "C:\Users\vorst\PythonProjects\ML\point_categorizer\ranking\final_model\Run_20191024002109model4:/models/model4" -e MODEL_NAME=model4 tensorflow/serving &`
Logs - 'Successfully loaded servable version {name: model4 version: 1572051525'
Sent a POST request `curl -d "{"instances":[1.0,2.0]}" -X POST http://localhost:8501/v1/models/model4`
Response - {"error": "Malformed request: POST /v1/models/model4"}
