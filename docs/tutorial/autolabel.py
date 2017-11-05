from glob import glob
import hvc

# 0. create training data
# In this case, we download already labeled data from an open repository.
# String in quotes matches with the name of one of the folders in the repository.
hvc.utils.fetch('gy6or6.032612')

# 1. pick a model and 2. extract features for that model
# Model and features are defined in extract.config.yml file.
hvc.extract('gy6or6_autolabel_example.knn.extract.config.yml')

# 3. pick hyperparameters for model
# Load summary feature file to use with helper functions for
# finding best hyperparameters.
summary_file = glob('./extract_output*/summary*')
summary_data = hvc.load_feature_file(summary_file)
# In this case, we picked a k-nearest neighbors model
# and we want to find what value of k will give us the highest accuracy
cv_scores, best_k = hvc.utils.find_best_k(summary_data['features'],
                                          summary_data['labels'],
                                          k_range=range(1, 11))

# 4. Fit the **model** to the data and 5. Select the **best** model
hvc.select('gy6or6_autolabel.example.select.knn.config.yml')

# 6. **Predict** labels for unlabeled data using the fit model.
hvc.predict('gy6or6_autolabel.example.predict.knn.config.yml')
