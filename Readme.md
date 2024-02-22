
### py-news-classification

This repo demonstrates machine learning for a news article classification
using the [Kaggle news category dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset)

The notebook tests 3 different techniques: nearest centroid, nearest neighbor, and svm. 
The results show svm performing best.

One interesting result, the fit results are better when limiting to the most popular
(ie most frequently occurring) categories. You can change the category range 
selection by changing the minimum number of articles per category for the dataset.
This is done in get_categories at line 26 in data.py

`category_list = category_counts.loc[category_counts["headline"] >= 2000].index.to_list()`

Please note that svm is incredibly slow. It took ~14hrs on an M2 Max with 32G.

### Further Experiments

attempt different ml algorithms
use different sample size
