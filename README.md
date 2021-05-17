# OkCupid [Dating a Scientist]

Here is what each file contains:
- **ok_cupid_cleaning_and_transformation1.ipynb** is PART 1 of data cleaning/ transformation/ EDA
- **ok_cupid_cleaning_and_transformation2.ipynb** is PART 2 of data cleaning/ transformation/ EDA
- **ok_cupid_models.ipynb** contains all machine learning models (clustering)

### This project is my capstone project for BrainStation's Data Science program. It explores the OkCupid dataset in which I try to answer if I can find meaningful clusters of users and answer the following questions:
1. Is there a lot of variety of people which OkCupid markets to its users or is everybody the same
2. Are there distinct groups on OkCupid in which the users can be separated into?

## Source of data
I extracted the OkCupid dataset from [Kaggle](https://www.kaggle.com/andrewmvd/okcupid-profiles) in CSV format. The single .csv file was used throughout my entire project. Age, income, and height were the only numeric features, and all others being type object.

## Missing Values
The OkCupid dataset had many rows and columns with missing values (i.e. null). The features income, and offspring were completely removed as they had over 59% missing values or values which were out of range (income: -1).

![null values](https://github.com/puneetsran/okcupid/blob/master/figures/null_values.png?raw=true)

1. `smokes`, `drinks`, `education`, `body_type`, and `drugs` were label encoded (i.e. doing drugs is 1, not doing drugs is 0)

2. `religion`, `speaks`, `diet`, and `sign` were transformed using CountVectorizer as well as some form of label encoding. All three of these features had either a level of seriousness or fluency which could not be ignored. For example, fluent in English is 3, poorly is 0. Seriousness about religion has a higher score versus not serious.
- `ethnicity`, all `essay` columns, and `pets` were also transformed using CountVectorizer (for pets only likes pet X, and has pet X were kept)
- For essays, the length of essay was also turned into a feature

![essay len](https://github.com/puneetsran/okcupid/blob/master/figures/essay_len.png?raw=true)

![ethnicities](https://github.com/puneetsran/okcupid/blob/master/figures/ethnicities.png?raw=true)

3. `status`, `orientation`, and `job` were transformed using OneHotEncoder
- `location` was split into `city`, `state`, and `country` and then one-hot-encoded

![map](https://github.com/puneetsran/okcupid/blob/master/figures/map.png?raw=true)

![relationship status](https://github.com/puneetsran/okcupid/blob/master/figures/status.png?raw=true)

![orientation](https://github.com/puneetsran/okcupid/blob/master/figures/orientaion.png?raw=true)

4. `last_online` was transformed into `days_since_last_online`

![days since last online](https://github.com/puneetsran/okcupid/blob/master/figures/days_since_last_online.png?raw=true)

Lastly, I was also able to extract missing values from the essay columns such as, if an essay contained the string ‘i am vegetarian’, then their diet was vegetarian. I applied this only if a user’s diet was unknown. Furthermore, some features had ‘rather not say’ as an option which I used if I could not find a users diet from the essay column. Since ‘rather not say’ is a valid response on OkCupid, it made sense to use it for some features.

## Modelling
For this project, the main modelling techniques were clustering while others (such as PLC, and TSNE) were simply for visualizations of different clusters. The main modelling techniques used were:

- KMeans clustering
- Agglomerative clustering
- DBSCAN

KMeans, Agglomerative, and DBSCAN are some of the main techniques we have learnt in this program and it made sense to try and use all for practice and to see if they would all indicate similar results. If they do indicate similar results, then perhaps I was on the right track.
For all clusterings, I used the following procedure:

1. Min-max Scale non-oneHotEncoded or non-transformed features (age, height, essay_len, and days_since_last_online), since clustering algorithms are distance based and features which are one-hot-encoded have a higher chance of not being represented in the final outcome.
2. Determine optimal number of clusters based on scree plots and silhouette scores (i.e. visual interpretation based on where the elbow lies).
3. Assign labels based on clusterings back into the data-frame, by creating new columns to hold those labels.
4. Analyze results based on:
5. Crosstab of both labels, adjusted_rand_score, and grouping by the labels

### Results
To summarize the results:
- All clusterings indicate 4 clusters based on where ‘elbow’ occurs (the purple line is drawn for reference):

![kmeans clusters](https://github.com/puneetsran/okcupid/blob/master/figures/k_means_scree_plot.png?raw=true)

![agg clusters](https://github.com/puneetsran/okcupid/blob/master/figures/agglomerative_scree_plot.png?raw=true)

![silhouette score kmeans](https://github.com/puneetsran/okcupid/blob/master/figures/silhouette_score_plot.png?raw=true)

![dbscan](https://github.com/puneetsran/okcupid/blob/master/figures/silhouette_scores.png?raw=true)

The adjusted_rand_score was 0.47 which means that the clusterings are in some agreement. The crosstab below illustrates that majority of the data falls within groups 1 and 2 (while also not being diagonal or cross-diagonal) which is not a bad thing but it would be ideal to have the data separated evenly across all labels. For reference 33.29% of data is in column 1 (row 1), while 28.86% is in column 2 (row 3).

![adj rand score](https://github.com/puneetsran/okcupid/blob/master/figures/adj_rand_score.png?raw=true)

Furthermore, a summary of the **labels** is as follows:
<p>

|                        | KMeans Groups |      |      |      | Agglomerative Groups |      |      |      |
|------------------------|:-------------:|:----:|:----:|:----:|:--------------------:|:----:|:----:|:----:|
|                        |       0       |   1  |   2  |   3  |           0          |   1  |   2  |   3  |
| white                  |      0.83     | 0.67 | 0.78 | 0.74 |         0.77         | 0.69 | 0.73 | 0.83 |
| anything_diet          |      0.91     | 0.83 | 0.92 | 0.87 |         0.92         |  0.8 | 0.91 | 0.95 |
| education              |      3.99     | 3.97 | 4.12 | 4.15 |         4.11         |  4.0 | 4.14 | 3.97 |
| smokes                 |      0.33     |  0.5 | 0.36 | 0.35 |         0.36         | 0.46 | 0.37 | 0.33 |
| agnosticism            |      0.61     | 0.45 | 0.61 | 0.53 |         0.59         | 0.43 | 0.58 | 0.64 |
| atheism                |      0.55     | 0.36 | 0.46 | 0.36 |         0.49         | 0.39 | 0.32 | 0.65 |
| catholicism            |      0.15     | 0.38 | 0.23 | 0.29 |          0.2         | 0.37 | 0.29 | 0.13 |
| christianity           |      0.23     | 0.55 | 0.33 | 0.43 |         0.31         | 0.52 | 0.45 |  0.2 |
| age                    |      2.9      | 2.94 | 2.69 | 3.14 |         2.69         | 3.31 | 2.74 | 2.69 |
| essay_len              |      1.04     | 0.14 | 0.56 | 0.34 |         0.59         | 0.17 | 0.35 | 0.94 |
| days_since_last_online |      0.85     |  1.5 | 0.87 | 0.81 |         0.73         | 1.79 | 0.48 | 0.81 |
</p>

From the summary table above, major differences can be seen in essay length, and days since last online. Overall, a higher score indicates likelihood of that feature (ex: 4.15 education indicates likelihood of user having higher education). Perhaps it can be said that:
- Users in KMeans group 0 write long essays, visits OkCupid more often (based on days since last online), and is the least into Catholicism (or not at all)
    - Perhaps this group is the most serious about finding a partner and hence they invest the time in telling as much as they can about themselves.
- Users in KMeans  group 3 are older, a little bit more into Catholicism
- Users in KMeans  Group 1 have not visited OkCupid for a long time, they write small essays, and are a little bit more into Catholicism
    - Perhaps this group has found a partner

While there might be **4** clusters, OkCupid does not have extremely distinct groups in which the users can be separated into (besides people that write long essays about themselves versus those that do not)

## Conclusion
In conclusion, this data offers limited insights in order to find meaningful clusters, and make meaningful recommendations. I was able to find clusters but they are certainly not meaningful. Besides `essay_len`, the difference in other features across various groups is not that distinct (i.e. `age` 2.9 vs 2.94). I think the fact that this data had to be transformed so drastically perhaps led to the results being less meaningful. Additionally, it can also be said that with so many missing values, the data itself was not great.

## Future outlook
If I am to go ahead with my clusterings and make recommendations, the only business application would be to group people based on essay length, days since user was last online, and religion. Perhaps I can help OkCupid determine if a user is serious about using the platform or if they have found a match based on days since last online. If a user has found a match from OkCupid and they have not used the app for a very long time, then perhaps it would not make sense for OkCupid to recommend that user to other users for potential matches. Furthermore, if a user has not visited the app for a long time and they have not found a match, then perhaps that user is not serious about finding a match in the first place.

Furthermore, the length of essay can indicate many different possibilities. Perhaps the user uses more words to express themselves or they prefer to not use too many words. Regardless, it is not likely to be used as a recommendation metric as preferences such as religion, orientation, age might be higher in the list.