# IMDB-Movies-Data-Mining-using-Clusetering
I used cosine similarity to identify similar movies based on their IMDB plot

## Introduction
Although it is called ‘the seventh art,’ cinema is the most powerful art form [1].  Many people watch a movie every day or at least every week. Therefore, it has a great effect on their lives. Throughout history, movies have been used for many purposes, including 1) Advertising different products, 2) expanding people’s knowledge, 3) Introducing a person to the world, etc. 
Drama, Comedy, and Sci-fi are the most popular genres among people. Although there are many available movies in each of these genres, one can hardly find similar films to watch. In this regard, many websites are now attempting to classify movies based on different features to help people find their most favorite movies. Some prefer to watch their movies directory-based, while some select their films based on the genre and plot summaries.  To address this issue, the authors attempted to suggest a method to identify similar movies by data mining on available datasets.   
In this project, two datasets have been used. One from the IMDB movies website and the other one from Wikipedia. These datasets are sounds interesting to the authors for two apparent reasons. First, they are publicly available. Second, almost all the previous studies on these datasets limits to data visualization and not data mining.

## Data Explanation
The IMDB dataset is driven from the Kaggle website. This dataset has been scraped from the publicly available IMDB website . It consists of all the movies with more than 100 votes until 17/11/2019. Our first dataset consists of more than 81,000 titles of movies from all around the world with different languages. Each row of this dataset is a data point, and the columns are their attributes. Moreover, another complementary dataset which is scraped from Wikipedia is used in this project. This data set contains plot descriptions of 34,886 different movies.

 ![image](https://user-images.githubusercontent.com/73087167/185809789-0d7a042d-5eb2-48f1-a2ab-ce6c23cbf5b7.png)
Figure 1- The process of preparing data

## Results (Part I)
## Recommend Movies Based on Similar Plots
In this section, we have used an exciting data mining approach that is capable of searching for similar plots based on user input. The user can insert his favorite movie as four inputs, and based on these inputs, the model tries to find similar movies. These four inputs are as follows: 1) his favorite movie name, 2) specific year of movie production, 3) movie rating threshold, and 4) the number of movies to suggest. For example, they can say that “I like movies similar to Inception movie from 2010, and I would like the model to suggest me top 20 movies with a rating above 8.0.”
The concept of our model is that it finds the most frequent words used in all of the movie plots, removes the stop words, chooses most frequent words, and creates a vector based on those words (the model creates Bag of Words for each plot). Then, it iterates through each plot and searches for that word. If that word exists in the plot, it assigns one. Otherwise, it assigns zero in the specific index of that word. Using the approach, it creates movie vectors for each movie, which contains values of one and zero. Finally, the model uses cosine similarity to find the most similar movies to our favorite movie. In the following sections, we are going to explain the detail of the model developed using Python programming language.
## Reading Data
First, our algorithm reads both datasets: 1) IMDB movie dataset with more than 81,000 movie titles and 2) Wikipedia Movie Plots with more than 34,000 movie titles. These two datasets are then merged together to have a single dataset. So, the final dataset consists of about 34,000 movie titles.
## Pre-processing
In this step, the data needs to be cleaned and organized. First, we remove duplicates and rows with empty values, so every row and column has values. Also, we iterate through each plot, convert the plot to lower case, and then remove the punctuation and empty spaces from the text.
## Finding Frequent Words
In this step, the algorithm loops over the plots and create Bag of Words for each movie. Using these bag of words for each plot, the algorithm again loops over the plots and find the most frequent words along with their frequency counts. The important point in this step is removing stop words. These words are usually the most common in any English language text, so they are not distinctive. In general, we are more interested in finding the words that will help us differentiate one plot from other plots that are about different subjects or genres. The output of this step is the frequent words with their frequency. In Figure 10, you can see the top 20 frequent words used int 34,000 movie plots in Wikipedia.
![image](https://user-images.githubusercontent.com/73087167/185809808-1bb42539-13fc-49a0-82e7-adbf61b32805.png)
Figure 2- Top 20 frequent words used int 34,000 movie plots in Wikipedia

## Creating Movie Vectors
In this step, our algorithm loops over the plots and determine if the plot contains any of the frequent words or not. The dimension of the movie vector can vary, and as the dimension increases, the accuracy of the movie suggestion increase. In other words, as more frequent words are used to create the movie vector, the accuracy of the model increases. In Figure 11, the movies’ vector for the top 20 frequent words is shown.
![image](https://user-images.githubusercontent.com/73087167/185809817-0f33aa30-914b-46e8-adb9-06ec2cd3738b.png)
Figure 3- Movies’ vector for top 20 frequent words

Finally, the algorithm loops over the data and finds n number of films with the closest cosine distance. The model suggests movies based on the top cosine similarity. In Table 1, the 20 suggested movie titles based on the movie name “Interstellar 2014”, and suggested movies later than 1930 and rating above 2.0 is shown. 
![image](https://user-images.githubusercontent.com/73087167/185809839-6bca11fe-2131-45a0-a7dd-50fc45f6f737.png)


## Results (Part II)
To reach our goal, we tried multiple methods of data mining, including Bag Of Words, most frequent items, correlation matrix, etc. In the following, the works that have been done will be elaborated, and the results will be presented.
As one of our first steps, we tried to find the most frequent genres among the movies. Figure 2 depicts the top 50 genres of movies in the IMDB dataset. 
 ![image](https://user-images.githubusercontent.com/73087167/185809873-b8186708-5ad7-499d-856c-703cf8555141.png)
Figure 4- Most frequent genres of movies in IMDB dataset
We also find the most frequent words in movies description. Figure 3 shows the results.

![image](https://user-images.githubusercontent.com/73087167/185809877-34e97c07-43b8-4f44-a2c7-d64e4bb9ea2d.png)
Figure 5- Most frequent words in movies descriptions
As can be seen in figure 3, almost all of the listed terms are the stopwords. Therefore, we used the “nltk” library of python to omit the stop words and other useless characters from movies description. Figure 4 shows the results of the analysis. 
![image](https://user-images.githubusercontent.com/73087167/185809882-af9b3848-67aa-4bdc-b1c7-c9e7c3c94eed.png)
Figure 6- Most frequent terms except stopwords
Finally, by using machine learning techniques, we trained a model to predict the genre of movies based on their description. Because of the great diversity in movie descriptions, the model has a low accuracy of around 50%. Figures 5 through 7 are some of the results of the model prediction.
 ![image](https://user-images.githubusercontent.com/73087167/185809887-267a3aa0-787c-4a33-81c0-ebf624a30b3f.png)
Figure 7- Model prediction 1

![image](https://user-images.githubusercontent.com/73087167/185809889-037ed275-6177-4420-8ede-dbb46213edc5.png)
![image](https://user-images.githubusercontent.com/73087167/185809892-6efd4cad-1bea-4bbf-98c3-2a825dceeb71.png)

Figure 8- Model prediction 2
![image](https://user-images.githubusercontent.com/73087167/185809897-edca4af0-ac0f-420f-900e-75c762c062b9.png)
Figure 9- Model prediction 3

Another method that we have tried on the dataset is the heat map, which shows the correlation between different features of the dataset. 
![image](https://user-images.githubusercontent.com/73087167/185809900-8a4f1332-7f19-4570-9a91-43ee7e3f1d09.png)

Figure 10- Heat map of the dataset
To figure out the relation of the production year and the average rate of each movie, we implemented a marginal plot analysis, as shown in Figure 9.
![Uploading image.png…]()

Figure 11- Marginal plot of the average rate of films based on their production year
As it is depicted in figure 9, most movies are rated around 6 to 7. Moreover, newer movies have a higher average IMDB rate.

## Conclusion and Findings
## Part I
As predicted, most of the movies have the same genre as “Interstellar,” which is Adventure, Drama, and Sci-Fi. This model has been generated using the top 3000 frequent words. To get better results, this number can be increased, but it has a negative impact on computational time. Furthermore, it seems that as we increase the dimension of the movie vectors, the cosine similarity decrease. Also, to evaluate the results, k-grams and other distance/similarity functions such as Lp distance and Mahalanobis distance can be used.
## Part II
As expected, most movies have a rating of around 6 and 7. Moreover, newer movies have higher scores due to multiple reasons like availability, production quality, actors and etc. One of the key findings of this project is that, if the descriptions of the movies are written in a more general way, it is possible to suggest similar movies to a person with a specific taste of movies. By “gerenal way,” the author means that the description only narrates the storyline and excluding the name of actors and proper nouns.








