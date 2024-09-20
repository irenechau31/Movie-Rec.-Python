import pandas as pd
movies=pd.read_csv(r"C:\Users\User\OneDrive\Desktop\Coding Learning\Movies Rec\movies.csv")
movies


#clean movie titles with regex
import re #python regular expression lib
def clean_title(title):
   # write a function called clean_title, take the inputed title and clean it
   # cuz there are some character (brackets, etc.) -> hard to search the movies
   return re.sub("[^a-zA-Z0-9 ]","", title) #re.sub(A, B, C) | Replace A with B in the string C.
   #Matches characters from a to z, A to Z and also from 0 to 9.
   #look for characters aren't space, digits, letter to be removed from the title

#apply clean_title method to the title to create new col "clean title" in the df using []
movies["clean title"]=movies["title"].apply(clean_title)
movies

#BUILD SEARCH ENGINE
from sklearn.feature_extraction.text import TfidfVectorizer
#instead of look for toy, story, 1995, also looks for Toy Story, Story 1995
#vectorizer will consider both unigrams (single words) and bigrams (sequences of two adjacent words) when generating features.
vectorizer=TfidfVectorizer(ngram_range=(1,2))
# Fit and transform the documents
tfidf_matrix =vectorizer.fit_transform(movies["clean title"])

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def search(title):
  title=clean_title(title)
#The transform() method converts the text data into a numerical representation 
#based on the TF-IDF (Term Frequency-Inverse Document Frequency) of the TfidfVectorizer 
  query_vec=vectorizer.transform([title]) #Transform Title into Query Vector:

#Calculate the cosine similarity between the query vectorand TF-IDF matrix 
#The resulting similarity scores are flattened to a 1D array using .flatten().
  similarity=cosine_similarity(query_vec, tfidf_matrix).flatten()

#Find the most 5 title has largest similarity
#finds the indices that would partition/separate the similarity array in such a way that the 5 largest elements are moved to the last 5 positions 
#[-5:] selects the last 5 indices from the result
  indices=np.argpartition(similarity,-5)[-5:]

#[::-1] reverse the results to have the most similarity is in the first in the list
  results=movies.iloc[indices][::-1] #selects rows from the movies DataFrame using the indices obtained
  return results


#BUILDING AN INTERACTIVE WIDGET SEARCH BOX
import ipywidgets as widgets
from IPython.display import display

movie_input = widgets.Text(value = "Toy Story", description = "Movie Title", disabled = False)

#MAKE THE BOX USEFUL
movie_list=widgets.Output()
def on_type(data): # run the function anytime something is inputted into the box
  with movie_list: #the output widget
    movie_list.clear_output() #clear anything is alredy in the output widget
    title=data["new"] #take the title form the input
    if len(title) > 5: 
      display(search(title)) # display into the output widget movie_list

#hook up the input widget to the on_type function
#whenever we type sth in, it will fun the on_type method
movie_input.observe(on_type, names='value')

display(movie_input,movie_list)

ratings=pd.read_csv('ratings.csv')
ratings

ratings.dtypes

movie_id=1

# #find users who also liked the inputted movie, then find the other movies they liked
# #FIND USERS WHO LIKED THE INPUTTED MOVIE

# #ratings["movieId"] == movie_id filters the ratings DataFrame to include only the rows where the "movieId" column matches the specified movie_id.
# #ratings["ratings"] >= 5 further filters the DataFrame to include only the rows where the "ratings" column has a value of 5 or greater.
# #selects the "userId" column from the filtered DataFrame containing ratings for the inputted movie with ratings of 5 or higher
# #.unique() returns an array of unique user IDs who rated the inputted movie with a rating of 5 or higher.
# similar_users=ratings[(ratings["movieId"]==movie_id)&(ratings["rating"]>4)]["userId"].unique()
# similar_users

# #FIND THE OTHER MOVIES THEY ALSO LIKED
# similar_user_recs=ratings[(ratings["userId"].isin(similar_users))&(ratings["rating"]>4)]["movieId"]
# similar_user_recs


# #find the movies thas had greater than 10% compare to number of user who liked the inputted movie
# similar_user_recs = similar_user_recs.value_counts()/len(similar_users)
# similar_user_recs = similar_user_recs[similar_user_recs > 0.1]

# similar_user_recs

# #FIND THE MOVIES SIMILAR TO INPUTTED MOVIE FROM THE SET OF THE OTHER MOVIES THEY LIKED

# #FIND HOW MUCH ALL USERS LIKE THE MOVIE
# all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index))&(ratings["rating"]>4)]
# all_users

#Find % of all users like these recommended movies similar_user_recs
#find the movies thas had greater % in similar_user_recs than in all_user_recs
#we only want movies from people who has similar taste to you vs everybody

#EX: you liked Avengers, people who liked Avengers rated Thor 100%, but of all the users who watched Thor, only 40% liked

# #.unique() returns an array of unique user IDs
# all_user_recs = all_users["movieId"].value_counts()/len(all_users["userId"].unique())
# all_user_recs

# #CREATE A REC SCORE

# #Compare the %
# rec_percentages=pd.concat([similar_user_recs,all_user_recs], axis =1)
# rec_percentages

# rec_percentages.columns=["similar", "all"]
# rec_percentages

# #Create a score col
# rec_percentages["score"]=rec_percentages["similar"]/rec_percentages["all"]

# #Sort the score high->low
# rec_percentages=rec_percentages.sort_values("score", ascending=False) 
# rec_percentages

# #Take the top 10 rec, merge into movie data to get the movie title

# #left_index=True cuz in the rec_percentage dateset, the left index is the movieid
# #Then merge it on the right with the movie ID from the movies dataset
# rec_percentages.head(10).merge(movies, left_index=True, right_on = "movieId")

#BUILD A REC FUNCTION

def find_similar_movies(movie_id):
#Find  rec from users similar to us
#ratings["movieId"] == movie_id filters the ratings DataFrame to include only the rows where the "movieId" column matches the specified movie_id.
#ratings["ratings"] >= 5 further filters the DataFrame to include only the rows where the "ratings" column has a value of 5 or greater.
#selects the "userId" column from the filtered DataFrame containing ratings for the inputted movie with ratings of 5 or higher
#.unique() returns an array of unique user IDs who rated the inputted movie with a rating of 4 or higher.
  similar_users=ratings[(ratings["movieId"]==movie_id)&(ratings["rating"]>4)]["userId"].unique()
  similar_user_recs=ratings[(ratings["userId"].isin(similar_users))&(ratings["rating"]>4)]["movieId"]

#Adjust to obtain only recs whereover 10% of similar users recommended
  similar_user_recs = similar_user_recs.value_counts()/len(similar_users)
  similar_user_recs = similar_user_recs[similar_user_recs > 0.1]

#Find rec among all users
  all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index))&(ratings["rating"]>4)]
  all_user_recs = all_users["movieId"].value_counts()/len(all_users["userId"].unique())

  rec_percentages=pd.concat([similar_user_recs,all_user_recs], axis =1)
  rec_percentages.columns=["similar", "all"]

#Create a score col
  rec_percentages["score"]=rec_percentages["similar"]/rec_percentages["all"]

#Sort the score high->low
  rec_percentages=rec_percentages.sort_values("score", ascending=False) 

#Return the merge
  return rec_percentages.head(10).merge(movies, left_index=True, right_on = "movieId")[["score","title","genres"]]
  

#CREATE AN INTERACTIVE REC WIDGETS

movie_name_input = widgets.Text(value='Toy Story',description='Movie Title:',disabled=False)
#MAKE THE BOX USEFUL
recommendation_list=widgets.Output()
def on_type(data): # run the function anytime something is inputted into the box
  with recommendation_list: #the output widget
    recommendation_list.clear_output() #clear anything is alredy in the output widget
    title=data["new"] #take the title form the input
    if len(title) > 5: 
      results=search(title) # search the inputted title
      movie_id=results.iloc[0]["movieId"] #Extract movie Id, (the movie has highest percentage is in the 1st row)
      display(find_similar_movies(movie_id))

movie_name_input.observe(on_type, names='value')

display(movie_name_input,recommendation_list)

