import numpy as np
import pandas as pd
import random
from numpy import array,identity,diagonal
from math import sqrt
import tkinter as tk


def top_cosine_similarity(topk_eigenvecs, movie_id, top_n):
    index = movie_id - 1 # Movie id starts from 1
    movie_row = topk_eigenvecs[index, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', topk_eigenvecs, topk_eigenvecs))
    similarity = np.dot(movie_row, topk_eigenvecs.T) / (magnitude[index] * magnitude)
    sort_indexes = np.argsort(-similarity)
    return sort_indexes[:top_n]

# Helper function to print top N similar movies
def print_similar_movies(movie_data, movie_id, top_indexes):
    recommendations = []
    for id in top_indexes + 1:
        recommendations.append(movie_data[movie_data.movie_id == id].title.values[0])
    return recommendations

def get_recommendations():
    movie_id = int(movie_id_entry.get())
    k = 10
    top_n = 5
   
    topk_eigenvecs = jacobi_eigen_vectors[:, :k]
    top_indexes = top_cosine_similarity(topk_eigenvecs, movie_id, top_n)
    recommendations = print_similar_movies(movie_data, movie_id, top_indexes)
   
    # Clear previous recommendations
    for widget in recommendations_frame.winfo_children():
        widget.destroy()
   
    # Display new recommendations
    for i, recommendation in enumerate(recommendations):
        label = tk.Label(recommendations_frame, text=f"{i+1}. {recommendation}", font=("Helvetica", 12))
        label.pack(pady=5)


# Load data
movie_data = pd.io.parsers.read_csv('C:/Users/babus/Downloads/movies.dat',
    names=['movie_id', 'title', 'genre'],
    engine='python', delimiter='::')

data = pd.DataFrame(columns = ['user_id', 'movie_id', 'rating', 'time'])

for i in range(200):
    userid = random.randint(1,15)
    movieid = random.randint(1,100)
    rating = random.randint(1,5)
    time = random.randint(30,150)
    row = [userid, movieid, rating, time]
    data.loc[len(data)] = row

# Data preprocessing
data.dropna(inplace=True)
data = data.astype('int64')
ratings = np.ndarray(shape=(np.max(data.movie_id.values), np.max(data.user_id.values)),dtype=np.uint8)
ratings[data.movie_id.values-1, data.user_id.values-1] = data.rating.values
normalized_ratings = ratings - np.asarray([(np.mean(ratings, axis=1))]).T
covariance = np.cov(normalized_ratings)

# Eigen pairs by Jacobi method
def Jacobi(a, limit):
    def maxElement(a):
        n = len(a)
        max_ = 0.0
        for i in range(n-1):
            for j in range(i+1, n):
                if abs(a[i,j]) >= max_:
                    max_ = abs(a[i,j])
                    k = i
                    l = j
        return max_, k, l

    def rotate(A, p, k, l):
        n = len(a)
        diff = a[l,l] - a[k,k]
        if abs(a[k,l]) < abs(diff)*1.0e-36:
            t = a[k,l]/diff
        else:
            phi = diff/(2.0*a[k,l])
            t = 1.0/(abs(phi) + sqrt(phi**2 + 1.0))
            if phi < 0.0:
                t = -t
        c = 1.0/sqrt(t**2 + 1.0)
        s = t*c
        tau = s/(1.0 + c)
        temp = a[k,l]
        a[k,l] = 0.0
        a[k,k] = a[k,k] - t*temp
        a[l,l] = a[l,l] + t*temp
        for i in range(k):
            temp = a[i,k]
            a[i,k] = temp - s*(a[i,l] + tau*temp)
            a[i,l] = a[i,l] + s*(temp - tau*a[i,l])
        for i in range(k+1, l):
            temp = a[k,i]
            a[k,i] = temp - s*(a[i,l] + tau*a[k,i])
            a[i,l] = a[i,l] + s*(temp - tau*a[i,l])
        for i in range(l+1, n):
            temp = a[k,i]
            a[k,i] = temp - s*(a[l,i] + tau*temp)
            a[l,i] = a[l,i] + s*(temp - tau*a[l,i])

        for i in range(n):
            temp = p[i,k]
            p[i,k] = temp - s*(p[i,l] + tau*p[i,k])
            p[i,l] = p[i,l] + s*(temp - tau*p[i,l])

    n = len(a)
    max_rotations = 5*(n**2)
    p = identity(n)*1.0
    for i in range(max_rotations):
        max_, k, l = maxElement(a)
        if max_ < limit:
            return diagonal(a), p
        rotate(a, p, k, l)
    print('Jacobi method did not converge')

jacobi_eigen_values, jacobi_eigen_vectors = Jacobi(covariance, limit=1.0e-9)
jacobi_eigen_values = list(jacobi_eigen_values)
jacobi_eigen_values.sort(reverse=True)
jacobi_eigen_values[0:10]

# Create GUI
root = tk.Tk()
root.title("Movie Recommender")

# Movie ID Entry
movie_id_label = tk.Label(root, text="Enter Movie ID:", font=("Helvetica", 16))
movie_id_label.pack(pady=10)

movie_id_entry = tk.Entry(root, font=("Helvetica", 16))
movie_id_entry.pack(pady=10)

# Get Recommendations Button
get_recommendations_button = tk.Button(root, text="Get Recommendations", font=("Helvetica", 16), command=get_recommendations)
get_recommendations_button.pack(pady=10)

# Recommendations Frame
recommendations_frame = tk.Frame(root)
recommendations_frame.pack(pady=10)

root.mainloop()


