
# SemesterProject

![image](https://github.com/Malvin-Maposa/SemesterProject/blob/main/Screenshot%20(35).png)

# Introduction

This project demonstrates the process of building a sentiment analysis system for product reviews. The dataset used contains 100,000 instances of reviews with various sentiments. We begin by preprocessing the data, including cleaning the text and mapping sentiments based on ratings. A sentiment analysis model is trained using logistic regression. Next, we create a user-friendly web API using Django, allowing users to input reviews and ratings for product sentiment prediction. The API employs the trained model to classify sentiment and provides real-time feedback. The entire project is containerized using Docker and deployed on Amazon Web Services (AWS) using Elastic Container Service (ECS) and Fargate. A continuous integration/continuous deployment (CI/CD) pipeline is set up with GitHub Actions to automate testing, building, and deployment processes, ensuring seamless project management.

# DataSet

The dataset is structured with six columns, each serving a specific purpose: 
- “review_id”: A unique identifier for each review
- “product_id”: The identifier of the sneaker product being reviewed
- “user_id”: The identifier of the user who provided the review.
- “rating”: The rating assigned to the sneaker by the user, indicating their satisfaction level.
- “review_text”: The rating assigned to the sneaker by the user, indicating their satisfaction level.
- “timestamp”: The timestamp indicating when the review was submitted.

This dataset is a valuable resource for various data analysis and machine learning tasks, including sentiment analysis, recommendation systems, and understanding customer feedback on sneaker products.
In addition to the dataset description, several tasks have been performed on this dataset, including data cleaning, feature engineering, and model training for sentiment analysis

# Requirements

- Django
- Djangorestframework
- ntlk
- scikit-learn

# How it works

Provided that django is already installed, and in the 'app' folder/directory.
You can accomplish the following:
- running 'python manage.py runserver' will run the html file before using docker or 'docker build -t image-name' and then 'docker-compose up' (outside 'app' directory)
- if running with docker, take note the url you cick will be '0.0.0.0:8000' simply change it to 'localhost:8000' to view the frontend
