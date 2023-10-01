from django.db import models

# Create your models here.
class Review(models.Model):
    review_text = models.TextField()
    rating = models.IntegerField()