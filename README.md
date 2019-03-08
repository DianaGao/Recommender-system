# Recommender-system
The social network recommender system can recommend the new friend to the platform users on the basis of user's common friends and common interests.

## Theory background
### Triadic closure 
Triadic closure is a concept in social network theory, first suggested by German sociologist Georg Simmel in the early 1900s. Triadic closure is the property among three nodes A, B, and C, such that if a strong tie exists between A-B and A-C, there is a weak or strong tie between B-C. 

In the example below, B and C follow the user A, regardless of whether user A follows back, user B and user C are the second-degree friends of user A.

![](Triadic%20closure.png)

## Normalization
Add a "weight" (normalization) to the similarity score reference the Jaccard coefficient.

### Jaccard Coefficient
![](Jaccard%20Coeffcient.png)

### User similarity score normalization
![](Interests%20rate%20normalization.png)

The user similarity score is calculated by dividing the number of “common friends” that both user i and j follow by the number of friends that both users follow.

Put the total number of following into the consideration, which means the score of the user would punishment the users following too many users. 
