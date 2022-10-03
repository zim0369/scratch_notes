# Machine Learning

## Unit I: Intro to Machine Learning

### Introduction to Machine Learning

**Definition of Machine Learning:** Arthur Samuel, an early American leader in the field of computer gaming and artificial intelligence, coined the term “Machine Learning ” in 1959 while at IBM. He defined machine learning as “the field of study that gives computers the ability to learn without being explicitly programmed “. However, there is no universally accepted definition for machine learning. Different authors define the term differently. We give below two more definitions.

*   Machine learning is programming computers to optimize a performance criterion using example data or past experience . We have a model defined up to some parameters, and learning is the execution of a computer program to optimize the parameters of the model using the training data or past experience. The model may be predictive to make predictions in the future, or descriptive to gain knowledge from data.

*   The field of study known as machine learning is concerned with the question of how to construct computer programs that automatically improve with experience.

### Comparison of Machine learning with traditional programming

Traditional computer programming has been around for more than a century, with the first known computer program dating back to the mid 1800s. Traditional Programming refers to any manually created program that uses input data and runs on a computer to produce the output.

But for decades now, an advanced type of programming has revolutionized business, particularly in the areas of intelligence and embedded analytics. In Machine Learning programming, also known as augmented analytics, the input data and output are fed to an algorithm to create a program. This yields powerful insights that can be used to predict future outcomes.

**Traditional Programming**

Traditional programming is a manual process—meaning a person (programmer) creates the program. But without anyone programming the logic, one has to manually formulate or code rules.

*   `Input + Program = Output`

In machine learning, on the other hand, the algorithm automatically formulates the rules from the data.

**Machine Learning Programming**

Unlike traditional programming, machine learning is an automated process. It can increase the value of your embedded analytics in many areas, including data prep, natural language interfaces, automatic outlier detection, recommendations, and causality and significance detection. All of these features help speed user insights and reduce decision bias.

*   `Input + Output = Program `

For example, if you feed in customer demographics and transactions as input data and use historical customer churn rates as your output data, the algorithm will formulate a program that can predict if a customer will churn or not. That program is called a predictive model.

*   `Input[demographics + transactions] + Output[churned or not] = Program[churn model]`

You can use this model to predict business outcomes in any situation where you have input and historical output data:

1.  Identify the business question you would like to ask.
2.  Identify the historical input.
3.  Identify the historically observed output (i.e., data samples for when the condition is true and for when it’s false).

For instance, if you want to predict who will pay the bills late, identify the input (customer demographics, bills) and the output (pay late or not), and let the machine learning use this data to create your model.

*   `Input + Output = Program `

As you can see, machine learning can turn your business data into a financial asset. You can point the algorithm at your data so it can learn powerful rules that can be used to predict future outcomes. It’s no wonder predictive analytics is now the number one capability on product roadmaps.

### ML vs AI vs Data Science

<!-- TODO -->

### Types of learning

**Definition of learning:** A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P , if its performance at tasks T, as measured by P , improves with experience E.

*   **Examples**

*   Handwriting recognition learning problem
    *   Task T :  Recognizing and classifying handwritten words within images
    *   Performance P : Percent of words correctly classified
    *   Training experience E : A dataset of handwritten words with given classifications

*   A robot driving learning problem
    *   Task T : Driving on highways using vision sensors
    *   Performance P : Average distance traveled before an error
    *   Training experience E : A sequence of images and steering commands recorded while observing a human driver

**Definition**: A computer program which learns from experience is called a machine learning program or simply a learning program .

**Classification of Machine Learning**

Machine learning implementations are classified into four major categories, depending on the nature of the learning “signal” or “response” available to a learning system which are as follows:

#### Supervised

Supervised learning is the machine learning task of learning a function that maps an input to  an output based on example input-output pairs. The given data is labeled . Both *classification and regression problems* are supervised learning problems.

*   Example —  Consider the following data regarding patients entering a clinic . The data consists of the gender and age of the patients and each patient is labeled as “healthy” or “sick”.

| gender | age | label   |
| ------ | --- | ------- |
| M      | 48  | sick    |
| M      | 67  | sick    |
| F      | 53  | healthy |
| M      | 49  | sick    |
| F      | 32  | healthy |
| M      | 34  | healthy |
| M      | 21  | healthy |

#### Unsupervised

Unsupervised learning is a type of machine learning algorithm used to draw inferences from datasets consisting of input data without labeled responses. In unsupervised learning algorithms, classification or categorization is not included in the observations. Example: Consider the following data regarding patients entering a clinic. The data consists of the gender and age of the patients.

| gender | age |
| ------ | --- |
| M      | 48  |
| M      | 67  |
| F      | 53  |
| M      | 49  |
| F      | 34  |
| M      | 21  |

As a kind of learning, it resembles the methods humans use to figure out that certain objects or events are from the same class, such as by observing the degree of similarity between objects. Some recommendation systems that you find on the web in the form of marketing automation are based on this type of learning.

**Supervised vs. Unsupervised Machine Learning**

| Parameters               | Supervised machine learning                                                                  | Unsupervised machine learning                                        |
| ------------------------ | -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| Input Data               | Algorithms are trained using labeled data.                                                   | Algorithms are used against data that is not labeled                 |
| Computational Complexity | Simpler method                                                                               | Computationally complex                                              |
| Accuracy                 | Highly accurate                                                                              | Less accurate                                                        |
| No. of classes           | No. of classes is known                                                                      | No. of classes is not known                                          |
| Data Analysis            | Uses offline analysis                                                                        | Uses real-time analysis of data                                      |
| Algorithms used          | Linear and Logistics regression, Random forest, Support Vector Machine, Neural Network, etc. | K-Means clustering, Hierarchical clustering, Apriori algorithm, etc. |

[READ MORE](https://www.geeksforgeeks.org/supervised-unsupervised-learning/)

#### Semi-supervised

Semi-Supervised learning is a type of Machine Learning algorithm that represents the intermediate ground between Supervised and Unsupervised learning algorithms. It uses the combination of labeled and unlabeled datasets during the training period.

<details>
<summary>More info</summary>

Before understanding the Semi-Supervised learning, you should know the main categories of Machine Learning algorithms. Machine Learning consists of three main categories: Supervised Learning, Unsupervised Learning, and Reinforcement Learning. Further, the basic difference between Supervised and unsupervised learning is that supervised learning datasets consist of an output label training data associated with each tuple, and unsupervised datasets do not consist the same. Semi-supervised learning is an important category that lies between the Supervised and Unsupervised machine learning. Although Semi-supervised learning is the middle ground between supervised and unsupervised learning and operates on the data that consists of a few labels, it mostly consists of unlabeled data. As labels are costly, but for the corporate purpose, it may have few labels.

The basic disadvantage of supervised learning is that it requires hand-labeling by ML specialists or data scientists, and it also requires a high cost to process. Further unsupervised learning also has a limited spectrum for its applications. To overcome these drawbacks of supervised learning and unsupervised learning algorithms, the concept of Semi-supervised learning is introduced. In this algorithm, training data is a combination of both labeled and unlabeled data. However, labeled data exists with a very small amount while it consists of a huge amount of unlabeled data. Initially, similar data is clustered along with an unsupervised learning algorithm, and further, it helps to label the unlabeled data into labeled data. It is why label data is a comparatively, more expensive acquisition than unlabeled data.

[READ MORE](https://www.javatpoint.com/semi-supervised-learning)

</details>

#### Reinforcement learning techniques

### Models of Machine learning

#### Geometric model

#### Probabilistic Models

#### Logical Models

#### Grouping and grading models

#### Parametric and non-parametric models

### Important Elements of Machine Learning

#### Data formats

#### Learnability

#### Statistical learning approaches
