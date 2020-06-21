# Introduction
Causality detection is a well known topic in the NLP and linguistic communities and has many applications in information retrieval. This shared task proposes data to experiment causality detection, and focuses on determining causality associated to an event. An event is defined as the arising or emergence of a new object or context in regard of a previous situation. So the task will emphasise the detection of causality associated with financial or economic analysis and resulting in a quantified output.

# Data Processing
The data are extracted from a corpus of 2019 financial news provided by QWAM. The original raw corpus is an ensemble of HTML pages corresponding to daily information retrieval from financial news feed. These news mostly inform on the 2019 financial landscape, but can also contain information related to politics, micro economics or other topic considered relevant for finance information. This raw set has been normalised as to ﬁt in the following format: Index; Text

# Task 1
Task 1 is a binary classification task. The dataset consists of a sample of text sections labeled with 1 if the text section is considered containing a causal relation, 0 otherwise. The dataset is by nature unbalanced, as to reﬂect the proportion of causal sentences extracted from the original news and SEC corpus, with provisional distribution approximately 5% 1 and 95% 0.

# Task 2
Task 2 is a relation extraction task. The text sections will correspond to the ones labeled as 1 in the Task 1 dataset, though for the purpose of results evaluation, they will not be exactly the same in the blind test set. The purpose of this task is to extract, in a causal text section, the sub-string identifying the causal elements and the sub-string describing the effects.

# Submission Details & Evaluation Criteria
We provide data sets for xxx-Task1 and xxx-Task2 respectively. (xxx = Trial, Practice, Evaluation)

Please note that you can only use the corresponding data set for xxx-Task1 to build models for Task1 and data set for xxx-Task2 to build models for Task2.

A valid submission zip file for CodaLab contains one of the following files:

task1.csv (directly zip it first and only submitted to "xxx-Task1" section)
task2.csv (directly zip it first and only submitted to "xxx-Task2" section)
Notes:

A .csv file with an incorrect file name (case is sensitive) will not be accepted.
A zip file containing both files will not be accepted.
Neither .csv nor .rar nor .7z file will be accepted, only .zip file is accepted.
Please zip your results file (e.g. task1.csv or task2.csv) directly without putting it into a folder and zipping the folder.
Submission format for xxx-Task1
The expected results would be provided by the participants in a csv ﬁle with headers: Index; Text; Prediction

The 'Index' must be in the same order as in 'xxx-task1.csv' for xxx-Task1.
The 'Prediction' must be labeled with '1' if the text section is considered containing a causal relation, 0 otherwise.
Submission example for xxx-Task1
Index; Text; Prediction
0001.00010; Nearly all of the victims had Latino last names.; 0
0001.00011; We have the highest child poverty rate of almost any country on Earth.; 0
...; ...; ...
Submission format for xxx-Task2
The expected results should be provided by the participants in a csv ﬁle with headers: Index; Text; Cause; Effect where Cause and Effect should be the sub-string identifying the causal elements and the sub-string describing the effects elements from a causal text section.

The 'Index' must be in the same order as in 'xxx-task2.csv' for xxx-Task2.
The 'Cause' should be a sub-string of the text section referencing the cause of an event (event or related object included)
The 'Effect' should be a sub-string referencing the effect of the cause
Submission example for xxx-Task2
Index; Text; Cause; Effect; Offset_Sentence2; Offset_Sentence3
0003.00117; Transat loss more than doubles as it works to complete Air Canada deal ; it works to complete Air Canada deal; Transat loss more than doubles; ;
...; ...; ...; ...; ...; ...
Note: the last 2 columns (sentence offsets) are not considered by the task 2 scoring program and can be omitted in the submission

## Evaluation Method
Participants can participate either to task 1 or 2, or both. The evaluation metrics that will be applied are:

### Task1: Precision, Recall, and F1
The evaluation script will verify whether the binary "Prediction" is the same as the desired "Prediction" which has been manually annotated, and then calculate its precision, recall, and F1 scores.

### Task2: Exact Match, Precision, Recall, and F1
Exact Match will represent what percentage of both your predicted cause and effect are exactly matched with the desired outcome that is annotated by human workers. F1 score is a token level metric and will be calculated according to the submitted cause, effect parts. Please refer to our baseline model for evaluation details.