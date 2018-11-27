### Problem Description

# July 2013 Pew Internet and American Life Project poll
# Internet anonymity and privacy Á¶»ç


### Variable Description

# Internet.Use: A binary variable indicating if the interviewee uses the Internet, at least occasionally (equals 1 if the interviewee uses the Internet, and equals 0 if the interviewee does not use the Internet).
# Smartphone: A binary variable indicating if the interviewee has a smartphone (equals 1 if they do have a smartphone, and equals 0 if they don't have a smartphone).
# Sex: Male or Female.
# Age: Age in years.
# State: State of residence of the interviewee.
# Region: Census region of the interviewee (Midwest, Northeast, South, or West).
# Conservativeness: Self-described level of conservativeness of interviewee, from 1 (very liberal) to 5 (very conservative).
# Info.On.Internet: Number of the following items this interviewee believes to be available on the Internet for others to see: (1) Their email address; (2) Their home address; (3) Their home phone number; (4) Their cell phone number; (5) The employer/company they work for; (6) Their political party or political affiliation; (7) Things they've written that have their name on it; (8) A photo of them; (9) A video of them; (10) Which groups or organizations they belong to; and (11) Their birth date.
# Worry.About.Info: A binary variable indicating if the interviewee worries about how much information is available about them on the Internet (equals 1 if they worry, and equals 0 if they don't worry).
# Privacy.Importance: A score from 0 (privacy is not too important) to 100 (privacy is very important), which combines the degree to which they find privacy important in the following: (1) The websites they browse; (2) Knowledge of the place they are located when they use the Internet; (3) The content and files they download; (4) The times of day they are online; (5) The applications or programs they use; (6) The searches they perform; (7) The content of their email; (8) The people they exchange email with; and (9) The content of their online chats or hangouts with others.
# Anonymity.Possible: A binary variable indicating if the interviewee thinks it's possible to use the Internet anonymously, meaning in such a way that online activities can't be traced back to them (equals 1 if he/she believes you can, and equals 0 if he/she believes you can't).
# Tried.Masking.Identity: A binary variable indicating if the interviewee has ever tried to mask his/her identity when using the Internet (equals 1 if he/she has tried to mask his/her identity, and equals 0 if he/she has not tried to mask his/her identity).
# Privacy.Laws.Effective: A binary variable indicating if the interviewee believes United States law provides reasonable privacy protection for Internet users (equals 1 if he/she believes it does, and equals 0 if he/she believes it doesn't).



### PROBLEM 1.1 - LOADING AND SUMMARIZING THE DATASET  
poll = read.csv("AnonymityPoll.csv")
summary(poll)
str(poll)
table(poll$Smartphone)
summary(poll$Smartphone)

## By using the table() function on two variables, we can tell how they are related. To use the table() function on two variables, just put the two variable names inside the parentheses, separated by a comma (don't forget to add poll$ before each variable name). In the output, the possible values of the first variable will be listed in the left, and the possible values of the second variable will be listed on the top. Each entry of the table counts the number of observations in the data set that have the value of the first value in that row, and the value of the second variable in that column. For example, suppose we want to create a table of the variables "Sex" and "Region". We would type
table(poll$Sex, poll$Region) # This table tells us that we have 123 people in our dataset who are female and from the Midwest, 116 people in our dataset who are male and from the Midwest, 90 people in our dataset who are female and from the Northeast, etc.
table(poll$State, poll$Region)


### PROBLEM 2.1 - INTERNET AND SMARTPHONE USERS  
## As mentioned in the introduction to this problem, many of the response variables (Info.On.Internet, Worry.About.Info, Privacy.Importance, Anonymity.Possible, and Tried.Masking.Identity) were not collected if an interviewee does not use the Internet or a smartphone, meaning the variables will have missing values for these interviewees.
table(poll$Smartphone, poll$Internet.Use)
summary(poll$Internet.Use)
summary(poll$Smartphone)

## Use the subset function to obtain a data frame called "limited", which is limited to interviewees who reported Internet use or who reported smartphone use. In lecture, we used the & symbol to use two criteria to make a subset of the data. To only take observations that have a certain value in one variable or the other, the | character can be used in place of the & symbol. This is also called a logical "or" operation.
limited = subset(poll, Internet.Use==1 | Smartphone==1)
str(limited)
nrow(limited) # How many interviewees are in the new data frame?


### PROBLEM 3.1 - SUMMARIZING OPINIONS ABOUT INTERNET PRIVACY
summary(limited)
mean(limited$Info.On.Internet) # What is the average number of pieces of personal information on the Internet, according to the Info.On.Internet variable?
summary(limited$Info.On.Internet)
str(limited$Info.On.Internet)
table(limited$Info.On.Internet) # How many interviewees reported a value of 0 for Info.On.Internet?, How many interviewees reported the maximum value of 11 for Info.On.Internet?

## What proportion of interviewees who answered the Worry.About.Info question worry about how much information is available about them on the Internet? Note that to compute this proportion you will be dividing by the number of people who answered the Worry.About.Info question, not the total number of people in the data frame.
table(limited$Worry.About.Info)
table(limited$Anonymity.Possible)
table(limited$Tried.Masking.Identity)
table(limited$Privacy.Laws.Effective)


### PROBLEM 4.1 - RELATING DEMOGRAPHICS TO POLLING RESULTS  
## Often, we are interested in whether certain characteristics of interviewees (e.g. their age or political opinions) affect their opinions on the topic of the poll (in this case, opinions on privacy). In this section, we will investigate the relationship between the characteristics Age and Smartphone and outcome variables Info.On.Internet and Tried.Masking.Identity, again using the limited data frame we built in an earlier section of this problem.
hist(limited$Age) # Build a histogram of the age of interviewees. What is the best represented age group in the population?
## Both Age and Info.On.Internet are variables that take on many values, so a good way to observe their relationship is through a graph. We learned in lecture that we can plot Age against Info.On.Internet with the command plot(limited$Age, limited$Info.On.Internet). However, because Info.On.Internet takes on a small number of values, multiple points can be plotted in exactly the same location on this graph.
max(table(limited$Age, limited$Info.On.Internet)) # What is the largest number of interviewees that have exactly the same value in their Age variable AND the same value in their Info.On.Internet variable?
plot(limited$Age, limited$Info.On.Internet) # In other words, what is the largest number of overlapping points in the plot plot(limited$Age, limited$Info.On.Internet)?

## To avoid points covering each other up, we can use the jitter() function on the values we pass to the plot function. Experimenting with the command jitter(c(1, 2, 3)), what appears to be the functionality of the jitter command?
jitter(c(1, 2, 3)) # jitter adds or subtracts a small amount of random noise to the values passed to it, and two runs will yield different results

plot(jitter(limited$Age), jitter(limited$Info.On.Internet)) # What relationship to you observe between Age and Info.On.Internet?
tapply(limited$Info.On.Internet, limited$Smartphone, summary) # What is the average Info.On.Internet value for smartphone users?, What is the average Info.On.Internet value for non-smartphone users?
tapply(limited$Tried.Masking.Identity, limited$Smartphone, table)



