### Problem Description

# look at how the stock dynamics
# monthly stock prices of five companies: IBM, General Electric (GE), Procter and Gamble, Coca Cola and Boeing
# Infochimps로부터 데이터 획득.


### Variable Description

# Date: the date of the stock price, always given as the first of the month
# StockPrice: the average stock price of the company in the given month





### PROBLEM 1.1 - SUMMARY STATISTICS
IBM = read.csv("IBMStock.csv")
GE = read.csv("GEStock.csv")
CocaCola = read.csv("CocaColaStock.csv")
ProcterGamble = read.csv("ProcterGambleStock.csv")
Boeing = read.csv("BoeingStock.csv")

## Before working with these data sets, we need to convert the dates into a format that R can understand. Take a look at the structure of one of the datasets using the str function. Right now, the date variable is stored as a factor. We can convert this to a "Date" object in R by using the following five commands (one for each data set):
IBM$Date = as.Date(IBM$Date, "%m/%d/%y")
GE$Date = as.Date(GE$Date, "%m/%d/%y")
CocaCola$Date = as.Date(CocaCola$Date, "%m/%d/%y")
ProcterGamble$Date = as.Date(ProcterGamble$Date, "%m/%d/%y")
Boeing$Date = as.Date(Boeing$Date, "%m/%d/%y")
## The first argument to the as.Date function is the variable we want to convert, and the second argument is the format of the Date variable. We can just overwrite the original Date variable values with the output of this function. Now, answer the following questions using the str and summary functions.

str(IBM) # Our five datasets all have the same number of observations. How many observations are there in each data set?

### PROBLEM 1.2 - SUMMARY STATISTICS
summary(IBM) # What is the latest year in our datasets?, What is the earliest year in our datasets?, What is the mean stock price of IBM over this time period?
summary(GE) # What is the minimum stock price of General Electric (GE) over this time period?
summary(CocaCola) # What is the maximum stock price of Coca-Cola over this time period?
summary(Boeing) # What is the median stock price of Boeing over this time period?
sd(ProcterGamble$StockPrice) # What is the standard deviation of the stock price of Procter & Gamble over this time period?

### PROBLEM 2.1 - VISUALIZING STOCK DYNAMICS
## Let's plot the stock prices to see if we can visualize trends in stock prices during this time period. Using the plot function, plot the Date on the x-axis and the StockPrice on the y-axis, for Coca-Cola.
## This plots our observations as points, but we would really like to see a line instead, since this is a continuous time period. To do this, add the argument type="l" to your plot command, and re-generate the plot (the character is quotes is the letter l, for line). You should now see a line plot of the Coca-Cola stock price.
plot(CocaCola$Date, CocaCola$StockPrice) 
## Now, let's add the line for Procter & Gamble too. You can add a line to a plot in R by using the lines function instead of the plot function. Keeping the plot for Coca-Cola open, type in your R console:
lines(ProcterGamble$Date, ProcterGamble$StockPrice)
## Unfortunately, it's hard to tell which line is which. Let's fix this by giving each line a color. First, re-run the plot command for Coca-Cola, but add the argument col="red". You should see the plot for Coca-Cola show up again, but this time in red. Now, let's add the Procter & Gamble line (using the lines function like we did before), adding the argument col="blue". You should now see in your plot the Coca-Cola stock price in red, and the Procter & Gamble stock price in blue.
## As an alternative choice to changing the colors, you could instead change the line type of the Procter & Gamble line by adding the argument lty=2. This will make the Procter & Gamble line dashed.

# 2000년 3월에 technology bubble burst와 stock market crash가 발생함.

plot(CocaCola$Date, CocaCola$StockPrice, col="red")
lines(ProcterGamble$Date, ProcterGamble$StockPrice, col="blue")
abline(v=as.Date(c("2000-03-01")), lwd=2)

### PROBLEM 3.1 - VISUALIZING STOCK DYNAMICS 1995-2005

# 빨간색 line은 1995년부터 2005년까지의 301~432번째의 observations들의 CocaCola stock prices를 나타냄.
# 2000년 3월 technology bubble burst가 발생한 이후 가장 고꾸라진 회사는?
# 1995-2005 중 가장 높은 값을 찍은 회사는?

## Let's take a look at how the stock prices changed from 1995-2005 for all five companies. In your R console, start by typing the following plot command:
plot(CocaCola$Date[301:432], CocaCola$StockPrice[301:432], type="l", col="red", ylim=c(0,210))
## Now, use the lines function to add in the other four companies, remembering to only plot the observations from 1995 to 2005, or [301:432]. You don't need the "type" or "ylim" arguments for the lines function, but remember to make each company a different color so that you can tell them apart. Some color options are "red", "blue", "green", "purple", "orange", and "black". To see all of the color options in R, type colors() in your R console.
lines(ProcterGamble$Date, ProcterGamble$StockPrice, col="blue")
lines(GE$Date, GE$StockPrice, col="green")
lines(IBM$Date, IBM$StockPrice, col="purple")
lines(Boeing$Date, Boeing$StockPrice, col="black")
## (If you prefer to change the type of the line instead of the color, here are some options for changing the line type: lty=2 will make the line dashed, lty=3 will make the line dotted, lty=4 will make the line alternate between dashes and dots, and lty=5 will make the line long-dashed.)

# 1997, Oct에 global stock market crash가 발생. 아시아의 economic crisis에 의해.
# 1997, Oct이후 주식이 감소되는 회사는?

abline(v=as.Date(c("1997-10-01")), lwd=2)
abline(v=as.Date(c("1997-09-01")), lwd=2)
abline(v=as.Date(c("1997-11-01")), lwd=2)

### PROBLEM 4.1 - MONTHLY TRENDS
## Lastly, let's see if stocks tend to be higher or lower during certain months. Use the tapply command to calculate the mean stock price of IBM, sorted by months. To sort by months, use
tapply(IBM$StockPrice, months(IBM$Date), mean) # For IBM, compare the monthly averages to the overall average stock price. In which months has IBM historically had a higher stock price (on average)? Select all that apply.
summary(IBM$StockPrice)

tapply(GE$StockPrice, months(IBM$Date), mean)
tapply(CocaCola$StockPrice, months(IBM$Date), mean)
tapply(ProcterGamble$StockPrice, months(IBM$Date), mean)
tapply(Boeing$StockPrice, months(IBM$Date), mean)












