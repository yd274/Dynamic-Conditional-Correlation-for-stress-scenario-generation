This project contains the class of Dynamic Conditional Correlation (DCC) (Engle, 2002)
and its application to stress scenario generation. The idea is straightforward. The essence 
of linear regression lies in the covariance and correlation between response variable and predictor.
This unconditional covariance and correlation captures the linear relations. However, financial returns
are known to exhibit non-linear relations, namely, the time-varying variance-covariance matrix. You can
think of this as regime changing. Naturally speaking, we do not expect that financial variables to behave
similarly during economic downturn and calm periods. Therefore, linear regression cannot capture this 
non-linearity stemming from regime changing.

A perfect example fo this is equity and interest rate. During a normal recession, equity typically goes down
and central banks slash interest rate to stimulate the economy. Whereas in a stagflationary scenario, equity
still goes down, but interest rates normally remain high or even increase to combat inflation.
As a result, there are periods when correlation is positive while other time, negative. This makes
linear regression almost powerless in predicting interest rate movement using equities.

In this project, the _DCC_class.py_ contains the class definition, including the loglikelihood function,
estimated variance-covariance matrix, as well as forecasting and conditional forecasting. Conditional 
forecasting means that given the realization of some of the variables, what would be expected value
of the other variables, which is essential in stress testing

_Utilities.py_ contains functions that call the DCC class, estimate the parameters and then make predictions
based on user defined input variable

_Demo.ipynb_ gives a simple demonstrate applying DCC to forecast US 10Y Treasury Yield using S&P 500 during
different regime periods

Requirements:

numpy == 2.1.3

scipy == 1.15.3

arch == 8.0.0

pandas == 2.3.1

statsmodels == 0.14.5
