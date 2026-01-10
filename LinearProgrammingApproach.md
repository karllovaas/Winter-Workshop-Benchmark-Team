# Approach

This is maybe a potential solution using a linear programming approach. An index/benchmark is some linear combination of stocks that summed together provide a price or return for the index. I couldnt determine if the price/value of an index is strictly a sum of stocks, if it uses fractional stocks, or if it uses fractional stocks and some scalars. It might be better to use returns since that would get rid of the exact scalings? 

We can use the top 190-1010 stocks by market cap as our possible choices (maybe picking slightly outside of the bound  of top 201-1000 will help us?). 

We can start to add constraints on the system using the fact sheet. We know the top 10 stocks in the index 

| Company | Industry |
| --------| -------- |
| Howmet Aerospace Inc | Industrials | 
|Royal Caribbean Group | Consumer Discretionary|
|Hilton Worldwide Holding | Consumer Discretionary|
|Cencora Inc | Consumer Staples|
|Cloudflare | Technology|
|Vertiv Holdings Co (a) | Technology|
|Carvana Co | Consumer Discretionary|
|Vistra Corp | Utilities|
|Idexx Labs Inc | Health Care|
|Alnylam Pharmaceuticals | Health Care|

That gives us some constraints where $x_1 > x_2 > x_3  > ... > x_{10} > x_{11..281} $
We also know from the fact sheet we need 281 holdings 
We know median market cap is ~$14 billion
We know largest stock by market cap is ~$100
We know Shapre Ratios
We know have positive values for those stocks/variables.

We can place reasonable guesses/bounds using fundamentals data too.

Is it computationally feasbile to find a solution with 281 holdings out of ~800 choices and ~500 days of close values (2yrs)? 

If we did get solutions, could we use some analysis on the fundementals data to define our own inclusion/exclusion criteria?

Can we backtest using those criteria and compare performance/benchmarks?
