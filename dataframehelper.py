from abc import abstractmethod, ABC 
import pandas as pd 
from typing import Iterable, List, Optional
from datetime import datetime
import numpy as np



class DataframeHelper:
    DATAOG: pd.DataFrame
    def __init__(self, data: pd.DataFrame) -> None:
        DataframeHelper.DATAOG = data 
        self.data = data 
    
    def slice_by_ticker(self,ticker: str) -> pd.DataFrame:
        ticker_mask = self.data.index.get_level_values("ticker") == ticker
        return self.data.loc[ticker_mask]

    def slice_by_day(self, day: datetime) -> pd.DataFrame: 
        day_mask = self.data.index.get_level_values("date") == day
        return self.data.loc[day_mask]
    
    def slice_any_row_with_na(self) -> pd.DataFrame:
        NA_mask = self.data.isna() 
        return self.data[NA_mask.any(axis=1)]
    
    def slice_complete_na_rows(self) -> pd.DataFrame: 
        #todo: 
        return pd.DataFrame()
    def slice_by_day_range(self, start_date: datetime, end_date: datetime) -> pd.DataFrame: 
        #todo 
        return pd.DataFrame() 
    
    @staticmethod
    def _rolling_growth_rate(price_to_sales_series : pd.Series) -> float:  
        lookback_days = 252 #one trading year 
        #see if there are NA values, there are growth is NA 
        NA_mask = price_to_sales_series.isna()
        if price_to_sales_series.loc[NA_mask].__len__() > 0:
            return np.nan
        # else slope of linear fit is growth rate
        x_arbitrary = range(price_to_sales_series.__len__())
        slope, intercept = np.polyfit(x_arbitrary, price_to_sales_series,1) 
        return slope*1000
        
    def add_roling_sales_growth_col(self):
        # add rolling sales growth , not very performant 
        self.data["1year_PtoS_growth"] = 0.0
        for ticker in self.data.index.get_level_values("ticker").unique():
            df_slice = self.slice_by_ticker(ticker)
            df_slice["rolling_growth"] = df_slice["price_to_sales"].rolling(252,min_periods= 252).apply(self._rolling_growth_rate)
            ticker_mask = df.index.get_level_values("ticker") == ticker
            self.data.loc[ticker_mask,"1year_PtoS_growth"] = df_slice["rolling_growth"]