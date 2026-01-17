
from dataframehelper import DataframeHelper

from abc import abstractmethod, ABC 
import pandas as pd 
from typing import Iterable, List, Optional
from datetime import datetime
import numpy as np

""" 
==========================================================
         Index Composition and Reconstitution Class
==========================================================
data. is our base dataframe 
"""
from dataframehelper import DataframeHelper


class IndexComposer:
    '''     
    Based on maria, filtering functions. Index Composer will work in conjuction with 
    the benchmark class. It's primary purpose is to take in a single day and
    and stock data universe as inputs and spit out subset of tickers that will constitute 
    the midcap growth index.  
    ''' 
    def __init__(self,data: pd.DataFrame, day: datetime, selection_method: str = "growth" ) -> None: 
        self.data: pd.DataFrame = data
        self._selection_method = selection_method
        self.df_helper = DataframeHelper(data) 
        self.day: datetime = day
        self.day_slice: pd.DataFrame = self.df_helper.slice_by_day(day)
        self._midcap_800: pd.DataFrame = self.get_midcap_800()
        self.growth_subset: pd.Series = self.growth_subset_filter_v1() #type: ignore 
        self.random_subset = self.random_subset_filter1() 
        self.growth_subset_weights = self.get_weights() 
        self.growth_subset_shares = self.get_share_count() 


        # ========================================
        #           CVS Class Attributes 
        # ========================================
        #self._cvs_price_to_book_median:float = self.cvs_price_to_book_median() 
        #self._cvs_price_to_book_spread: float = self.cvs
        self._cvs_book_to_price_median: int
        self._cvs_sales_to_price_median: int 
        self._cvs_book_to_price_spread: int
        self._cvs_sales_to_price_spread: int 
        self.cvs_subset_s = self.cvs_subset()

    def get_midcap_800(self) -> pd.DataFrame: 
        ''' This needs to be date based: we only want to drop NA values 
            for a given day
        '''
        # get all of the data for that day 
        df = self.day_slice 
        # filter only for stocks that are not NA. 
        df = df.dropna(subset = ["market_cap"]).copy()
        #sort from largest to smallest 
        df.sort_values(by = "market_cap", ascending= False, inplace= True)
        # filter out the largest 200 stocks 
        df = df.iloc[199:,]
        return df


    def compute_growth_probability(self, k=5.0) -> pd.Series:
        '''
        Computing Growth Probability

        Since right now we only have P/B data, we can approximate growth
        classification using inverse price-to-book (B/P) mapped into a 
        smooth probability via a logistic function.
        '''
        midcap = self._midcap_800
        pb = midcap["price_to_book"].astype(float)

        #convert P/B to B/P since Russell uses B/P
        bp = np.where((pb > 0 ) & np.isfinite(pb), 1.0 / pb, np.nan)
        bp = pd.Series(bp, index=midcap.index)
        bp = bp.fillna(bp.median()) # to fill empty values (for later computation)

        z = (bp -bp.mean()) / (bp.std(ddof=0) + 1e-12) # z-score for standardization
        z_growth = -z # low B/P -> growth, so now high z_growth -> more growthlike

        #normalizing
        #very neg z_growth -> 0 (value), very pos z_growth -> 1 (growth)
        p = 1 / (1 + np.exp(-k * z_growth)) 

        return pd.Series(p, index=midcap.index, name="p_growth")
    
    def growth_subset_filter_v1(self) -> pd.Series:
        growth_subset_mask = self.compute_growth_probability() > .85
        growth_subset = self._midcap_800.loc[growth_subset_mask]
        return pd.Series(growth_subset.index.get_level_values("ticker"))
    
    def get_weights(self) -> pd.Series: 
        df = self.day_slice
        df = df.reset_index() 
        df = df.set_index("ticker")
        if self._selection_method == "growth":
            subset = self.growth_subset
        if self._selection_method == "random":
            subset = self.random_subset
        if self._selection_method == "cvs":
            subset = self.cvs_subset() 
        df = df.loc[subset]
        total_cap = df["market_cap"].sum() 
        return df["market_cap"] / total_cap
    
    def get_share_count(self) -> pd.Series: 
        df = self.day_slice
        df = df.reset_index() 
        df = df.set_index("ticker")
        if self._selection_method == "growth":
            subset = self.growth_subset
        if self._selection_method == "random":
            subset = self.random_subset
        if self._selection_method == "cvs":
            subset = self.cvs_subset() 
        df = df.loc[subset]
        return df["market_cap"] / df["close_price"]

    # ===============================================
    #     Methods for Creating a Random Portfolio 
    # ===============================================

    def random_subset_filter1(self) ->pd.Series:
        df_day = self._midcap_800.copy()
        random_selection = np.random.randint(0,self._midcap_800.__len__(),300)
        df_day = df_day.reset_index()
        subset = df_day.loc[random_selection]
        return subset["ticker"]
    # ==============================================
    #       CVS Calculation Methods: 
    # ==============================================
    # (1): append to the midcap_800 dataframe the cvs score stuff 
    # (2): then once we have the score we can filter out for the stocks 
    # (3): that are needed. 
     
    def _get_book_to_price_rank(self, ticker: str) -> float: 
        df = self._midcap_800.copy()
        df = df.reset_index()
        na_mask = df["price_to_book"].isna()
        # set NA values to zero #todo: something to do later probably.         df = df.loc[na_mask]
        df = df.loc[~na_mask]
        price_to_book_s = pd.Series(
                data = (df["price_to_book"]).to_list(),
                index = df["ticker"])  # small means growth stock
        price_to_book_s = price_to_book_s.sort_values(ascending=False)       
        self._cvs_book_to_price_median = price_to_book_s.__len__() // 2
        self._cvs_book_to_price_spread = price_to_book_s.__len__()
        #print(price_to_book_s)
        try: 
            out = price_to_book_s.index.get_loc(ticker)
        except: 
            out = np.nan 
        return out # type: ignore 
    
    def _get_price_to_sales_rank(self, ticker: str) -> float: 
        df = self._midcap_800.copy() 
        df = df.reset_index()
        na_mask = df["price_to_sales"].isna()
        df = df.loc[~na_mask]
        # set NA values to zero #todo: something to do later probably.         df = df.loc[na_mask]
        price_to_sales_s = pd.Series(
                data = (df["price_to_sales"]**-1).to_list(),
                index = df["ticker"])  # small means growth stock
        price_to_sales_s = price_to_sales_s.sort_values(ascending= False)       
        self._cvs_sales_to_price_median = price_to_sales_s.__len__() // 2 
        self._cvs_sales_to_price_spread = price_to_sales_s.__len__()
        #print(price_to_sales_s)
        try: 
            out = price_to_sales_s.index.get_loc(ticker)
        except: 
            out = np.nan 
        return out 

    def create_cvs_dataframe(self) -> pd.DataFrame:  
        df = self._midcap_800.copy() 
        df = df.reset_index() 
        # this will probably error out on NA data, might have to just remove the NA data entirely. 
        df["book_to_price_rank"] = df["ticker"].apply(self._get_book_to_price_rank)
        df["cvs_book_to_price"] = (df["book_to_price_rank"] - self._cvs_book_to_price_median) / self._cvs_book_to_price_spread
        df["sales_to_price_rank"] = df["ticker"].apply(self._get_price_to_sales_rank)
        df["cvs_sales_to_price"] = (df["book_to_price_rank"] - self._cvs_sales_to_price_median) / self._cvs_sales_to_price_spread
        df["cvs_score"] = df["cvs_book_to_price"]*.5 + df["cvs_sales_to_price"]*.5
        return df
    
    def cvs_subset(self):
        df = self.create_cvs_dataframe() 
        df.set_index("ticker",inplace= True) 
        cvs_sorted = df["cvs_score"].sort_values().iloc[0:282]
        return pd.Series(cvs_sorted.index)

class Benchmark:
    """_summary_
    """
    def __init__(self, data: pd.DataFrame, benchmark_start_val: float): 
        self.data: pd.DataFrame = data
        self.cur_constituents: pd.Series #this is all of current member of the index. 
        self.dates: pd.Series = pd.Series(self.data.index.get_level_values("date").unique())
        self.cur_date: datetime  = self.dates[0]
        self._i: int = 0 
        self.bechmark_timeseries = pd.Series(data = 0.0, index = self.dates)
        self.benchmark_divisor: float = 1
        self.quarterly_recon_dates: List[datetime] = []
        self._get_quarterly_reconst_dates()
        self.annual_recon_dates = self._get_annual_reconstitution_dates() 
        self.benchmark_makeup_dict: dict = {}
        self.cur_divisor: float = 0.0
        self.DataHelper = DataframeHelper(self.data)

    #=========================================================
    #          Benchmark Constitution Methods
    #========================================================
    def calculate_benchmark(self, cap_weighted: bool = False, random: bool = False): 
        # set the benchmark constitution on the first day of the calculation
        composer = IndexComposer(self.data, self.cur_date)
        self.cur_constituents = composer.growth_subset
        self.benchmark_makeup_dict[self.cur_date] = composer.growth_subset
        
        # calculate benchmark for the first day
        print(f"calculating benchmark for the first day {self.cur_date}")
        day_df = composer.day_slice
        day_df = day_df.reset_index() 
        day_df.set_index("ticker",inplace= True)
        day_df = day_df.loc[composer.growth_subset]

        # add columns to day df that we need for calculating the cap adjusted benchmark 
        day_df["weights"] = composer.growth_subset_weights
        day_df["shares"] = composer.growth_subset_shares
        day_df["weight_adjusted_cap"] = day_df["weights"] * day_df["shares"] * day_df["close_price"]
        
        cap_weighted_benchmark_numerator = day_df["weight_adjusted_cap"].sum() 
        self.cur_divisor = cap_weighted_benchmark_numerator / 1839.00  # hardcoded for now #todo will updated soon 
        first_benchmark_price = cap_weighted_benchmark_numerator / self.cur_divisor
        self.bechmark_timeseries.loc[self.cur_date] = first_benchmark_price #type: ignore

        while self.next_date() is not None:
            if self.cur_date in self.annual_recon_dates:
                print(f"{self.cur_date} is an reconsitution date" )
                composer = IndexComposer(self.data,self.cur_date)
                self.benchmark_makeup_dict[self.cur_date] = composer.growth_subset
                # when there is recomposition we'll need to rescale the divisor, we'll take the new bench mark
                # constitution, weights and share counts and calculate what divisor makes its such that the 
                # new constitution equals the previous constitutions benchmark value
                prev_date = self.dates[self._i - 2]

                # add columns to day df that we need for calculating the cap adjusted benchmark 
                prev_day_df = self.DataHelper.slice_by_day(prev_date)
                prev_day_df = prev_day_df.reset_index() 
                prev_day_df.set_index("ticker",inplace= True)
                prev_day_df = prev_day_df.loc[composer.growth_subset]
                prev_day_df["weights"] = composer.growth_subset_weights
                prev_day_df["shares"] = composer.growth_subset_shares
                prev_day_df["weight_adjusted_cap"] = prev_day_df["weights"] * prev_day_df["shares"] * prev_day_df["close_price"]
                cap_weighted_benchmark_numerator = prev_day_df["weight_adjusted_cap"].sum()

                self.cur_divisor =  cap_weighted_benchmark_numerator / self.bechmark_timeseries.loc[prev_date]
                print(f" cur_divisor is {self.cur_divisor},  self.bechmark_timeseries.loc[prev_date])")

            if not cap_weighted: 
                # add columns to day df that we need for calculating the cap adjusted benchmark 
                day_df = self.DataHelper.slice_by_day(self.cur_date)
                day_df.reset_index(inplace= True)
                day_df.set_index("ticker",inplace= True)
                day_df = day_df.loc[composer.growth_subset]
                day_df["weights"] = composer.growth_subset_weights
                day_df["shares"] = composer.growth_subset_shares
                day_df["weight_adjusted_cap"] = day_df["weights"] * day_df["shares"] * day_df["close_price"]
                cap_weighted_benchmark_numerator = day_df["weight_adjusted_cap"].sum() 
                benchmark_price = cap_weighted_benchmark_numerator / self.cur_divisor
                self.bechmark_timeseries.loc[self.cur_date] = benchmark_price #type: ignore
            else: 
                day_df = self.DataHelper.slice_by_day(self.cur_date)
                day_df.reset_index(inplace= True)
                day_df.set_index("ticker",inplace= True)
                day_df = day_df.loc[composer.growth_subset]
                day_df["weights"] = composer.growth_subset_weights
                day_df["weight_adjusted_cap"] = day_df["weights"] * day_df["market_cap"]
                cap_weighted_benchmark_numerator = day_df["weight_adjusted_cap"].sum() 
                benchmark_price = cap_weighted_benchmark_numerator / self.cur_divisor
                self.bechmark_timeseries.loc[self.cur_date] = benchmark_price #type: ignore

    def calculate_benchmark_random(self, cap_weighted: bool = False, random: bool = False): 
        #todo: to be refactored.... and deleted. temp solution

        composer = IndexComposer(self.data, self.cur_date,selection_method= "random")
        self.cur_constituents = composer.random_subset
        self.benchmark_makeup_dict[self.cur_date] = composer.random_subset
        
        # calculate benchmark for the first day
        print(f"calculating benchmark for the first day {self.cur_date}")
        day_df = composer.day_slice
        day_df = day_df.reset_index() 
        day_df.set_index("ticker",inplace= True)
        day_df = day_df.loc[composer.random_subset]

        # add columns to day df that we need for calculating the cap adjusted benchmark 
        day_df["weights"] = composer.growth_subset_weights # weights are actually random weights here 
        day_df["shares"] = composer.growth_subset_shares   # weights are random wrights actually 
        day_df["weight_adjusted_cap"] = day_df["weights"] * day_df["shares"] * day_df["close_price"]
        
        cap_weighted_benchmark_numerator = day_df["weight_adjusted_cap"].sum() 
        self.cur_divisor = cap_weighted_benchmark_numerator / 1839.00  # hardcoded for now #todo will updated soon 
        first_benchmark_price = cap_weighted_benchmark_numerator / self.cur_divisor
        self.bechmark_timeseries.loc[self.cur_date] = first_benchmark_price #type: ignore

        while self.next_date() is not None:
            if self.cur_date in self.annual_recon_dates:
                print(f"{self.cur_date} is an reconsitution date" )
                composer = IndexComposer(self.data,self.cur_date, selection_method= "random")
                self.benchmark_makeup_dict[self.cur_date] = composer.random_subset
                # when there is recomposition we'll need to rescale the divisor, we'll take the new bench mark
                # constitution, weights and share counts and calculate what divisor makes its such that the 
                # new constitution equals the previous constitutions benchmark value
                prev_date = self.dates[self._i - 2]

                # add columns to day df that we need for calculating the cap adjusted benchmark 
                prev_day_df = self.DataHelper.slice_by_day(prev_date)
                prev_day_df = prev_day_df.reset_index() 
                prev_day_df.set_index("ticker",inplace= True)
                prev_day_df = prev_day_df.loc[composer.random_subset]
                prev_day_df["weights"] = composer.growth_subset_weights
                prev_day_df["shares"] = composer.growth_subset_shares
                prev_day_df["weight_adjusted_cap"] = prev_day_df["weights"] * prev_day_df["shares"] * prev_day_df["close_price"]
                cap_weighted_benchmark_numerator = prev_day_df["weight_adjusted_cap"].sum()

                self.cur_divisor =  cap_weighted_benchmark_numerator / self.bechmark_timeseries.loc[prev_date]
                print(f" cur_divisor is {self.cur_divisor},  self.bechmark_timeseries.loc[prev_date])")

            if not cap_weighted: 
                # add columns to day df that we need for calculating the cap adjusted benchmark 
                day_df = self.DataHelper.slice_by_day(self.cur_date)
                day_df.reset_index(inplace= True)
                day_df.set_index("ticker",inplace= True)
                day_df = day_df.loc[composer.random_subset]
                day_df["weights"] = composer.growth_subset_weights
                day_df["shares"] = composer.growth_subset_shares
                day_df["weight_adjusted_cap"] = day_df["weights"] * day_df["shares"] * day_df["close_price"]
                cap_weighted_benchmark_numerator = day_df["weight_adjusted_cap"].sum() 
                benchmark_price = cap_weighted_benchmark_numerator / self.cur_divisor
                self.bechmark_timeseries.loc[self.cur_date] = benchmark_price #type: ignore
            else: 
                day_df = self.DataHelper.slice_by_day(self.cur_date)
                day_df.reset_index(inplace= True)
                day_df.set_index("ticker",inplace= True)
                day_df = day_df.loc[composer.growth_subset]
                day_df["weights"] = composer.growth_subset_weights
                day_df["weight_adjusted_cap"] = day_df["weights"] * day_df["market_cap"]
                cap_weighted_benchmark_numerator = day_df["weight_adjusted_cap"].sum() 
                benchmark_price = cap_weighted_benchmark_numerator / self.cur_divisor
                self.bechmark_timeseries.loc[self.cur_date] = benchmark_price #type: ignore

    def calculate_benchmark_cvs(self, cap_weighted: bool = False, random: bool = False): 
        #todo: to be refactored.... and deleted. temp solution

        composer = IndexComposer(self.data, self.cur_date,selection_method= "cvs")
        self.cur_constituents = composer.cvs_subset()
        self.benchmark_makeup_dict[self.cur_date] = composer.cvs_subset_s
        
        # calculate benchmark for the first day
        print(f"calculating benchmark for the first day {self.cur_date}")
        day_df = composer.day_slice
        day_df = day_df.reset_index() 
        day_df.set_index("ticker",inplace= True)
        day_df = day_df.loc[composer.cvs_subset()]

        # add columns to day df that we need for calculating the cap adjusted benchmark 
        day_df["weights"] = composer.growth_subset_weights # weights are actually random weights here 
        day_df["shares"] = composer.growth_subset_shares   # weights are random wrights actually 
        day_df["weight_adjusted_cap"] = day_df["weights"] * day_df["shares"] * day_df["close_price"]
        
        cap_weighted_benchmark_numerator = day_df["weight_adjusted_cap"].sum() 
        self.cur_divisor = cap_weighted_benchmark_numerator / 1839.00  # hardcoded for now #todo will updated soon 
        first_benchmark_price = cap_weighted_benchmark_numerator / self.cur_divisor
        self.bechmark_timeseries.loc[self.cur_date] = first_benchmark_price #type: ignore

        while self.next_date() is not None:
            if self.cur_date in self.annual_recon_dates:
                print(f"{self.cur_date} is an reconsitution date" )
                composer = IndexComposer(self.data,self.cur_date, selection_method= "cvs")
                self.benchmark_makeup_dict[self.cur_date] = composer.cvs_subset() 
                # when there is recomposition we'll need to rescale the divisor, we'll take the new bench mark
                # constitution, weights and share counts and calculate what divisor makes its such that the 
                # new constitution equals the previous constitutions benchmark value
                prev_date = self.dates[self._i - 2]

                # add columns to day df that we need for calculating the cap adjusted benchmark 
                prev_day_df = self.DataHelper.slice_by_day(prev_date)
                prev_day_df = prev_day_df.reset_index() 
                prev_day_df.set_index("ticker",inplace= True)
                prev_day_df = prev_day_df.loc[composer.cvs_subset()]
                prev_day_df["weights"] = composer.growth_subset_weights
                prev_day_df["shares"] = composer.growth_subset_shares
                prev_day_df["weight_adjusted_cap"] = prev_day_df["weights"] * prev_day_df["shares"] * prev_day_df["close_price"]
                cap_weighted_benchmark_numerator = prev_day_df["weight_adjusted_cap"].sum()

                self.cur_divisor =  cap_weighted_benchmark_numerator / self.bechmark_timeseries.loc[prev_date]
                print(f" cur_divisor is {self.cur_divisor},  self.bechmark_timeseries.loc[prev_date])")

            if not cap_weighted: 
                # add columns to day df that we need for calculating the cap adjusted benchmark 
                day_df = self.DataHelper.slice_by_day(self.cur_date)
                day_df.reset_index(inplace= True)
                day_df.set_index("ticker",inplace= True)
                day_df = day_df.loc[composer.cvs_subset_s]
                day_df["weights"] = composer.growth_subset_weights
                day_df["shares"] = composer.growth_subset_shares
                day_df["weight_adjusted_cap"] = day_df["weights"] * day_df["shares"] * day_df["close_price"]
                cap_weighted_benchmark_numerator = day_df["weight_adjusted_cap"].sum() 
                benchmark_price = cap_weighted_benchmark_numerator / self.cur_divisor
                self.bechmark_timeseries.loc[self.cur_date] = benchmark_price #type: ignore
            else: 
                day_df = self.DataHelper.slice_by_day(self.cur_date)
                day_df.reset_index(inplace= True)
                day_df.set_index("ticker",inplace= True)
                day_df = day_df.loc[composer.cvs_subset_s]
                day_df["weights"] = composer.growth_subset_weights
                day_df["weight_adjusted_cap"] = day_df["weights"] * day_df["market_cap"]
                cap_weighted_benchmark_numerator = day_df["weight_adjusted_cap"].sum() 
                benchmark_price = cap_weighted_benchmark_numerator / self.cur_divisor
                self.bechmark_timeseries.loc[self.cur_date] = benchmark_price #type: ignore
    

    #========================================================
    #           Date Handling Methods: 
    #========================================================
    
    def _get_quarterly_reconst_dates(self) -> None: 
        """
        Last trading of the months January, April, July, October
        """
        s_dates = self.dates
        l_dates: list[datetime] = self.dates.to_list() 
        for year in range(l_dates[0].year, l_dates[-1].year + 1):
            for month in [1,4,7,10]:
                year_month_mask = (s_dates.dt.year == year) &  (s_dates.dt.month == month) #type: ignore 
                self.quarterly_recon_dates += [s_dates[year_month_mask].iloc[-1]]

    def _get_annual_reconstitution_dates(self) -> List[datetime]:
        """Maria: method"""
        dates = self.data.index.get_level_values("date").unique()
        recon = []
        years = pd.DatetimeIndex(dates).year.unique()
        for year in years:
            june = [d for d in dates if d.year == year and d.month == 6]
            if not june:
                continue
            fridays = [d for d in june if pd.Timestamp(d).weekday() == 4] # Monday=0 ... Friday=4
            if not fridays:
                continue

            fridays = sorted(fridays)
            if len(fridays) >= 4:
                recon.append(fridays[3]) # 4th Friday (0-indexed)
            else:
                recon.append(fridays[-1]) # fallback: last Friday available
        return recon


    def is_reconstitution_date(self) -> bool:  
        out = False
        annual_recon_dates = self._get_annual_reconstitution_dates()
        if self.cur_date in annual_recon_dates: 
            return True
        if self.cur_date in self.quarterly_recon_dates: 
            return True 
        return False 

    def set_cur_date(self, day: datetime):
        if day not in self.dates.to_list(): 
            raise ValueError("Invalid Day Selected") 
        self.cur_date = day
        self._i = self.dates.to_list().index(day)
        

    def next_date(self) -> Optional[datetime]:
        if self._i < self.dates.__len__(): 
            self.cur_date = self.dates[self._i]
            self._i += 1 
            return self.cur_date
        else:
            return None 
    



if __name__ == "__main__":

    df = pd.read_csv("data_with_growth_v1.0.csv")

    #set date to datetime object 
    df["date"] = pd.to_datetime(df["date"])
    df.set_index(["date","ticker"],inplace= True)
    pd.set_option('display.float_format', '{:,.4f}'.format)

    #bench = Benchmark(df, 1000)
    #print(bench.calculate_benchmark())
    #print(bench.bechmark_timeseries)
    #bench.calculate_benchmark_random() 
    #print(bench.bechmark_timeseries)
    IC  = IndexComposer(df,datetime(2022,1,3),selection_method="cvs")
    
    # test out some of the CVS methods quickly. 
    #print(IC._get_price_to_sales_rank("SMMT"))
    #print(IC._get_book_to_price_rank("BHF")) 
    #print(IC.create_cvs_dataframe()["cvs_score"].sort_values()) 
    #print(IC.cvs_subset().iloc[0:280])
    
    bench = Benchmark(df, 1000)
    print(bench.calculate_benchmark_cvs())
    print(bench.bechmark_timeseries)


