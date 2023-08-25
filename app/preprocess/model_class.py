import pandas as pd
import numpy as np

class Models:
    """This Class is for using the complexity and time estimation model 
    """
    def __init__(self, complexity_model, time_model):
        """_summary_

        Args:
            complexity_model (Pretrained RandomForest Model): Pretrained RandomForest Model for Complexity Model
            time_model (Pretrained RandomForest Model): Pretrained RandomForest Model for Time Estimation Model
        """
        self.complexity_model = complexity_model
        self.time_model = time_model
        
    def _get_complexity_df(self, dev_type:str, dev_category:str) -> pd.DataFrame:
        """internal function that is used to create the complexity df to be used within the 
           complexity ML model

        Args:
            dev_type (str): development type category. 
            dev_category (str): development type category 

        Returns:
            pd.DataFrame: returns a dataframe with one example to be used in the ml model.
        """
        columns = ['Back-end Development', 'Front-end Development',
           'Infrastructure / Devops', 'Accessibility',
           'Content Management and E-Commerce', 'Custom API Development',
           'Custom CSS/HTML Styling', 'Custom Functionality',
           'Database Management', 'Domain Management', 'Email Setup',
           'Form Development', 'Hosting Management', 'Other',
           'Page Builder Development', 'Performance Optimization',
           'Plugin Development', 'Responsive Design', 'Security',
           'Security Enhancement']

        df = pd.DataFrame(0.0, index=range(1), columns=columns)

        if dev_category not in columns:
            dev_category = 'Other'
        for item in columns:
            if (dev_type == item):
                df.loc[0,item] = 1
            if (dev_category == item):
                df.loc[0,item] = 1

        return df
    
    def complexity_predict(self, dev_type:str, dev_category:str) -> float:
        """Predicts the complexity of a task

        Args:
            dev_type (str): development type category. 
            dev_category (str): development type category 

        Returns:
            float: complexity float between 0-5
        """
        df = self._get_complexity_df(dev_type, dev_category)
        predict = self.complexity_model.predict(df)
        return predict[0]
    
    def _get_time_df(self, dev_type: str, dev_category:str , complexity:float) -> pd.DataFrame:
        """internal function that is used to create the time estimation df to be used within the 
           time estimation ML model

        Args:
            dev_type (str): development type category. 
            dev_category (str): development type category 
            complexity (float): task complexity float

        Returns:
            pd.DataFrame: returns a dataframe with one example to be used in the ml model.
        """

        columns = ['Back-end Development', 'Front-end Development',
           'Infrastructure / Devops', 'Accessibility',
           'Content Management and E-Commerce', 'Custom API Development',
           'Custom CSS/HTML Styling', 'Custom Functionality',
           'Database Management', 'Domain Management', 'Email Setup',
           'Form Development', 'Hosting Management', 'Other',
           'Page Builder Development', 'Performance Optimization',
           'Plugin Development', 'Responsive Design', 'Security',
           'Security Enhancement', 'complexity']
        df = pd.DataFrame(0.0, index=range(1), columns=columns)


        if dev_category not in columns:
            dev_category = 'Other'
        for item in columns:
            if (dev_type == item):
                df.loc[0,item] = 1
            if (dev_category == item):
                df.loc[0,item] = 1
        df.loc[0,'complexity'] = complexity

        return df
    
    def _convert_exp(self, prediction: float) -> float:
        """ Internal function that is used to reverse the log function used in the ML model for time estimate.
            This is used to get the time estimation predicted value to the right units
        Args:
            prediction (float): time estimate prediction

        Returns:
            float: returns the exponential value of the time estimate prediction
        """
        return np.expm1(prediction)
    
    def _min_max_scaler(self, x: float, minimum: int = 0, maximum:int  = 5) -> float:
        """Transform features by scaling each feature to a given range.
           This estimator scales and translates each feature individually such that it is in the given range on the training set.
           The transformation is between 0 - 1. 

           Tranformation given by :
           X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
           X_scaled = X_std * (max - min) + min


        Args:
            x (float): x to me converted
            minimum (int, optional): minimum of the X values range. Defaults to 0.
            maximum (int, optional): maximum of the X values range. Defaults to 5.

        Returns:
            float: transformed value between 0 - 1
        """
        range = maximum - minimum
        std = (x - minimum) / range
        maxx = 1
        minn = 0
        x_scaled = std * (maxx - minn) + minn
        return x_scaled

    
    def time_predict(self, dev_type: str, dev_category: str, complexity: float) -> float:
        """time estimation function that estimated the task time to completion.

        Args:
            dev_type (str): development type category. 
            dev_category (str): development type category 
            complexity (float): task complexity float

        Returns:
            float: time estimation 
        """

        scaled_complexity = self._min_max_scaler(complexity)
        df = self._get_time_df(dev_type, dev_category, scaled_complexity)
        predict = self.time_model.predict(df)
        return self._convert_exp(predict[0]) / 60
    
    
    def returnModel(self):
        return self.complexity_model