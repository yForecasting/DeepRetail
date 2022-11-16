










class Evaluator(object):

    def __init__(self, df, res_df, freq):


        self.res_df = res_df
        self.df = df
        self.freq = freq

        # Get the total number of fh and cv
        self.total_cv = len(self.res_df['cv'].unique())
        self.total_fh = len(self.res_df['fh'].unique())





    def evaluate(self, metrics):
        """Returns a df with the evaluation scores

        Args:
            metrics (metric-type): Imports from the metrics file
        """
        
        # for metric in metrics!
        # Edit the self.df in the right format for rmsse and mase and pass it!
        ...




    def evaluate_plot(self, metrics = None, eval_df = None):

        # Either metrics or eval_df needs to be passed
        # If metrics is passed then it generates the eval df
        # Otherwise it gets the passed eval_df
        # A plot with boxplots (for every model) for every metric!
        ...


    def add_forecast(self, new_fc):
        # adds a new forecast object!
        ...


    def plot_forecasts(self, models = None):
        # Plots for the passed models
        # Models should be included in the res_df 
