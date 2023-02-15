import pandas as pd
from DeepRetail.transformations.formats import transaction_df
from DeepRetail.data.casereaders import read_case_0


class Reader(object):
    """
    Reads data of different formats from the provided filepath.
    Converts data into a universal format used throught the package.

    """

    def __init__(self, filepath, calendar_filepath=None, case=0):
        """
        Initializes a Reader object.

        Args:
            filepath (str): Path to the file to be read.
            calendar_filepath (str, optional): Path to the calendar dataframe.
                Required for case 3. Defaults to None.
            case (int): Specifies the format of the data to be read.
                Defaults to 0.

        Raises:
            AssertionError: If case is not an integer.
        """

        assert type(case) == int, "case should be an integer"
        self.case = case
        self.filepath = filepath
        self.calendar_filepath = calendar_filepath

    def call_in_memory(
        self,
        filter_negatives=True,
        filter_negatives_strategy="constant",
        filter_negatives_constant=0,
        filter_missing=True,
        filter_missing_strategy="constant",
        filter_missing_constant=0,
    ):
        """
        Reads data from the provided filepath and stores it in memory.

        Args:
            filter_negatives (bool, optional): Whether to filter negative values.
                Defaults to True.
            filter_negatives_strategy (str, optional): Strategy to use for filtering
                negative values. Either "constant" or "drop". Defaults to "constant".
            filter_negatives_constant (float, optional): The constant to replace
                negative values with if the filter_negatives_strategy is "constant".
                Defaults to 0.
            filter_missing (bool, optional): Whether to filter missing values.
                Defaults to True.
            filter_missing_strategy (str, optional): Strategy to use for filtering
                missing values. Either "constant" or "drop". Defaults to "constant".
            filter_missing_constant (float, optional): The constant to replace missing
                values with if the filter_missing_strategy is "constant".
                Defaults to 0.
        """
        # Call an individual reading function for every case
        if self.case == 0:
            # for case 3 -> read the data (we require the calendar data here)
            if self.calendar_filepath is None:
                raise ValueError("Case 0 requires the calendar dataframe")
            # read
            temp_df = read_case_0(self.filepath, self.calendar_filepath)

        # filter negatives
        if filter_negatives:
            if filter_negatives_strategy == "constant":
                temp_df[temp_df < 0] = filter_negatives_constant
            elif filter_negatives_strategy == "drop":
                temp_df = temp_df[temp_df > 0]

        # filter missing
        if filter_missing:
            if filter_missing_strategy == "constant":
                temp_df = temp_df.fillna(filter_missing_constant)
            elif filter_missing_strategy == "drop":
                temp_df = temp_df.dropna()

        # convert columns to datetime
        temp_df.columns = pd.to_datetime(temp_df.columns)

        # add to the object
        self.temp_df = temp_df

    def save(self, save_filepath, frequency, format, **kwargs):
        """
        Saves the data in a specified format.

        Args:
            save_filepath (str): Path to the file to be saved.
            frequency (str): The selected frequency. Aligned with pandas frequency format:
                'PF' with P the number of periods and F the frequency. Examples, 'M', '2W',
                'D' for monthly, bi-weekly and daily.
            format (str): The format of the saved dataframe. Supports "pivoted" and
                "transaction".
            **kwargs: Additional keyword arguments to be passed to the transaction_df function.

        Raises:
            ValueError: If the format argument is not "pivoted" or "transaction".
        """
        # Add attributes
        self.save_filepath = save_filepath
        self.frequency = frequency
        self.format = format

        # Call the dataframe on memory
        self.call_in_memory()

        # fills missing values
        print("Loading completed. Saving.")

        # Converts in the right format
        if format == "pivoted":
            self.temp_df.to_csv(save_filepath)

        elif format == "transaction":
            # Make the conversion to transaction
            temp_df = transaction_df(self.temp_df, **kwargs)
            temp_df.to_csv(save_filepath)

        else:
            raise ValueError(
                "Define a proper output format. Supporting pivoted and transaction"
            )

    def load(self, frequency, format, calendar_filepath=None, **kwargs):
        """
        Loads the selected dataframe.

        Args:
            frequency (str): The selected frequency. Aligned with pandas frequency format:
                'PF' with P the number of periods and F the frequency. Examples, 'M', '2W',
                'D' for monthly, bi-weekly and daily.
            format (str): The format of the saved dataframe. Supports "pivoted" and
                "transaction".
            calendar_filepath (str, optional): Path to the calendar dataframe. Required for
                case 3. Defaults to None.
            **kwargs: Additional keyword arguments to be passed to the transaction_df function.

        Raises:
            ValueError: If the format argument is not "pivoted" or "transaction".

        Returns:
            pd.DataFrame: The loaded dataframe.
        """

        # Add some attributes
        self.frequency = frequency
        self.format = format

        # Call the dataframe on memory
        temp_df = self.call_in_memory()

        # fills missing values
        temp_df = temp_df.fillna(0)
        print("Loading completed.")

        # Converts in the right format
        if format == "transaction":
            # Make the conversion to transaction
            temp_df = transaction_df(temp_df, **kwargs)

        elif (format != "pivoted") & format != "transaction":
            raise ValueError(
                "Define a proper output format. Supporting pivoted and transaction"
            )

        return temp_df
