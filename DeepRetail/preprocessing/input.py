# Reads the given data

# read all case specific function
from DeepRetail.preprocessing.casespecific import (
    read_case_1,
    read_case_2,
    read_case_3,
    read_case_4,
    read_case_5,
    read_case_6,
)
from DeepRetail.preprocessing.converters import (
    pivoted_df,
    fix_duplicate_cols,
    transaction_df,
)

import pandas as pd


class Reader(object):
    def __init__(self, case, filepath):
        """Reads data from the different companies on the provided filepath.
        Each case requires different preprocessing.

        Args:
            case (int): The unique number of the case.
            filepath (str): Location of the data file
        """

        # Assert case is an integer between 1 and 6
        assert type(case) == int, "case should be an integer between 1 and 6"
        assert case > 0, "case should be an integer between 1 and 6"
        assert case < 7, "case should be an integer between 1 and 6"

        self.case = case
        self.filepath = filepath

    def call_in_memory(self):

        """
        Reads the file on the given filepath on memory.
         Then saves it at the provided filepath

        """
        # An individual function for each case study
        if self.case == 1:
            print("Start")
            # For case 1 -> we first read the data in the folder
            temp_df = read_case_1(
                self.filepath, self.save_filepath, self.frequency, self.temporary_save
            )
            # converts columns to str to fix a bug
            temp_df.columns = pd.to_datetime(temp_df.columns).astype(str)

            # Fix duplicates
            temp_df = fix_duplicate_cols(temp_df)
            print("End")
        elif self.case == 2:

            # For case 2 -> read
            temp_df = read_case_2(self.filepath)

            # pivot and resample
            temp_df = pivoted_df(
                df=temp_df,
                target_frequency=self.frequency,
                agg_func="sum",
                fill_values=True,
            )

        elif self.case == 3:

            # for case 3 -> read the data (we require the calendar data here)
            if self.calendar_filepath is None:
                raise ValueError("Case 3 requires the calendar dataframe")
            # read
            temp_df = read_case_3(self.filepath, self.calendar_filepath)

            # Resample
            temp_df = temp_df.resample(self.frequency, axis=1).sum()

        elif self.case == 4:

            # read data
            temp_df = read_case_4(self.filepath)

            # pivot and resample
            temp_df = pivoted_df(
                df=temp_df,
                target_frequency=self.frequency,
                agg_func="sum",
                fill_values=True,
            )

        elif self.case == 5:

            # read
            temp_df = read_case_5(self.filepath)

            # pivot and resample
            temp_df = pivoted_df(
                df=temp_df,
                target_frequency=self.frequency,
                agg_func="sum",
                fill_values=True,
            )

        elif self.case == 6:

            temp_df = read_case_6(self.filepath)

            # pivot and resample
            temp_df = pivoted_df(
                df=temp_df,
                target_frequency=self.frequency,
                agg_func="sum",
                fill_values=True,
            )

        # fills missing values
        temp_df = temp_df.fillna(0)
        temp_df.columns = pd.to_datetime(temp_df.columns)

        # Deals with negative values
        temp_df[temp_df < 0] = 0

        # includes
        self.dataset = temp_df

    def save(
        self,
        save_filepath,
        frequency,
        format,
        temporary_save=False,
        calendar_filepath=None,
        **kwargs
    ):
        """Converts the read data in the selected dataframe format.
         Then saves it at the provided filepath

        Args:
            save_filepath (str): Location for saving the dataframe
            frequency (str): The selected frequency.
                            Aligned with pandas frequency format:
                            'PF' with P the number of periods and F the frequency.
                            Examples, 'M', '2W', 'D' for monthly, bi-weekly and daily
            format (str): The format of the saved dataframe.
                        Supports pivoted, and transaction
            temporary_save (bool, Optional): If true it saves the dataframe on chunks
                                         Deals with memory breaks.
            calendar_filepath(str): Location of the calendar dataframe.
                                    Required for case 3
        """

        # Add some attributes
        self.save_filepath = save_filepath
        self.frequency = frequency
        self.format = format
        self.temporary_save = temporary_save
        self.calendar_filepath = calendar_filepath

        # Call the dataframe on memory
        self.call_in_memory()

        # fills missing values

        temp_df = self.dataset.fillna(0)
        print("Loading completed. Saving.")

        # Converts in the right format
        if format == "pivoted":
            temp_df.to_csv(save_filepath)

        elif format == "transaction":
            # Make the conversion to transaction
            temp_df = transaction_df(temp_df, **kwargs)
            temp_df.to_csv(save_filepath)

        else:
            raise ValueError(
                "Define a proper output format. Supporting pivoted and transaction"
            )

    def load(self, frequency, format, calendar_filepath=None, **kwargs):
        """Loads the selected dataframe

        Args:
            frequency (str): The selected frequency.
                            Aligned with pandas frequency format:
                            'PF' with P the number of periods and F the frequency.
                            Examples, 'M', '2W', 'D' for monthly, bi-weekly and daily
            format (str): The format of the saved dataframe.
                        Supports pivoted, and transaction
            calendar_filepath(str): Location of the calendar dataframe.
                                    Required for case 3
        """

        # Add some attributes
        self.frequency = frequency
        self.format = format
        self.calendar_filepath = calendar_filepath

        # Call the dataframe on memory
        temp_df = self.call_in_memory()

        # fills missing values
        temp_df = temp_df.fillna(0)
        print("Loading completed. Saving.")

        # Converts in the right format
        if format == "transaction":
            # Make the conversion to transaction
            temp_df = transaction_df(temp_df, **kwargs)

        elif (format != "pivoted") & format != "transaction":
            raise ValueError(
                "Define a proper output format. Supporting pivoted and transaction"
            )

        return temp_df
