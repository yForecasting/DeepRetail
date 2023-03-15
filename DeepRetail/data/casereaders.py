import pandas as pd
import numpy as np
import gc
import datetime
import re


def read_case_0(read_filepath, calendar_filepath):
    """Reads data for case 0

    Args:
        read_filepath (str): Existing location of the data file.
        calendar_filepath (str): Existing location of the calendar file.
            Required for reading.

    Returns:
        pandas.DataFrame: A dataframe with the loaded data.

    Example usage:
    >>> df = read_case_0('data.csv', 'calendar.csv')

    """

    # read the data file and the calendar
    df = pd.read_csv(read_filepath)
    calendar = pd.read_csv(calendar_filepath)

    # Drop some columns
    # Hierarchy is defined as:
    # State -> Store -> Category -> Department -> Item
    to_drop = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    df = df.drop(to_drop, axis=1)

    # Modify the id and set it as index
    df["id"] = ["_".join(d.split("_")[:-1]) for d in df["id"].values]
    df = df.rename(columns={"id": "unique_id"})
    df = df.set_index("unique_id")

    # Prepare the dates from the calendar
    dates = calendar[["d", "date"]]

    # find the total days
    total_days = df.shape[1]
    dates = dates.iloc[:total_days]["date"].values

    # Replace on the columns
    df.columns = dates

    # Convert to datetime
    df.columns = pd.to_datetime(df.columns)

    # drop columns with only zeros
    df = df.loc[~(df == 0).all(axis=1)]

    return df


def read_case_1(read_filepath, write_filepath, frequency, temporary_save):
    """Reads data for case 1

    Args:
        read_filepath (str): Existing loocation of the data file
        write_filepath (str): Location to save the new file
        frequency (str): The selected frequency.
                    Note: Due to size issues, for case 1 only supports W and M
        temporary_save (bool, Optional): If true it saves the dataframe on chunks
                                         Deals with memory breaks.
    """
    # Initialize parameters to ensure stable loading

    chunksize = 10**6
    dict_dtypes = {
        "CENTRALE": "category",
        "FILIAAL": "category",
        "ALDIARTNR": "category",
        "ST": np.float32,
        "VRD": np.float16,
    }

    # Initialize the reading itterator
    tp = pd.read_csv(
        read_filepath,
        iterator=True,
        chunksize=chunksize,
        sep=";",
        dtype=dict_dtypes,
        parse_dates=["DATUM"],
        infer_datetime_format=True,
        decimal=",",
    )

    # Drop stock column for now
    df = pd.concat(tp, ignore_index=True).drop("VRD", axis=1)

    # Name the columns
    cols = ["DC", "Shop", "Item", "date", "y"]
    df.columns = cols

    # Delete the itterator to release some memory
    del tp
    gc.collect()
    # Main loading idea!
    # Process the df in chunks: -> At each chunk sample to the given frequency
    # Then concat!

    # Initialize chunk size based on the frequency
    if frequency == "W":
        chunk_size = 14
    elif frequency == "M":
        chunk_size = 59
    else:
        raise ValueError(
            "Currently supporting only Weekly(W) and Monthly(M) frequencies for case 1"
        )
    # Initialize values for the chunks
    start_date = df["date"].min()
    chunk_period = datetime.timedelta(days=chunk_size)
    temp_date = start_date + chunk_period

    # Initialize the df on the first chunk!
    out_df = df[(df["date"] < temp_date) & (df["date"] > start_date)]
    start_date = temp_date - datetime.timedelta(days=1)

    # Initialize the names on the unique_id
    # Lower level is the product-level
    out_df = out_df.drop(["DC", "Shop"], axis=1)
    out_df = out_df.rename(columns={"Item": "unique_id"})
    # Pivot and resample to the given frequency
    out_df = (
        pd.pivot_table(
            out_df, index="unique_id", columns="date", values="y", aggfunc="sum"
        )
        .resample(frequency, axis=1)
        .sum()
    )
    # Itterate over the other chunks:
    while start_date + chunk_period < df["date"].max():
        # Update the date
        temp_date = start_date + chunk_period

        # Filter on the given period
        temp_df = df[(df["date"] < temp_date) & (df["date"] > start_date)]
        start_date = temp_date - datetime.timedelta(days=1)

        # Update names on the unique_id, drop columns, pivot & resample
        temp_df = temp_df.drop(["DC", "Shop"], axis=1)
        temp_df = temp_df.rename(columns={"Item": "unique_id"})

        temp_df = (
            pd.pivot_table(
                temp_df, index="unique_id", columns="date", values="y", aggfunc="sum"
            )
            .resample(frequency, axis=1)
            .sum()
        )

        # Add to the main df
        out_df = pd.concat([out_df, temp_df], axis=1)

        # Save at each itteration to deal with memory breaks
        if temporary_save:
            out_df.to_csv(write_filepath)
    # Final save
    # out_df = fix_duplicate_cols(out_df)
    return out_df


def read_case_2(read_filepath):
    """Reads data for case 2

    Args:
        read_filepath (str): Existing loocation of the data file
    """

    # Read data from an excel format
    xl = pd.ExcelFile(read_filepath)
    df = pd.read_excel(
        xl, "data"
    )  # data is the name of the tab with the time series data

    # Rename columns
    df = df.rename(
        columns={
            "Verkoopdoc": "OrderNum",
            "klantnr.": "CostumerNum",
            "artikelnr.": "ID",
            "orderhoeveelheid VE": "y",
            "Gecr. op": "date",
            "GewLevrDat": "DeliveryDate",
            "land": "Country",
            "productfamilie": "ProductFamily",
            "internalgroup": "Internal",
            "segment": "Segment",
        }
    )

    # Change some data types
    items = df["ID"].values
    # Splitting the string,
    # Getting the 2nd value after the split and converting to a number
    items_num = [int(single_item.split(" ")[1]) for single_item in items]
    df["ID"] = items_num
    # Replace delimiters
    df["y"] = [
        float(val.replace(",", "."))
        if type(val) == str
        else float(str(val).replace(",", "."))
        for val in df["y"].values
    ]
    # Convert to datetime
    df["date"] = [pd.Timestamp(date, freq="D") for date in df["date"].values]

    # prepare the unique_id col
    # Format: Product Family + ID
    df["unique_id"] = [
        "-".join([product, str(id)])
        for product, id in zip(df["ProductFamily"], df["ID"])
    ]

    # keeping only specific columns
    cols_to_keep = ["date", "y", "unique_id"]
    df = df[cols_to_keep]

    return df


def read_case_3(
    read_filepath,
):
    """Reads data for case 4

    Args:
        read_filepath (str): Existing loocation of the data file
    """

    # Loading
    df = pd.read_csv(read_filepath)

    # Removing instances with bad status
    ids = df[(df["Status"] == 4) | (df["Status"].isna()) | (df["Exclincl"] != 1)].index
    df = df.drop(ids)

    # Convert date to datetime
    df["Datum"] = pd.to_datetime(df["Datum"])

    # keep only sales
    df = df[df["Trans"] == "VK"]
    df["Inuit"] = df["Inuit"].astype(float)

    # Group
    df = df.groupby(["Groep", "Resource", "Datum"]).agg({"Inuit": "sum"}).reset_index()

    # Convert to positive
    df["Inuit"] = df["Inuit"] * -1

    # Merge on the names to make the unique_id
    # format: Shop - group
    df["unique_id"] = [
        str(shop) + "-" + str(group) for shop, group in zip(df["Resource"], df["Groep"])
    ]

    # Drop cols
    df = df.drop(["Groep", "Resource"], axis=1)

    # Change column names
    cols = ["date", "y", "unique_id"]
    df.columns = cols

    return df


def read_case_4(read_filepath):
    """Reads data for case 5

    Args:
        read_filepath (str): Existing loocation of the data file
    """
    # Loading
    df = pd.read_excel(read_filepath, skiprows=9)

    # Drop two columns
    df = df.drop(["Unnamed: 0", "Unnamed: 2"], axis=1)

    # We focus on sales
    df = df[df["Status"] == "Gewonnen"]

    # Editting the items names
    # Convert to str and titlecase
    df.loc[:, "Modellen van interesse"] = (
        df["Modellen van interesse"].astype(str).str.title()
    )
    df["Merk"] = df["Merk"].astype(str).str.title()

    # Keep only the last item shown in case it is of the right brand
    df.loc[:, "Modellen van interesse"] = [
        [i for i in a.split(",") if str(b) in i]
        for a, b in zip(df["Modellen van interesse"], df["Merk"])
    ]
    df.loc[:, "Keep"] = [len(a) for a in df["Modellen van interesse"].values]
    df = df[df["Keep"] > 0]
    df["Modellen van interesse"] = [
        item[0] for item in df["Modellen van interesse"].values
    ]

    # Remove brands without naming
    df.loc[:, "Keep"] = [len(a.split(" ")) for a in df["Modellen van interesse"].values]
    df = df[df["Keep"] > 1]

    # Fix issues with some items
    df.loc[:, "Modellen van interesse"] = [
        " ".join(a.split(" ")[1:]) if a.split(" ")[1] == "Audi" else a
        for a in df["Modellen van interesse"].values
    ]

    # fix the issue with RS 6 and RS6, merge characters if a number is after a letter
    df.loc[:, "Modellen van interesse"] = [
        " ".join(
            [
                "".join([a.split(" ")[i], a.split(" ")[i + 1]])
                if a.split(" ")[i + 1].isdigit()
                else a.split(" ")[i]
                for i in range(len(a.split(" ")) - 1)
            ]
            + [a.split(" ")[-1]]
        )
        for a in df["Modellen van interesse"].values
    ]

    # Manualy replace some values for a car
    df.loc[:, "Modellen van interesse"] = df.loc[
        :, "Modellen van interesse"
    ].str.replace("!", " ")
    df.loc[:, "Modellen van interesse"] = df.loc[
        :, "Modellen van interesse"
    ].str.replace("Multivan77", "Multivan")
    df.loc[:, "Modellen van interesse"] = df.loc[
        :, "Modellen van interesse"
    ].str.replace("Multivan7", "Multivan")

    # keep onl standard models not extras
    df.loc[:, "Modellen van interesse"] = [
        " ".join(a.split(" ")[:2]) for a in df["Modellen van interesse"].values
    ]

    # Add the sale
    df.loc[:, "Sale"] = 1

    # Edit dates
    df.loc[:, "Aanmaakdatum"] = pd.to_datetime(df["Aanmaakdatum"], format="%d/%m/%Y")

    # Pick a specific start date
    start_date = datetime.datetime.strptime("01-12-2016", "%d-%m-%Y")
    df = df[df["Aanmaakdatum"] > start_date]

    # Keep only relevant columns
    cols = ["Aanmaakdatum", "Modellen van interesse", "Sale"]
    renamed_cols = ["date", "unique_id", "y"]
    df = df[cols]
    df.columns = renamed_cols

    # Aggregate
    df = df.groupby(["date", "unique_id"]).sum().reset_index()

    return df


def remove_gifts(p, q):
    """A function to compare values on case 5

    Args:
        p (float): Value 1 to compare
        q (float): value 2 to compare

    Returns:
        int: A flag 1 or 2 on weather to keep or not a specific column
    """
    if (q > 0) & (p == 0):
        return 1
    else:
        return 0


def read_case_5(read_filepath):
    """Reads data for case 5

    Args:
        read_filepath (str): Existing loocation of the data file
    """
    # read
    xl = pd.ExcelFile(read_filepath)
    # take the sheets
    df = pd.read_excel(xl, "TRANSACTIONS")
    products = pd.read_excel(xl, "PRODUCT_MASTER")

    # clean memory
    del xl
    gc.collect()

    # Fix the dates on order/invoice
    df["ORDER_DATE"] = pd.to_datetime(
        [item.split(" ")[0] for item in df["ORDER_DATE"].values]
    )
    df["INVOICE_DATE"] = pd.to_datetime(
        [item.split(" ")[0] for item in df["INVOICE_DATE"].values]
    )

    # Following steps clean data
    # For detailed explanations head to the case specific read notebook

    # Remove some special items delivered to a single person
    df = df[~df["PRODUCT_CODE"].str.startswith("D")]

    # remove some promotion items based on a flag
    df["gift"] = df.apply(
        lambda row: remove_gifts(row["EUR_INVOICED"], row["QTY_INVOICED"]), axis=1
    )
    df = df[df["gift"] == 0]
    df = df.drop("gift", axis=1)

    # summing to drop some negative alues
    groupby_to = [
        "ORDER_DATE",
        "INVOICE_DATE",
        "ORDER_LINE",
        "PRODUCT_CODE",
        "SHIPTO_CODE",
    ]
    df = (
        df.groupby(groupby_to)
        .agg(
            {
                "QTY_INVOICED": "sum",
                "EUR_INVOICED": "sum",
                "ORDERNR": "first",
                "BILLTO_CODE": "first",
            }
        )
        .reset_index()
    )

    # The idea is to correct negative values by grouping on specific columns
    # Details on the full dataframe

    df["FullOrder"] = [re.sub("\D", "", d) for d in df["ORDERNR"]]  # noqa: W605
    to_group = ["ORDER_DATE", "ORDER_LINE", "PRODUCT_CODE", "SHIPTO_CODE", "FullOrder"]
    df = df.groupby(to_group).agg({"QTY_INVOICED": "sum"}).reset_index()

    # Repeat
    to_group = ["ORDER_LINE", "PRODUCT_CODE", "SHIPTO_CODE", "FullOrder"]
    df = (
        df.groupby(to_group)
        .agg({"QTY_INVOICED": "sum", "ORDER_DATE": "last"})
        .reset_index()
    )

    # Group again
    to_group = ["PRODUCT_CODE", "SHIPTO_CODE", "FullOrder", "ORDER_DATE"]
    df = df.groupby(to_group).agg({"QTY_INVOICED": "sum"}).reset_index()

    # Group again
    to_group = ["PRODUCT_CODE", "SHIPTO_CODE", "FullOrder"]
    df = (
        df.groupby(to_group)
        .agg({"QTY_INVOICED": "sum", "ORDER_DATE": "first"})
        .reset_index()
    )

    # Group again
    to_group = ["PRODUCT_CODE", "FullOrder"]
    df = (
        df.groupby(to_group)
        .agg({"QTY_INVOICED": "sum", "ORDER_DATE": "first", "SHIPTO_CODE": "first"})
        .reset_index()
    )

    # After making the corrections droping some extra negative values
    df = df[df["QTY_INVOICED"] > 0]

    # correcting some typos
    # Make the corrections manualy as they are only 3
    change_date = "12-05-2021"
    change_date = datetime.datetime.strptime(change_date, "%d-%m-%Y")
    df.loc[665397, "ORDER_DATE"] = change_date

    change_date = "16-12-2021"
    change_date = datetime.datetime.strptime(change_date, "%d-%m-%Y")
    df.loc[416822, "ORDER_DATE"] = change_date

    change_date = "03-11-2014"
    change_date = datetime.datetime.strptime(change_date, "%d-%m-%Y")
    df.loc[4237, "ORDER_DATE"] = change_date
    df.loc[142, "ORDER_DATE"] = change_date

    # I am interested not in boxes but in items.
    # So I have to get how many items in each box

    # Del first line in products
    products = products.iloc[1:, :]
    # remove bad products
    products = products[~products["PRODUCT_CODE"].str.startswith("D")]

    # Make some splits based on the id format
    ids = products["PRODUCT_CODE"].values

    # Appending the info to the df
    products.loc[:, "Family_Code"] = [id[:2] for id in ids]
    products.loc[:, "Sr_Num"] = [id[2:5] for id in ids]
    products.loc[:, "Cigars_Num"] = [int(id[5:8]) for id in ids]

    # Adding any extra information. For example DF -> duty free
    products.loc[:, "Extras"] = [id[11:] if len(id) > 11 else np.nan for id in ids]

    # Drop a duplicate column
    products = products.drop("AANTAL_SIG", axis=1)

    # take items per products
    print(products.columns)
    item_per_prod = products[["PRODUCT_CODE", "Cigars_Num", "FAM"]]

    # Adding the number of cigars per product!
    df = pd.merge(df, item_per_prod, on="PRODUCT_CODE", how="left")
    # Multiply to get the cigars => the actual sales!
    df["Sales"] = df["QTY_INVOICED"] * df["Cigars_Num"]
    # Drop used columns
    df = df.drop(["Cigars_Num", "QTY_INVOICED"], axis=1)

    # groupby on product level
    df["product_id"] = [id.split("-")[0] for id in df["PRODUCT_CODE"]]

    # Add the family:
    df["product_id"] = [
        str(fam) + str(id) for fam, id in zip(df["FAM"], df["product_id"])
    ]

    df = df.groupby(["product_id", "ORDER_DATE"]).agg({"Sales": "sum"}).reset_index()

    # keep columns
    cols = ["unique_id", "date", "y"]
    df.columns = cols
    return df
