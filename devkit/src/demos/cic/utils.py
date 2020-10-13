def long_to_wide(df, unstack_var='date_time'):
    
    df = df.reset_index(level=['date_time'])
    
    df['hour'] = df['date_time'].dt.hour
    df['date_time'] = df['date_time'].dt.date
    df = df.set_index(['hour', 'date_time'], append=True)
    
    df = df.unstack(['hour'])
    new_cols = ["{0}_{1:02d}".format(*kk) for kk in df.columns]
    df.columns = new_cols
    return df


def wide_to_long(df_input, col_wide=None, dt_name="date_time", sep='_', maxsplit=1, unstack_level=1):
    """Convert wide table to long table

    Args:
        **df_input** (pandas): input dataframe - wide table
        index include col: date_time

        **col_wide** (list str): Column(s) to use to make long

    Kwargs:
        **dt_name": the name of datetime in multiindex 
        **sep** (char): A character indicating the separation of the variable
        names in the wide format, to be stripped from the names in the
        long format

        **maxsplit** (int) : It is a number, which tells us to split the string
        into maximum of provided number of times.
        If it is not provided then use 1
        表示split次数 1表示根据分割符划分一次(2部分) 2表示划分两次(3部分)

        **unstack_level** (int): level/s to unstack from column to rows
        (wide to long). Value must be between range(0,maxsplit)
        unstack_level=1 表示对mutiindex的第二部分进行unstack

        **right_dir** (bool): True - split starting from right
        False - split starting from left

    Returns:
        dataframe (pandas): long table
    """
    cols_order = [col[:-3] for col in df_input.columns.tolist()]
    cols_order = [cols_order[i] for i in range(0, len(cols_order),24)]
    
    if col_wide == None:
        col_wide = df_input.columns.tolist()
        
    df = df_input[col_wide].copy()

    new_cols = df.columns.str.rsplit(sep, maxsplit, expand=True)
    new_cols.names = [None, 'hour']
    df.columns = new_cols

    df_long = df.stack(level=unstack_level)

    df_long = df_long.reset_index(level=[dt_name, 'hour'])

    df_long[dt_name] = df_long[dt_name].dt.strftime("%Y-%m-%d") + " " +  df_long['hour']
    df_long[dt_name] = pd.to_datetime(df_long[dt_name])
    del df_long['hour']

    df_long = df_long.set_index([dt_name], append=True)
    df_long = df_long.reindex(columns=cols_order)
    return df_long