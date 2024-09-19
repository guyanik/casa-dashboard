import pandas as pd
import plotly.express as px  # interactive charts
import streamlit as st

st.set_page_config(
    page_title="Project Data Science Dashboard",
    page_icon="images/becris.png",
    layout="wide",
)

tab1, tab2, tab3, tab4 = st.tabs(["Sales Stock", "WO", "WL", "Forecast & Prediction"])  

with tab1:
    # read csv
    df = pd.read_csv("data/daily_total.csv")
    df.index = pd.to_datetime(df["DATE"], format="%Y-%m-%d")
    df["HOURS SPENT"] = df["SECONDSSPENT"] / 3600
    df["USER COUNT"] = df["USERCOUNT"]
    df = df.drop(columns=["DATE", "SECONDSSPENT", "USERCOUNT"])
    df = df[["HOURS SPENT", "USER COUNT", "QUANTITY", "VOLUME", "WEIGHT", "PRICE"]]

    # top-level filters
    col1, col2, col3 = st.columns(3)

    # this will put a button in the middle column
    with col2:
        # dashboard title
        st.title("Sales Stock Data")

        # choose daily, weekly, monthly, yearly
        timeframe = st.selectbox("Select the timeframe", ["Daily", "Weekly", "Monthly", "Yearly"])

    if timeframe == "Weekly":
        # first day of the week
        df = df.resample("W").sum().round(2)
    elif timeframe == "Monthly":
        # first day of the month
        df = df.resample("MS").sum().round(2)
    elif timeframe == "Yearly":
        df = df.resample("YS").sum().round(2)

    # creating a single-element container
    sales_stock1 = st.empty()

    with sales_stock1.container():
        date = st.date_input("Select the date", 
                            min_value=pd.to_datetime("2018-01-01"), 
                            max_value=pd.to_datetime("2023-12-31"), 
                            value=df.index[-1])

        date = pd.Timestamp(date)
        if date in df.index.to_list():
            # create columns
            seconds_spent, user_count, quantity, volume, weight, price = st.columns(6)

            if date != df.index[0]:
                prev_date = date - pd.Timedelta(days=1)
                while prev_date not in df.index.to_list():
                    prev_date -= pd.Timedelta(days=1)

            # fill in columns with respective values
            seconds_spent.metric(
                label="Hours Spent ⏱️",
                value=round(df.loc[str(date), "HOURS SPENT"], 2),
                delta=round((df.loc[str(date), "HOURS SPENT"] - df.loc[str(prev_date), "HOURS SPENT"]), 2) if date > df.index[0] else 0,
            )

            user_count.metric(
                label="User Count 👥",
                value=int(df.loc[str(date), "USER COUNT"]),
                delta=int(df.loc[str(date), "USER COUNT"] - df.loc[str(prev_date), "USER COUNT"]) if date > df.index[0] else 0,
            )
            
            quantity.metric(
                label="Quantity 📈",
                value=int(df.loc[str(date), "QUANTITY"]),
                delta=int(df.loc[str(date), "QUANTITY"] - df.loc[str(prev_date), "QUANTITY"]) if date > df.index[0] else 0,
            )
            
            volume.metric(
                label="Volume 📦",
                value=round(df.loc[str(date), "VOLUME"], 2),
                delta=round(df.loc[str(date), "VOLUME"] - df.loc[str(prev_date), "VOLUME"], 2) if date > df.index[0] else 0,
            )

            weight.metric(
                label="Weight ⚖️",
                value=round(df.loc[str(date), "WEIGHT"], 2),
                delta=round(df.loc[str(date), "WEIGHT"] - df.loc[str(prev_date), "WEIGHT"], 2) if date > df.index[0] else 0,
            )

            price.metric(
                label="Price 💰",
                value=round(df.loc[str(date), "PRICE"], 2),
                delta=round(df.loc[str(date), "PRICE"] - df.loc[str(prev_date), "PRICE"], 2) if date > df.index[0] else 0,
            )
        
        else:
            st.error("No data for this date")


    sales_stock2 = st.empty()

    with sales_stock2.container():
        column = st.selectbox("Select the column", df.columns)

        # scatter plot
        fig = px.scatter(round(df, 2), x=df.index, y=column, color=column, template="plotly_dark")
        fig.update_xaxes(rangeslider_visible=True)
        fig.update_layout(transition_duration=500)
        st.plotly_chart(fig, use_container_width=True)


    sales_stock3 = st.empty()

    with sales_stock3.container():
        # 2 columns
        col1, col2 = st.columns(2)

        # plot correlation heatmap
        with col1:
            if timeframe == "Daily":
                st.markdown("### Correlation Heatmap After Data Breach")
                fig = px.imshow(df[df.index > '2022-08-10'].corr(), template="plotly_dark")
                fig.update_layout(transition_duration=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown("### Correlation Heatmap")
                fig = px.imshow(df.corr(), template="plotly_dark")
                fig.update_layout(transition_duration=500)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Detailed Data View")
            st.dataframe(df)

with tab2:
    df = pd.read_csv("data/daily_wo.csv")
    df['DATE'] = pd.to_datetime(df['DATE'])
    df_total = df.drop(columns=["LOCUS"]).groupby("DATE").sum().round(2).reset_index()

    # top-level filters
    col1, col2, col3 = st.columns(3)

    # this will put a button in the middle column
    with col2:
        # dashboard title
        st.title("WO Data")

        # choose daily, weekly, monthly, yearly
        timeframe = st.selectbox("Select the timeframe", ["Daily WO", "Weekly WO", "Monthly WO", "Yearly WO"])
    
    locus = st.multiselect("Select the locus", ["All"] + df["LOCUS"].unique().tolist(), default=["All"])

    if timeframe == "Weekly WO":
        # First day of the week
        df = df.set_index('DATE').groupby("LOCUS").resample("W").sum().round(2).reset_index()
        df_total = df_total.set_index('DATE').resample("W").sum().round(2).reset_index()
    elif timeframe == "Monthly WO":
        # First day of the month
        df = df.set_index('DATE').groupby("LOCUS").resample("MS").sum().round(2).reset_index()
        df_total = df_total.set_index('DATE').resample("MS").sum().round(2).reset_index()
    elif timeframe == "Yearly WO":
        df = df.set_index('DATE').groupby("LOCUS").resample("YS").sum().round(2).reset_index()
        df_total = df_total.set_index('DATE').resample("YS").sum().round(2).reset_index()
    else:  # Daily WO
        df = df.sort_values('DATE')
        df_total = df_total.sort_values('DATE')
    
    wo1 = st.empty()

    with wo1.container():

        # scatter plot
        if "All" not in locus:
            df_locus = df[df["LOCUS"].isin(locus)]
            fig = px.scatter(df_locus, x="DATE", y="WOANBL", color="LOCUS", template="plotly_dark")
            fig.update_xaxes(rangeslider_visible=True)
            fig.update_layout(transition_duration=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.scatter(df_total, x=df_total.index, y="WOANBL", template="plotly_dark")
            fig.update_xaxes(rangeslider_visible=True)
            fig.update_layout(transition_duration=500)
            st.plotly_chart(fig, use_container_width=True)
    
    wo2 = st.empty()

    with wo2.container():
        col1, col2 = st.columns(2)

        with col1:
            date = st.date_input("Select the date", 
                                min_value=pd.to_datetime("2018-01-01"), 
                                max_value=pd.to_datetime("2023-12-31"), 
                                value=pd.to_datetime("2023-01-01"),
                                key="wo_date")
            # bar plot with locuses for the selected date
            fig = px.bar(df[df["DATE"] == str(date)], x="LOCUS", y="WOANBL", template="plotly_dark")
            fig.update_layout(transition_duration=500)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            col1, col2 = st.columns(2)
            with col1:                
                st.markdown("### Detailed Data View")
                st.dataframe(df.set_index('DATE'))
            with col2:
                st.markdown("### Detailed Data View (Total)")
                st.dataframe(df_total.set_index('DATE'))

with tab3:
    df = pd.read_csv("data/daily_wl.csv")
    df['DATE'] = pd.to_datetime(df['DATE'])
    df_total = df.drop(columns=["LOCUS"]).groupby("DATE").sum().round(2).reset_index()

    # top-level filters
    col1, col2, col3 = st.columns(3)

    # this will put a button in the middle column
    with col2:
        # dashboard title
        st.title("WL Data")

        # choose daily, weekly, monthly, yearly
        timeframe = st.selectbox("Select the timeframe", ["Daily WL", "Weekly WL", "Monthly WL", "Yearly WL"])
    
    var = st.selectbox("Select the variable", ["ORDER", "PRICE"])
    locus = st.multiselect("Select the locus", ["All"] + df["LOCUS"].unique().tolist(), default=["All"])

    if timeframe == "Weekly WL":
        df = df.set_index('DATE').groupby("LOCUS").resample("W").sum().round(2).reset_index()
        df_total = df_total.set_index('DATE').resample("W").sum().round(2).reset_index()
    elif timeframe == "Monthly WL":
        df = df.set_index('DATE').groupby("LOCUS").resample("MS").sum().round(2).reset_index()
        df_total = df_total.set_index('DATE').resample("MS").sum().round(2).reset_index()
    elif timeframe == "Yearly WL":
        df = df.set_index('DATE').groupby("LOCUS").resample("YS").sum().round(2).reset_index()
        df_total = df_total.set_index('DATE').resample("YS").sum().round(2).reset_index()
    else:  # Daily WL
        df = df.sort_values('DATE')
        df_total = df_total.sort_values('DATE')
    
    wl1 = st.empty()

    with wl1.container():

        # scatter plot
        if "All" not in locus:
            df_locus = df[df["LOCUS"].isin(locus)]
            fig = px.scatter(df_locus, x="DATE", y=var, color="LOCUS", template="plotly_dark")
            fig.update_xaxes(rangeslider_visible=True)
            fig.update_layout(transition_duration=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.scatter(df_total, x=df_total.index, y=var, template="plotly_dark")
            fig.update_xaxes(rangeslider_visible=True)
            fig.update_layout(transition_duration=500)
            st.plotly_chart(fig, use_container_width=True)
        
    wl2 = st.empty()

    with wl2.container():
        col1, col2 = st.columns(2)

        with col1:
            date = st.date_input("Select the date", 
                                min_value=pd.to_datetime("2018-01-01"), 
                                max_value=pd.to_datetime("2023-12-31"), 
                                value=pd.to_datetime("2023-01-01"),
                                key="wl_date")
            # bar plot with locuses for the selected date
            fig = px.bar(df[df["DATE"] == str(date)], x="LOCUS", y=var, template="plotly_dark")
            fig.update_layout(transition_duration=500)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            col1, col2 = st.columns(2)
            with col1:                
                st.markdown("### Detailed Data View")
                st.dataframe(df.set_index('DATE'))
            with col2:
                st.markdown("### Detailed Data View (Total)")
                st.dataframe(df_total.set_index('DATE'))


with tab4:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Forecast")
        model1 = st.selectbox("Select the model", ["ARIMA", "Holt-Winters"])

    fc = st.empty()
    with fc.container():
        col1, col2 = st.columns([2.5, 1])

        if model1 == "ARIMA":
            with col1:
                # display image
                st.image("images/arima_forecast.png", caption="Mean Absolute Error: 32.6 Hours")

            with col2:
                df = pd.read_csv("data/arima_forecast.csv")[['DATE', 'SECONDSSPENT', 'FORECAST']].set_index('DATE')
                df[['SECONDSSPENT', 'FORECAST']] = df[['SECONDSSPENT', 'FORECAST']] / 3600
                st.markdown("### Last 14 days (7 days forecasted)")
                st.dataframe(df.tail(14))
        else:
            with col1:
                # display image
                st.image("images/hw_forecast.png", caption="Mean Absolute Error: 110.2 Hours")

            with col2:
                df = pd.read_csv("data/hw_forecast.csv")[['DATE', 'SECONDSSPENT', 'FORECAST']].set_index('DATE')
                df[['SECONDSSPENT', 'FORECAST']] = df[['SECONDSSPENT', 'FORECAST']] / 3600
                st.markdown("### Last 14 days (7 days forecasted)")
                st.dataframe(df.tail(14))

    with col1:
        st.markdown("### Prediction")
        model2 = st.selectbox("Select the model", ["LSTM", "LSTM_total"])
    
    # new row for prediction similar to the forecast
    fc2 = st.empty()
    with fc2.container():
        col1, col2 = st.columns([2.5, 1])

        if model2 == "LSTM":
            with col1:
                # display image
                st.image("images/lstm.png", caption="Mean Absolute Error: 14.8 Hours")

            with col2:
                df = pd.read_csv("data/lstm_prediction.csv")[['DATE', 'DP2', 'DP1', 'HP1', 'HP2', 'DD', 'HD', 
                                                              'DP2_pred', 'DP1_pred', 'HP1_pred', 'HP2_pred', 'DD_pred', 'HD_pred']].set_index('DATE')
                st.markdown("### Last 41 days predicted")
                st.dataframe(df)
        else:
            with col1:
                # display image
                st.image("images/lstm_total.png", caption="Mean Absolute Error: 33.1 Hours")

            with col2:
                df = pd.read_csv("data/lstm_total_prediction.csv")[['DATE', 'QUANTITY', 'VOLUME', 'WEIGHT', 'PRICE', 'SECONDSSPENT', 'PREDICTION']].set_index('DATE')
                df[['SECONDSSPENT', 'PREDICTION']] = df[['SECONDSSPENT', 'PREDICTION']] / 3600
                st.markdown("### Last 41 days predicted")
                st.dataframe(df.tail(41))

