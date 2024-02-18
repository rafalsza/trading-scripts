import pandas as pd
import numpy as np
from binance.client import Client
import logging
import plotly.graph_objects as go

# Set logging configuration
logging.basicConfig(level=logging.INFO)
pd.set_option("display.max_columns", 500)
pd.set_option("display.max_rows", 1000)


def import_data(symbol, interval, start_date):
    client = Client()
    df = pd.DataFrame(
        client.get_historical_klines(
            symbol, start_str=start_date, interval=interval, limit=5000
        )
    ).astype(float)

    df = df.iloc[:, :6]
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, unit="ms")

    return df


def find_ob_swings(df, length):
    ob_swings = {"top": [], "bottom": []}
    swing_type = 0

    # Calculate the highest high and lowest low for the specified length
    upper = df["high"].rolling(window=length).max()
    lower = df["low"].rolling(window=length).min()

    # Concatenate upper and lower to df
    df["upper"] = upper
    df["lower"] = lower

    # Initialize the previous swing type
    prev_swing_type = 0

    # Iterate over each index in the dataframe
    for i in range(len(df)):
        try:
            # Determine the swing type
            if df["high"].iloc[i] > upper.iloc[i + length]:
                swing_type = 0
            elif df["low"].iloc[i] < lower.iloc[i + length]:
                swing_type = 1

            # Check if it's a new top or bottom
            if swing_type == 0 and prev_swing_type != 0:
                ob_swings["top"].append(
                    {
                        "index": i,
                        "loc": df.index[i],
                        "price": df["high"].iloc[i],
                        "volume": df["volume"].iloc[i],
                        "crossed": False,
                    }
                )
            elif swing_type == 1 and prev_swing_type != 1:
                ob_swings["bottom"].append(
                    {
                        "index": i,
                        "loc": df.index[i],
                        "price": df["low"].iloc[i],
                        "volume": df["volume"].iloc[i],
                        "crossed": False,
                    }
                )

            # Update the previous swing type
            prev_swing_type = swing_type
        except IndexError:
            pass

    return ob_swings["top"], ob_swings["bottom"]


def findOrderBlocks(df, maxDistanceToLastBar, swingLength, obEndMethod, maxOrderBlocks):
    bullishOrderBlocksList = []
    bearishOrderBlocksList = []

    last_bar_index = len(df)
    bar_index = max(0, last_bar_index - maxDistanceToLastBar)
    top, btm = find_ob_swings(df, swingLength)

    if bar_index >= 0:

        useBody = False

        # Bullish Order Block
        for close_index in range(bar_index, last_bar_index):
            close_price = df["close"].iloc[close_index]

            bullishBreaked = 0
            if len(bullishOrderBlocksList) > 0:
                for i in range(len(bullishOrderBlocksList) - 1, -1, -1):
                    currentOB = bullishOrderBlocksList[i]
                    if not currentOB["breaker"]:
                        if (
                            obEndMethod == "Wick"
                            and df.low.iloc[close_index - 1] < currentOB["bottom"]
                        ) or (
                            obEndMethod != "Wick"
                            and df.low.iloc[close_index - 1]
                            < currentOB[["open", "close"]].min()
                        ):
                            currentOB["breaker"] = True
                            currentOB["breakTime"] = df.index[close_index - 1]
                            currentOB["bbvolume"] = df["volume"].iloc[close_index - 1]
                    else:
                        if df.high.iloc[close_index] > currentOB["top"]:
                            bullishOrderBlocksList.pop(i)
                        elif (
                            i < 10
                            and top[i]["price"] < currentOB["top"]
                            and top[i]["price"] > currentOB["bottom"]
                        ):
                            bullishBreaked = 1

            last_top_index = None
            for i in range(len(top)):
                if top[i]["index"] < close_index:
                    last_top_index = i
            if last_top_index is not None:
                swing_top_price = top[last_top_index]["price"]
                if close_price > swing_top_price and not top[last_top_index]["crossed"]:
                    top[last_top_index]["crossed"] = True
                    boxBtm = df.high.iloc[close_index - 1]
                    boxTop = df.low.iloc[close_index - 1]
                    boxLoc = df.index[close_index - 1]
                    for j in range(1, close_index - top[last_top_index]["index"]):
                        boxBtm = min(
                            df.low.iloc[top[last_top_index]["index"] + j], boxBtm
                        )
                        if boxBtm == df.low.iloc[top[last_top_index]["index"] + j]:
                            boxTop = df.high.iloc[top[last_top_index]["index"] + j]
                        boxLoc = (
                            df.index[top[last_top_index]["index"] + j]
                            if boxBtm == df.low.iloc[top[last_top_index]["index"] + j]
                            else boxLoc
                        )

                    newOrderBlockInfo = {
                        "top": boxTop,
                        "bottom": boxBtm,
                        "volume": df["volume"].iloc[close_index]
                        + df["volume"].iloc[close_index - 1]
                        + df["volume"].iloc[close_index - 2],
                        "type": "Bull",
                        "loc": boxLoc,
                        "loc_number": close_index,
                        "index": len(bullishOrderBlocksList),
                        "oblowvolume": df["volume"].iloc[close_index - 2],
                        "obhighvolume": (
                            df["volume"].iloc[close_index]
                            + df["volume"].iloc[close_index - 1]
                        ),
                        "breaker": False,
                    }
                    bullishOrderBlocksList.insert(0, newOrderBlockInfo)
                    if len(bullishOrderBlocksList) > maxOrderBlocks:
                        bullishOrderBlocksList.pop()
                        break

        for close_index in range(bar_index, last_bar_index):
            close_price = df["close"].iloc[close_index]

            # Bearish Order Block
            bearishBreaked = 0
            if len(bearishOrderBlocksList) > 0:
                for i in range(len(bearishOrderBlocksList) - 1, -1, -1):
                    currentOB = bearishOrderBlocksList[i]
                    if not currentOB["breaker"]:
                        if (
                            obEndMethod == "Wick"
                            and df.high.iloc[close_index] > currentOB["top"]
                        ) or (
                            obEndMethod != "Wick"
                            and df.high.iloc[close_index - 1]
                            > currentOB[["open", "close"]].max()
                        ):
                            currentOB["breaker"] = True
                            currentOB["breakTime"] = df.index[close_index]
                            currentOB["bbvolume"] = df["volume"].iloc[close_index]
                    else:
                        if df.low.iloc[close_index] < currentOB["bottom"]:
                            bearishOrderBlocksList.pop(i)
                        elif (
                            i < 10
                            and btm[i]["price"] > currentOB["bottom"]
                            and btm[i]["price"] < currentOB["top"]
                        ):
                            bearishBreaked = 1

            last_btm_index = None
            for i in range(len(btm)):
                if btm[i]["index"] < close_index:
                    last_btm_index = i
            if last_btm_index is not None:
                swing_btm_price = btm[last_btm_index]["price"]
                if close_price < swing_btm_price and not btm[last_btm_index]["crossed"]:
                    btm[last_btm_index]["crossed"] = True
                    boxBtm = df.low.iloc[close_index - 1]
                    boxTop = df.high.iloc[close_index - 1]
                    boxLoc = df.index[close_index - 1]
                    for j in range(1, close_index - btm[last_btm_index]["index"]):
                        boxTop = max(
                            df.high.iloc[btm[last_btm_index]["index"] + j], boxTop
                        )
                        boxBtm = (
                            df.low.iloc[btm[last_btm_index]["index"] + j]
                            if boxTop == df.high.iloc[btm[last_btm_index]["index"] + j]
                            else boxBtm
                        )
                        boxLoc = (
                            df.index[btm[last_btm_index]["index"] + j]
                            if boxTop == df.high.iloc[btm[last_btm_index]["index"] + j]
                            else boxLoc
                        )

                    newOrderBlockInfo = {
                        "top": boxTop,
                        "bottom": boxBtm,
                        "volume": df["volume"].iloc[close_index]
                        + df["volume"].iloc[close_index - 1]
                        + df["volume"].iloc[close_index - 2],
                        "type": "Bear",
                        "loc": boxLoc,
                        "loc_number": close_index,
                        "index": len(bearishOrderBlocksList),
                        "oblowvolume": (
                            df["volume"].iloc[close_index]
                            + df["volume"].iloc[close_index - 1]
                        ),
                        "obhighvolume": df["volume"].iloc[close_index - 2],
                        "breaker": False,
                    }
                    bearishOrderBlocksList.insert(0, newOrderBlockInfo)
                    if len(bearishOrderBlocksList) > maxOrderBlocks:
                        bearishOrderBlocksList.pop()
                        break

    return bullishOrderBlocksList, bearishOrderBlocksList, top, btm


# Fetch historical data from Binance
symbol = "BTTCUSDT"
interval = "1d"
df = import_data(symbol, interval, "2021-12-01")
swing_length = 10
maxDistanceToLastBar = 1750


# Detect bullish and bearish order blocks
bullish_order_blocks, bearish_order_blocks, top, btm = findOrderBlocks(
    df, maxDistanceToLastBar, swing_length, "Wick", 30
)
print("Bullish Order Blocks:")
print(pd.DataFrame(bullish_order_blocks))

print("Bearish Order Blocks:")
print(pd.DataFrame(bearish_order_blocks))


def format_volume(volume):
    if volume >= 1e12:
        return f"{volume / 1e12:.3f}T"
    elif volume >= 1e9:
        return f"{volume / 1e9:.3f}B"
    elif volume >= 1e6:
        return f"{volume / 1e6:.3f}M"
    elif volume >= 1e3:
        return f"{volume / 1e3:.3f}k"
    else:
        return f"{volume:.2f}"


# Plotting
fig = go.Figure(
    data=[
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            increasing=dict(line=dict(width=1.5, color="#26a69a"), fillcolor="#26a69a"),
            decreasing=dict(line=dict(width=1.5, color="#ef5350"), fillcolor="#ef5350"),
            showlegend=False,
        )
    ]
)

fig.add_trace(
    go.Scatter(
        x=[point["loc"] for point in top],
        y=[point["price"] for point in top],
        mode="markers",
        marker=dict(color="green", size=10),
        name="Tops",
        legendgroup="tops",
    )
)

fig.add_trace(
    go.Scatter(
        x=[point["loc"] for point in btm],
        y=[point["price"] for point in btm],
        mode="markers",
        marker=dict(color="darkred", size=10),
        name="Bottoms",
        legendgroup="bottoms",
    )
)

# Add rectangles for bullish order blocks
for ob in bullish_order_blocks:
    fig.add_shape(
        type="rect",
        x0=ob["loc"],
        y0=ob["bottom"],
        x1=ob["breakTime"] if "breakTime" in ob else df.index[-1],
        y1=ob["top"],
        line=dict(color="Green"),
        fillcolor="Green",
        opacity=0.3,
        name="Bullish OB",
        legendgroup="bullish ob",
        showlegend=True,
    )

    if "breakTime" in ob:
        x_center = ob["loc"] + (ob["breakTime"] - ob["loc"]) / 2
    else:
        x_center = ob["loc"] + (df.index[-1] - ob["loc"]) / 2

    # Convert y-value to log scale
    y_center = np.log10((ob["bottom"] + ob["top"]) / 2)
    volume_text = format_volume(ob["volume"])
    percentage = int(
        (
            min(ob["obhighvolume"], ob["oblowvolume"])
            / max(ob["obhighvolume"], ob["oblowvolume"])
        )
        * 100.0
    )

    # Add annotation text
    annotation_text = f"{volume_text} ({percentage}%)"

    fig.add_annotation(
        x=x_center,
        y=y_center,
        xref="x",
        yref="y",
        align="center",
        text=annotation_text,
        font=dict(color="white", size=13),
        showarrow=False,
    )

# Add rectangles for bearish order blocks
for ob in bearish_order_blocks:
    fig.add_shape(
        type="rect",
        x0=ob["loc"],
        y0=ob["bottom"],
        x1=ob["breakTime"] if "breakTime" in ob else df.index[-1],
        y1=ob["top"],
        line=dict(color="Red"),
        fillcolor="Red",
        opacity=0.3,
        name="Bearish OB",
        legendgroup="bearish ob",
        showlegend=True,
    )

    if "breakTime" in ob:
        x_center = ob["loc"] + (ob["breakTime"] - ob["loc"]) / 2
    else:
        x_center = ob["loc"] + (df.index[-1] - ob["loc"]) / 2

    # Convert y-value to log scale
    y_center = np.log10((ob["bottom"] + ob["top"]) / 2)
    percentage = int(
        (
            min(ob["obhighvolume"], ob["oblowvolume"])
            / max(ob["obhighvolume"], ob["oblowvolume"])
        )
        * 100.0
    )

    volume_text = format_volume(ob["volume"])

    # Add annotation text
    annotation_text = f"{volume_text} ({percentage}%)"

    fig.add_annotation(
        x=x_center,
        y=y_center,
        xref="x",
        yref="y",
        align="center",
        text=annotation_text,
        font=dict(color="white", size=13),
        showarrow=False,
    )

# Update layout
fig.update_layout(
    title=f"Volumized Order Blocks {symbol} interval: {interval}",
    xaxis=dict(
        title="Time",
        automargin=True,
        rangeslider_visible=False,
        showspikes=True,
        spikesnap="cursor",
        spikedash="dash",
        spikemode="toaxis+across",
        spikethickness=-3,
    ),
    yaxis=dict(
        title="Price",
        side="right",
        type="log",
        automargin=True,
        showspikes=True,
        spikesnap="cursor",
        spikedash="dash",
        spikemode="toaxis+across",
        spikethickness=-3,
        tickformatstops=[
            dict(dtickrange=[None, 0.0001], value=".6f"),
            dict(dtickrange=[0.0001, 0.001], value=".5f"),
            dict(dtickrange=[0.001, 0.01], value=".4f"),
            dict(dtickrange=[0.01, 0.1], value=".3f"),
            dict(dtickrange=[0.1, 1], value=".2f"),
            dict(dtickrange=[1, None], value=".2f"),
        ],
    ),
    template="plotly_dark",
    hovermode="x",
    showlegend=True,
    autosize=True,
)

fig.show()