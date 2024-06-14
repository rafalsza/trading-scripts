from functools import wraps
import pandas as pd
import numpy as np
from pandas import DataFrame
from numba import njit


def input_validator(input_="ohlc"):
    def df_check(func):
        @wraps(func)
        def wrap(*args, **kwargs):
            args = list(args)
            i = 0 if isinstance(args[0], pd.DataFrame) else 1
            args[i] = args[i].rename(columns={c: c.lower() for c in args[i].columns})

            inputs = {
                "o": "open",
                "h": "high",
                "l": "low",
                "c": kwargs.get("column", "close").lower(),
                "v": "volume",
            }

            if inputs["c"] != "close":
                kwargs["column"] = inputs["c"]

            for l in input_:
                if inputs[l] not in args[i].columns:
                    raise LookupError(
                        'Must have a dataframe column named "{0}"'.format(inputs[l])
                    )

            return func(*args, **kwargs)

        return wrap

    return df_check


def apply(decorator):
    def decorate(cls):
        for attr in cls.__dict__:
            if callable(getattr(cls, attr)):
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls

    return decorate


@njit
def calculate_swing_tops_bottoms(high, low, swing_length):
    swing_tops_bottoms = np.zeros(len(high), dtype=np.int32)
    swing_type = 0
    prev_swing_type = 0
    upper = np.empty(len(high))
    lower = np.empty(len(low))

    for i in range(len(high)):
        upper[i] = np.max(high[max(0, i - swing_length + 1) : i + 1])
        lower[i] = np.min(low[max(0, i - swing_length + 1) : i + 1])

    for i in range(len(high) - swing_length):
        if high[i] > upper[i + swing_length]:
            swing_type = 0
        elif low[i] < lower[i + swing_length]:
            swing_type = 1

        if swing_type == 0 and prev_swing_type != 0:
            swing_tops_bottoms[i] = 1
        elif swing_type == 1 and prev_swing_type != 1:
            swing_tops_bottoms[i] = -1

        prev_swing_type = swing_type

    return swing_tops_bottoms, upper, lower


@njit
def numba_volumized_ob_processing(
    ohlc_close,
    ohlc_high,
    ohlc_low,
    ohlc_open,
    ohlc_volume,
    ob_swing,
    swing_length,
    close_mitigation,
):

    n = len(ohlc_close)
    crossed = np.full(n, False, dtype=bool)
    ob = np.zeros(n, dtype=np.int32)
    top = np.zeros(n, dtype=np.float32)
    bottom = np.zeros(n, dtype=np.float32)
    obVolume = np.zeros(n, dtype=np.float32)
    lowVolume = np.zeros(n, dtype=np.float32)
    highVolume = np.zeros(n, dtype=np.float32)
    percentage = np.zeros(n, dtype=np.float32)
    mitigated_index = np.zeros(n, dtype=np.int32)
    breaker = np.full(n, False, dtype=bool)

    for i in range(n):
        close_index = i
        close_price = ohlc_close[close_index]

        # Bullish Order Block
        if len(ob[ob == 1]) > 0:
            for j in range(n - 1, -1, -1):
                if ob[j] == 1:
                    currentOB = j
                    if not breaker[currentOB]:
                        if (
                            not close_mitigation
                            and ohlc_low[close_index] < bottom[currentOB]
                        ) or (
                            close_mitigation
                            and min(
                                ohlc_open[close_index],
                                ohlc_close[close_index],
                            )
                            < bottom[currentOB]
                        ):
                            breaker[currentOB] = True
                            mitigated_index[currentOB] = close_index - 1
                    else:
                        if ohlc_high[close_index] > top[currentOB]:
                            ob[j] = top[j] = bottom[j] = obVolume[j] = lowVolume[j] = (
                                highVolume[j]
                            ) = mitigated_index[j] = percentage[j] = 0.0

        last_top_indices = np.nonzero(
            (ob_swing == 1) & (np.arange(len(ob_swing)) < close_index)
        )[0]
        if len(last_top_indices) > 0:
            last_top_index = last_top_indices[-1]
            swing_top_price = ohlc_high[last_top_index]
            if close_price > swing_top_price and not crossed[last_top_index]:
                crossed[last_top_index] = True
                obBtm = ohlc_high[close_index - 1]
                obTop = ohlc_low[close_index - 1]
                obIndex = close_index - 1
                for j in range(1, close_index - last_top_index):
                    obBtm = min(
                        ohlc_low[last_top_index + j],
                        obBtm,
                    )
                    if obBtm == ohlc_low[last_top_index + j]:
                        obTop = ohlc_high[last_top_index + j]
                    obIndex = (
                        last_top_index + j
                        if obBtm == ohlc_low[last_top_index + j]
                        else obIndex
                    )

                ob[obIndex] = 1
                top[obIndex] = obTop
                bottom[obIndex] = obBtm
                obVolume[obIndex] = (
                    ohlc_volume[close_index]
                    + ohlc_volume[close_index - 1]
                    + ohlc_volume[close_index - 2]
                )
                lowVolume[obIndex] = ohlc_volume[close_index - 2]
                highVolume[obIndex] = (
                    ohlc_volume[close_index] + ohlc_volume[close_index - 1]
                )
                percentage[obIndex] = np.round(
                    (
                        min(highVolume[obIndex], lowVolume[obIndex])
                        / max(highVolume[obIndex], lowVolume[obIndex])
                    )
                    * 100.0,
                    0,
                )

    for i in range(n):
        close_index = i
        close_price = ohlc_close[close_index]

        # Bearish Order Block
        if len(ob[ob == -1]) > 0:
            for j in range(n - 1, -1, -1):
                if ob[j] == -1:
                    currentOB = j
                    if not breaker[currentOB]:
                        if (
                            not close_mitigation
                            and ohlc_high[close_index] > top[currentOB]
                        ) or (
                            close_mitigation
                            and max(
                                ohlc_open[close_index],
                                ohlc_close[close_index],
                            )
                            > top[currentOB]
                        ):
                            breaker[currentOB] = True
                            mitigated_index[currentOB] = close_index
                    else:
                        if ohlc_low[close_index] < bottom[currentOB]:
                            ob[j] = top[j] = bottom[j] = obVolume[j] = lowVolume[j] = (
                                highVolume[j]
                            ) = mitigated_index[j] = percentage[j] = 0.0

        last_btm_indices = np.nonzero(
            (ob_swing == -1) & (np.arange(len(ob_swing)) < close_index)
        )[0]
        if len(last_btm_indices) > 0:
            last_btm_index = last_btm_indices[-1]
            swing_btm_price = ohlc_low[last_btm_index]
            if close_price < swing_btm_price and not crossed[last_btm_index]:
                crossed[last_btm_index] = True
                obBtm = ohlc_low[close_index - 1]
                obTop = ohlc_high[close_index - 1]
                obIndex = close_index - 1
                for j in range(1, close_index - last_btm_index):
                    obTop = max(ohlc_high[last_btm_index + j], obTop)
                    obBtm = (
                        ohlc_low[last_btm_index + j]
                        if obTop == ohlc_high[last_btm_index + j]
                        else obBtm
                    )
                    obIndex = (
                        last_btm_index + j
                        if obTop == ohlc_high[last_btm_index + j]
                        else obIndex
                    )

                ob[obIndex] = -1
                top[obIndex] = obTop
                bottom[obIndex] = obBtm
                obVolume[obIndex] = (
                    ohlc_volume[close_index]
                    + ohlc_volume[close_index - 1]
                    + ohlc_volume[close_index - 2]
                )
                lowVolume[obIndex] = (
                    ohlc_volume[close_index] + ohlc_volume[close_index - 1]
                )
                highVolume[obIndex] = ohlc_volume[close_index - 2]
                percentage[obIndex] = np.round(
                    (
                        min(highVolume[obIndex], lowVolume[obIndex])
                        / max(highVolume[obIndex], lowVolume[obIndex])
                    )
                    * 100.0,
                    0,
                )
    return ob, top, bottom, obVolume, lowVolume, highVolume, mitigated_index, percentage


@apply(input_validator(input_="ohlc"))
class Vob:
    swing_length = 10
    close_mitigation = False

    @classmethod
    def swing_tops_bottoms(
        cls, ohlc: DataFrame, swing_length: int = swing_length
    ) -> DataFrame:
        high = ohlc["high"].values
        low = ohlc["low"].values
        swing_tops_bottoms, upper, lower = calculate_swing_tops_bottoms(
            high, low, swing_length
        )

        ohlc["upper"] = upper
        ohlc["lower"] = lower

        levels = np.where(
            swing_tops_bottoms != 0,
            np.where(swing_tops_bottoms == 1, high, low),
            np.nan,
        )

        swing_tops_bottoms = pd.Series(swing_tops_bottoms, name="SwingTopsBottoms")
        levels = pd.Series(levels, name="Levels")

        return pd.concat([swing_tops_bottoms, levels], axis=1)

    @classmethod
    def volumized_ob(
        cls,
        ohlc: DataFrame,
        swing_length: int = swing_length,
        close_mitigation: bool = close_mitigation,
    ) -> DataFrame:

        ohlc_close = ohlc["close"].values
        ohlc_high = ohlc["high"].values
        ohlc_low = ohlc["low"].values
        ohlc_open = ohlc["open"].values
        ohlc_volume = ohlc["volume"].values

        ob_swing_df = cls.swing_tops_bottoms(ohlc, swing_length)
        ob_swing = ob_swing_df["SwingTopsBottoms"].values

        (
            ob,
            top,
            bottom,
            obVolume,
            lowVolume,
            highVolume,
            mitigated_index,
            percentage,
        ) = numba_volumized_ob_processing(
            ohlc_close,
            ohlc_high,
            ohlc_low,
            ohlc_open,
            ohlc_volume,
            ob_swing,
            swing_length,
            close_mitigation,
        )

        ob_series = pd.Series(ob, name="OB")
        top_series = pd.Series(top, name="Top")
        bottom_series = pd.Series(bottom, name="Bottom")
        obVolume_series = pd.Series(obVolume, name="OBVolume")
        mitigated_index_series = pd.Series(mitigated_index, name="MitigatedIndex")
        percentage_series = pd.Series(percentage, name="Percentage")

        return pd.concat(
            [
                ob_series,
                top_series,
                bottom_series,
                obVolume_series,
                mitigated_index_series,
                percentage_series,
            ],
            axis=1,
        )
