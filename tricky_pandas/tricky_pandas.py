"""A collection of tricky pandas functions."""
from typing import List

import numpy as np
import pandas as pd


def cumulative_differences(df: pd.DataFrame, diff_on: str, join_on: List[str], diff_cols: List[str]) -> pd.DataFrame:
    """Running differences since a reference point via self-join.

    This is a very specific-looking function that is handy in the following situation:
    You have yearly values and want to take the differences in each year since the starting year.

    Often we will then want relative differences and to express those relative differences in percent, so those
    columns are provided for convenience.

    Args:
        df: The DataFrame on which to compute differences.
        diff_on: The column on which to difference; this would be e.g. year in our example.
        join_on: Additional join columns. This allows to difference by group.
        diff_cols: Columns to difference. These will be metrics like sales, clicks, temperature, voltage etc.

    Returns:
        New DataFrame containing the differences. Each record will have "starting" values suffixed with `_0` and
        "current" values suffixed by `_1`. Differences will be suffixed by `_diff`.
    """
    diffs_df = (
        df[df[diff_on] == df[diff_on].min()]
        .drop(columns=[diff_on])
        .merge(df[df[diff_on] > df[diff_on].min()], on=join_on, suffixes=['_0', '_1'])
        .sort_values([diff_on] + join_on, ascending=True)
    )

    for col in diff_cols:
        diffs_df[f'{col}_diff'] = diffs_df[f'{col}_1'] - diffs_df[f'{col}_0']
        diffs_df[f'{col}_0_size'] = 10 * diffs_df[f'{col}_0']
        diffs_df[f'rel_{col}_diff'] = diffs_df[f'{col}_diff'] / (0.5 * (diffs_df[f'{col}_1'] + diffs_df[f'{col}_0']))
        diffs_df[f'rel_{col}_diff_pct'] = diffs_df[f'rel_{col}_diff'] * 100

    return diffs_df


def augment_with_backward_diffs(df: pd.DataFrame, diff_cols: List[str]) -> pd.DataFrame:
    """Add columns of backward differences to a DataFrame.

    The columns to difference are presumed to be numeric, and since we fill in the first difference with a `nan` ints
    are likely to be upcast, so use caution.

    Args:
        df: DataFrame to augment.
        diff_cols: Columns to difference.

    Returns:
        A new DataFrame with the same columns as the old, plus additional difference columns suffixed by `_diff`.

    """
    new_df = df.copy()
    for c in diff_cols:
        new_df[f'{c}_diff'] = np.concatenate([np.array([np.nan]), np.diff(new_df[c])], axis=0)
        new_df[f'{c}_rel_diff'] = np.concatenate(
            [np.array([np.nan]), np.array(new_df[f'{c}_diff'][1:]) / np.array(new_df[c][:-1])], axis=0
        )

    return new_df


def augment_with_rank(df: pd.DataFrame, measure_col: str, rank_col: str, ascending: bool = False) -> pd.DataFrame:
    """Add ranking columns to a DataFrame.

    Args:
        df: DataFrame to rank.
        measure_col: Column on which to rank.
        rank_col: Name of column to store the rank.
        ascending: If True, rank least to greatest.

    Returns:
        A new DataFrame with the same columns as the old, plus an additional column storing the rank.

    """
    new_df = df.copy()
    f = 1 if ascending else -1
    order = np.argsort(f * new_df[measure_col])
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(ranks))
    new_df[rank_col] = ranks
    return new_df


def normalized_sum(
    df: pd.DataFrame, sum_by: List[str], norm_by: List[str], measure_col: str, rank_by: List[str] = None
) -> pd.DataFrame:
    """Flexible computation of normalized sums.

    This operation computes the sum of `measure_col` grouped by the columns of `set(sum_by_) | set(norm_by_)`, and then
    normalizes those sums by a sum over the columns of `norm_by_`. So suppose that we had a dataframe `df` of
    "population" by "year", "country", "age_group", "gender", and "ethnicity". Then the invocation:

    ```python
        s = normalized_sum(
            df,
            sum_by_=['age_group', 'gender'],
            norm_by_=['country', 'year'],
            measure_col='population',
            rank_by_=['country'],
        )
    ```

    Would produce a DataFrame `s` endowed with a column "normed_population_sum" representing the fraction of each
    country's population having a particular age and gender in a given year ('ethnicity' would be contracted out).
    In other words, `s.groupby(['country', 'year']).sum()['normed_population_sum']` would be equal to 1.0 for all
    'country', 'year' pairs. In this invocation, we would also assign each record a rank according to the population of
    each age group and gender averaged over year within each country.

    Another way to say it might be that we will compute `sum_by_`'s share of `norm_by_`'s `measure_col`.

    Args:
        df: DataFrame to compute normalized sums.
        sum_by_: Columns that, together with  `norm_by_` will determine the "numerator" of the sum. Other columns will
            be contracted out.
        norm_by_: Columns that will form the "denominator" to normalize the sum.
        measure_col: The column to be summed.
        rank_by_: Optional subset of `norm_by_` columns by which to rank the normalized sums. Remaining columns of
            `norm_by_` will be contracted out. Defaults to `norm_by_`

    Returns:
        A DataFrame with dimensions `norm_by_ + sum_by_` and measures:
            `f'total_{measure_col}'`: The raw sum of `measure_col`.
            `f'normed_{measure_col}_sum'`: The normalized sum of `measure_col`.
            `f'avg_normed_{measure_col}_sum'`: The mean normalized `measure_col` by columns in `rank_by_`.
            `f'avg_normed_{measure_col}_sum_rank'`: The rank of the record's mean normalized `measure_col` within
                `rank_by_`
    """
    total_col = f'total_{measure_col}'
    normed_measure = f'normed_{measure_col}_sum'
    avg_normed_measure = f'avg_{normed_measure}'
    rank_col = f'{avg_normed_measure}_rank'

    agg_spec = {measure_col: 'sum'}

    norm_by_ = set(norm_by)
    sum_by_ = set(sum_by) - norm_by_
    rank_by_ = norm_by_ if rank_by is None else set(rank_by)

    assert sum_by_, 'dimension to sum by cannot be empty or subset of norm_by_'
    assert rank_by_ <= norm_by_, 'ranking dimensions must be a subset of norm dimensions'

    summed = df.groupby(list(sum_by_ | norm_by_)).aggregate(agg_spec).reset_index()
    norms = summed.groupby(list(norm_by_)).aggregate(agg_spec).reset_index()
    norms[total_col] = norms[measure_col]
    normed = summed.merge(norms.drop(measure_col, axis=1), on=list(norm_by_))
    normed[normed_measure] = normed[measure_col] / normed[total_col]

    rank_agg = (
        normed.groupby(list(rank_by_ | sum_by_))
        .aggregate({normed_measure: 'mean'})
        .reset_index()
        .rename(columns={normed_measure: avg_normed_measure})
    )

    if rank_by_:
        rank_groups = [g for _, g in rank_agg.groupby(list(rank_by_))]
    else:
        rank_groups = [rank_agg]

    augmented = [augment_with_rank(g, avg_normed_measure, rank_col, ascending=False) for g in rank_groups]
    for a in augmented:
        assert (a.shape[0] - 1) == np.max(a[rank_col])
    ranked = pd.concat(augmented, axis=0, ignore_index=True)

    result = normed.merge(
        ranked.drop(list(set(ranked.columns) - rank_by_ - sum_by_ - {rank_col, avg_normed_measure}), axis=1),
        on=list(sum_by_ | rank_by_),
    )

    return result
