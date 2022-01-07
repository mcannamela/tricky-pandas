"""Tests for tricky pandas functions."""
import numpy as np
import pandas as pd

from tricky_pandas.tricky_pandas import augment_with_backward_diffs, cumulative_differences, normalized_sum


def test_cumulative_differences():
    """Ensure we match manually computed differences."""
    df = pd.DataFrame(dict(a=[3, 2, 1, 2, 3, 1], b=[1.1, 0.9, 0.6, 0.2, 0.4, 0.8], c=['x', 'x', 'x', 'y', 'y', 'y']))

    exp_diffs_df = pd.DataFrame(
        dict(
            a=[3, 2, 2, 3],
            b_0=[0.6, 0.6, 0.8, 0.8],
            b_1=[1.1, 0.9, 0.2, 0.4],
            b_diff=[0.5, 0.3, -0.6, -0.4],
            rel_b_diff=[0.5 / 0.85, 0.3 / 0.75, -0.6 / 0.5, -0.4 / 0.6],
            rel_b_diff_pct=[100 * x for x in [0.5 / 0.85, 0.3 / 0.75, -0.6 / 0.5, -0.4 / 0.6]],
            b_0_size=[10 * x for x in [0.6, 0.6, 0.8, 0.8]],
            c=['x', 'x', 'y', 'y'],
        )
    ).sort_values(['a', 'c'])

    diffs_df = cumulative_differences(df, diff_on='a', join_on=['c'], diff_cols='b')

    diffs_df = diffs_df[exp_diffs_df.columns]

    pd.testing.assert_frame_equal(diffs_df, exp_diffs_df)


def test_augment_with_backward_diffs():
    """Ensure we match manually computed differences."""
    df = pd.DataFrame(dict(a=[3, 2, 1], b=[1.1, 0.9, 0.6]))

    exp_aug = pd.DataFrame(
        dict(a=[1, 2, 3], b=[0.6, 0.9, 1.1], b_diff=[np.nan, 0.3, 0.2], b_rel_diff=[np.nan, 0.3 / 0.6, 0.2 / 0.9])
    ).reset_index(drop=True)

    aug = augment_with_backward_diffs(df.sort_values('a'), ['b']).reset_index(drop=True)
    pd.testing.assert_frame_equal(aug, exp_aug)


def test_normalized_sum():
    """Ensure we computed ranks correctly."""
    n = 3
    df = pd.DataFrame(
        dict(
            a=list(('abcdef' * n)),
            b=sum([[i] * n * 2 for i in range(3)], []),
            c=sum([[i] * 9 for i in range(2)], []),
        )
    )
    vals = dict(a=0.2, b=0.4, c=0.3, d=0.6, e=0.5, f=0.1)
    v = df.c + df.b + np.array([vals[k] for k in df.a])
    df['v'] = v

    x = normalized_sum(df, ['a'], ['b', 'c'], 'v', rank_by=['c'])

    avg_normed_measure = 'avg_normed_v_sum'
    rank_col = 'avg_normed_v_sum_rank'

    for _, g in x.groupby(['c']):
        for _, r in g.iterrows():
            for _, q in g.iterrows():
                if r['a'] == q['a']:
                    assert r[rank_col] == q[rank_col]
                if r[avg_normed_measure] < q[avg_normed_measure]:
                    assert r[rank_col] > q[rank_col]


def test_normalized_sum_again():
    """Ensure ranks are consistent with normed sums."""
    n = 3
    df = pd.DataFrame(
        dict(
            a=list(('abcdef' * n)),
            b=sum([[i] * n * 2 for i in range(3)], []),
            c=sum([[i] * 9 for i in range(2)], []),
        )
    )
    vals = dict(a=0.2, b=0.4, c=0.3, d=0.6, e=0.5, f=0.1)
    v = df.c + df.b + np.array([vals[k] for k in df.a])
    df['v'] = v

    x = normalized_sum(df, ['a'], ['b', 'c'], 'v', rank_by=[])

    avg_normed_measure = 'avg_normed_v_sum'
    rank_col = 'avg_normed_v_sum_rank'

    for _, r in x.iterrows():
        for _, q in x.iterrows():
            if r['a'] == q['a']:
                assert r[rank_col] == q[rank_col]
            if r[avg_normed_measure] < q[avg_normed_measure]:
                assert r[rank_col] > q[rank_col]
