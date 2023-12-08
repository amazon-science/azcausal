def f_power(dx):
    return (dx['sign']
            .value_counts(normalize=True)
            .reindex(['+', '+/-', '-'])
            .fillna(0.0)
            )
