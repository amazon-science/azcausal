def plot_hist_contr_treat(ax, T=None, CF=None, W=None, others=list(), start_time=None, arrows=True):
    if T is not None:
        ax.plot(T.index, T, label="T", color="blue")

    if CF is not None:
        ax.plot(CF.index, CF.index, "--", color="blue", alpha=0.5)

        if arrows:
            for t, origin, delta in zip((W==1).index, CF[W==1], (CF - T)[W == 1]):
                ax.arrow(t, origin, 0, delta, color="black",
                         length_includes_head=True, head_width=0.3, width=0.01, head_length=2)

    for col in others:
        ax.plot(col.index, col, label=col.name, color="red")

    if start_time is not None:
        ax.axvline(start_time, color="black", alpha=0.3)
