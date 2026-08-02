from matplotlib.axes import Axes


def plot_loss_curves(
    train_loss: list[float], eval_loss: list[float], ax: Axes, title: str
):
    ax.plot(train_loss, color="tab:blue")
    ax.plot(eval_loss, color="tab:orange")
    ax.set_title(title)
