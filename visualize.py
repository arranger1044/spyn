import numpy

import matplotlib
import matplotlib.pyplot as pyplot
from matplotlib.backends.backend_pdf import PdfPages

import seaborn

#
# changing font size
seaborn.set_context("poster", font_scale=1.7, rc={'font.size': 32,
                                                  # 'axes.labelsize': fontSize,
                                                  # 'xtick.labelsize': fontSize,
                                                  # 'ytick.labelsize': fontSize,
                                                  # 'legend.fontsize': fontSize,
                                                  'text.usetex': True
                                                  })

# matplotlib.rcParams.update({'font.size': 22})


def beautify_with_seaborn():
    #

    seaborn.set_style('white')
    seaborn.despine(trim=True)
    seaborn.set_context('poster')


def visualize_curves(curves,
                     output=None,
                     labels=None,
                     lines=None,
                     linestyles=None,
                     linewidths=None,
                     palette='hls',
                     markers=None,
                     loc=None):
    """
    WRITEME
    """

    seaborn.set_style('white')
    # seaborn.set_context('poster')

    n_curves = len(curves)
    n_lines = len(lines)

    #
    # default legend location, upper right
    if loc is None:
        loc = 3

    #
    # setting the palette
    seaborn.set_palette(palette, n_colors=(n_curves + n_lines))

    #
    # default linestyle
    default_linestyle = '-'
    if linestyles is None:
        linestyles = [default_linestyle for i in range(n_curves)]
    default_width = 5
    if linewidths is None:
        linewidths = [default_width for i in range(n_curves)]
    if markers is None:
        markers = ['o', 'v', '1', '2', '3']

    for i, curve in enumerate(curves):

        curve_x, curve_y = curve
        if labels is not None:
            label = labels[i]
            line = pyplot.plot(curve_x, curve_y,
                               label=label,
                               linestyle=linestyles[i],
                               linewidth=linewidths[i],
                               # marker=markers[i]
                               )
        else:
            line = pyplot.plot(curve_x, curve_y,
                               linestyle=linestyles[i],
                               linewidth=linewidths[i],
                               # marker=markers[i]
                               )

    #
    # lastly plotting straight lines, if present
    if lines is not None:
        default_linestyle = '--'
        for i, line_y in enumerate(lines):
            #
            # this feels a little bit hackish, assuming all share the same axis
            prototypical_x_axis = curves[0][0]
            start_x = prototypical_x_axis[0]
            end_x = prototypical_x_axis[-1]
            pyplot.plot([start_x, end_x],
                        [line_y, line_y],
                        linestyle=default_linestyle,
                        linewidth=default_width)  # linestyles[i + n_curves])

    #
    # setting up the legend
    if labels is not None:
        legend = pyplot.legend(labels, loc=loc)

    seaborn.despine()

    if output is not None:
        fig = pyplot.gcf()
        fig_width = 18.5
        fig_height = 10.5
        dpi = 100
        fig.set_size_inches(fig_width, fig_height)
        fig.savefig(output,
                    # additional_artists=[legend],
                    dpi=dpi,
                    bbox_inches='tight')
        pyplot.close(fig)
    else:
        #
        # shall this be mutually exclusive with file saving?
        pyplot.show()


DATASET_LIST = ['nltcs', 'msnbc', 'kdd',
                'plants', 'baudio', 'jester', 'bnetflix',
                'accidents', 'tretail', 'pumsb_star',
                'dna', 'kosarek', 'msweb',
                'book', 'tmovie', 'cwebkb',
                'cr52', 'c20ng', 'bbc', 'ad']


def visualize_histograms(histograms,
                         output=None,
                         labels=DATASET_LIST,
                         linestyles=None,
                         rotation=90,
                         legend=None,
                         y_log=False,
                         colors=['seagreen', 'orange', 'cornflowerblue']):
    """
    Plotting histograms one near the other
    """

    n_histograms = len(histograms)
    #
    # assuming homogeneous data leengths
    # TODO: better error checking
    n_ticks = len(histograms[0])

    bin_width = 1 / (n_histograms + 1)
    bins = [[i + j * bin_width for i in range(n_ticks)]
            for j in range(1, n_histograms + 1)]

    #
    # setting up seaborn
    seaborn.set_style("white")
    seaborn.set_context("poster")
    # seaborn.set_palette(palette, n_colors=n_histograms)

    fig, ax = pyplot.subplots()

    if legend is not None:
        _legend = pyplot.legend(legend)
    #
    # setting labels
    middle_histogram = n_histograms // 2 + 1  # if n_histograms > 1 else 0
    pyplot.xticks(bins[middle_histogram], DATASET_LIST)
    if rotation is not None:
        locs, labels = pyplot.xticks()
        pyplot.setp(labels, rotation=90)

    #
    # actual plotting
    print(histograms)
    for i, histogram in enumerate(histograms):
        ax.bar(bins[i], histogram, width=bin_width,
               facecolor=colors[i], edgecolor="none",
               log=y_log)

    seaborn.despine()

    if output is not None:

        pp = PdfPages(output)
        pp.savefig(fig)
        pp.close()

if __name__ == '__main__':

    labels = [i for i in range(-10, 10)]
    points = [numpy.exp(i) for i in labels]

    visualize_curves([(labels, points)], labels=['a', 'b'])
