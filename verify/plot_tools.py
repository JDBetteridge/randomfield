import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from matplotlib.patheffects import withStroke


class PlotStyle(object):
    def __init__(self, colours=None, markers=None, linestyles=None):
        if colours is None:
            self.colours = ['C'+str(ii) for ii in range(10)]
        else:
            self.colours = colours

        if markers is None:
            self.markers= ['+', '^', 'd', '*', 's', 'x', 'o', 'P', '1', 'v']
        else:
            self.markers = markers

        if linestyles is None:
            self.linestyles = ['-', '--', ':', '-.'] + [':']*5
        else:
            self.linestyles = linestyles

def data_legend(ax, select=None, pos=None, title=None):
    legend_label = []
    for item in select:
        legend_label.append('{}'.format(item))
    if pos is None:
        legend = ax.legend(legend_label,
                           title=title,
                           loc='upper left',
                           fontsize=10,
                           bbox_to_anchor=(1.05, 1),
                           borderaxespad=0.)
    else:
        legend = ax.legend(legend_label,
                           title=title,
                           loc=pos,
                           fontsize=10)

    ax.add_artist(legend)
    return ax

def custom_legend(ax, style, methods, pos=None, title=None):
    custom_lines = []
    for line in style.linestyles:
        custom_lines.append(Line2D([0], [0], color='black', linestyle=line, lw=1))

    custom = ax.legend(custom_lines,
                       methods,
                       title=title,
                       loc=pos,
                       fontsize=10)

    ax.add_artist(custom)
    return ax

#####
# Scaling
#####
def _scaling_plot(ax, data, style=None, select=None, weak=False):
    if style is None:
        style = PlotStyle()

    if select is None:
        raise ValueError('Must select some fields to plot')

    for item, col, marker in zip(select, style.colours, style.markers):
        fmt = col + marker + style.linestyles[0]
        data[item].plot(style=fmt, ax=ax, logx=True, logy=(not weak), legend=False, clip_on=False)

    ax.set_axisbelow(False)
    return ax

def weak_scaling_plot(ax, data, style=None, select=None, legends=None, procpernode=16):
    # Plot data
    _scaling_plot(ax, data, style=style, select=select, weak=True)

    # Format axes
    ax.set_xscale('log', base=2)
    ax.set_xticks(data.index)
    ax.set_xlim(data.index[[0, -1]])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_ylabel('runtime', fontsize=10)

    # Position of legend depends on strong/weak scaling
    pos = 'upper left'
    ax.axvspan(1, procpernode, facecolor='k', alpha=0.25)

    ax.yaxis.set_major_formatter(ScalarFormatter())

    # Add "data" legend
    if legends is not None:
        data_legend(ax, select=select, pos=pos)
    return ax

def strong_scaling_plot(ax, data, style=None, select=None, legends=None,
                        dofs=None, annotate_color=('k', 'w'),
                        dof_series=0, reverse_last=1):
    # Plot data
    _scaling_plot(ax, data, style=style, select=select, weak=False)

    # Add comparison lines and format axes
    scaling_comparison_lines(ax, data, select=select)
    ax.set_xscale('log', base=2)
    ax.set_xticks(data.index)
    ax.set_xlim(data.index[[0, -1]])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_ylabel('runtime', fontsize=10)

    # Position of legend depends on strong/weak scaling
    pos = 'lower left'
    ax.set_yscale('log', base=10)

    #ax.yaxis.set_major_formatter(ScalarFormatter())

    # Add "data" legend
    if legends is not None:
        data_legend(ax, select=select, pos=pos, title=legends)
    # Add dofs per core
    if dofs is not None:
        if style is None:
            style = PlotStyle()
        # ~ sorteddata = data[select].loc[data.index[0]].sort_values()
        # ~ item = sorteddata.index[dof_series]
        #breakpoint()
        item = select[dof_series]
        if isinstance(annotate_color, tuple):
            col = annotate_color[0]
            out = annotate_color[1]
        else:
            col = annotate_color
            out = None
        add_dofs(ax, data, dofs, item, color=col, outline=out,
                 reverse_last=reverse_last)
    return ax

def scaling_comparison_lines(ax, data, select=None):
    ''' Add comparison lines
    '''
    eps = 1.0
    for item in select:
        # First non NaN value
        # ~ for xa in data.index.values:
            # ~ ya = data[item].loc[xa] * eps
            # ~ if ya == ya:
                # ~ break
        xa = data.index.values[0]
        ya = data[item].loc[xa] * eps
        C = ya*xa
        xb = data.index.values[-1]
        yb = C/xb
        # Line
        ax.plot([xa, xb], [ya, yb], 'k--', zorder=1)
    return ax

def add_dofs(ax, data, dof_data, item, color='k', outline=None, reverse_last=1):
    ''' Annotate data points with dofs/core
    '''
    if outline:
        patheff = [withStroke(linewidth=2, foreground=outline)]
    else:
        patheff = []
    last = sum(1 for _ in data[item].items()) - reverse_last
    for ii, (index, time) in enumerate(data[item].items()):
        if ii >= last:
            anchor='right'
            offset = (-3, -8)
        else:
            anchor = 'left'
            offset = (3, 2)
        ax.annotate(str(int(dof_data[item][index])//index),
             xy=(index, time), xycoords='data',
             xytext=offset, textcoords='offset points',
             fontsize='small', color=color, ha=anchor,
             path_effects=patheff)
    return ax
