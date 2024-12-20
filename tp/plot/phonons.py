"""Tools for dealing with phonon dispersions.

Contains the traditional phonon dispersion plotter, as well as various
ways of projecting other quantitites onto a high-symmetry path.
"""

#Functions
#---------
#
#    add_dispersion:
#        phonon dispersion.
#    add_multi:
#        phonon dispersions.
#    add_alt_dispersion:
#        phono3py quantity against high symmetry path.
#    add_projected_dispersion:
#        phonon dispersion with phono3py quantity on colour axis.
#    add_alt_projected_dispersion:
#        alt_dispersion + projection.
#    add_wideband:
#        phonon dispersion broadened according to scattering.
#
#    get_equivalent_qpoint:
#        converts phonopy to phono3py qpoints.
#
#    formatting:
#        formatting axes.
#    tile_properties:
#        tiling properties semi-intelligently.
#"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import tp
import warnings
import yaml

warnings.filterwarnings('ignore', module='matplotlib')
warnings.filterwarnings('ignore', module='scipy')

try:
    filename = '{}/.config/tprc.yaml'.format(os.path.expanduser("~"))
    with open(filename, 'r') as f:
        conf = yaml.safe_load(f)
except yaml.parser.ParserError:
    warnings.warn('Failed to read ~/.config/tprc.yaml')
    conf = None
except FileNotFoundError:
    conf = None

workers = tp.settings.get_workers()

def add_dispersion(ax, data, sdata=None, bandmin=None, bandmax=None, main=True,
                   label=None, colour='#800080', linestyle='solid', marker=None,
                   xmarkkwargs={}, **kwargs):
    """Adds a phonon band structure to a set of axes.

    Labels, colours and linestyles can be given one for the whole
    dispersion, or one for each band, with the last entry filling all
    remaining bands.

    Arguments
    ---------
        ax : axes
            axes to plot on.
        data : dict
            dispersion data containing:

                x : array-like
                    high-symmetry path.
                frequency : array-like
                    phonon frequencies.
                tick_position : array-like
                    x-tick positions.
                tick_label : array-like
                    x-tick labels.

        sdata : dict, optional
            dispersion data to scale to, same format as data.
            Default: None.
        bandmin : int, optional
            zero-indexed minimum band index to plot. Default: None.
        bandmax : int, optional
            zero-indexed maximum band index to plot. Default: None.

        main : bool, optional
            set axis ticks, labels, limits. Default: True.
        label : str, optional
            legend label(s). Default: None

        colour : str or array-like, optional
            line colour(s). Note if retrieved from a colourmap or in
            [r,g,b] notation, the colour should be enclosed in [], e.g.
            colour = plt.get_cmap('winter_r')(linspace(0, 1, len(files)))
            tp.plot.phonons.add_dispersion(..., colour=[colour[0]], ...)
            Default: #800080.
        linestyle : str or array-like, optional
            linestyle(s) ('-', '--', '.-', ':'). Default: solid.
        marker : str or array-like, optional
            marker(s). Default: None.

        xmarkkwargs : dict, optional
            keyword arguments for x markers passed to
            matplotlib.pyplot.axvline. Set color to None to turn off.
            Defaults are defined below, which are overridden by those in
            ``~/.config/tprc.yaml``, both of which are overridden by
            arguments passed to this function.
            Defaults:

                color:      black
                linewidth:  axis line width
                rasterized: False

        kwargs
            keyword arguments passed to matplotlib.pyplot.plot.
            Defaults are defined below, which are overridden by those in
            ``~/.config/tprc.yaml``, both of which are overridden by
            arguments passed to this function.
            Defaults:

                rasterized: False

    Returns
    -------

        None
            adds plot directly to ax.
    """

    # defaults

    defkwargs = {'rasterized': False}

    if conf is None or 'dispersion_kwargs' not in conf or \
       conf['dispersion_kwargs'] is None:
        kwargs = {**defkwargs, **kwargs}
    else:
        kwargs = {**defkwargs, **conf['dispersion_kwargs'], **kwargs}

    # check inputs

    assert isinstance(main, bool), 'main must be True or False'

    # data selection

    if bandmin is None:
        bandmin = 0
    else:
        bandmin = np.amax([0, bandmin])
    if bandmax is None:
        bandmax = len(data['frequency'][0])
    else:
        bandmax = np.amin([len(data['frequency'][0]), bandmax])
    f = np.array(data['frequency'])[:,bandmin:bandmax]

    # line appearance

    colour = tile_properties(colour, bandmin, bandmax)
    linestyle = tile_properties(linestyle, bandmin, bandmax)
    marker = tile_properties(marker, bandmin, bandmax)

    # prevents unintentionally repeated legend entries
    if isinstance(label, np.ndarray):
        label = list(label)
    elif not isinstance(label, list):
        label = [label]
    label.append(None)
    label = tile_properties(label, bandmin, bandmax)

    # data scaling

    d2 = data['tick_position']
    x = data['x']
    if sdata is not None:
        d1 = sdata['tick_position']
        n = 0
        for i, d0 in enumerate(x):
            while n <= len(d2) and not (d0 >= d2[n] and d0 <= d2[n+1]):
                n += 1
            x[i] = d1[n] + ((d0 - d2[n]) * (d1[n+1] - d1[n]) / \
                                           (d2[n+1] - d2[n]))
    else:
        d1 = d2

    # plotting

    # avoid connecting bands at disconnected q-points
    split_indices = [0, *np.where(np.diff(x) == 0)[0] + 1, len(x)]
    for n in range(len(f[0])):
        for i in range(len(split_indices)-1):
            starting_index = split_indices[i]
            ending_index = split_indices[i+1]
            x_i = x[starting_index:ending_index]
            f_ni = f[starting_index:ending_index, n]
            label_ni = label[n] if i == 0 else None
            ax.plot(x_i, f_ni, color=colour[n], linestyle=linestyle[n],
                    label=label_ni, marker=marker[n], **kwargs)

    # axes formatting

    if main:
        if round(np.amin(f), 1) == 0:
            ax.set_ylim(bottom=0)
        if sdata is None: sdata = data
        formatting(ax, sdata, 'frequency', **xmarkkwargs)

    return

def add_multi(ax, data, bandmin=None, bandmax=None, main=True, label=None,
              colour='winter_r', linestyle='solid', marker=None,
              xmarkkwargs={}, **kwargs):
    """Adds multiple phonon band structures to a set of axes.

    Scales the x-scales to match.

    Arguments
    ---------

        ax : axes
            axes to plot on.
        data : array-like
            dictionaries of dispersion data containing:

                x : array-like
                    high-symmetry path.
                frequency : array-like
                    phonon frequencies.
                tick_position : array-like
                    x-tick positions.
                tick_label : array-like
                    x-tick labels.

        bandmin : int, optional
            Zero-indexed minimum band index to plot. Default: 0.
        bandmax : int, optional
            Zero-indexed maximum band index to plot. Default: max index.

        main : bool, optional
            set axis ticks, labels, limits. Default: True.
        label : array-like, optional
            legend labels. Default: None

        colour : colourmap or str or array-like or dict, optional
            colourmap or colourmap name or list of colours, one for
            each dispersion or a min and max colour to generate a
            linear colourmap between or a dictionary with cmin and cmax
            keys. Note [r,g,b] format colours should be enclosed in an
            additional [], i.e. [[[r,g,b]],...].
            Default: winter_r.
        linestyle : str or array-like, optional
            linestyle(s) ('-', '--', '.-', ':'). Default: solid.
        marker : str or array-like, optional
            marker(s). Default: None.

        xmarkkwargs : dict, optional
            keyword arguments for x markers passed to
            matplotlib.pyplot.axvline. Set color to None to turn off.
            Defaults are defined below, which are overridden by those in
            ``~/.config/tprc.yaml``, both of which are overridden by
            arguments passed to this function.
            Defaults:

                color:      black
                linewidth:  axis line width
                rasterized: False

        kwargs
            keyword arguments passed to matplotlib.pyplot.plot.
            Defaults are defined below, which are overridden by those in
            ``~/.config/tprc.yaml``, both of which are overridden by
            arguments passed to this function.
            Defaults:

                rasterized: False

    Returns
    -------

        None
            adds plot directly to ax.
    """

    # defaults

    defkwargs = {'rasterized': False}

    if conf is None or 'multi_kwargs' not in conf or \
       conf['multi_kwargs'] is None:
        kwargs = {**defkwargs, **kwargs}
    else:
        kwargs = {**defkwargs, **conf['multi_kwargs'], **kwargs}

    # check inputs

    assert isinstance(main, bool), 'main must be True or False'

    # line appearance

    try:
        try:
            colours = mpl.cm.get_cmap(colour)(np.linspace(0, 1, len(data)))
        except AttributeError:
            colours = mpl.colormaps[colour](np.linspace(0, 1, len(data)))
        colours = [[c] for c in colours]
    except ValueError:
        if isinstance(colour, mpl.colors.ListedColormap):
            colours = [[colour(i)] for i in np.linspace(0, 1, len(data))]
        elif isinstance(colour, str) and colour == 'skelton':
            colour = tp.plot.colour.skelton()
            colours = [[colour(i)] for i in np.linspace(0, 1, len(data))]
        elif isinstance(colour, list) and len(colour) == 2 and len(data) != 2:
            colour = tp.plot.colour.linear(*colour)
            colours = [[colour(i)] for i in np.linspace(0, 1, len(data))]
        elif isinstance(colour, dict):
            colour = tp.plot.colour.linear(**colour)
            colours = [[colour(i)] for i in np.linspace(0, 1, len(data))]
        elif isinstance(colour, str):
            colours = [colour]
        else:
            colours = colour

    if label is None or label == [None]:
        label = np.full(len(data), None)

    if isinstance(linestyle, str) or len(linestyle) == 1:
        linestyle = np.repeat(linestyle, len(data))
    elif len(linestyle) < len(data):
        while len(linestyle) < len(data):
            linestyle.append(linestyle[-1])

    if marker is None or isinstance(marker, (str, tuple)) or len(marker) == 1:
        marker = np.repeat(marker, len(data))
    elif len(marker) < len(data):
        while len(marker) < len(data):
            marker.append(marker[-1])

    # plotting

    for i in range(len(data)):
        add_dispersion(ax, data[i], sdata=data[0], bandmin=bandmin,
                       bandmax=bandmax, main=False, label=label[i],
                       colour=colours[i], linestyle=linestyle[i],
                       marker=marker[i], **kwargs)

    # axes formatting

    if main:
        if bandmin is None:
            bandmin = 0
        else:
            bandmin = np.amax([0, bandmin])
        
        if bandmax is None:
            bandmax = len(data[0]['frequency'][0])
        else:
            bandmax = np.amin([len(data[0]['frequency'][0]), bandmax])

        if bandmin < 3:
            f = [d['frequency'] for d in data]
            noim = True
            for ff in f:
                ff = np.array(ff)[:,bandmin:bandmax]
                if round(np.amin(ff), 1) < 0:
                         noim = False
            if noim == True:
                ax.set_ylim(bottom=0)
        formatting(ax, data[0], 'frequency', **xmarkkwargs)

    return

@tp.docstring_replace(workers=str(workers))
def add_alt_dispersion(ax, data, pdata, quantity, bandmin=None, bandmax=None,
                       temperature=300, direction='avg', label=['Longitudinal',
                       'Transverse$_1$', 'Transverse$_2$', 'Optic'],
                       poscar='POSCAR', scatter=False, main=True, log=False,
                       interpolate=10000, smoothing=5, colour=['#44ffff',
                       '#ff8044', '#ff4444', '#00000010'], linestyle='-',
                       marker=None, workers=workers, xmarkkwargs={}, verbose=False,
                       **kwargs):
    """Plots a phono3py quantity on a high-symmetry path.

    Labels, colours and linestyles can be given one for the whole
    dispersion, or one for each band, with the last entry filling all
    remaining bands. Requires a POSCAR.

    Arguments
    ---------

        ax : axes
            axes to plot on.
        data : dict
            Phono3py-like data containing:

                qpoint : array-like
                    q-point locations.
                (quantity) : array-like
                    plotted quantity.
                temperature : array-like, optional
                    temperature, if necessary for quantity.

        pdata : dict
            phonon dispersion data containing:

                q-point : array-like
                    qpoint locations.
                x : array-like
                    high-symmetry path.
                tick_position : array-like
                    x-tick positions.
                tick_label : array-like
                    x-tick labels.

        quantity : str
            quantity to plot.

        bandmin : int, optional
            Zero-indexed minimum band index to plot. Default: 0.
        bandmax : int, optional
            Zero-indexed maximum band index to plot. Default: max index.
        temperature : float, optional
            approximate temperature in K (finds closest). Default: 300.
        direction : str, optional
            direction from anisotropic data, accepts x-z/ a-c,
            average/ avg or normal/ norm. Default: average.
        label : array-like, optional
            labels per line. A single dataset could have a single label,
            or the default labels the lines by type. You'll want to
            change this if a minimum band index is set.
            Default: ['Longitudinal', 'Transverse$_1$', 'Transverse$_2$',
            'Optic'].
        scatter : bool, optional
            plot scatter rather than line graph. Default: False.

        poscar : str, optional
            VASP POSCAR filepath. Default: POSCAR.

        main : bool, optional
            set axis ticks, label, limits. Default: True.
        log : bool, optional
            log the y scale. Default: False.
        interpolate : int, optional
            number of points per line. Default: 10,000.
        smoothing : int, optional
            every n points to sample. Default: 5.

        colour : str or array-like, optional
            line colour(s). Note if retrieved from a colourmap or in
            [r,g,b] notation, the colour should be enclosed in [], e.g.
            colour = plt.get_cmap('winter_r')(linspace(0, 1, len(files)))
            tp.plot.phonons.add_dispersion(..., colour=[colour[0]], ...)
            Default: ['#44ffff', '#ff8044', '#ff4444', '#00000010'].
        linestyle : str or array-like, optional
            linestyle(s) ('-', '--', '.-', ':'). Default: solid.
        marker : str or array-like, optional
            marker(s). Default: None.

        workers : int, optional
            number of workers for paralellised section. Default: {workers}.
        verbose : bool, optional
            Write actual temperature used if applicable.
            Default: False.

        xmarkkwargs : dict, optional
            keyword arguments for x markers passed to
            matplotlib.pyplot.axvline. Set color to None to turn off.
            Defaults are defined below, which are overridden by those in
            ``~/.config/tprc.yaml``, both of which are overridden by
            arguments passed to this function.
            Defaults:

                color:      black
                linewidth:  axis line width
                rasterized: False

        kwargs
            keyword arguments passed to matplotlib.pyplot.plot.
            Defaults are defined below, which are overridden by those in
            ``~/.config/tprc.yaml``, both of which are overridden by
            arguments passed to this function.
            Defaults:

                rasterized: False

    Returns
    -------

        None
            adds plot directly to ax.
    """

    from functools import partial
    from multiprocessing import Pool
    from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
    from pymatgen.io.vasp.inputs import Poscar
    from scipy.interpolate import interp1d

    # defaults

    defkwargs = {'rasterized': False}

    if conf is None or 'alt_dispersion_kwargs' not in conf or \
       conf['alt_dispersion_kwargs'] is None:
        kwargs = {**defkwargs, **kwargs}
    else:
        kwargs = {**defkwargs, **conf['alt_dispersion_kwargs'], **kwargs}

    # check inputs

    for name, value in zip(['main', 'log'],
                           [ main,   log]):
        assert isinstance(value, bool), '{} must be True or False'.format(name)

    # Phono3py data formatting

    tnames = tp.settings.to_tp()
    if quantity in tnames: quantity = tnames[quantity]
    if quantity == 'kappa': quantity = 'mode_kappa'

    data = tp.data.utilities.resolve(data, quantity, temperature=temperature,
                                   direction=direction)
    if verbose and 'temperature' in data['meta']:
        print('Using {} {}.'.format(data['meta']['temperature'],
                                    data['meta']['units']['temperature']))
    y = data[quantity]
    if bandmin is None:
        bandmin = 0
    else:
        bandmin = np.amax([0, bandmin])
    if bandmax is None:
        bandmax = len(y[0])
    else:
        bandmax = np.amin([len(y[0]), bandmax])
    y = np.array(y)[:,bandmin:bandmax]
    qk = data['qpoint']

    # Phonopy data formatting

    qp = pdata['qpoint']
    x = pdata['x']

    if len(x) > 100:
        xi = [x[i] for i in range(len(x)) if i % smoothing == 0]
        qpi = [qp[i] for i in range(len(qp)) if i % smoothing == 0]
    else:
        xi = x
        qpi = qp
    if xi[-1] != x[-1]: qpi.append(qp[-1]); xi.append(x[-1])

    # map Phono3py to Phonopy

    struct = Poscar.from_file(poscar).structure
    sg = SpacegroupAnalyzer(struct)
    symops = sg.get_point_group_operations(cartesian=False)
    geq = partial(get_equivalent_qpoint, np.array(qk), symops)
    with Pool(processes=workers) as pool:
        min_id = pool.map(geq, qpi)
    y2 = y[min_id, :]

    x, indices = np.unique(xi, return_index=True)
    y2 = np.array(y2)[indices]

    # interpolate

    x2 = np.linspace(min(x), max(x), interpolate)
    yinterp = interp1d(x, y2, kind='linear', axis=0)
    y2 = np.abs(yinterp(x2))
    ysort = np.ravel(y2)
    ysort = ysort[ysort.argsort()]
    ymin = ysort[int(round(len(ysort)/100 - 1, 0))]
    ymax = ysort[int(round(len(ysort)*99.9/100 - 1, 0))]

    # line appearance

    colour = tile_properties(colour, bandmin, bandmax)
    linestyle = tile_properties(linestyle, bandmin, bandmax)
    marker = tile_properties(marker, bandmin, bandmax)

    # prevents unintentionally repeated legend entries
    label.append(None)
    label = tile_properties(label, bandmin, bandmax)

    # plotting

    # avoid connecting bands at disconnected q-points
    split_indices = [0, *np.where(np.diff(x2) == 0)[0] + 1, len(x2)]
    for n in range(len(y2[0])):
        if scatter:
            ax.scatter(x2, y2[:,n], color=colour[n], linestyle=linestyle[n],
                       label=label[n], marker=marker[n], **kwargs)
        else:
            for i in range(len(split_indices)-1):
                starting_index = split_indices[i]
                ending_index = split_indices[i+1]
                x2_i = x2[starting_index:ending_index]
                y2_ni = y2[starting_index:ending_index, n]
                label_ni = label[n] if i == 0 else None
                ax.plot(x2_i, y2_ni, color=colour[n], linestyle=linestyle[n],
                        label=label_ni, marker=marker[n], **kwargs)

    # axes formatting

    if main:
        ax.set_ylim(ymin, ymax)
        formatting(ax, pdata, quantity, log=log, **xmarkkwargs)

    return

@tp.docstring_replace(workers=str(workers))
def add_projected_dispersion(ax, data, pdata, quantity, bandmin=None,
                             bandmax=None, temperature=300, direction='avg',
                             poscar='POSCAR', main=True, interpolate=500,
                             colour='viridis_r', cmin=None, cmax=None,
                             cscale=None, unoccupied='grey', workers=workers,
                             xmarkkwargs={}, verbose=False, **kwargs):
    """Plots a phonon dispersion with projected colour.

    Plots a phonon dispersion, and projects a quantity onto the colour
    axis. Requires a POSCAR.

    Arguments
    ---------

        ax : axes
            axes to plot on.
        data : dict
            Phono3py-like data containing:

                qpoint : array-like
                    q-point locations.
                (quantity) : array-like
                    projected quantity.
                temperature : array-like, optional
                    temperature, if necessary for quantity.

        pdata : dict
            phonon dispersion data containing:

                q-point : array-like
                    qpoint locations.
                x : array-like
                    high-symmetry path.
                frequency : array-like
                    phonon frequencies.
                tick_position : array-like
                    x-tick positions.
                tick_label : array-like
                    x-tick labels.

        quantity : str
            quantity to project.

        bandmin : int, optional
            Zero-indexed minimum band index to plot. Default: 0.
        bandmax : int, optional
            Zero-indexed maximum band index to plot. Default: max index.
        temperature : float, optional
            approximate temperature in K (finds closest). Default: 300.
        direction : str, optional
            direction from anisotropic data, accepts x-z/ a-c,
            average/ avg or normal/ norm. Default: average.

        poscar : str, optional
            VASP POSCAR filepath. Default: POSCAR.

        main : bool, optional
            set axis ticks, label, limits. Default: True.
        interpolate : int, optional
            number of points per path (e.g. gamma -> X) per line.
            Default: 500.

        colour : colourmap or str or array-like or dict, optional
            colourmap or colourmap name or highlight colour or
            highlight, min, max colours in that order, or dictionary
            with cmid and cmin and/or cmax keys. Colour format must be
            hex or rgb (array) or a named colour recognised by
            matplotlib. Default: viridis_r.
        cmin : float, optional
            colour scale minimum. Default: display 99 % data.
        cmax : float, optional
            colour scale maximum. Default: display 99.9 % data.
        cscale : str, optional
            override colour scale (linear/ log). Default: None.
        unoccupied : str or array-like, optional
            if the colour variable is occuption, values below 1 are
            coloured in this colour. If set to None, or cmin is set,
            this feature is turned off. Default: grey.

        workers : int, optional
            number of workers for paralellised section. Default: {workers}.
        verbose : bool, optional
            Write actual temperature used if applicable.
            Default: False.

        xmarkkwargs : dict, optional
            keyword arguments for x markers passed to
            matplotlib.pyplot.axvline. Set color to None to turn off.
            Defaults are defined below, which are overridden by those in
            ``~/.config/tprc.yaml``, both of which are overridden by
            arguments passed to this function.
            Default

            color:      black
            linewidth:  axis line width
            rasterized: False

        kwargs
            keyword arguments passed to matplotlib.pyplot.scatter.
            Defaults are defined below, which are overridden by those in
            ``~/.config/tprc.yaml``, both of which are overridden by
            arguments passed to this function.
            Defaults:

            marker:     .
            rasterized: True

    Returns
    -------

        colorbar
            colour bar for projected data.
    """

    from functools import partial
    from multiprocessing import Pool
    from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
    from pymatgen.io.vasp.inputs import Poscar
    from scipy.interpolate import interp1d

    # defaults

    defkwargs = {'marker':     '.',
                 'rasterized': True}

    if conf is None or 'projected_dispersion_kwargs' not in conf or \
       conf['projected_dispersion_kwargs'] is None:
        kwargs = {**defkwargs, **kwargs}
    else:
        kwargs = {**defkwargs, **conf['projected_dispersion_kwargs'], **kwargs}

    # check inputs

    assert isinstance(main, bool), 'main must be True or False'

    # Phono3py data formatting

    tnames = tp.settings.to_tp()
    quantity = tnames[quantity] if quantity in tnames else quantity
    if quantity == 'kappa': quantity = 'mode_kappa'

    data = tp.data.utilities.resolve(data, quantity, temperature=temperature,
                                   direction=direction)
    if verbose and 'temperature' in data['meta']:
        print('Using {} {}.'.format(data['meta']['temperature'],
                                    data['meta']['units']['temperature']))
    c = data[quantity]
    if bandmin is None:
        bandmin = 0
    else:
        bandmin = np.amax([0, bandmin])
    if bandmax is None:
        bandmax = len(c[0])
    else:
        bandmax = np.amin([len(c[0]), bandmax])
    c = np.array(c)[:, bandmin:bandmax]
    qk = data['qpoint']

    # Phonopy data formatting

    qp = pdata['qpoint']
    x = pdata['x']
    f = np.array(pdata['frequency'])[:, bandmin:bandmax]

    # map Phono3py to Phonopy

    struct = Poscar.from_file(poscar).structure
    sg = SpacegroupAnalyzer(struct)
    symops = sg.get_point_group_operations(cartesian=False)
    geq = partial(get_equivalent_qpoint, np.array(qk), symops)
    with Pool(processes=workers) as pool:
        min_id = pool.map(geq, qp)
    c = c[min_id, :]

    x, indices = np.unique(x, return_index=True)
    f = np.array(f)[indices]
    c = np.array(c)[indices]

    # interpolate

    index = [0, 0]
    x2, f2, c2 = [], [], []
    for d in pdata['tick_position'][1:]:
        index[0] = index[1] + 1
        index[1] = next(i[0] for i in enumerate(x) if i[1] == d)
        xtemp = np.linspace(x[index[0]], x[index[1]], interpolate)
        finterp = interp1d(x[index[0]:index[1]], f[index[0]:index[1]],
                           kind='cubic', axis=0, fill_value='extrapolate')
        x2.append(xtemp)
        f2.append(finterp(xtemp))

        cinterp = interp1d(x[index[0]:index[1]], c[index[0]:index[1]],
                           kind='linear', axis=0, fill_value='extrapolate')
        c2.append(np.abs(cinterp(xtemp)))

    cmap = tp.plot.utilities.parse_colours(colour)
    cnorm, extend = tp.plot.utilities.colour_scale(c2, quantity, cmap, cmin,
                                                   cmax, cscale, unoccupied)

    # plotting

    for n in range(len(f2[0][0])):
        for i in range(len(x2)):
            line = ax.scatter(x2[i], np.array(f2[i])[:,n], c=np.array(c2[i])[:,n], cmap=cmap, norm=cnorm,
                              **kwargs)

    # axes formatting

    axlabels = tp.settings.labels()
    fig = ax.get_figure()
    if 'dos' in fig.__dict__ and fig.__dict__['dos']:
        # place colourbar outside dos
        cbar = plt.colorbar(line, extend=extend)
    else:
        cbar = plt.colorbar(line, ax=ax, extend=extend)
    cbar.set_alpha(1)
    cbar.set_label(axlabels[quantity])
    tp.plot.utilities.set_locators(cbar.ax, y=cbar.ax.yaxis.get_scale())
    cbar._draw_all()

    if main:
        if round(np.amin(f), 1) == 0:
            ax.set_ylim(bottom=0)
        formatting(ax, pdata, 'frequency', **xmarkkwargs)

    return cbar

@tp.docstring_replace(workers=str(workers))
def add_alt_projected_dispersion(ax, data, pdata, quantity, projected,
                                 bandmin=None, bandmax=None, temperature=300,
                                 direction='avg', poscar='POSCAR', main=True,
                                 log=False, interpolate=10000, smoothing=10,
                                 colour='viridis_r', cmin=None, cmax=None,
                                 cscale=None, unoccupied='grey',
                                 workers=workers, xmarkkwargs={},
                                 verbose=False, **kwargs):
    """Plots a phono3py quantity on a high-symmetry path and projection.

    Just because you can, doesn't mean you should. A maxim I may fail to
    live up to, so I leave it to you, dear reader, to decide for
    yourself. Requires a POSCAR.

    Arguments
    ---------

        ax : axes
            axes to plot on.
        data : dict
            Phono3py-like data containing:

                qpoint : array-like
                    q-point locations.
                (quantities) : array-like
                    plotted and projected quantities.
                temperature : array-like, optional
                    temperature, if necessary for quantities.

        pdata : dict
            phonon dispersion data containing:

                q-point : array-like
                    qpoint locations.
                x : array-like
                    high-symmetry path.
                tick_position : array-like
                    x-tick positions.
                tick_label : array-like
                    x-tick labels.

        quantity : str
            quantity to plot.
        projected: str
            quantity to project.

        bandmin : int, optional
            Zero-indexed minimum band index to plot. Default: 0.
        bandmax : int, optional
            Zero-indexed maximum band index to plot. Default: max index.
        temperature : float, optional
            approximate temperature in K (finds closest). Default: 300.
        direction : str, optional
            direction from anisotropic data, accepts x-z/ a-c,
            average/ avg or normal/ norm. Default: average.

        poscar : str, optional
            VASP POSCAR filepath. Default: POSCAR.

        main : bool, optional
            set axis ticks, label, limits. Default: True.
        log : bool, optional
            log the y scale. Default: False.
        interpolate : int, optional
            number of points per line. Default: 10,000.
        smoothing : int, optional
            every n points to sample. Default: 10.

        colour : colourmap or str or array-like or dict, optional
            colourmap or colourmap name or highlight colour or
            highlight, min, max colours in that order, or dictionary
            with cmid and cmin and/or cmax keys. Colour format must be
            hex or rgb (array) or a named colour recognised by
            matplotlib. Default: viridis_r.
        cmin : float, optional
            colour scale minimum. Default: display 99 % data.
        cmax : float, optional
            colour scale maximum. Default: display 99.9 % data.
        cscale : str, optional
            override colour scale (linear/ log). Default: None.
        unoccupied : str, optional
            if the colour variable is occuption, values below 1 are
            coloured in this colour. Id set to None, or cmin is set,
            this feature is turned off. Default: grey.

        workers : int, optional
            number of workers for paralellised section. Default: {workers}.
        verbose : bool, optional
            Write actual temperature used if applicable.
            Default: False.

        xmarkkwargs : dict, optional
            keyword arguments for x markers passed to
            matplotlib.pyplot.axvline. Set color to None to turn off.
            Defaults are defined below, which are overridden by those in
            ``~/.config/tprc.yaml``, both of which are overridden by
            arguments passed to this function.
            Defaults:

            color:      black
            linewidth:  axis line width
            rasterized: False

        kwargs
            keyword arguments passed to matplotlib.pyplot.scatter.
            Defaults are defined below, which are overridden by those in
            ``~/.config/tprc.yaml``, both of which are overridden by
            arguments passed to this function.
            Defaults:

                marker:     .
                rasterized: True

    Returns
    -------

        colorbar
            colour bar for projected data.
    """

    from functools import partial
    from multiprocessing import Pool
    from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
    from pymatgen.io.vasp.inputs import Poscar
    from scipy.interpolate import interp1d

    # defaults

    defkwargs = {'marker':     '.',
                 'rasterized': True}

    if conf is None or 'alt_projected_dispersion_kwargs' not in conf or \
       conf['alt_projected_dispersion_kwargs'] is None:
        kwargs = {**defkwargs, **kwargs}
    else:
        kwargs = {**defkwargs, **conf['alt_projected_dispersion_kwargs'], **kwargs}

    # check inputs

    for name, value in zip(['main', 'log'],
                           [ main,   log]):
        assert isinstance(value, bool), '{} must be True or False'.format(name)

    # Phono3py data formatting

    tnames = tp.settings.to_tp()
    if quantity in tnames: quantity = tnames[quantity]
    if projected in tnames: projected = tnames[projected]
    if quantity == 'kappa': quantity = 'mode_kappa'
    if projected == 'kappa': projected = 'mode_kappa'
    qs = quantity if quantity == projected else [quantity, projected]

    data = tp.data.utilities.resolve(data, qs, temperature=temperature,
                                   direction=direction)
    if verbose and 'temperature' in data['meta']:
        print('Using {} {}.'.format(data['meta']['temperature'],
                                    data['meta']['units']['temperature']))
    y = data[quantity]
    c = data[projected]
    if bandmin is None:
        bandmin = 0
    else:
        bandmin = np.amax([0, bandmin])
    if bandmax is None:
        bandmax = len(y[0])
    else:
        bandmax = np.amin([len(y[0]), bandmax])
    y = np.array(y)[:, bandmin:bandmax]
    c = np.array(c)[:, bandmin:bandmax]
    qk = data['qpoint']

    # Phonopy data formatting

    qp = pdata['qpoint']
    x = pdata['x']

    # map Phono3py to Phonopy

    struct = Poscar.from_file(poscar).structure
    sg = SpacegroupAnalyzer(struct)
    symops = sg.get_point_group_operations(cartesian=False)
    geq = partial(get_equivalent_qpoint, np.array(qk), symops)
    with Pool(processes=workers) as pool:
        min_id = pool.map(geq, qp)
    y2 = y[min_id, :]
    c2 = c[min_id, :]

    x, indices = np.unique(x, return_index=True)
    y2 = np.array(y2)[indices]
    c2 = np.array(c2)[indices]
    if len(x) > 100:
        xi = [x[i] for i in range(len(x)) if i % smoothing == 0]
        y2 = [y2[i] for i in range(len(y2)) if i % smoothing == 0]
    else:
        xi = x
        y2 = y2
    if xi[-1] != x[-1]: y2.append(y2[-1]); xi.append(x[-1])

    # interpolate

    x2 = np.linspace(min(x), max(x), interpolate)
    yinterp = interp1d(xi, y2, kind='cubic', axis=0)
    y2 = np.abs(yinterp(x2))
    ysort = np.ravel(y2)
    ysort = ysort[ysort.argsort()]
    ymin = ysort[int(round(len(ysort)/100 - 1, 0))]
    ymax = ysort[int(round(len(ysort)*99.9/100 - 1, 0))]

    cinterp = interp1d(x, c2, kind='cubic', axis=0)
    c2 = np.abs(cinterp(x2))
    cmap = tp.plot.utilities.parse_colours(colour)
    cnorm, extend = tp.plot.utilities.colour_scale(c2, projected, cmap, cmin,
                                                   cmax, cscale, unoccupied)

    # plotting

    for n in range(len(y2[0])):
        line = ax.scatter(x2, y2[:,n], c=c2[:,n], cmap=cmap, norm=cnorm,
                          **kwargs)

    # axes formatting

    axlabels = tp.settings.labels()
    fig = ax.get_figure()
    if 'dos' in fig.__dict__ and fig.__dict__['dos']:
        # place colourbar outside dos
        cbar = plt.colorbar(line, extend=extend)
    else:
        cbar = plt.colorbar(line, ax=ax, extend=extend)
    cbar.set_alpha(1)
    cbar.set_label(axlabels[projected])
    tp.plot.utilities.set_locators(cbar.ax, y=cbar.ax.yaxis.get_scale())
    cbar._draw_all()

    if main:
        ax.set_ylim(ymin, ymax)
        formatting(ax, pdata, quantity, log=log, **xmarkkwargs)

    return cbar

@tp.docstring_replace(workers=str(workers))
def add_wideband(ax, kdata, pdata, temperature=300, bandmin=None, bandmax=None,
                 ymin=None, ymax=None, poscar='POSCAR', main=True, smoothing=5,
                 colour='viridis', workers=workers, xmarkkwargs={},
                 verbose=False, **kwargs):
    """Plots a phonon dispersion with broadened bands.

    Requires a POSCAR.

    Arguments
    ---------

        ax : axes
            axes to plot on.
        kdata : dict
            Phono3py-like data containing:

                qpoint : array-like
                    q-point locations.
                gamma : array-like
                    imaginary component of the self-energy.
                temperature : array-like
                    temperature.

        pdata : dict
            phonon dispersion data containing:

                q-point : array-like
                    qpoint locations.
                x : array-like
                    high-symmetry path.
                frequency : array-like
                    phonon frequencies.
                tick_position : array-like
                    x-tick positions.
                tick_label : array-like
                    x-tick labels.

        temperature : float, optional
            approximate temperature in K (finds closest). Default: 300.
        bandmin : int, optional
            zero-indexed minimum band index to plot. Default: None.
        bandmax : int, optional
            zero-indexed maximum band index to plot. Default: None.
        ymin : float, optional
            minimum y-value plotted.
        ymax : float, optional
            maximum y-value plotted.

        poscar : str, optional
            VASP POSCAR filepath. Default: POSCAR.

        main : bool, optional
            set axis ticks, label, limits. Default: True.
        smoothing : int, optional
            every n points to sample. Default: 5.

        colour : colormap or str or list, optional
            colourmap or colourmap name or max colour (fades to white)
            or min and max colours or dictionary with cmin and cmax
            keys. Colour format must be hex or rgb (array) or a named
            colour recognised by matplotlib. Default: viridis.

        workers : int, optional
            number of workers for paralellised section. Default: {workers}.
        verbose : bool, optional
            Write actual temperature used if applicable.
            Default: False.

        xmarkkwargs : dict, optional
            keyword arguments for x markers passed to
            matplotlib.pyplot.axvline. Set color to None to turn off.
            Defaults are defined below, which are overridden by those in
            ``~/.config/tprc.yaml``, both of which are overridden by
            arguments passed to this function.
            Defaults:

                color:      None
                linewidth:  axis line width
                rasterized: False

        kwargs
            keyword arguments passed to matplotlib.pyplot.pcolormesh.
            Defaults are defined below, which are overridden by those in
            ``~/.config/tprc.yaml``, both of which are overridden by
            arguments passed to this function.
            Defaults:

                rasterized: True
                shading:    auto

    Returns
    -------

        None
            adds plot directly to ax.
    """

    from functools import partial
    from multiprocessing import Pool
    from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
    from pymatgen.io.vasp.inputs import Poscar
    from scipy.interpolate import interp1d
    from tp.calculate import lorentzian

    # defaults

    defkwargs = {'rasterized': True,
                 'shading':    'auto'}

    if conf is None or 'wideband_kwargs' not in conf or \
       conf['wideband_kwargs'] is None:
        kwargs = {**defkwargs, **kwargs}
    else:
        kwargs = {**defkwargs, **conf['wideband_kwargs'], **kwargs}

    defxmarkkwargs = {'color': None}
    xmarkkwargs = {**defxmarkkwargs, **xmarkkwargs}

    # check inputs

    assert isinstance(main, bool), 'main must be True or False'

    # Phono3py data formatting

    kdata = tp.data.utilities.resolve(kdata, 'gamma', temperature=temperature)
    if verbose:
        print('Using {} {}.'.format(kdata['meta']['temperature'],
                                    kdata['meta']['units']['temperature']))
    c = np.array(kdata['gamma'])
    qk = kdata['qpoint']

    # data selection

    if bandmin is None:
        bandmin = 0
    else:
        bandmin = np.amax([0, bandmin])
    if bandmax is None:
        bandmax = len(pdata['frequency'][0])
    else:
        bandmax = np.amin([len(pdata['frequency'][0]), bandmax])

    c = c[:,bandmin:bandmax]

    # Phonopy data formatting

    qp = pdata['qpoint']
    x = pdata['x']

    qpi = [qp[i] for i in range(len(qp)) if i % smoothing == 0]
    xi = [x[i] for i in range(len(x)) if i % smoothing == 0]
    if xi[-1] != x[-1]: qpi.append(qp[-1]); xi.append(x[-1])

    # map Phono3py to Phonopy

    struct = Poscar.from_file(poscar).structure
    sg = SpacegroupAnalyzer(struct)
    symops = sg.get_point_group_operations(cartesian=False)
    geq = partial(get_equivalent_qpoint, np.array(qk), symops)
    with Pool(processes=workers) as pool:
        min_id = pool.map(geq, qpi)
    c2 = c[min_id, :]

    # interpolate

    x, indices = np.unique(x, return_index=True)
    f = np.array(pdata['frequency'])[indices,bandmin:bandmax]
    x2 = np.linspace(min(x), max(x), 2500)
    finterp = interp1d(x, f, kind='cubic', axis=0)
    f = finterp(x2)

    cinterp = interp1d(xi, c2, kind='cubic', axis=0)
    c2 = np.abs(cinterp(x2))
    fmax = np.amax(np.add(f, c2))
    fmin = np.amin(np.subtract(f, c2))
    margin = (fmax - fmin) * 0.05
    fmax = fmax + margin if ymax is None else ymax
    fmin = fmin - margin if ymin is None else ymin

    c2 = np.where(c2==0, np.nanmin(c2[np.nonzero(c2)]), c2)
    f2 = np.linspace(fmin, fmax, 2500)

    # broadening

    area = np.zeros((len(x2), len(f2)))
    for q in range(len(area)):
        for b in range(len(c2[q])):
            area[q] = np.add(area[q], lorentzian(f2, f[q][b], c2[q][b]))

    cnorm = mpl.colors.LogNorm(vmin=np.nanmin(area), vmax=np.nanmax(area))

    # colours
    # Tries colourmap name or colourmap object, then tries a single
    # colour as the max of a linear colourmap, then tries two colours as
    # the min and max values.

    try:
        try:
            cmap = mpl.cm.get_cmap(colour)
        except AttributeError:
            cmap = mpl.colormaps[colour]
    except ValueError:
        if isinstance(colour, mpl.colors.ListedColormap):
            cmap = colour
        elif isinstance(colour, str):
            cmap = tp.plot.colour.linear(colour)
        elif isinstance(colour, list):
            cmap = tp.plot.colour.linear(colour[1], colour[0])
        elif isinstance(colour, dict):
            cmap = tp.plot.colour.linear(**colour)
        else:
            raise Exception('colour must be a colourmap, colourmap '
                            'name, single #rrggbb max colour or min '
                            'and max #rrggbb colours, or a dictionary '
                            'with cmin and cmax keys.')

    # plotting

    ax.pcolormesh(x2, f2, np.transpose(area), cmap=cmap, norm=cnorm, **kwargs)

    # axes formatting

    if main:
        if round(np.amin(f), 1) == 0:
            ax.set_ylim(bottom=0, top=fmax)
        else:
            ax.set_ylim(bottom=fmin, top=fmax)
        formatting(ax, pdata, 'frequency', **xmarkkwargs)

    return

def get_equivalent_qpoint(qk, symops, qp, tol=1e-2):
    """Finds the closest phono3py qpoint to a phonopy qpoint.

    Arguments
    ---------

        qk : array-like
            all qpoints from the phono3py kappa file.
        symmops
            symmetry operations (e.g. from Pymatgen)
        qp : array-like
            single qpoint from the phonon dispersion.

        tol : float, optional
            tolerance. Default: 1e-2.

    Returns
    -------

        int
            nearest qpoint index.
    """

    from pymatgen.util.coord import pbc_diff

    # Equivalent qpoints

    points = np.dot(qp, [m.rotation_matrix for m in symops])

    # Remove duplicates

    rm_list = []
    for i in range(len(points) - 1):
        for j in range(i + 1, len(points)):
            if np.allclose(pbc_diff(points[i], points[j]), [0, 0, 0], tol):
                rm_list.append(i)
                break
    seq = np.delete(points, rm_list, axis=0)

    # Find nearest

    dists = [np.min(np.sum(np.power(k - seq, 2), axis=1)) for k in qk]

    return dists.index(np.min(dists))

def formatting(ax, data, yquantity='frequency', log=False, **kwargs):
    """Formats the axes of phonon plots.

    Arguments
    ---------

        ax : axes
            axes to format.
        pdata : dict
            phonon dispersion data containing:

                x : array-like
                    high-symmetry path.
                tick_position : array-like
                    x-tick positions.
                tick_label : array-like
                    x-tick labels.

        yquantity : str, optional
            y variable. Default: frequency.
        log : bool, optional
            log the y scale. Default: False.

        kwargs
            keyword arguments passed to matplotlib.pyplot.axvline.
            Set color to None to turn off.
            Defaults are defined below, which are overridden by those in
            ``~/.config/tprc.yaml``, both of which are overridden by
            arguments passed to this function.
            Defaults:

                color:      black
                linewidth:  axis line width
                rasterized: False

    Returns
    -------

        None
            formats ax directly.
    """

    # defaults

    defkwargs = {'color':      'black',
                 'linewidth':  ax.spines['bottom'].get_linewidth(),
                 'rasterized': False}

    if conf is None or 'xmarkkwargs' not in conf or \
       conf['xmarkkwargs'] is None:
        kwargs = {**defkwargs, **kwargs}
    else:
        kwargs = {**defkwargs, **conf['xmarkkwargs'], **kwargs}

    # lines

    if kwargs['color'] is not None:
        for d in data['tick_position']:
            ax.axvline(d, **kwargs)

    # formatting

    axlabels = tp.settings.labels()
    ax.set_xlabel(axlabels['wavevector'])
    ax.set_ylabel(axlabels[yquantity])

    ax.set_xticks(data['tick_position'])
    ax.set_xticklabels(data['tick_label'])
    ax.tick_params(axis='x', which='minor', top=False, bottom=False)
    ax.set_xlim(data['x'][0], data['x'][-1])
    plt.axhline(linestyle=':', linewidth=ax.spines['bottom'].get_linewidth())
    scale = 'log' if log else 'linear'
    tp.plot.utilities.set_locators(ax, y=scale)

    return

def tile_properties(properties, bandmin, bandmax):
    """Tiles properties for dispersion and alt_dispersion

    Allows for different colour formats and fills out arrays with the
    last element or selects a subset.

    Arguments
    ---------

        property : array-like or str
            array or string to tile.
        bandmin : int
            minimum band.
        bandmax : int
            maximum band.

    Returns
    -------

        tiled
            array.
    """

    bdiff = bandmax - bandmin
    if isinstance(properties, str) or properties is None:
        tiled = np.repeat(properties, bdiff)
    elif len(properties) == 1:
        tiled = np.tile(properties, (bdiff, 1))
    elif len(properties) == bdiff:
        tiled = properties
    elif len(properties) > bdiff:
        if len(properties) >= bandmax:
            tiled = properties[bandmin:bandmax]
        else:
            tiled = properties[:bdiff]
    elif len(properties) < bdiff:
        tiled = list(properties)
        while len(tiled) < bdiff:
            tiled.append(tiled[-1])

    return tiled
