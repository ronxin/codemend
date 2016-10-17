Includes annotations created manually to polish documentations. Mainly used by
`../polish.py`.

** About `stype_map.tsv` **
These sets of transformations are for the convenience of calculating statistics and making code suggestions.

Just be cautious that when collecting value statistics we need to prevent stuff like add_subplot's args to be populated into plt.gca()'s actual arguments (none).