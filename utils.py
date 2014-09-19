#!/usr/bin/env python
# File: utils.py

scene_abbr_dict = {
        'bar': 'bar',
        'cafe': 'cafe',
        'elevator': 'elev',
        'library': 'lib',
        'office': 'offi', 
        'subwaystation': 'subw'
}
def abbreviate_names(names, abbrdict):
    abbrs = []
    for name in names:
        abbrs.append(abbrdict[name])
    return abbrs

